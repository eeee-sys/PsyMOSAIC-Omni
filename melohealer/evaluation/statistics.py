"""Corpus statistics for the six MusicCaps experiments.

The PANNs inputs must be pre-sigmoid classifier logits, not clipwise sigmoid
probabilities. The primary KL is the categorical KL(reference || generated)
after softmax. ``KL_sigmoid`` reproduces AudioLDM_eval's supplementary,
unnormalized sigmoid expression, including its 1e-6 denominator offset; that
expression is not a complete Bernoulli KL and can be slightly negative.

IS follows AudioLDM_eval's seeded shuffle and integer-boundary split means.
Stable log-softmax replaces the upstream KL denominator offset for the primary
KL, so identical distributions have zero KL without an epsilon-induced bias.
FD and FAD use sample covariance (N - 1). FAD inputs contain every VGGish
patch, without averaging the patches within each audio clip first.

Primary implementation references:
https://github.com/haoheliu/audioldm_eval/blob/main/audioldm_eval/metrics/kl.py
https://github.com/haoheliu/audioldm_eval/blob/main/audioldm_eval/metrics/isc.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _CorePath
for _core_root in _CorePath(__file__).resolve().parents:
    if (_core_root / "project_paths.py").is_file():
        if str(_core_root) not in _sys.path:
            _sys.path.insert(0, str(_core_root))
        break
from project_paths import (load_config, expand_paths, get_api_key, LORA_ROOT,
                           MODEL_ROOT, OUTPUT_ROOT, RAW_DATA_ROOT, VENDOR_ROOT,
                           CODE_ROOT as PROJECT_CODE_ROOT, DATA_ROOT as PROJECT_DATA_ROOT)


import numpy as np
from scipy import linalg
from scipy.special import log_expit, log_softmax


def _matrix(value, name: str, *, min_rows: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < min_rows or array.shape[1] < 1:
        raise ValueError(f"{name} must be a 2D array with at least {min_rows} rows")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _vector(value, name: str, *, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a nonempty 1D array")
    if size is not None and array.size != size:
        raise ValueError(f"{name} has {array.size} rows, expected {size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def feature_statistics(features) -> tuple[np.ndarray, np.ndarray]:
    """Return the FP64 feature mean and unbiased sample covariance."""
    values = _matrix(features, "features", min_rows=2)
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    return mean, (covariance + covariance.T) * 0.5


def _psd_eigenvalues(values: np.ndarray, name: str) -> np.ndarray:
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = 1e-9 * scale
    if float(values.min()) < -tolerance:
        raise ValueError(f"{name} is not positive semidefinite")
    return np.maximum(values, 0.0)


def frechet_distance_from_statistics(mean_reference, covariance_reference,
                                     mean_generated, covariance_generated) -> float:
    """Compute Gaussian Frechet distance with symmetric PSD square roots.

    Unlike a square root of the nonsymmetric covariance product, the symmetric
    sandwich has real, nonnegative eigenvalues, including for singular sample
    covariances. Tiny negative eigenvalues caused by roundoff are clipped.
    """
    mean_ref = _vector(mean_reference, "mean_reference")
    mean_gen = _vector(mean_generated, "mean_generated", size=mean_ref.size)
    cov_ref = _matrix(covariance_reference, "covariance_reference")
    cov_gen = _matrix(covariance_generated, "covariance_generated")
    expected = (mean_ref.size, mean_ref.size)
    if cov_ref.shape != expected or cov_gen.shape != expected:
        raise ValueError("Covariances must be square and match the mean dimension")
    if not np.allclose(cov_ref, cov_ref.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance_reference must be symmetric")
    if not np.allclose(cov_gen, cov_gen.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance_generated must be symmetric")
    cov_ref = (cov_ref + cov_ref.T) * 0.5
    cov_gen = (cov_gen + cov_gen.T) * 0.5
    difference = mean_ref - mean_gen
    mean_term = float(difference @ difference)
    if np.array_equal(cov_ref, cov_gen):
        eigenvalues = linalg.eigvalsh(cov_ref, check_finite=False)
        _psd_eigenvalues(eigenvalues, "covariance_reference")
        return mean_term

    values_ref, vectors_ref = linalg.eigh(cov_ref, check_finite=False, driver="evd")
    values_ref = _psd_eigenvalues(values_ref, "covariance_reference")
    values_gen = linalg.eigvalsh(cov_gen, check_finite=False)
    _psd_eigenvalues(values_gen, "covariance_generated")
    square_root_ref = (vectors_ref * np.sqrt(values_ref)) @ vectors_ref.T
    sandwich = square_root_ref @ cov_gen @ square_root_ref
    sandwich = (sandwich + sandwich.T) * 0.5
    product_values = linalg.eigvalsh(sandwich, check_finite=False)
    product_values = _psd_eigenvalues(product_values, "covariance sandwich")
    trace_ref = float(np.trace(cov_ref))
    trace_gen = float(np.trace(cov_gen))
    trace_square_root = float(np.sqrt(product_values).sum())
    distance = mean_term + trace_ref + trace_gen - 2.0 * trace_square_root
    tolerance = 1e-6 * max(1.0, mean_term + abs(trace_ref) + abs(trace_gen))
    if distance < -tolerance:
        raise FloatingPointError(f"Frechet distance is materially negative: {distance}")
    return float(max(0.0, distance))


def frechet_distance(reference_features, generated_features) -> float:
    """Compute corpus FD/FAD; feature row counts may differ."""
    reference = _matrix(reference_features, "reference_features", min_rows=2)
    generated = _matrix(generated_features, "generated_features", min_rows=2)
    if reference.shape[1] != generated.shape[1]:
        raise ValueError("Reference and generated feature dimensions differ")
    return frechet_distance_from_statistics(
        *feature_statistics(reference), *feature_statistics(generated)
    )


def paired_kl(reference_logits, generated_logits) -> dict[str, float]:
    """Return mean paired KL(ref || gen), and the legacy sigmoid supplement."""
    reference = _matrix(reference_logits, "reference_logits")
    generated = _matrix(generated_logits, "generated_logits")
    if reference.shape != generated.shape:
        raise ValueError("Paired reference and generated logits must have identical shape")
    log_reference = log_softmax(reference, axis=1)
    log_generated = log_softmax(generated, axis=1)
    primary = np.mean(np.sum(np.exp(log_reference) *
                             (log_reference - log_generated), axis=1))
    log_sigmoid_reference = log_expit(reference)
    log_sigmoid_generated = log_expit(generated)
    legacy_denominator = np.logaddexp(log_sigmoid_generated, np.log(1e-6))
    sigmoid = np.mean(np.sum(np.exp(log_sigmoid_reference) *
                             (log_sigmoid_reference - legacy_denominator), axis=1))
    return {"KL": float(max(0.0, primary)), "KL_sigmoid": float(sigmoid)}


def inception_score(generated_logits, *, splits: int = 10,
                    seed: int = 2020) -> dict[str, float]:
    """Return mean and population SD of the seeded split Inception Scores."""
    logits = _matrix(generated_logits, "generated_logits")
    if not isinstance(splits, (int, np.integer)) or not 1 <= splits <= logits.shape[0]:
        raise ValueError("splits must be an integer between 1 and the sample count")
    permutation = np.random.RandomState(seed).permutation(logits.shape[0])
    log_probability = log_softmax(logits[permutation], axis=1)
    probability = np.exp(log_probability)
    count = logits.shape[0]
    scores = []
    for index in range(splits):
        start, stop = index * count // splits, (index + 1) * count // splits
        p_chunk = probability[start:stop]
        log_chunk = log_probability[start:stop]
        marginal = p_chunk.mean(axis=0, keepdims=True)
        # Underflowed all-zero classes contribute zero rather than 0 * -inf.
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p_chunk > 0.0,
                             p_chunk * (log_chunk - np.log(marginal)), 0.0)
        scores.append(np.exp(terms.sum(axis=1).mean()))
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Inception Score contains a nonfinite value")
    return {"IS": float(scores.mean()), "IS_std": float(scores.std(ddof=0))}


def calculate_metrics(reference_panns_embeddings, generated_panns_embeddings,
                      reference_logits, generated_logits,
                      reference_vggish_patches, generated_vggish_patches,
                      pc, pq, align, align_prompt) -> dict[str, float]:
    """Calculate all Table 1 metrics from already extracted feature arrays.

    Expected feature widths are 2048 (PANNs), 527 (logits), and 128 (VGGish).
    Widths are inferred so the same functions can be numerically tested with
    small arrays. Reference and generated PANNs rows must be paired in the same
    manifest order. VGGish patch counts may differ. ``align`` always uses the
    original MusicCaps caption; ``align_prompt`` uses the actual input prompt.
    """
    reference_panns = _matrix(reference_panns_embeddings,
                             "reference_panns_embeddings", min_rows=2)
    generated_panns = _matrix(generated_panns_embeddings,
                             "generated_panns_embeddings", min_rows=2)
    reference_classes = _matrix(reference_logits, "reference_logits")
    generated_classes = _matrix(generated_logits, "generated_logits")
    count = generated_panns.shape[0]
    if any(values.shape[0] != count for values in
           (reference_panns, reference_classes, generated_classes)):
        raise ValueError("PANNs embeddings and paired logits must share one sample count")
    vectors = {"PC": _vector(pc, "pc", size=count),
               "PQ": _vector(pq, "pq", size=count),
               "Align": _vector(align, "align", size=count),
               "Align_prompt": _vector(align_prompt, "align_prompt", size=count)}
    result = paired_kl(reference_classes, generated_classes)
    result.update(inception_score(generated_classes))
    result["FD"] = frechet_distance(reference_panns, generated_panns)
    result["FAD"] = frechet_distance(reference_vggish_patches, generated_vggish_patches)
    result.update({name: float(values.mean()) for name, values in vectors.items()})
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("A reported metric is nonfinite")
    return result
