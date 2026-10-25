"""Exact ace_native_peak_v2 function from the completed MusicCaps reevaluation."""
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

def normalize_waveform(audio, target_db=-1.0):
    """Native decode peak scaling followed by native normalize_audio (NumPy).

    The gain is shared by both stereo channels; samples are never hard-clipped.
    Exactly silent or near-silent waveforms retain their original amplitudes.
    """
    audio = np.asarray(audio)
    if audio.dtype != np.float32 or audio.ndim not in (1, 2) or not audio.size:
        raise ValueError("Expected a nonempty FP32 waveform with 1 or 2 dimensions")
    if not np.isfinite(audio).all() or not np.isfinite(target_db) or target_db > 0:
        raise ValueError("Waveform/target must be finite and target_db must be <= 0")
    # Matches pred_wavs /= peak.clamp(min=1.0), without modifying the input.
    decoded = audio / np.maximum(np.max(np.abs(audio)), np.float32(1.0))
    peak = np.max(np.abs(decoded))
    if peak < 1e-6:
        return decoded
    # Matches acestep.audio_utils.normalize_audio's NumPy branch literally.
    gain = 10 ** (target_db / 20.0) / peak
    return (decoded.copy() * gain).astype(np.float32, copy=False)

