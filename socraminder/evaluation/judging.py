"""Combine one shared evidence-based success decision with separate quality ratings."""
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


import copy

from control_validation import validate_closure_review


QUALITY_SCALES = {
    'ctr': (0, 10),
    'empathy': (1, 5),
    'coherence': (1, 5),
    'helpfulness': (1, 5),
    'safety': (1, 5),
}


def validate_quality(value):
    """Validate native integer scales without changing or renormalizing any score."""
    expected = set(QUALITY_SCALES) | {'metric_evidence', 'overall_comment'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('Quality judgment must contain exactly the seven declared fields')
    for name, (minimum, maximum) in QUALITY_SCALES.items():
        if type(value[name]) is not int or not minimum <= value[name] <= maximum:
            raise ValueError(f'Quality score {name} must be an integer in [{minimum},{maximum}]')
    evidence = value['metric_evidence']
    if not isinstance(evidence, dict) or set(evidence) != set(QUALITY_SCALES):
        raise ValueError('Quality judgment requires evidence for all five scores')
    if any(not isinstance(text, str) or not text.strip() for text in evidence.values()):
        raise ValueError('Quality evidence must be nonempty text')
    if not isinstance(value['overall_comment'], str) or not value['overall_comment'].strip():
        raise ValueError('Quality judgment requires a nonempty overall comment')
    return copy.deepcopy(value)


def combine_judgment(closure_review, quality, dialogue):
    """Combine one shared success review with anonymous quality ratings."""
    review = validate_closure_review(copy.deepcopy(closure_review), dialogue)
    quality = validate_quality(quality)
    success = review['ready']
    if success:
        insight, action = review['insight'], review['action_plan']
        success_evidence = {
            'insight_quote': insight['quote'],
            'insight_round': insight['round'],
            'action_plan_quote': action['quote'],
            'action_plan_round': action['round'],
        }
        first_success_round = max(insight['round'], action['round'])
    else:
        success_evidence = {
            'insight_quote': None,
            'insight_round': None,
            'action_plan_quote': None,
            'action_plan_round': None,
        }
        first_success_round = None
    return {
        'success': success,
        'first_success_round': first_success_round,
        **quality,
        'success_evidence': success_evidence,
    }
