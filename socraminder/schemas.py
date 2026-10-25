"""Strict provider schemas keep annotation field names stable."""
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


from contracts import DIMENSIONS


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def bundle_schema(stage, count):
    text = {'type': 'string'}
    understanding = object_schema({key: text for key in
        ('patient_anchor', 'current_belief_or_emotion', 'information_gap')})
    plan = object_schema({'question_strategy': text, 'question_target': text})
    update = {'anyOf': [object_schema({}),
        object_schema({stage: {'type': ['string', 'null']}})]}
    turn = object_schema({'patient_response': text, 'patient_understanding': understanding,
        'question_plan': plan, 'state_update': update, 'response': text,
        'stage_status': {'type': 'string', 'enum': ['stay', 'advance', 'complete']},
        'patient_passed': {'type': 'boolean'}})
    return object_schema({'stage': {'type': 'string', 'enum': [stage]},
        'new_simulated_details': {'type': 'array', 'items': text},
        'turns': {'type': 'array', 'items': turn, 'minItems': count, 'maxItems': count}})


def quality_schema():
    return object_schema({'scores': object_schema({key: {'type': 'integer', 'minimum': 0, 'maximum': 4}
        for key in DIMENSIONS}), 'hard_failure': {'type': 'boolean'}, 'feedback': {'type': 'string'}})


def handoff_schema():
    text = {'type': 'string'}
    return object_schema({'response': text, 'question_plan': object_schema({
        'question_strategy': text, 'question_target': text})})
