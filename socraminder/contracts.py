"""Teacher annotations, causal SFT inputs, and program-controlled CBT stages."""
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
import hashlib
import json
import math
import re

STAGES = ('situation', 'automatic_thought', 'emotion', 'behavior', 'evidence_for',
          'evidence_against', 'alternative_thought', 'action_plan')
DIMENSIONS = ('grounding', 'socratic', 'empathy', 'progress', 'coherence')


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def next_stage(stage):
    index = STAGES.index(stage)
    return STAGES[index + 1] if index + 1 < len(STAGES) else None


def validate_bundle(bundle, stage, turn_count):
    if set(bundle) != {'stage', 'new_simulated_details', 'turns'} or bundle['stage'] != stage:
        raise ValueError('Teacher stage bundle has unexpected keys or stage')
    if not isinstance(bundle['new_simulated_details'], list) or not all(
            isinstance(v, str) for v in bundle['new_simulated_details']):
        raise ValueError('Expected a list of simulated-detail annotations')
    turns = bundle['turns']
    if not isinstance(turns, list) or len(turns) != turn_count:
        raise ValueError(f'Expected exactly {turn_count} therapy turns')
    for index, turn in enumerate(turns):
        required = {'patient_response', 'patient_understanding', 'question_plan', 'state_update',
                    'response', 'stage_status', 'patient_passed'}
        if set(turn) != required:
            raise ValueError(f'Unexpected teacher turn keys at turn {index + 1}')
        for key in ('patient_response', 'response'):
            if not isinstance(turn[key], str) or not turn[key].strip():
                raise ValueError(f'Missing {key}')
        understanding = turn['patient_understanding']
        if not isinstance(understanding, dict) or set(understanding) != {
                'patient_anchor', 'current_belief_or_emotion', 'information_gap'}:
            raise ValueError('Invalid patient-understanding annotation')
        if not all(isinstance(v, str) and v.strip() for v in understanding.values()):
            raise ValueError('Empty patient-understanding field')
        if understanding['patient_anchor'] not in turn['patient_response']:
            raise ValueError('Patient anchor is not an exact substring of the current patient utterance')
        plan = turn['question_plan']
        if not isinstance(plan, dict) or set(plan) != {'question_strategy', 'question_target'}:
            raise ValueError('Invalid question-plan annotation')
        if not isinstance(plan['question_strategy'], str) or not re.fullmatch(
                '[a-z][a-z0-9_]*', plan['question_strategy']):
            raise ValueError('Question strategy must be a snake-case name')
        if not isinstance(plan['question_target'], str) or not plan['question_target'].strip():
            raise ValueError('Empty question target')
        update = turn['state_update']
        if not isinstance(update, dict) or not set(update) <= {stage}:
            raise ValueError('State update can update only the current stage')
        if any(v is not None and (not isinstance(v, str) or not v.strip()) for v in update.values()):
            raise ValueError('State values must be nonempty strings or null')
        if type(turn['patient_passed']) is not bool:
            raise ValueError('patient_passed must be boolean')
        final = index == turn_count - 1
        if turn['patient_passed'] and not final:
            raise ValueError('An explicit patient pass must transition immediately, not stay')
        expected = ('complete' if stage == STAGES[-1] else 'advance') if final else 'stay'
        if turn['stage_status'] != expected:
            raise ValueError(f'Turn {index + 1} must use {expected}')
        questions = turn['response'].count('?') + turn['response'].count('？')
        if expected == 'complete':
            if questions > 1:
                raise ValueError('Closing reply contains multiple questions')
        elif questions != 1:
            raise ValueError('Therapist reply must contain one core question')
        if final and not turn['patient_passed'] and not update.get(stage):
            raise ValueError('Stage transition requires patient-grounded content or an explicit pass')
        if turn['patient_passed'] and update.get(stage) is not None:
            raise ValueError('A skipped stage must preserve its null value')
    return bundle


def validate_quality(value, threshold=0.75):
    if set(value) != {'scores', 'hard_failure', 'feedback'}:
        raise ValueError('Invalid quality-judge fields')
    scores = value['scores']
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise ValueError('Invalid quality dimensions')
    if any(type(v) is not int or not 0 <= v <= 4 for v in scores.values()):
        raise ValueError('Quality scores must be integers in [0,4]')
    if type(value['hard_failure']) is not bool or not isinstance(value['feedback'], str):
        raise ValueError('Invalid quality judgment types')
    normalized = {key: scores[key] / 4 for key in DIMENSIONS}
    score = sum(normalized.values()) / len(normalized)
    accepted = (not value['hard_failure'] and score >= threshold
                and all(scores[k] >= 3 for k in ('grounding', 'socratic', 'coherence'))
                and all(scores[k] >= 2 for k in ('empathy', 'progress')))
    return {**value, 'normalized_scores': normalized, 'normalized_mean': score, 'accepted': accepted}


def render_target(turn):
    return ('<state_update>' + json.dumps(turn['state_update'], ensure_ascii=False) + '</state_update>\n'
            '<response>' + turn['response'] + '</response>\n'
            '<stage_status>' + turn['stage_status'] + '</stage_status>')


def build_stage_samples(profile, stage, bundle, state, history, knowledge, system):
    """Construct each turn before applying its update; future text never enters inputs."""
    state = copy.deepcopy(state)
    history = copy.deepcopy(history)
    samples = []
    for index, turn in enumerate(bundle['turns']):
        payload = {'patient_record': profile['patient_record'], 'therapy_state': copy.deepcopy(state),
                   'current_stage': stage, 'stage_knowledge': knowledge,
                   'following_stage': next_stage(stage), 'recent_dialogue': copy.deepcopy(history[-8:]),
                   'latest_patient_response': turn['patient_response']}
        samples.append({'patient_id': profile['patient_id'], 'split': profile['split'],
                        'session_id': f"edaic_{profile['patient_id']}_cbt1", 'stage': stage,
                        'turn_index': index + 1, 'synthetic_dialogue': True,
                        'messages': [{'role': 'system', 'content': system},
                                     {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
                                     {'role': 'assistant', 'content': render_target(turn)}]})
        state.update(turn['state_update'])
        history.extend([{'role': 'user', 'content': turn['patient_response']},
                        {'role': 'assistant', 'content': turn['response']}])
    return samples, state, history
