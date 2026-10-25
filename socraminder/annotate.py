#!/usr/bin/env python3
"""Build one continuous eight-stage teacher CBT session per official patient."""
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

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import time

from project_paths import DATA_ROOT, OUTPUT_ROOT, load_config

from contracts import (STAGES, build_stage_samples, next_stage, stable_hash,
                       validate_bundle, validate_quality)
from teacher_api import APIBlocked, TeacherAPI, atomic_json
from schemas import bundle_schema, quality_schema, handoff_schema

ROOT = Path(__file__).resolve().parent
DATA_DIR = DATA_ROOT / "socraminder"
OUTPUT_DIR = OUTPUT_ROOT / "socraminder"
STYLES = ('specific but initially incomplete', 'initially vague, then more concrete',
          'uncertain and reflective', 'brief and practical', 'ambivalent about a belief',
          'asks for clarification before answering', 'hesitant but willing to explore')
HANDOFF_GOALS = {
    'situation': 'Ask what thought or meaning went through the patient\'s mind in this specific situation.',
    'automatic_thought': 'Ask which emotion the patient noticed when having this automatic thought.',
    'emotion': 'Ask what the patient did or felt an urge to do in response to this emotion.',
    'behavior': 'Ask what the patient actually observed that seemed to support the target automatic thought.',
    'evidence_for': 'Ask about an observation or experience that does not fully fit or limits that automatic thought.',
    'evidence_against': 'Invite the patient to formulate their own balanced interpretation using both sides of the evidence.',
    'alternative_thought': 'Invite the patient to choose one small feasible action to try in response to the situation.',
    'action_plan': 'Briefly consolidate the patient\'s agreed plan and close the session; no further question is required.'}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def protocol(config):
    files = [ROOT / 'annotate.py', ROOT / 'contracts.py', ROOT / 'teacher_api.py', ROOT / 'schemas.py',
             *(ROOT / 'prompts').glob('*.txt'), *(ROOT / 'knowledge').glob('*.md')]
    identities = {str(path.relative_to(ROOT)): stable_hash(path.read_text()) for path in files}
    identity = {'annotation': config['annotation'], 'api_model': config['api']['model'],
                'api_base_url': config['api']['base_url'], 'sources': identities}
    return stable_hash(identity)


def export_sft(profiles, protocol_id, require_complete=True):
    knowledge = {stage: (ROOT / f'knowledge/{stage}.md').read_text() for stage in STAGES}
    student_system = (ROOT / 'prompts/student_system.txt').read_text().strip()
    grouped = {split: [] for split in ('train', 'dev', 'test')}
    patient_counts = Counter()
    states, skipped, qualities, status_counts = [], Counter(), [], Counter()
    incomplete = []
    for profile in profiles:
        session = DATA_DIR / 'annotations' / profile['split'] / profile['patient_id']
        state, history, rows = dict.fromkeys(STAGES), [], []
        finished = True
        for stage in STAGES:
            path = session / f'{stage}.json'
            if not path.exists():
                finished = False
                break
            saved = json.loads(path.read_text())
            if saved['protocol_id'] != protocol_id or saved['profile_sha256'] != profile['profile_sha256']:
                raise ValueError('Annotation cache identity mismatch during export')
            if saved['starting_state_hash'] != stable_hash(state) or saved['starting_history_hash'] != stable_hash(history):
                raise ValueError('Cached stage is detached from its preceding dialogue')
            validate_bundle(saved['bundle'], stage, saved['turn_count'])
            if not saved['quality']['accepted']:
                raise ValueError('Rejected teacher data found in accepted annotations')
            samples, state, history = build_stage_samples(profile, stage, saved['bundle'], state,
                                                         history, knowledge[stage], student_system)
            rows.extend(samples)
            qualities.append(saved['quality']['normalized_mean'])
            for turn in saved['bundle']['turns']:
                status_counts[turn['stage_status']] += 1
                if turn['patient_passed']:
                    skipped[stage] += 1
        if not finished:
            incomplete.append({'patient_id': profile['patient_id'], 'split': profile['split']})
            continue
        if rows[-1]['stage'] != STAGES[-1] or history[-1]['role'] != 'assistant':
            raise ValueError('Session did not finish the eight-stage curriculum')
        grouped[profile['split']].extend(rows)
        patient_counts[profile['split']] += 1
        states.append({'patient_id': profile['patient_id'], 'split': profile['split'],
                       'therapy_state': state, 'turns': len(rows), 'synthetic_dialogue': True})
    if require_complete and incomplete:
        raise ValueError(f'{len(incomplete)} patient sessions are incomplete; SFT export is blocked')
    for split, rows in grouped.items():
        write_jsonl(DATA_DIR / f'sft/{split}.jsonl', rows)
    write_jsonl(DATA_DIR / 'session_states.jsonl', states)
    report = {'status': 'complete' if not incomplete else 'partial', 'protocol_id': protocol_id,
              'patients': dict(patient_counts), 'samples': {key: len(value) for key, value in grouped.items()},
              'total_samples': sum(map(len, grouped.values())), 'incomplete': incomplete,
              'stage_status_counts': dict(status_counts), 'explicit_pass_counts': dict(skipped),
              'mean_normalized_quality': sum(qualities) / len(qualities) if qualities else None,
              'assistant_target_blocks': ['state_update', 'response', 'stage_status'],
              'auxiliary_annotations_in_assistant_targets': False,
              'split_policy': 'Official patient split preserved; only train is optimized'}
    atomic_json(DATA_DIR / 'sft/manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/annotation.json')
    parser.add_argument('--limit-patients', type=int)
    parser.add_argument('--stages', type=int, default=8, help='Pilot stage count; full export requires eight')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--retry-round', type=int, default=0,
                        help='Controlled fresh revisions for failed stages; accepted stages are reused')
    args = parser.parse_args()
    config = load_config(args.config)
    run = OUTPUT_DIR / 'run'
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'annotation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not 1 <= args.stages <= 8 or args.retry_round < 0 or (args.limit_patients is not None and args.limit_patients < 1):
        raise ValueError('Invalid pilot limits')
    profiles = [row for split in ('train', 'dev', 'test')
                for row in read_jsonl(DATA_DIR / f'profiles/{split}.jsonl')]
    all_profiles = profiles
    if args.limit_patients:
        profiles = profiles[:args.limit_patients]
    protocol_id = protocol(config)
    manifest_path = run / 'annotation_protocol.json'
    identity = {'protocol_id': protocol_id, 'profile_manifest_hash': stable_hash(
                    json.loads((DATA_DIR / 'profiles/manifest.json').read_text())),
                'model': config['api']['model'], 'stages': list(STAGES),
                'sessions_per_patient': 1, 'turns_per_stage': [2, 3]}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError('Annotation sources, prompts, or data changed; use a new run directory')
    atomic_json(manifest_path, identity)
    if args.export_only:
        print(json.dumps(export_sft(all_profiles, protocol_id)), flush=True)
        return
    teacher = (ROOT / 'prompts/teacher_system.txt').read_text().strip()
    judge = (ROOT / 'prompts/quality_judge_system.txt').read_text().strip()
    handoff_prompt = (ROOT / 'prompts/handoff_system.txt').read_text().strip()
    knowledge = {stage: (ROOT / f'knowledge/{stage}.md').read_text() for stage in STAGES}
    api = TeacherAPI(config['api'], OUTPUT_DIR)
    started = time.monotonic()
    progress_lock = threading.Lock()
    done = 0
    errors = []
    stage_done = 0

    def status(state, **extra):
        value = {'state': state, 'pid': os.getpid(), 'time_utc': datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds': time.monotonic() - started, 'patients_completed': done,
                 'patients_requested': len(profiles), 'stages_completed': stage_done,
                 'stages_requested': len(profiles) * args.stages, 'api_calls_this_process': api.calls,
                 'api_cache_hits_this_process': api.cache_hits, 'api_usage_this_process': dict(api.usage),
                 'failed_patients': len(errors), **extra}
        atomic_json(run / 'annotation_status.json', value)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)

    def annotate_patient(profile):
        nonlocal stage_done
        state, history = dict.fromkeys(STAGES), []
        session = DATA_DIR / 'annotations' / profile['split'] / profile['patient_id']
        for stage in STAGES[:args.stages]:
            if api.blocked.is_set():
                raise APIBlocked('Teacher API unavailable')
            number = int(stable_hash({'patient': profile['patient_id'], 'stage': stage,
                                      'seed': config['annotation']['seed']})[:8], 16)
            count = 2 + number % 2
            base_payload = {'patient_record': profile['patient_record'], 'therapy_state': state,
                            'current_stage': stage, 'stage_knowledge': knowledge[stage],
                            'following_stage': next_stage(stage), 'recent_dialogue': history[-8:],
                            'requested_therapist_turns': count, 'patient_response_style': STYLES[number % len(STYLES)],
                            'final_turn_question_goal': HANDOFF_GOALS[stage]}
            context_hash = stable_hash(base_payload)
            path = session / f'{stage}.json'
            saved = None
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['protocol_id'] != protocol_id or saved['profile_sha256'] != profile['profile_sha256'] or saved['context_hash'] != context_hash:
                    raise ValueError('Existing patient-stage annotation has incompatible context')
                validate_bundle(saved['bundle'], stage, count)
                if not saved['quality']['accepted']:
                    raise ValueError('Existing stage failed quality selection')
            else:
                feedback = ''
                offset = args.retry_round * config['annotation']['stage_attempts']
                for revision in range(offset, offset + config['annotation']['stage_attempts']):
                    payload = {**base_payload, 'revision': revision, 'revision_feedback': feedback}
                    attempt_file = session / 'attempts' / f'{stage}-{revision}.json'
                    try:
                        bundle, generation_info = api.complete('stage_generation', teacher, payload,
                            temperature=config['annotation']['temperature'],
                            max_tokens=config['annotation']['max_generation_tokens'],
                            schema=bundle_schema(stage, count))
                        if generation_info['finish_reason'] == 'length':
                            raise ValueError('Teacher stage generation reached its output token limit')
                        validate_bundle(bundle, stage, count)
                        handoff_info = None
                        following = next_stage(stage)
                        if following is not None:
                            handoff_state, handoff_history = dict(state), list(history)
                            for index, turn in enumerate(bundle['turns']):
                                handoff_state.update(turn['state_update'])
                                if index + 1 < len(bundle['turns']):
                                    handoff_history.extend([{'role': 'user', 'content': turn['patient_response']},
                                                            {'role': 'assistant', 'content': turn['response']}])
                            handoff, handoff_info = api.complete('stage_handoff', handoff_prompt,
                                {'patient_record': profile['patient_record'], 'therapy_state': handoff_state,
                                 'next_stage': following, 'stage_knowledge': knowledge[following],
                                 'recent_dialogue': handoff_history[-8:],
                                 'latest_patient_response': bundle['turns'][-1]['patient_response'],
                                 'next_stage_goal': HANDOFF_GOALS[stage]},
                                temperature=0.4, max_tokens=512, schema=handoff_schema())
                            bundle['turns'][-1]['response'] = handoff['response']
                            bundle['turns'][-1]['question_plan'] = handoff['question_plan']
                            validate_bundle(bundle, stage, count)
                        judgment, judge_info = api.complete('stage_quality', judge,
                            {**base_payload, 'candidate': bundle}, temperature=0,
                            max_tokens=config['annotation']['max_judge_tokens'], schema=quality_schema())
                        quality = validate_quality(judgment, config['annotation']['quality_threshold'])
                        atomic_json(attempt_file, {'revision': revision, 'bundle': bundle,
                            'generation': generation_info, 'handoff': handoff_info,
                            'judge': judge_info, 'quality': quality})
                        if not quality['accepted']:
                            feedback = quality['feedback']
                            continue
                        saved = {'patient_id': profile['patient_id'], 'split': profile['split'],
                            'stage': stage, 'turn_count': count, 'protocol_id': protocol_id,
                            'profile_sha256': profile['profile_sha256'], 'context_hash': context_hash,
                            'starting_state_hash': stable_hash(state), 'starting_history_hash': stable_hash(history),
                            'bundle': bundle, 'quality': quality, 'generation': generation_info, 'judge': judge_info,
                            'handoff': handoff_info,
                            'accepted_revision': revision, 'synthetic_dialogue': True}
                        atomic_json(path, saved)
                        break
                    except (ValueError, KeyError, TypeError) as error:
                        feedback = str(error)
                        atomic_json(attempt_file, {'revision': revision, 'validation_error': feedback})
                if saved is None:
                    raise ValueError(f'{stage}: all bounded annotation attempts failed; last feedback: {feedback}')
            for turn in saved['bundle']['turns']:
                state.update(turn['state_update'])
                history.extend([{'role': 'user', 'content': turn['patient_response']},
                                {'role': 'assistant', 'content': turn['response']}])
            with progress_lock:
                stage_done += 1
                status('annotating', latest_patient_id=profile['patient_id'], latest_split=profile['split'], latest_stage=stage)
        return profile['patient_id']

    status('annotating')
    with ThreadPoolExecutor(max_workers=config['annotation']['workers']) as pool:
        futures = {pool.submit(annotate_patient, profile): profile for profile in profiles}
        for future in as_completed(futures):
            profile = futures[future]
            with progress_lock:
                try:
                    future.result()
                    done += 1
                except Exception as error:
                    errors.append({'patient_id': profile['patient_id'], 'split': profile['split'],
                                   'error_type': type(error).__name__, 'error': str(error)})
                    atomic_json(run / 'annotation_errors.json', errors)
                    if isinstance(error, APIBlocked):
                        for pending in futures:
                            pending.cancel()
                status('annotating' if not api.blocked.is_set() else 'api_blocked')
    if errors:
        status('failed', errors=errors)
        raise RuntimeError(f'{len(errors)} patient sessions failed; saved stages can be resumed')
    pilot = args.limit_patients is not None or args.stages < 8
    if pilot:
        status('pilot_complete', protocol_id=protocol_id)
    else:
        report = export_sft(all_profiles, protocol_id)
        status('complete', dataset=report, protocol_id=protocol_id)


if __name__ == '__main__':
    main()
