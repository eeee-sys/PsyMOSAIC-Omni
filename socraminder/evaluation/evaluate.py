"""Resumable therapist evaluation with one shared, anonymous judging rubric."""
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid

from project_paths import OUTPUT_ROOT, load_config

from api_client import APICache, APIBlocked, APIError, APITruncated
from control_validation import explicit_patient_pass, transition_supported as state_transition_supported, validate_closure_review
from judging import validate_quality, combine_judgment

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = OUTPUT_ROOT / "socraminder" / "evaluation"
AGENTS = ('ours', 'omni_base', 'qwen37', 'cbt')
STAGES = ('situation', 'automatic_thought', 'emotion', 'behavior', 'evidence_for',
          'evidence_against', 'alternative_thought', 'action_plan')
ACTIVE_WORKER = None
STOP = threading.Event()
REVIEW_LOCK = threading.Lock()


class ResponseValidationError(ValueError):
    """A model response exhausted bounded evidence/schema validation retries."""
    def __init__(self, message, metadata=None):
        super().__init__(message)
        self.metadata = metadata or {}


class ProtocolIntegrityError(ValueError):
    """Stored results belong to a different experiment and cannot be skipped."""


def runtime_gpu(config):
    value = os.environ.get('PSYMOSAIC_EVAL_GPU', str(config.get('gpu', 0)))
    if not re.fullmatch(r'\d+', value):
        raise ValueError('PSYMOSAIC_EVAL_GPU must be one physical GPU index')
    return int(value)


def dialogue_round_limit(config, agent, namespace='outputs'):
    if namespace == 'smoke':
        return config.get('smoke_rounds', 3)
    return config['max_rounds'] if agent == 'ours' else min(
        config['max_rounds'], config.get('baseline_max_rounds', 40))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_prompt(name):
    return (ROOT / 'prompts' / name).read_text()


def patient_reply_valid(value):
    if not isinstance(value, dict) or set(value) != {'response', 'session_end_requested', 'closure_ready'}:
        raise ValueError('Invalid patient reply fields')
    if not isinstance(value['response'], str) or not value['response'].strip():
        raise ValueError('Empty patient response')
    if any(type(value[key]) is not bool for key in ('session_end_requested', 'closure_ready')):
        raise ValueError('Patient flags must be boolean')
    return value


def normalize_quote(text):
    return ' '.join(text.split())


def judgment_valid(value, dialogue):
    fields = {'success', 'first_success_round', 'ctr', 'empathy', 'coherence', 'helpfulness',
              'safety', 'success_evidence', 'metric_evidence', 'overall_comment'}
    if not isinstance(value, dict) or set(value) != fields or type(value['success']) is not bool:
        raise ValueError('Invalid judgment fields')
    for key, low, high in (('ctr', 0, 10), ('empathy', 1, 5), ('coherence', 1, 5),
                           ('helpfulness', 1, 5), ('safety', 1, 5)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError('Invalid judgment scale: ' + key)
    evidence = value['success_evidence']
    if not isinstance(evidence, dict) or set(evidence) != {
            'insight_quote', 'insight_round', 'action_plan_quote', 'action_plan_round'}:
        raise ValueError('Invalid success evidence')
    if not isinstance(value['metric_evidence'], dict) or set(value['metric_evidence']) != {
            'ctr', 'empathy', 'coherence', 'helpfulness', 'safety'}:
        raise ValueError('Invalid metric evidence fields')
    if any(not isinstance(text, str) or not text.strip()
           for text in value['metric_evidence'].values()):
        raise ValueError('Metric explanations must be nonempty')
    if not isinstance(value['overall_comment'], str):
        raise ValueError('Invalid overall comment')
    if value['success']:
        rounds = []
        for prefix in ('insight', 'action_plan'):
            index, quote = evidence[prefix + '_round'], evidence[prefix + '_quote']
            if type(index) is not int or not 1 <= index <= len(dialogue):
                raise ValueError('Success evidence has invalid patient turn')
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError('Missing patient evidence quote')
            if normalize_quote(quote) not in normalize_quote(dialogue[index - 1]['patient_response']):
                raise ValueError('Success evidence is not an exact patient quote')
            rounds.append(index)
        if type(value['first_success_round']) is not int or value['first_success_round'] != max(rounds):
            raise ValueError('First success round must match both evidence components')
    elif value['first_success_round'] is not None or any(item is not None for item in evidence.values()):
        raise ValueError('An unsuccessful session has no success round or success evidence')
    if not value['overall_comment'].strip():
        raise ValueError('Overall comment must be nonempty')
    return value


def parse_ours(text, stage):
    pattern = (r'\A\s*<state_update>(.*?)</state_update>\s*'
               r'<response>(.*?)</response>\s*'
               r'<stage_status>\s*(stay|advance|complete)\s*</stage_status>\s*\Z')
    match = re.fullmatch(pattern, text, re.DOTALL)
    if match is None:
        raise ValueError('Invalid three-block therapist response')
    update = json.loads(match.group(1))
    if not isinstance(update, dict) or not set(update) <= {stage}:
        raise ValueError('State update must address only the current stage')
    if any(item is not None and (not isinstance(item, str) or not item.strip()) for item in update.values()):
        raise ValueError('Invalid therapy state value')
    response = match.group(2).strip()
    if not response or any(tag in response for tag in ('<state_update>', '<stage_status>')):
        raise ValueError('Invalid public therapist response')
    status = match.group(3)
    if status == 'complete' and stage != STAGES[-1] or status == 'advance' and stage == STAGES[-1]:
        raise ValueError('Stage status is incompatible with the current stage')
    return update, response, status


def explicit_no_supporting_evidence(text):
    """Recognize the evidence-for knowledge's explicit no-example completion."""
    if not isinstance(text, str):
        return False
    text = text.replace('\u2019', "'")
    pattern = (r'\bno\s+(?:(?:specific|recent|concrete|actual|real|supporting|clear)\s+){0,3}'
               r'(evidence|examples?|observations?|interactions?)\b')
    for match in re.finditer(pattern, text, re.IGNORECASE):
        prefix = re.split(r'[.!?;\n]', text[:match.start()])[-1]
        if re.search(r"\b(?:not\s+saying|did\s+not\s+say|didn't\s+say|don't\s+think|"
                     r'do\s+not\s+think|rather\s+than|not\s+true\s+that|whether|what\s+if)\b',
                     prefix, re.IGNORECASE):
            continue
        suffix = text[match.end():]
        if re.match(r'\s+against\b', suffix, re.IGNORECASE):
            continue
        # An absent interaction alone can describe isolation. Require an
        # explicit supporting relation and an assertion, not a hypothetical.
        if '?' in re.split(r'[.!;\n]', suffix, maxsplit=1)[0]:
            continue
        supporting = re.search(r'\bsupporting\s+', match.group(0), re.IGNORECASE)
        linked = re.match(r'\s+(?:(?:that|which)\s+)?(?:support|prov|show|confirm|suggest)\w*\b',
                          suffix, re.IGNORECASE)
        asserted = re.search(r'\b(?:there\s+(?:is|are)|I\s+(?:have|found|can\s+find))\s+$',
                             prefix, re.IGNORECASE)
        if not supporting and not (asserted and linked):
            continue
        return True
    return False


def transition_supported(state, stage, update, proposed_status, latest_patient):
    if state_transition_supported(state, stage, update, proposed_status, latest_patient):
        return True
    return (stage == 'evidence_for' and proposed_status == 'advance'
            and isinstance(update, dict) and set(update).issubset({stage})
            and update.get(stage) is None and explicit_no_supporting_evidence(latest_patient))


def apply_state(state, stage, update, proposed_status, latest_patient):
    state = copy.deepcopy(state)
    state.update(update)
    explicit_pass = explicit_patient_pass(latest_patient)
    supported = transition_supported(state, stage, {}, proposed_status, latest_patient)
    transition = proposed_status in ('advance', 'complete') and supported
    next_index = min(STAGES.index(stage) + 1, len(STAGES) - 1) if transition else STAGES.index(stage)
    control = {
        'proposed_status': proposed_status, 'transition_applied': transition,
        'explicit_no_supporting_evidence': stage == 'evidence_for' and explicit_no_supporting_evidence(latest_patient),
        'explicit_patient_pass': explicit_pass, 'stage_before': stage, 'stage_after': STAGES[next_index]}
    if transition and not state_transition_supported(state, stage, {}, proposed_status, latest_patient):
        control.update(completion_basis='explicit_no_supporting_evidence',
                       patient_evidence_quote=latest_patient,
                       raw_current_stage_value=state.get(stage))
    return state, STAGES[next_index], control


def stop(signum, frame):
    STOP.set()
    if ACTIVE_WORKER is not None:
        ACTIVE_WORKER.close()
    raise KeyboardInterrupt('User or system stopped evaluation')


class LocalWorker:
    def __init__(self, config, agent, protocol_id):
        self.config, self.agent, self.protocol_id = config, agent, protocol_id
        python = config['cbt_python'] if agent == 'cbt' else config['python']
        model = config['cbt_model'] if agent == 'cbt' else config['base_model']
        command = [python, '-u', str(ROOT / 'worker.py'), '--backend', agent,
                   '--base-model', model, '--device', 'cuda:0',
                   '--source-project', config['source_project']]
        if agent == 'ours':
            command += ['--adapter', config['adapter']]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(runtime_gpu(config)), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        logs = OUTPUT_DIR / 'run'
        logs.mkdir(exist_ok=True)
        self.log = (logs / f'worker_{agent}.log').open('ab', buffering=0)
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            start_new_session=True, close_fds=True)
        self.sequence = 0
        ready = self.receive(600)
        if ready.get('event') != 'ready':
            self.close()
            raise RuntimeError('Worker did not signal readiness')
        self.ready = ready
        atomic_json(OUTPUT_DIR / 'run' / f'worker_{agent}_ready.json', ready)

    def receive(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise RuntimeError('Local model worker timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Local model worker exited; inspect its worker log')
        return json.loads(line)

    def request(self, payload):
        if STOP.is_set():
            raise KeyboardInterrupt()
        self.sequence += 1
        request_hash = digest({'protocol_id': self.protocol_id, 'agent': self.agent, 'request': payload})
        cache = OUTPUT_DIR / 'local_cache' / self.agent / (request_hash + '.json')
        if cache.exists():
            saved = read_json(cache)
            return {**saved['response'], 'id': self.sequence, 'cached': True,
                    'local_request_hash': request_hash}
        original_payload = copy.deepcopy(payload)
        payload = {'id': self.sequence, **payload}
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        result = self.receive(self.config.get('local_generation_timeout_seconds', 300))
        if result.get('id') != self.sequence:
            raise RuntimeError('Worker response identity mismatch')
        if 'error' in result:
            raise WorkerError(result['error'])
        atomic_json(cache, {'protocol_id': self.protocol_id, 'request': original_payload,
                            'response': result})
        return {**result, 'cached': False, 'local_request_hash': request_hash}

    def close(self):
        if getattr(self, 'process', None) is not None and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({'op': 'shutdown'}) + '\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        if getattr(self, 'log', None):
            self.log.close()


class WorkerError(RuntimeError):
    def __init__(self, error):
        self.error = error
        super().__init__('Local model error: ' + error.get('code', 'unknown'))


class Evaluation:
    def __init__(self, config, mode='production'):
        self.config = config
        self.mode = mode
        self.started = time.monotonic()
        self.cases = [json.loads(line) for line in Path(config['cases']).read_text().splitlines() if line.strip()]
        if len(self.cases) != 56 or len({row['patient_id'] for row in self.cases}) != 56:
            raise ValueError('Expected all 56 distinct official test patients')
        assets = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((ROOT / 'prompts').glob('*')) if path.is_file()}
        for name in ('evaluate.py', 'worker.py', 'api_client.py', 'prepare_cases.py',
                     'control_validation.py', 'judging.py'):
            assets[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        adapter = Path(config['adapter'])
        self.source = Path(config['source_project'])
        for relative in ('prompts/student_system.txt', 'training/model.py',
                         *(f'knowledge/{stage}.md' for stage in STAGES)):
            assets['source/' + relative] = hashlib.sha256((self.source / relative).read_bytes()).hexdigest()
        self.protocol_id = digest({'config': config, 'assets': assets, 'mode': mode,
            'cases_sha256': hashlib.sha256(Path(config['cases']).read_bytes()).hexdigest(),
            'adapter_config_sha256': hashlib.sha256((adapter / 'adapter_config.json').read_bytes()).hexdigest(),
            'adapter_sha256': hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()})
        protocol_path = OUTPUT_DIR / 'run' / ('smoke_protocol.json' if mode == 'smoke' else 'evaluation_protocol.json')
        protocol = {'protocol_id': self.protocol_id, 'config': config, 'asset_sha256': assets,
                    'patient_count': 56, 'dialogues_per_patient': len(config.get('active_agents', AGENTS)),
                    'active_agents': config.get('active_agents', list(AGENTS)), 'mode': mode,
                    'judge_schemes': ['fair_shared'],
                    'judge_is_anonymous': True,
                    'at_definition': 'Mean actual completed rounds among successful sessions',
                    'patient_record_policy': {'ours': 'full', 'omni_base': 'full',
                                              'qwen37': 'full', 'cbt': 'record_update_only'},
                    'original_sft_test_dialogues_used': False}
        protocol.update(version='shared_anonymous_evaluation',
                        closure_policy='Verify simulator closure candidates using actual patient insight and adopted action with timing',
                        ours_normal_completion='action_plan stage complete proposal plus independently verified patient insight and action',
                        current_execution_scope='All configured agents',
                        shared_success_judge='One common evidence-based success decision per session',
                        unsupported_stage_transition='Retry before sending a public therapist response',
                        model_repetition_policy='Preserve natural repeated responses and score them')
        if protocol_path.exists() and read_json(protocol_path).get('protocol_id') != self.protocol_id:
            raise ProtocolIntegrityError('Cached evaluation uses different inputs, weights, or prompts.')
        protocol['effective_max_rounds'] = {
            agent: dialogue_round_limit(config, agent) for agent in AGENTS}
        atomic_json(protocol_path, protocol)
        self.api = APICache(config['api'], OUTPUT_DIR, self.protocol_id)
        self.ours_system = (self.source / 'prompts/student_system.txt').read_text()
        self.knowledge = {stage: (self.source / 'knowledge' / (stage + '.md')).read_text() for stage in STAGES}
        self.lock = threading.Lock()

    def status(self, state, **extra):
        with self.lock:
            value = {'state': state, 'pid': os.getpid(), 'protocol_id': self.protocol_id,
                     'physical_gpu': runtime_gpu(self.config),
                     'time_utc': datetime.now(timezone.utc).isoformat(),
                     'elapsed_seconds': round(time.monotonic() - self.started, 1), **extra}
            filename = 'smoke_status.json' if self.mode == 'smoke' else 'evaluation_status.json'
            atomic_json(OUTPUT_DIR / 'run' / filename, value)
            print(json.dumps(value, ensure_ascii=False), flush=True)

    def classify_invalid_json_exhaustion(self, error):
        """Identify saved malformed model text without hiding transport failures."""
        if isinstance(error, APIBlocked):
            return None
        match = re.fullmatch(r'Qwen request failed after (\d+) attempts; '
                             r'inspect api_cache artifacts for request ([a-f0-9]{64})', str(error))
        if match is None:
            return None
        attempts, request_hash = int(match.group(1)), match.group(2)
        if attempts < 1 or attempts != self.config.get('api', {}).get('max_attempts', 4):
            return None
        records = []
        for path in (OUTPUT_DIR / 'api_cache').glob(request_hash + '.attempt-*.error.json'):
            try:
                value = read_json(path)
            except (OSError, ValueError):
                return None
            records.append((value, path))
        records.sort(key=lambda item: item[0].get('time_utc', ''))
        latest = records[-attempts:]
        if len(latest) != attempts or {value.get('attempt') for value, _ in latest} != set(range(1, attempts + 1)):
            return None
        requests = [value.get('request') for value, _ in latest]
        if not all(isinstance(request, dict) for request in requests) or len({digest(request) for request in requests}) != 1:
            return None
        for value, _ in latest:
            if (value.get('request_hash') != request_hash or value.get('protocol_id') != self.protocol_id
                    or value.get('error_type') != 'JSONDecodeError'
                    or value.get('status_code') is not None):
                return None
            try:
                envelope = json.loads(value['response_raw'])
                content = envelope['choices'][0]['message']['content']
            except (ValueError, KeyError, IndexError, TypeError):
                return None
            if not isinstance(content, str) or value.get('response_text', content) != content:
                return None
            try:
                json.loads(content)
            except json.JSONDecodeError:
                pass
            else:
                return None
        return {'request_hash': request_hash, 'attempts': attempts,
                'failure_kind': 'malformed_json_model_response',
                'raw_error_artifacts': [str(path) for _, path in latest]}

    def json_call(self, kind, model, system, payload, validator, max_tokens):
        payload = copy.deepcopy(payload)
        retry_cycle = uuid.uuid4().hex
        for attempt in range(3):
            result, meta = None, {}
            try:
                result, meta = self.api.complete(model,
                    [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                    temperature=0.6 if kind.startswith('patient') else 0.0,
                    max_tokens=max_tokens, kind=kind, json_mode=True)
                if meta['finish_reason'] == 'length':
                    raise APITruncated(meta)
                return validator(result), meta
            except APITruncated:
                if attempt == 2:
                    raise
                max_tokens = min(max_tokens * 2, 4096)
            except APIError as error:
                metadata = self.classify_invalid_json_exhaustion(error)
                if metadata is None:
                    raise
                raise ResponseValidationError('Qwen returned malformed JSON after bounded retries', metadata) from error
            except (ValueError, KeyError, TypeError) as error:
                if attempt == 2:
                    raise ResponseValidationError(str(error), meta) from error
                payload['format_validation_feedback'] = {
                    'attempt': attempt + 1, 'error': str(error),
                    'retry_cycle': retry_cycle,
                    'invalid_response_request_hash': meta.get('request_hash'),
                    'instruction': 'Correct only the invalid fields using actual quoted dialogue evidence. '
                                   'Do not invent a quote, action, or future intention.'}
                if isinstance(result, dict) and 'ready' in result and 'action_plan' in result:
                    feedback = payload['format_validation_feedback']
                    feedback['instruction'] += (
                        ' Only patient_response text can support evidence. A time proposed by the '
                        'therapist is not patient evidence. If a required component is unsupported, '
                        'use null for that component and ready=false. Do not copy the unsupported phrase.')
                    plan = result.get('action_plan')
                    rows = payload.get('public_dialogue', [])
                    index = plan.get('round') if isinstance(plan, dict) else None
                    if type(index) is int and 1 <= index <= len(rows):
                        feedback['referenced_patient_response'] = rows[index - 1]['patient_response']
        raise RuntimeError('Bounded JSON validation attempts exhausted')

    def patient(self, case, opening, dialogue, therapist=None, opening_response=None):
        payload = {'mode': 'opening' if opening else 'reply',
                   'patient_record': case['patient_record'],
                   'public_dialogue': dialogue}
        if not opening:
            payload.update(therapist_response=therapist, round_index=len(dialogue) + 1,
                           opening_patient_response=opening_response)
        instruction = load_prompt('patient_opening_user.txt' if opening else 'patient_turn_user.txt')
        payload['instruction'] = instruction
        return self.json_call('patient_opening' if opening else 'patient_reply',
            self.config['simulator_model'], load_prompt('patient_simulator_system.txt'),
            payload, patient_reply_valid, 512)

    def record_for(self, case, agent):
        if agent != 'cbt':
            return case['patient_record']
        return {stage: {'record_update': value['record_update'], 'text_evidence': []}
                for stage, value in case['patient_record'].items()}

    def baseline_messages(self, case, agent, opening, dialogue):
        system = load_prompt('baseline_therapist_system.txt')
        system += '\n\nPatient background:\n' + json.dumps(self.record_for(case, agent),
                                                        ensure_ascii=False, separators=(',', ':'))
        messages = [{'role': 'system', 'content': system}]
        recent = [{'role': 'user', 'content': opening}]
        for turn in dialogue:
            recent += [{'role': 'assistant', 'content': turn['therapist_response']},
                       {'role': 'user', 'content': turn['patient_response']}]
        messages += recent[-9:]
        if messages[1]['role'] == 'assistant':
            messages.pop(1)
        return messages

    def ours_reply(self, worker, case, state, stage, opening, dialogue):
        history = []
        for index, turn in enumerate(dialogue):
            prior_patient = opening if index == 0 else dialogue[index - 1]['patient_response']
            history.extend([{'role': 'user', 'content': prior_patient},
                            {'role': 'assistant', 'content': turn['therapist_response']}])
        latest = dialogue[-1]['patient_response'] if dialogue else opening
        following = STAGES[STAGES.index(stage) + 1] if stage != STAGES[-1] else None
        base_payload = {'patient_record': case['patient_record'], 'therapy_state': state,
                        'current_stage': stage, 'stage_knowledge': self.knowledge[stage],
                        'following_stage': following, 'latest_patient_response': latest}
        last_error = None
        rejected_attempts = []
        for recent_count in (8, 6, 4, 2, 0):
            payload = {**base_payload, 'recent_dialogue': history[-recent_count:] if recent_count else []}
            messages = [{'role': 'system', 'content': self.ours_system},
                        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
            for attempt in range(3):
                result = None
                try:
                    request = {'messages': messages, 'seed': self.config['seed'] + len(dialogue),
                               'temperature': 0.0, 'top_p': .9,
                               'max_new_tokens': 1024 if attempt == 0 else 2048,
                               'history_policy': 'error'}
                    result = worker.request(request)
                    update, response, proposed = parse_ours(result['text'], stage)
                    if result['finish_reason'] == 'length':
                        raise ValueError('Structured output was truncated')
                    if not transition_supported(state, stage, update, proposed, latest):
                        raise ValueError('Cannot advance or complete without a nonempty current-stage update '
                                         'or an explicit patient choice to pass the current topic')
                    return update, response, proposed, {
                        **result, 'recent_dialogue_messages': len(payload['recent_dialogue']),
                        'format_retry': attempt, 'rejected_attempts': rejected_attempts}
                except WorkerError as error:
                    last_error = error
                    if 'context_overflow' in error.error.get('code', ''):
                        break
                    raise
                except (ValueError, json.JSONDecodeError) as error:
                    last_error = error
                    rejected_attempts.append({'error': str(error), 'raw_output': result})
                    if attempt < 2:
                        messages[0]['content'] += ('\nOutput validation feedback: ' + str(error) +
                            '. Return exactly state_update, response, stage_status blocks. '
                            'Update only the current stage from the patient response. '
                            'If the current stage is not supported, use stay and ask a question for that stage.')
                    else:
                        raise
        raise last_error or RuntimeError('Unable to fit the complete clinical profile')

    def therapist(self, agent, worker, case, state, stage, opening, dialogue):
        if agent == 'ours':
            return self.ours_reply(worker, case, state, stage, opening, dialogue)
        messages = self.baseline_messages(case, agent, opening, dialogue)
        budget = self.config.get('baseline_max_new_tokens', 256)
        if agent in ('cbt', 'omni_base'):
            result = worker.request({'messages': messages, 'seed': self.config['seed'] + len(dialogue),
                'temperature': 0.0, 'top_p': .9, 'max_new_tokens': budget,
                'history_policy': 'drop_oldest_pairs'})
            response = result['text']
        else:
            response, result = self.api.complete(self.config['baseline_api_model'], messages,
                temperature=0.0, max_tokens=budget, kind='baseline_therapist', json_mode=False)
        result['completion_attempts'] = [{'budget': budget, 'finish_reason': result['finish_reason'],
            'request_hash': result.get('request_hash', result.get('local_request_hash'))}]
        result['truncated_at_budget'] = result['finish_reason'] == 'length'
        if not isinstance(response, str) or not response.strip():
            raise ValueError('Empty therapist reply')
        return {}, response.strip(), None, result

    def closure_review(self, dialogue):
        payload = {'public_dialogue': dialogue, 'completed_rounds': len(dialogue)}
        try:
            return self.json_call('closure_verification', self.config['closure_verifier_model'],
                load_prompt('closure_verifier_system.txt'), payload,
                lambda value: validate_closure_review(value, dialogue), 1536)
        except ResponseValidationError as error:
            # An unverified stopping candidate cannot terminate the conversation.
            # Its raw attempts remain in the API cache; it is not a therapy score.
            review = {'ready': False, 'insight': None, 'action_plan': None,
                      'reason': 'Closure evidence remains unverified; continue the conversation.'}
            meta = {'validation_failed': True, 'validation_error': str(error),
                    'last_response': error.metadata, 'decision': 'unverified_continue'}
            print(json.dumps({'event': 'closure_review_unverified',
                              'completed_rounds': len(dialogue), 'error': str(error)}), flush=True)
            return review, meta

    def run_session(self, agent, worker, case, namespace='outputs'):
        round_limit = dialogue_round_limit(self.config, agent, namespace)
        path = OUTPUT_DIR / namespace / 'dialogues' / agent / (case['patient_id'] + '.json')
        if path.exists():
            session = read_json(path)
            if session['protocol_id'] != self.protocol_id:
                raise ProtocolIntegrityError('Session protocol mismatch')
            if session.get('max_rounds') != round_limit:
                raise ProtocolIntegrityError('Saved session round budget differs from the active budget')
            if session['state'] == 'complete':
                return session
        else:
            opening, opening_meta = self.patient(case, True, [])
            session = {'state': 'running', 'protocol_id': self.protocol_id,
                'agent': agent, 'patient_id': case['patient_id'], 'split': 'test',
                'opening_patient_response': opening['response'],
                'opening_api': opening_meta, 'public_dialogue': [],
                'private_turn_records': [], 'therapy_state': {stage: None for stage in STAGES},
                'current_stage': STAGES[0], 'completed_rounds': 0, 'max_rounds': round_limit,
                'therapist_input_policy': 'record_update_only' if agent == 'cbt' else 'full_record'}
            atomic_json(path, session)
        while len(session['public_dialogue']) < round_limit:
            if STOP.is_set():
                raise KeyboardInterrupt()
            dialogue = session['public_dialogue']
            if 'pending_therapist' not in session:
                update, spoken, proposed, metadata = self.therapist(agent, worker, case,
                    session['therapy_state'], session['current_stage'],
                    session['opening_patient_response'], dialogue)
                session['pending_therapist'] = {'update': update, 'response': spoken,
                    'proposed_status': proposed, 'metadata': metadata}
                atomic_json(path, session)
            pending = session['pending_therapist']
            update, spoken, proposed, metadata = (pending['update'], pending['response'],
                                                   pending['proposed_status'], pending['metadata'])
            reply, patient_meta = self.patient(case, False, dialogue, spoken,
                                               session['opening_patient_response'])
            previous_patient = dialogue[-1]['patient_response'] if dialogue else session['opening_patient_response']
            state, stage, transition = apply_state(session['therapy_state'], session['current_stage'],
                                                  update, proposed, previous_patient) if agent == 'ours' else (
                session['therapy_state'], session['current_stage'], {})
            index = len(dialogue) + 1
            session['public_dialogue'].append({'round_index': index,
                'therapist_response': spoken, 'patient_response': reply['response']})
            session['private_turn_records'].append({'round_index': index,
                'therapist_output': metadata, 'state_update': update,
                'stage_transition': transition, 'patient_control': {
                    'session_end_requested': reply['session_end_requested'],
                    'closure_ready': reply['closure_ready']}, 'patient_api': patient_meta})
            session.update(therapy_state=state, current_stage=stage, completed_rounds=index)
            model_completed_stages = (agent == 'ours' and proposed == 'complete'
                                      and transition.get('stage_before') == STAGES[-1])
            closure = {'candidate': reply['closure_ready'], 'accepted': False,
                       'review': None, 'api': None,
                       'model_completed_stages': model_completed_stages,
                       'deferred_for_model_completion': False}
            if (reply['closure_ready'] or model_completed_stages) and not reply['session_end_requested']:
                review, review_meta = self.closure_review(session['public_dialogue'])
                closure.update(accepted=review['ready'], review=review, api=review_meta)
            closure['deferred_for_model_completion'] = (
                agent == 'ours' and closure['accepted'] and not model_completed_stages)
            session['private_turn_records'][-1]['closure_verification'] = closure
            completed_with_pass = model_completed_stages and transition.get('explicit_patient_pass', False)
            normal_end = (closure['accepted'] and (agent != 'ours' or model_completed_stages)) or completed_with_pass
            if reply['session_end_requested'] or normal_end:
                session.update(state='complete', end_reason='patient_requested_end' if
                    reply['session_end_requested'] else 'patient_completed_with_action_plan_pass' if
                    completed_with_pass else 'verified_complete_eight_stage_therapy' if
                    agent == 'ours' else 'verified_patient_ready_for_closure')
            elif index == round_limit:
                session.update(state='complete', end_reason='max_rounds')
            del session['pending_therapist']
            atomic_json(path, session)
            print(json.dumps({'event': 'dialogue_turn', 'agent': agent, 'patient_id': case['patient_id'],
                              'round': index, 'session_state': session['state']}, ensure_ascii=False), flush=True)
            if session['state'] == 'complete':
                break
        return session

    def run_session_or_defer(self, agent, worker, case, namespace='outputs'):
        """Keep a failing condition's checkpoint and move on without retrying it."""
        patient_id = case['patient_id']
        path = OUTPUT_DIR / namespace / 'dialogues' / agent / (patient_id + '.json')
        deferred = OUTPUT_DIR / namespace / 'deferred_samples' / agent / (patient_id + '.json')
        for artifact in (deferred, path):
            if artifact.exists():
                saved = read_json(artifact)
                if (saved.get('protocol_id') != self.protocol_id
                        or saved.get('agent') != agent or saved.get('patient_id') != patient_id):
                    raise ProtocolIntegrityError('Deferred sample or session identity mismatch')
        if deferred.exists():
            print(json.dumps({'event': 'deferred_sample_skipped', 'agent': agent,
                              'patient_id': patient_id}), flush=True)
            return None
        try:
            return self.run_session(agent, worker, case, namespace)
        except ProtocolIntegrityError:
            raise
        except (ValueError, APITruncated, WorkerError) as error:
            if (isinstance(error, WorkerError) and error.error.get('code')
                    not in ('context_overflow', 'protected_context_overflow')):
                raise
            # Reading a broken checkpoint is an integrity error, not a model failure.
            session = read_json(path) if path.exists() else {}
            atomic_json(deferred, {'state': 'deferred', 'protocol_id': self.protocol_id,
                'agent': agent, 'patient_id': patient_id,
                'completed_rounds': session.get('completed_rounds', 0),
                'current_stage': session.get('current_stage'),
                'session_path': str(path),
                'session_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(),
                'time_utc': datetime.now(timezone.utc).isoformat(),
                'action': 'Skip this agent/patient condition on resume; retain raw outputs for later repair.'})
            print(json.dumps({'event': 'sample_deferred', 'agent': agent,
                              'patient_id': patient_id, 'error_type': type(error).__name__,
                              'error': str(error)}), flush=True)
            return None

    def success_review(self, case, session, namespace='outputs'):
        path = OUTPUT_DIR / namespace / 'success_reviews' / session['agent'] / (case['patient_id'] + '.json')
        dialogue_hash = digest({'opening': session['opening_patient_response'],
                                'dialogue': session['public_dialogue']})
        with REVIEW_LOCK:
            if path.exists():
                saved = read_json(path)
                if saved['protocol_id'] != self.protocol_id or saved['dialogue_hash'] != dialogue_hash:
                    raise ProtocolIntegrityError('Shared success review identity changed')
                validate_closure_review(saved['review'], session['public_dialogue'])
                return saved
            payload = {'opening_patient_response': session['opening_patient_response'],
                       'public_dialogue': session['public_dialogue'],
                       'completed_rounds': session['completed_rounds']}
            review, meta = self.json_call('judge_shared_success', self.config['success_judge_model'],
                load_prompt('judge_success_system.txt'), payload,
                lambda value: validate_closure_review(value, session['public_dialogue']), 1536)
            saved = {'protocol_id': self.protocol_id, 'agent': session['agent'],
                     'patient_id': case['patient_id'], 'dialogue_hash': dialogue_hash,
                     'review': review, 'api': meta}
            atomic_json(path, saved)
            return saved

    def judge(self, case, session, scheme, namespace='outputs'):
        agent = session['agent']
        if scheme != 'fair_shared':
            raise ValueError('Only shared anonymous judging is supported')
        filename = 'judge_quality_shared_system.txt'
        path = OUTPUT_DIR / namespace / 'judgments' / scheme / agent / (case['patient_id'] + '.json')
        dialogue_hash = digest({'opening': session['opening_patient_response'],
                                'dialogue': session['public_dialogue']})
        if path.exists():
            result = read_json(path)
            if result['protocol_id'] != self.protocol_id or result['dialogue_hash'] != dialogue_hash:
                raise ProtocolIntegrityError('Judgment source identity changed')
            return result
        payload = {'patient_record': case['patient_record'],
                   'opening_patient_response': session['opening_patient_response'],
                   'public_dialogue': session['public_dialogue'],
                   'completed_rounds': session['completed_rounds']}
        success = self.success_review(case, session, namespace)
        quality, meta = self.json_call('judge_quality_shared', self.config['judge_model'],
            load_prompt(filename), payload, validate_quality, 2048)
        value = combine_judgment(success['review'], quality, session['public_dialogue'])
        judgment_valid(value, session['public_dialogue'])
        result = {'protocol_id': self.protocol_id, 'scheme': scheme,
                  'is_fair_comparison': True, 'agent': agent,
                  'patient_id': case['patient_id'], 'dialogue_hash': dialogue_hash,
                  'completed_rounds': session['completed_rounds'], 'judgment': value, 'api': meta,
                  'shared_success_review_hash': digest(success['review']),
                  'success_api': success['api']}
        result['max_rounds'] = session['max_rounds']
        atomic_json(path, result)
        print(json.dumps({'event': 'judgment_complete', 'scheme': scheme, 'agent': agent,
                          'patient_id': case['patient_id']}, ensure_ascii=False), flush=True)
        return result

    def judge_or_defer(self, case, session, scheme, namespace='outputs', retry_pending=True):
        path = OUTPUT_DIR / namespace / 'pending_judgments' / scheme / session['agent'] / (case['patient_id'] + '.json')
        if path.exists() and not retry_pending:
            pending = read_json(path)
            if pending.get('protocol_id') != self.protocol_id:
                raise ProtocolIntegrityError('Pending judgment protocol mismatch')
            return None
        try:
            result = self.judge(case, session, scheme, namespace)
        except (ResponseValidationError, APITruncated) as error:
            atomic_json(path, {'state': 'pending_unscored', 'protocol_id': self.protocol_id,
                'agent': session['agent'], 'patient_id': case['patient_id'], 'scheme': scheme,
                'error': str(error), 'last_response': error.metadata,
                'time_utc': datetime.now(timezone.utc).isoformat(),
                'action': 'Retry evidence extraction; do not assign a clinical failure or zero score.'})
            print(json.dumps({'event': 'judgment_deferred', 'agent': session['agent'],
                              'patient_id': case['patient_id'], 'scheme': scheme,
                              'error': str(error)}), flush=True)
            return None
        path.unlink(missing_ok=True)
        return result

    def write_coverage(self, cases, agents, namespace='outputs'):
        rows, deferred = [], []
        for agent in agents:
            row = {'agent': agent, 'expected_patients': len(cases), 'completed_dialogues': 0,
                   'deferred_dialogues': 0, 'valid_judgments': {}, 'pending_judgments': {}}
            for case in cases:
                patient_id = case['patient_id']
                session_path = OUTPUT_DIR / namespace / 'dialogues' / agent / (patient_id + '.json')
                if session_path.exists() and read_json(session_path).get('state') == 'complete':
                    row['completed_dialogues'] += 1
                marker = OUTPUT_DIR / namespace / 'deferred_samples' / agent / (patient_id + '.json')
                if marker.exists():
                    row['deferred_dialogues'] += 1
                    deferred.append({'kind': 'dialogue', **read_json(marker)})
            for scheme in ('fair_shared',):
                row['valid_judgments'][scheme] = sum(
                    (OUTPUT_DIR / namespace / 'judgments' / scheme / agent / (case['patient_id'] + '.json')).exists()
                    for case in cases)
                pending_paths = [OUTPUT_DIR / namespace / 'pending_judgments' / scheme / agent / (case['patient_id'] + '.json')
                                 for case in cases]
                row['pending_judgments'][scheme] = sum(path.exists() for path in pending_paths)
                deferred.extend({'kind': 'judgment', **read_json(path)} for path in pending_paths if path.exists())
            rows.append(row)
        complete = all(row['completed_dialogues'] == len(cases) and not row['deferred_dialogues']
                       and all(count == len(cases) for count in row['valid_judgments'].values())
                       and not any(row['pending_judgments'].values()) for row in rows)
        coverage = {'state': 'complete' if complete else 'completed_with_deferred_samples',
                    'protocol_id': self.protocol_id, 'conditions': rows,
                    'full_cohort_metrics_ready': complete, 'deferred_items': len(deferred),
                    'missing_results_are_zero_scores': False}
        atomic_json(OUTPUT_DIR / namespace / 'results' / 'coverage.json', coverage)
        atomic_json(OUTPUT_DIR / namespace / 'results' / 'deferred_samples.json', deferred)
        return coverage

    def aggregate(self, namespace='outputs', expected_patients=56, agents=None):
        agents = tuple(agents or AGENTS)
        results = OUTPUT_DIR / namespace / 'results'
        all_rows = []
        for scheme in ('fair_shared',):
            rows = []
            for agent in agents:
                values = [read_json(path) for path in sorted(
                    (OUTPUT_DIR / namespace / 'judgments' / scheme / agent).glob('*.json'))]
                if len(values) != expected_patients:
                    raise ValueError(f'Incomplete judgments: {scheme}/{agent}')
                successes = [value for value in values if value['judgment']['success']]
                row = {'scheme': scheme, 'agent': agent, 'patients': len(values),
                       'max_dialogue_rounds': dialogue_round_limit(self.config, agent, namespace),
                       'successful_sessions': len(successes), 'SR': len(successes) / len(values),
                       'AT': statistics.mean(v['completed_rounds'] for v in successes) if successes else None}
                sessions = [read_json(path) for path in sorted(
                    (OUTPUT_DIR / namespace / 'dialogues' / agent).glob('*.json'))]
                row['truncated_therapist_turns'] = sum(
                    turn['therapist_output'].get('truncated_at_budget', False)
                    for session in sessions for turn in session['private_turn_records'])
                for output, source in (('CTR', 'ctr'), ('Empathy', 'empathy'), ('Coherence', 'coherence'),
                                       ('Helpfulness', 'helpfulness'), ('Safety', 'safety')):
                    scores = [v['judgment'][source] for v in values]
                    row[output] = statistics.mean(scores)
                    row[output + '_std'] = statistics.stdev(scores) if len(scores) > 1 else 0.0
                rows.append(row)
            atomic_json(results / (scheme + '_metrics.json'), rows)
            results.mkdir(parents=True, exist_ok=True)
            fields = ['scheme', 'agent', 'max_dialogue_rounds', 'patients', 'successful_sessions', 'SR', 'AT',
                      'CTR', 'Empathy', 'Coherence', 'Helpfulness', 'Safety', 'truncated_therapist_turns']
            with (results / (scheme + '_metrics.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            all_rows += rows
        atomic_json(results / 'all_metrics.json', all_rows)
        return all_rows


def main():
    global ACTIVE_WORKER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/eval.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--patient-limit', type=int)
    parser.add_argument('--agents', nargs='+', choices=AGENTS, default=AGENTS)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get('therapist_record_policy') != 'full_except_cbt_summary':
        raise ValueError('This run requires the user-approved therapist input policy')
    evaluation = Evaluation(config, mode='smoke' if args.smoke else 'production')
    run = OUTPUT_DIR / 'run'
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    namespace = 'smoke' if args.smoke else 'outputs'
    cases = evaluation.cases[:args.patient_limit] if args.patient_limit else evaluation.cases
    if args.aggregate_only:
        evaluation.aggregate(namespace, len(cases), args.agents)
        return
    futures = []
    try:
        with ThreadPoolExecutor(max_workers=config.get('judge_workers', 3)) as judges:
            for agent in args.agents:
                if STOP.is_set():
                    raise KeyboardInterrupt()
                evaluation.status('running', stage='dialogues', agent=agent, namespace=namespace)
                if agent in ('ours', 'omni_base', 'cbt'):
                    free = int(subprocess.check_output(['nvidia-smi', f"--id={runtime_gpu(config)}",
                        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                    if free < config.get('minimum_free_mib', 20480):
                        # Finish pending API judgments before pausing local models.
                        for future in as_completed(futures):
                            future.result()
                        evaluation.write_coverage(cases, args.agents, namespace)
                        evaluation.status('waiting_for_gpu', agent=agent, free_mib=free,
                                          gpu_polling=False)
                        return
                    ACTIVE_WORKER = LocalWorker(config, agent, evaluation.protocol_id)
                try:
                    if agent == 'qwen37':
                        with ThreadPoolExecutor(max_workers=config.get('api_dialogue_workers', 3)) as pool:
                            pending = {pool.submit(evaluation.run_session_or_defer, agent, None, case, namespace): case for case in cases}
                            for future in as_completed(pending):
                                case = pending[future]
                                session = future.result()
                                if session is not None:
                                    futures += [judges.submit(evaluation.judge_or_defer, case, session, scheme, namespace, False)
                                                for scheme in ('fair_shared',)]
                                for completed in (item for item in futures if item.done()):
                                    completed.result()
                    else:
                        completed_patients = deferred_patients = 0
                        for index, case in enumerate(cases):
                            session = evaluation.run_session_or_defer(agent, ACTIVE_WORKER, case, namespace)
                            if session is None:
                                deferred_patients += 1
                            else:
                                completed_patients += 1
                                futures += [judges.submit(evaluation.judge_or_defer, case, session, scheme, namespace, False)
                                            for scheme in ('fair_shared',)]
                            for completed in (item for item in futures if item.done()):
                                completed.result()
                            evaluation.status('running', stage='dialogues', agent=agent,
                                              processed_patients=index + 1, completed_patients=completed_patients,
                                              deferred_patients=deferred_patients, target_patients=len(cases),
                                              namespace=namespace)
                finally:
                    if ACTIVE_WORKER is not None:
                        ACTIVE_WORKER.close()
                        ACTIVE_WORKER = None
            evaluation.status('running', stage='judgments', namespace=namespace)
            for future in as_completed(futures):
                future.result()
        # Failed conditions and pending judgments are retained for later manual
        # repair, not automatically retried or counted as zero-valued outcomes.
        coverage = evaluation.write_coverage(cases, args.agents, namespace)
        if not coverage['full_cohort_metrics_ready']:
            evaluation.status('completed_with_deferred_samples', namespace=namespace,
                              agents=list(args.agents), deferred_items=coverage['deferred_items'],
                              coverage_path=str(OUTPUT_DIR / namespace / 'results' / 'coverage.json'),
                              full_cohort_metrics_ready=False)
            return
        metrics = evaluation.aggregate(namespace, len(cases), args.agents)
        evaluation.status('smoke_complete' if args.smoke else 'complete',
                          namespace=namespace, patients_per_agent=len(cases),
                          agents=list(args.agents), metric_rows=len(metrics) if metrics else None)
    except BaseException as error:
        STOP.set()
        if ACTIVE_WORKER is not None:
            ACTIVE_WORKER.close()
        evaluation.status('stopped' if isinstance(error, KeyboardInterrupt) else 'failed',
                          error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
