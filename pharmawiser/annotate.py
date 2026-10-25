"""Resumable Qwen teacher reconstruction of Case1-30; never relabel test cases."""
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
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from common import ROOT, digest, load_json, parse_recommendations, publish_shared_data_permissions, read_jsonl, sha256_file, write_json, write_jsonl


class APIBlocked(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def reconstruct(config):
    run = Path(config['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'annotation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = Path(config['data_dir'])
    source = read_jsonl(data / 'source_cases.jsonl')
    train = sorted((x for x in source if x['split'] == 'train'), key=lambda x: x['case_number'])
    tests = read_jsonl(data / 'test_original.jsonl')
    assert [x['case_number'] for x in train] == list(range(1, 31))
    assert [x['case_number'] for x in tests] == list(range(31, 41))
    assert all(x['split'] == 'test' and 'target' not in x and 'reference_medications' not in x for x in tests)
    test_hash = sha256_file(data / 'test_original.jsonl')
    api = config['api']
    teacher_prompt = (ROOT / 'prompts/teacher_system.txt').read_text()
    student_prompt = (ROOT / 'prompts/student_system.txt').read_text()
    candidates = load_json(config['candidate_file'])['medications']
    assert len(candidates) >= 30 and len(candidates) == len(set(candidates))
    for case in train:
        model_input = case['model_input']
        assert set(model_input) == {'patient_historical_record', 'current_situation', 'candidate_medications'}
        assert all(isinstance(model_input[field], str) and model_input[field].strip()
                   for field in ('patient_historical_record', 'current_situation'))
        assert model_input['candidate_medications'] == candidates
    protocol = {'model': api['model'], 'base_url': api['base_url'],
                'temperature': api['temperature'], 'top_p': api['top_p'],
                'max_tokens': api['max_tokens'], 'teacher_prompt': teacher_prompt,
                'student_prompt': student_prompt, 'candidates': candidates,
                'train_input_hash': digest([{'case_id': x['case_id'], 'model_input': x['model_input']} for x in train]),
                'test_original_sha256': test_hash, 'reward': '0.4*TCAS+0.3*MMS+0.3*RCR'}
    protocol_id = digest(protocol)
    protocol_path = run / 'annotation_protocol.json'
    if protocol_path.exists():
        assert load_json(protocol_path)['protocol_id'] == protocol_id, 'Annotation protocol changed; keep existing successful labels unchanged.'
    else:
        write_json(protocol_path, {'protocol_id': protocol_id, 'created_utc': now(), 'protocol': protocol})
    labels = data / 'train_teacher_records'
    labels.mkdir(exist_ok=True)
    cache = run / 'teacher_api_cache'
    cache.mkdir(exist_ok=True)
    completed, guard = set(), threading.Lock()
    stop = threading.Event()
    started = time.monotonic()
    key = get_api_key(api)
    if not key:
        raise APIBlocked('The configured private API key file is empty.')

    def status(state, **extra):
        obj = {'state': state, 'pid': os.getpid(), 'time_utc': now(),
               'protocol_id': protocol_id, 'completed': len(completed), 'total': 30,
               'test_cases_preserved': 10, 'elapsed_seconds': round(time.monotonic() - started, 1), **extra}
        write_json(run / 'annotation_status.json', obj)
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    def fetch(body):
        request_hash = digest(body)
        path = cache / (request_hash + '.json')
        if path.exists():
            saved = load_json(path)
            assert saved['request_hash'] == request_hash and saved['protocol_id'] == protocol_id
            return saved['response'], {'request_hash': request_hash, 'cached': True}
        request = urllib.request.Request(api['base_url'].rstrip('/') + '/chat/completions',
                    data=json.dumps(body).encode(), method='POST',
                    headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        for attempt in range(api['max_retries']):
            if stop.is_set():
                raise RuntimeError('Annotation stopped after another request failed.')
            try:
                began = time.monotonic()
                with urllib.request.urlopen(request, timeout=api['timeout_seconds']) as response:
                    result = json.loads(response.read())
                write_json(path, {'protocol_id': protocol_id, 'request_hash': request_hash,
                                 'request_body': body, 'response': result, 'time_utc': now()})
                return result, {'request_hash': request_hash, 'cached': False,
                                'elapsed_seconds': round(time.monotonic() - began, 3)}
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors='replace')[:1000]
                if error.code not in (408, 429, 500, 502, 503, 504):
                    raise APIBlocked(f'Qwen API HTTP {error.code}: {detail}') from error
                if attempt + 1 == api['max_retries']:
                    raise RuntimeError(f'Qwen API retry limit, HTTP {error.code}: {detail}') from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt + 1 == api['max_retries']:
                    raise RuntimeError(f'Qwen API network retry limit: {type(error).__name__}') from error
            time.sleep(min(2 ** (attempt + 1), 20))
        raise AssertionError('Unreachable retry state')

    def annotate(case):
        assert 1 <= case['case_number'] <= 30 and case['split'] == 'train'
        path = labels / (case['case_id'] + '.json')
        input_hash = digest(case['model_input'])
        if path.exists():
            saved = load_json(path)
            assert saved['teacher_metadata']['protocol_id'] == protocol_id
            assert digest(saved['model_input']) == input_hash and saved['case_id'] == case['case_id']
            parse_recommendations(json.dumps(saved['target']), candidates)
            return saved
        error = None
        for attempt in range(3):
            messages = [{'role': 'system', 'content': teacher_prompt},
                        {'role': 'user', 'content': json.dumps(case['model_input'], ensure_ascii=False)}]
            if attempt:
                messages.append({'role': 'user', 'content':
                    'Return the complete JSON annotation again. The previous attempt failed output validation: '
                    + str(error)[:250] + '. Use only the requested schema and exact candidate names.'})
            body = {'model': api['model'], 'messages': messages,
                    'temperature': api['temperature'], 'top_p': api['top_p'],
                    'max_tokens': api['max_tokens'] + 1024 * attempt,
                    'response_format': {'type': 'json_object'}, 'enable_thinking': False}
            response, metadata = fetch(body)
            try:
                choice = response['choices'][0]
                if choice.get('finish_reason') == 'length':
                    raise ValueError('Generation reached its token limit')
                text = choice['message']['content']
                target = parse_recommendations(text, candidates)
            except (ValueError, KeyError, IndexError, TypeError) as invalid:
                error = invalid
                write_json(run / 'invalid_teacher_outputs' / f"{case['case_id']}_attempt{attempt+1}.json",
                           {'case_id': case['case_id'], 'error': str(invalid), 'api': metadata})
                continue
            record = {'case_id': case['case_id'], 'case_number': case['case_number'], 'split': 'train',
                      'model_input': case['model_input'], 'target': target,
                      'reference_medications': [x['medication'] for x in target['recommendations']],
                      'source': case['source'],
                      'teacher_metadata': {'protocol_id': protocol_id, 'model': api['model'],
                          'response_model': response.get('model'), 'usage': response.get('usage'),
                          'finish_reason': choice.get('finish_reason'), 'input_hash': input_hash,
                          'time_utc': now(), **metadata}}
            write_json(path, record)
            return record
        raise RuntimeError(f"{case['case_id']}: teacher output remained invalid after three attempts: {error}")

    for case in train:
        path = labels / (case['case_id'] + '.json')
        if path.exists():
            annotate(case)
            completed.add(case['case_id'])
    status('annotating')
    futures = {}
    try:
        with ThreadPoolExecutor(max_workers=api['concurrency']) as pool:
            for case in train:
                if case['case_id'] not in completed:
                    futures[pool.submit(annotate, case)] = case['case_id']
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException:
                    stop.set()
                    for pending in futures:
                        pending.cancel()
                    raise
                with guard:
                    completed.add(futures[future])
                    status('annotating', last_completed=futures[future])
    except BaseException as error:
        stop.set()
        status('blocked_api' if isinstance(error, APIBlocked) else 'failed',
               error=f'{type(error).__name__}: {str(error)[:1000]}')
        raise
    assert len(completed) == 30
    records = [load_json(labels / (case['case_id'] + '.json')) for case in train]
    write_jsonl(data / 'train.jsonl', records)
    sft = [{'case_id': x['case_id'], 'split': 'train', 'messages': [
                {'role': 'system', 'content': student_prompt},
                {'role': 'user', 'content': json.dumps(x['model_input'], ensure_ascii=False)},
                {'role': 'assistant', 'content': json.dumps(x['target'], ensure_ascii=False)}]}
           for x in records]
    write_jsonl(data / 'train_sft.jsonl', sft)
    assert sha256_file(data / 'test_original.jsonl') == test_hash
    write_json(data / 'manifest.json', {'protocol_id': protocol_id, 'created_utc': now(),
        'train_case_ids': [x['case_id'] for x in records], 'test_case_ids': [x['case_id'] for x in tests],
        'train_count': 30, 'test_count': 10, 'candidate_count': len(candidates),
        'test_reconstructed': False, 'input_fields': list(train[0]['model_input']),
        'teacher_model': api['model'], 'test_reference_note':
        'Original free-text replies are preserved; textual drug occurrence order is not a ranked gold answer.',
        'sha256': {name: sha256_file(data / name) for name in
                   ('source_cases.jsonl', 'test_original.jsonl', 'train.jsonl', 'train_sft.jsonl')}})
    publish_shared_data_permissions(data)
    status('complete', data_dir=str(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config/train.json'))
    args = parser.parse_args()
    reconstruct(load_json(args.config))


if __name__ == '__main__':
    main()
