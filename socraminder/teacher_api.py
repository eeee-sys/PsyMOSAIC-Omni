"""Cached, bounded Qwen calls with usage logs and private file-based credentials."""
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

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from contracts import stable_hash


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class APIBlocked(RuntimeError):
    pass


class TeacherAPI:
    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.key = get_api_key(config)
        if not self.key:
            raise ValueError('Empty teacher API credential file')
        self.lock = threading.Lock()
        self.blocked = threading.Event()
        self.calls = self.cache_hits = 0
        self.usage = {}

    def log(self, event):
        with self.lock:
            path = self.root / 'run/api_events.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), **event},
                                        ensure_ascii=False, allow_nan=False) + '\n')

    def complete(self, kind, system, payload, temperature, max_tokens, schema=None):
        body = {'model': self.config['model'], 'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                'temperature': temperature, 'top_p': 0.9, 'enable_thinking': False,
                'max_tokens': max_tokens, 'response_format': {'type': 'json_object'}}
        if schema is not None:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': kind, 'strict': True, 'schema': schema}}
        endpoint = self.config['base_url'].rstrip('/') + '/chat/completions'
        request_hash = stable_hash({'endpoint': endpoint, 'body': body})
        path = self.root / 'api_cache' / f'{request_hash}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            with self.lock:
                self.cache_hits += 1
            return cached['parsed'], {'request_hash': request_hash, 'cached': True,
                                      'finish_reason': cached['finish_reason'], 'usage': cached['usage']}
        for attempt in range(self.config.get('max_attempts', 4)):
            if self.blocked.is_set():
                raise APIBlocked('Teacher service requires attention; no further calls dispatched')
            started = time.monotonic()
            request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                        headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=self.config.get('timeout_seconds', 180)) as response:
                    result = json.load(response)
                choice = result['choices'][0]
                content = choice['message']['content']
                parsed = json.loads(content)
                cached = {'request_hash': request_hash, 'kind': kind, 'request': body,
                          'parsed': parsed, 'response_text': content, 'provider_id': result.get('id'),
                          'response_model': result.get('model'),
                          'finish_reason': choice.get('finish_reason'), 'usage': result.get('usage', {}),
                          'elapsed_seconds': time.monotonic() - started}
                atomic_json(path, cached)
                with self.lock:
                    self.calls += 1
                    for key, value in cached['usage'].items():
                        if isinstance(value, (int, float)):
                            self.usage[key] = self.usage.get(key, 0) + value
                self.log({'kind': kind, 'request_hash': request_hash, 'state': 'success',
                          'finish_reason': cached['finish_reason'], 'usage': cached['usage'],
                          'elapsed_seconds': cached['elapsed_seconds']})
                return parsed, {'request_hash': request_hash, 'cached': False,
                                'finish_reason': cached['finish_reason'], 'usage': cached['usage']}
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors='replace').replace(self.key, '[redacted]')[:1200]
                self.log({'kind': kind, 'request_hash': request_hash, 'state': 'http_error',
                          'status_code': error.code, 'detail': detail})
                if error.code in (401, 402, 403) or any(s.lower() in detail.lower() for s in
                        ('Arrearage', 'insufficient_quota', 'InvalidApiKey', 'insufficient balance')):
                    self.blocked.set()
                    raise APIBlocked(f'Teacher API blocked (HTTP {error.code}): {detail}') from error
                if error.code != 429 and not 500 <= error.code < 600:
                    raise RuntimeError(f'Teacher API HTTP {error.code}: {detail}') from error
                if attempt + 1 == self.config.get('max_attempts', 4):
                    raise
            except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError, TypeError) as error:
                self.log({'kind': kind, 'request_hash': request_hash, 'state': 'request_error',
                          'error_type': type(error).__name__, 'attempt': attempt + 1})
                if attempt + 1 == self.config.get('max_attempts', 4):
                    raise
            time.sleep(min(2 ** attempt, 15))
        raise RuntimeError('Teacher request attempts exhausted')
