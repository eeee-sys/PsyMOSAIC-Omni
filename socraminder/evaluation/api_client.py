"""Private, resumable Qwen requests with bounded retries and cross-process deduplication."""
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
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import uuid


class APIError(RuntimeError):
    """A bounded request failed. Messages intentionally contain no clinical content."""


class APIBlocked(APIError):
    """Account or credential intervention is required before further network calls."""


class APITruncated(APIError):
    def __init__(self, metadata):
        super().__init__('Qwen response reached its token limit; retry with a larger budget')
        self.metadata = metadata


_LOCK_GUARD = threading.Lock()
_REQUEST_LOCKS = {}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _load_json(text):
    def reject_constant(value):
        raise ValueError('Non-finite JSON value')
    return json.loads(text, parse_constant=reject_constant)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)




class APICache:
    def __init__(self, config, root, protocol_id):
        self.config = dict(config)
        self.root, self.protocol_id = Path(root), str(protocol_id)
        self.cache_dir = self.root / 'api_cache'
        self.run_dir = self.root / 'run'
        for directory in (self.cache_dir, self.run_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blocked_path = self.run_dir / 'api_blocked.json'
        self.lock = threading.Lock()
        self.blocked = threading.Event()
        self.calls = self.cache_hits = 0
        self.usage = {}
        self.endpoint = self.config['base_url'].rstrip('/') + '/chat/completions'
        self.max_attempts = int(self.config.get('max_attempts', 4))
        if self.max_attempts < 1:
            raise ValueError('max_attempts must be positive')
        try:
            self._key = get_api_key(self.config)
        except OSError:
            raise APIBlocked('Qwen credential file is unavailable') from None
        if not self._key:
            raise APIBlocked('Qwen credential file is empty')
        if self.blocked_path.exists():
            self.blocked.set()

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self._key, '[redacted]')
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()
                    if key.lower() not in ('authorization', 'api_key', 'apikey')}
        return value

    def _save(self, path, value):
        _atomic_json(path, self._redact(value))

    def _check_blocked(self):
        if self.blocked.is_set() or self.blocked_path.exists():
            self.blocked.set()
            raise APIBlocked('Qwen API is paused; inspect run/api_blocked.json before resuming')

    def _block(self, status_code, code, request_hash):
        self.blocked.set()
        self._save(self.blocked_path, {'state': 'blocked', 'time_utc': self._now(),
                   'status_code': status_code, 'error_code': code,
                   'request_hash': request_hash,
                   'action': 'Resolve account or credentials; then remove this flag to resume.'})

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def log(self, event):
        """Append non-clinical usage metadata; separate processes share the file lock."""
        path = self.run_dir / 'api_events.jsonl'
        with self.lock:
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, 'a', encoding='utf-8') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write(_canonical(self._redact({'time_utc': self._now(), **event})) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                fcntl.flock(stream, fcntl.LOCK_UN)

    @staticmethod
    def _retry_delay(error, attempt):
        retry_after = error.headers.get('Retry-After') if error.headers else None
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                try:
                    date = parsedate_to_datetime(retry_after)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(2 ** (attempt - 1), 30)

    def _pause(self, seconds):
        # Honor Retry-After without an uninterruptible long sleep.
        if not math.isfinite(seconds):
            seconds = 30
        while seconds > 0:
            self._check_blocked()
            part = min(seconds, 30)
            time.sleep(part)
            seconds -= part

    def _metadata(self, artifact, cached):
        return {key: artifact.get(key) for key in
                ('request_hash', 'finish_reason', 'usage', 'response_model', 'provider_id',
                 'elapsed_seconds', 'cache_valid')} | {'cached': cached}

    def _read_cache(self, path, request_hash, body, json_mode):
        if not path.exists():
            return None
        try:
            cached = _load_json(path.read_text())
            if (cached['request_hash'] != request_hash or cached['protocol_id'] != self.protocol_id
                    or cached['request'] != self._redact(body) or not cached['cache_valid']
                    or cached['finish_reason'] == 'length'):
                raise ValueError('Cache does not match this request')
            parsed = _load_json(cached['response_text']) if json_mode else cached['response_text']
            if json_mode and not isinstance(parsed, dict):
                raise ValueError('Cached JSON is not an object')
            if parsed != cached['parsed']:
                raise ValueError('Cached parsed value does not match response')
        except (OSError, ValueError, KeyError, TypeError):
            path.rename(path.with_name(path.name + f'.invalid-{uuid.uuid4().hex}'))
            return None
        with self.lock:
            self.cache_hits += 1
        self.log({'state': 'cache_hit', 'kind': cached['kind'], 'request_hash': request_hash})
        return parsed, self._metadata(cached, True)

    def complete(self, model, messages, temperature, max_tokens, kind,
                 json_mode=True, top_p=0.9):
        body = {'model': model, 'messages': messages, 'temperature': temperature,
                'top_p': top_p, 'max_tokens': int(max_tokens), 'enable_thinking': False}
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        request_hash = hashlib.sha256(_canonical({'endpoint': self.endpoint, 'body': body,
                         'protocol_id': self.protocol_id}).encode()).hexdigest()
        path = self.cache_dir / f'{request_hash}.json'
        with _LOCK_GUARD:
            request_lock = _REQUEST_LOCKS.setdefault(str(path.resolve()), threading.Lock())
        with request_lock:
            descriptor = os.open(path.with_suffix('.lock'), os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, 'a') as lock_stream:
                fcntl.flock(lock_stream, fcntl.LOCK_EX)
                cached = self._read_cache(path, request_hash, body, json_mode)
                if cached is not None:
                    return cached
                return self._request(body, request_hash, path, kind, json_mode)

    def _request(self, body, request_hash, cache_path, kind, json_mode):
        for attempt in range(1, self.max_attempts + 1):
            self._check_blocked()
            started = time.monotonic()
            attempt_id = f'{request_hash}.attempt-{uuid.uuid4().hex}'
            request = urllib.request.Request(self.endpoint, data=_canonical(body).encode(),
                       headers={'Authorization': 'Bearer ' + self._key,
                                'Content-Type': 'application/json'})
            record = {'request_hash': request_hash, 'protocol_id': self.protocol_id,
                      'kind': kind, 'request': body, 'time_utc': self._now(),
                      'attempt': attempt, 'cache_valid': False}
            retry_delay = min(2 ** (attempt - 1), 30)
            try:
                with urllib.request.urlopen(request, timeout=self.config.get('timeout_seconds', 180)) as reply:
                    raw = reply.read().decode('utf-8')
                elapsed = time.monotonic() - started
                record.update(response_raw=raw, elapsed_seconds=elapsed)
                self._save(self.cache_dir / f'{attempt_id}.response.json', record)
                result = _load_json(raw)
                choice = result['choices'][0]
                content = choice['message']['content']
                record.update(response=result, response_text=content, usage=result.get('usage', {}),
                              finish_reason=choice.get('finish_reason'),
                              response_model=result.get('model'), provider_id=result.get('id'))
                with self.lock:
                    self.calls += 1
                    for name, value in record['usage'].items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            self.usage[name] = self.usage.get(name, 0) + value
                self.log({'state': 'response', 'kind': kind, 'request_hash': request_hash,
                          'attempt': attempt, 'usage': record['usage'],
                          'response_model': record['response_model'],
                          'finish_reason': record['finish_reason'], 'elapsed_seconds': elapsed})
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('Missing textual response')
                try:
                    parsed = _load_json(content) if json_mode else content
                    if json_mode and not isinstance(parsed, dict):
                        raise ValueError('Response JSON must be an object')
                except ValueError:
                    if record['finish_reason'] == 'length':
                        self._save(self.cache_dir / f'{attempt_id}.response.json', record)
                        raise APITruncated(self._metadata(record, False)) from None
                    raise
                record['parsed'] = parsed
                record['cache_valid'] = record['finish_reason'] == 'stop'
                self._save(self.cache_dir / f'{attempt_id}.response.json', record)
                if record['cache_valid']:
                    self._save(cache_path, record)
                return parsed, self._metadata(record, False)
            except urllib.error.HTTPError as error:
                detail = error.read().decode('utf-8', errors='replace')
                code = 'http_error'
                try:
                    payload = _load_json(detail)
                    code = str(payload.get('error', payload).get('code', 'http_error'))
                except (ValueError, TypeError, AttributeError):
                    pass
                # Provider code goes in events only if it has a short identifier form.
                safe_code = code if re.fullmatch(r'[A-Za-z0-9_.:-]{1,96}', code) else 'http_error'
                record.update(status_code=error.code, error_code=safe_code, response_raw=detail,
                              elapsed_seconds=time.monotonic() - started)
                self._save(self.cache_dir / f'{attempt_id}.error.json', record)
                self.log({'state': 'http_error', 'kind': kind, 'request_hash': request_hash,
                          'attempt': attempt, 'status_code': error.code, 'error_code': safe_code})
                blocked_markers = ('arrearage', 'invalidapikey', 'invalid_api_key', 'quota',
                                   'insufficient balance', 'payment_required', 'accountoverdue')
                if error.code in (401, 402, 403) or any(s in detail.lower() for s in blocked_markers):
                    self._block(error.code, safe_code, request_hash)
                    raise APIBlocked(f'Qwen API account/credential blocked (HTTP {error.code})') from None
                if error.code != 429 and not 500 <= error.code < 600:
                    raise APIError(f'Qwen API rejected request (HTTP {error.code}, code {safe_code})') from None
                retry_delay = self._retry_delay(error, attempt)
            except (TimeoutError, urllib.error.URLError, OSError, ValueError,
                    KeyError, IndexError, TypeError) as error:
                record.update(error_type=type(error).__name__,
                              elapsed_seconds=time.monotonic() - started)
                self._save(self.cache_dir / f'{attempt_id}.error.json', record)
                self.log({'state': 'request_error', 'kind': kind, 'request_hash': request_hash,
                          'attempt': attempt, 'error_type': type(error).__name__})
            if attempt < self.max_attempts:
                self._pause(retry_delay)
        raise APIError(f'Qwen request failed after {self.max_attempts} attempts; '
                       f'inspect api_cache artifacts for request {request_hash}')
