"""Shared data contracts and deterministic medication-ranking rewards."""
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


import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import unicodedata

ROOT = Path(__file__).resolve().parent


def load_json(path):
    return load_config(path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        os.chmod(temporary, 0o600)
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        os.chmod(temporary, 0o600)
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_shared_data_permissions(data_dir):
    """Create the configured dataset location using the current user's permissions."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)

def canonical_name(value):
    if not isinstance(value, str):
        raise ValueError('Medication name must be a string')
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def parse_recommendations(text, candidate_medications):
    """Require one complete JSON object, canonical names, and a genuine ranking."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'recommendations'}:
        raise ValueError('Output must contain exactly recommendations')
    records = value['recommendations']
    if not isinstance(records, list) or not 1 <= len(records) <= 5:
        raise ValueError('Recommend between one and five medications')
    allowed = {canonical_name(name): name for name in candidate_medications}
    seen, normalized = set(), []
    for rank, item in enumerate(records, 1):
        if not isinstance(item, dict) or set(item) != {'rank', 'medication', 'rationale'}:
            raise ValueError('Each recommendation requires rank, medication, and rationale')
        if type(item['rank']) is not int or item['rank'] != rank:
            raise ValueError('Ranks must be consecutive integers starting at one')
        name = canonical_name(item['medication'])
        if name not in allowed or name in seen:
            raise ValueError('Each medication must be a unique candidate name')
        if not isinstance(item['rationale'], str) or not item['rationale'].strip():
            raise ValueError('Each recommendation requires a nonempty rationale')
        seen.add(name)
        normalized.append({'rank': rank, 'medication': allowed[name],
                           'rationale': item['rationale'].strip()})
    return {'recommendations': normalized}


def score_completion(text, reference_medications, candidate_medications):
    """TCAS + precision + recall, all positive; invalid generations receive zero."""
    reference = [canonical_name(name) for name in reference_medications]
    candidates = {canonical_name(name) for name in candidate_medications}
    if not reference or len(reference) != len(set(reference)) or not set(reference) <= candidates:
        raise ValueError('Invalid reference medication list')
    try:
        target = parse_recommendations(text, candidate_medications)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return {'reward': 0.0, 'TCAS': 0.0, 'MMS': 0.0, 'RCR': 0.0,
                'valid': False, 'predicted_medications': [], 'error': str(error)}
    predicted = [canonical_name(item['medication']) for item in target['recommendations']]
    overlap = len(set(predicted) & set(reference))
    tcas = float(predicted[0] == reference[0])
    precision = overlap / len(set(predicted))
    recall = overlap / len(set(reference))
    reward = round(0.4 * tcas + 0.3 * precision + 0.3 * recall, 12)
    assert math.isfinite(reward) and 0.0 <= reward <= 1.0
    return {'reward': reward, 'TCAS': tcas, 'MMS': precision, 'RCR': recall,
            'valid': True, 'predicted_medications': predicted, 'error': None}
