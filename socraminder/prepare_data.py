"""Prepare text-only patient profiles or export accepted CBT annotations to SFT."""
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
import json
from pathlib import Path

from project_paths import DATA_ROOT, load_config
from prepare_profiles import prepare_profiles
from annotate import export_sft, protocol, read_jsonl


ROOT = Path(__file__).resolve().parent
DATA_DIR = DATA_ROOT / "socraminder"


def annotation_identity(profiles, config):
    """Preserve the identity of accepted source annotations during re-export."""
    identities = set()
    for profile in profiles:
        folder = DATA_DIR / "annotations" / profile["split"] / profile["patient_id"]
        for path in folder.glob("*.json"):
            saved = json.loads(path.read_text())
            if "protocol_id" in saved:
                identities.add(saved["protocol_id"])
    if len(identities) > 1:
        raise ValueError("Accepted annotations contain different protocol identities")
    return next(iter(identities)) if identities else protocol(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DATA_ROOT / "edaic")
    parser.add_argument("--config", type=Path, default=ROOT / "config/annotation.json")
    parser.add_argument("--export-sft", action="store_true",
                        help="Export complete accepted sessions after profile preparation")
    args = parser.parse_args()
    profiles_dir = DATA_DIR / "profiles"
    prepared = prepare_profiles(args.source_root, profiles_dir)
    report = {"patients": prepared["counts"]}
    if args.export_sft:
        profiles = [row for split in ("train", "dev", "test")
                    for row in read_jsonl(profiles_dir / f"{split}.jsonl")]
        config = load_config(args.config)
        report["sft"] = export_sft(profiles, annotation_identity(profiles, config))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
