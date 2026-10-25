#!/usr/bin/env python3
"""Extract MedicationAssistant cases without executing notebook code.

Only historical record, current situation, and the common candidate list enter
``model_input``. Original questions and references remain audit metadata. Cases
31--40 are exported verbatim as a held-out test source, never as teacher targets.
"""

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
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
CASE_HEADING = re.compile(r"^#{1,6}\s+Case\s*(\d+)\b", re.IGNORECASE | re.MULTILINE)
FIELD_PATTERN = re.compile(
    r"\*\*Patient historical record\*\*:\s*(?P<history>.*?)\s*"
    r"\*\*Current situation\*\*:\s*(?P<situation>.*?)\s*"
    r"\*\*Query\*\*:\s*(?P<query>.*?)\s*^---\s*$\s*"
    r"\*\*\*expected reply\*\*\*:\s*(?P<reference>.*)\Z",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
CODE_HISTORY = re.compile(
    r"Patient historical record:\s*(.*?)\s*End of patient historical record\.",
    re.IGNORECASE | re.DOTALL,
)
CODE_SITUATION = re.compile(
    r"Current situation:\s*(.*?)\s*End of current situation\.",
    re.IGNORECASE | re.DOTALL,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def read_candidates(path: Path) -> list[str]:
    """Accept a string list or an object whose ``medications`` is a string list."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("medications")
    if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
        raise ValueError("candidate_file must contain a nonempty string list or {medications: [...]}.")
    if any(x != x.strip() for x in value):
        raise ValueError("Candidate names must not contain surrounding whitespace.")
    if len(value) < 30 or len({x.casefold() for x in value}) != len(value):
        raise ValueError("At least 30 distinct case-insensitive candidate names are required.")
    return value


def cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(x, str) for x in source):
        return "".join(source)
    raise ValueError("Invalid notebook cell source representation.")


def query_literal(cell: dict[str, Any], cell_index: int) -> str | None:
    """Read an app.query literal via AST; never run or import notebook code."""
    if cell.get("cell_type") != "code":
        return None
    tree = ast.parse(cell_text(cell), filename=f"notebook-cell-{cell_index}")
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "query"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
    ]
    if not matches:
        return None
    if len(matches) != 1 or not matches[0].args:
        raise ValueError(f"Cell {cell_index}: expected exactly one literal app.query call.")
    literal = ast.literal_eval(matches[0].args[0])
    if not isinstance(literal, str) or not literal.strip():
        raise ValueError(f"Cell {cell_index}: app.query input is not a nonempty string literal.")
    return literal


def normalize_source_markup(text: str) -> str:
    # This is used only to compare two copies of a source field. The exported
    # model input retains the original markdown, including its emphasis marks.
    text = re.sub(r"\*Current medication\*", "Current medication", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def embedded_medication(history: str) -> str | None:
    match = re.search(r"(?:\*\s*)?Current medication(?:\s*\*)?\s*:\s*(.*)\Z", history,
                      re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\bNo current medication\b[^.]*\.?", history, re.IGNORECASE)
    return match.group(0) if match else None


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def extract_notebook(notebook_path: Path, candidates: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw = notebook_path.read_bytes()
    notebook_hash = sha256_bytes(raw)
    notebook = json.loads(raw)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Source is not a notebook with a cells list.")
    headings: list[tuple[int, int]] = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        found = list(CASE_HEADING.finditer(cell_text(cell)))
        if len(found) > 1:
            raise ValueError(f"Cell {index}: several case headings in a single cell.")
        if found:
            headings.append((int(found[0].group(1)), index))
    numbers = [number for number, _ in headings]
    if len(numbers) != 40 or sorted(numbers) != list(range(1, 41)):
        raise ValueError(f"Expected each Case1--40 exactly once; found case numbers {numbers}.")

    rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    audit_cases: list[dict[str, Any]] = []
    for position, (number, markdown_index) in enumerate(headings):
        next_index = headings[position + 1][1] if position + 1 < len(headings) else len(cells)
        indices = list(range(markdown_index, next_index))
        markdown = cell_text(cells[markdown_index])
        parsed = FIELD_PATTERN.search(markdown)
        if not parsed:
            raise ValueError(f"Case{number}: required markdown sections are missing or ambiguous.")
        fields = {key: value.strip() for key, value in parsed.groupdict().items()}
        if not all(fields.values()):
            raise ValueError(f"Case{number}: empty required source field.")
        query_cells = [(index, query_literal(cells[index], index)) for index in indices]
        query_cells = [(index, value) for index, value in query_cells if value is not None]
        if len(query_cells) != 1:
            raise ValueError(f"Case{number}: expected one source query literal, found {len(query_cells)}.")
        query_index, raw_query = query_cells[0]
        code_history, code_situation = CODE_HISTORY.search(raw_query), CODE_SITUATION.search(raw_query)
        if code_history is None or code_situation is None:
            raise ValueError(f"Case{number}: query literal lacks historical record or situation markers.")
        history_equal = fields["history"] == code_history.group(1).strip()
        situation_equal = fields["situation"] == code_situation.group(1).strip()
        if normalize_source_markup(fields["history"]) != normalize_source_markup(code_history.group(1)):
            raise ValueError(f"Case{number}: factual historical-record mismatch between markdown and query code.")
        if normalize_source_markup(fields["situation"]) != normalize_source_markup(code_situation.group(1)):
            raise ValueError(f"Case{number}: factual current-situation mismatch between markdown and query code.")

        current_medication = embedded_medication(fields["history"])
        source = {
            "notebook_path": str(notebook_path),
            "notebook_sha256": notebook_hash,
            "markdown_cell_index": markdown_index,
            "query_cell_index": query_index,
            "cell_indices": indices,
            "field_authority": "original_markdown",
            "original_markdown": markdown,
            "original_app_query_literal": raw_query,
        }
        row = {
            "case_id": f"Case{number}",
            "case_number": number,
            "split": "train" if number <= 30 else "test",
            "model_input": {
                "patient_historical_record": fields["history"],
                "current_situation": fields["situation"],
                "candidate_medications": list(candidates),
            },
            "original_current_medication": current_medication,
            "original_query": fields["query"],
            "original_reference": fields["reference"],
            "source": source,
        }
        if set(row["model_input"]) != {"patient_historical_record", "current_situation", "candidate_medications"}:
            raise AssertionError("Unexpected model-input field.")
        rows.append(row)
        if number >= 31:
            test_rows.append({
                "case_id": f"Case{number}", "case_number": number, "split": "test",
                "patient_historical_record": fields["history"],
                "current_situation": fields["situation"],
                "original_current_medication": current_medication,
                "original_query": fields["query"],
                "original_reference": fields["reference"],
                "original_markdown": markdown,
                "source": source,
                "original_notebook_cells": [{"cell_index": index, "cell": cells[index]} for index in indices],
                "teacher_reconstruction": False,
            })
        audit_cases.append({
            "case_id": row["case_id"], "split": row["split"],
            "markdown_cell_index": markdown_index, "query_cell_index": query_index,
            "history_chars": len(fields["history"]), "situation_chars": len(fields["situation"]),
            "original_reference_chars": len(fields["reference"]),
            "current_medication_available": current_medication is not None,
            "history_literal_exact_match": history_equal,
            "situation_literal_exact_match": situation_equal,
            "source_facts_match_after_markup_normalization": True,
        })

    rows.sort(key=lambda row: row["case_number"])
    test_rows.sort(key=lambda row: row["case_number"])
    audit = {
        "source_notebook": str(notebook_path), "source_notebook_sha256": notebook_hash,
        "source_notebook_bytes": len(raw), "source_notebook_cells": len(cells),
        "case_count": len(rows), "train_count": 30, "test_count": 10,
        "train_case_ids": [f"Case{n}" for n in range(1, 31)],
        "test_case_ids": [f"Case{n}" for n in range(31, 41)],
        "candidate_count": len(candidates),
        "model_input_fields": ["patient_historical_record", "current_situation", "candidate_medications"],
        "field_authority": "original_markdown", "notebook_executed": False,
        "test_teacher_reconstruction": False,
        "source_reference_is_ranked_target": False,
        "notes": [
            "Current-medication facts already embedded in historical records are preserved verbatim.",
            "Original questions and expected replies are audit metadata, not teacher inputs.",
            "Cases1--30 original references concern dose, adverse effects, suitability or contraindications and are not ranked recommendation targets.",
            "Some Cases31--40 references offer unordered alternatives or medication adjustments; textual mention order is not a gold priority ranking.",
            "Original test references are preserved even when their source terminology or spelling is imperfect.",
        ],
        "cases": audit_cases,
    }
    return rows, test_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.json")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    for key in ("source_notebook", "data_dir", "candidate_file"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Configuration field {key} must be a nonempty path string.")
    source_path = project_path(config["source_notebook"])
    candidate_path = project_path(config["candidate_file"])
    data_dir = project_path(config["data_dir"])
    candidates = read_candidates(candidate_path)
    rows, test_rows, audit = extract_notebook(source_path, candidates)
    audit["candidate_file"] = str(candidate_path)
    audit["candidate_file_sha256"] = sha256_bytes(candidate_path.read_bytes())
    source_output, test_output = data_dir / "source_cases.jsonl", data_dir / "test_original.jsonl"
    write_jsonl(source_output, rows)
    write_jsonl(test_output, test_rows)
    audit["outputs"] = {
        "source_cases": {"path": str(source_output), "sha256": sha256_bytes(source_output.read_bytes()), "rows": len(rows)},
        "test_original": {"path": str(test_output), "sha256": sha256_bytes(test_output.read_bytes()), "rows": len(test_rows)},
    }
    atomic_write(PROJECT_DIR / "run" / "source_audit.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"state": "source_extracted", "train": 30, "test": 10,
                      "candidate_count": len(candidates), "source_sha256": audit["source_notebook_sha256"]}))


if __name__ == "__main__":
    main()
