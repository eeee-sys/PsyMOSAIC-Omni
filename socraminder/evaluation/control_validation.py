"""Patient-grounded transition and evaluation validation for controls V2.

Validators never modify the input objects. Quotes are checked against actual
patient turns after whitespace normalization; no language model is called here.
"""
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


from collections.abc import Mapping
import copy
import re


def normalized_quote(text):
    return " ".join(text.split())


# A bare occurrence of "skip" or "pass" is not a decision about therapy. Limit
# affirmative proposals to the current question/topic, or an unambiguous short
# refusal. The native text may contain curly apostrophes.
_TARGET = (
    r"(?:(?:this|that)(?:\s+(?:stage|question|topic|part|section|discussion|exercise|one))?"
    r"|the\s+(?:current\s+)?(?:stage|question|topic|part|section|discussion|exercise))"
)
_TARGET_END = r"(?=\s*(?:[.!?,;:]|$|for\s+now\b|today\b|please\b|and\b|because\b))"
_AFFIRMATIVE_PREFIX = (
    r"(?:\bI\s+(?:(?:would\s+)?(?:like|prefer)\s+to|want\s+to|would\s+rather|"
    r"am\s+going\s+to)|\bI'd\s+(?:like\s+to|prefer\s+to|rather)|"
    r"\bI(?:'ll|\s+will)|\b(?:can|could|may)\s+(?:we|I)|\blet(?:'s|\s+us))\s+"
)
_PASS_PROPOSAL = re.compile(
    _AFFIRMATIVE_PREFIX + r"(?:skip|pass\s+on)\s+" + _TARGET + _TARGET_END,
    re.IGNORECASE,
)
_MOVE_ON = re.compile(
    _AFFIRMATIVE_PREFIX + r"move\s+on(?:\s+from\s+" + _TARGET + r")?" + _TARGET_END,
    re.IGNORECASE,
)
_DECLINE = re.compile(
    r"\bI(?:\s+(?:do\s+not|don't|would\s+rather\s+not|prefer\s+not\s+to|"
    r"would\s+prefer\s+not\s+to)|'d\s+(?:rather\s+not|prefer\s+not\s+to))\s+"
    r"(?:want\s+to\s+)?(?:discuss|answer|share|talk\s+about|go\s+into|explore|address)\b",
    re.IGNORECASE,
)
_SHORT_PASS = re.compile(
    r"\s*(?:(?:no|sorry|thanks|thank\s+you)[,;:]?\s+)?"
    r"(?:" + _AFFIRMATIVE_PREFIX + r"|I\s+)?"
    r"(?:pass(?:\s+on\s+" + _TARGET + r")?|skip\s+" + _TARGET + r")"
    r"(?:\s+(?:for\s+now|today|please|thanks))?[.!?]?\s*",
    re.IGNORECASE,
)


def explicit_patient_pass(text):
    """Recognize an affirmative therapy-topic pass, rather than the word skip.

    This narrow text rule handles explicit choices. It does not infer refusal
    from uncertainty, symptoms, unrelated skipped activities, or negated passes.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    if _SHORT_PASS.fullmatch(text):
        return True
    for pattern in (_DECLINE, _PASS_PROPOSAL, _MOVE_ON):
        for match in pattern.finditer(text):
            # Exclude quoted/negated assertions such as "I am not saying I want
            # to skip this" without treating a different preceding sentence's
            # unrelated negation as a veto of the current affirmative choice.
            prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
            if re.search(r"\b(?:not\s+saying|did\s+not\s+say|didn't\s+say|"
                         r"don't\s+think|do\s+not\s+think|rather\s+than)\b",
                         prefix, re.IGNORECASE):
                continue
            return True
    return False


def transition_supported(state, stage, update, proposed_status, latest_patient):
    """Check support for a proposed transition without mutating therapy state."""
    if proposed_status == "stay":
        return True
    if proposed_status not in ("advance", "complete"):
        return False
    if not isinstance(state, Mapping) or not isinstance(update, Mapping):
        return False
    if not set(update).issubset({stage}):
        return False
    value = update.get(stage, state.get(stage))
    has_state = isinstance(value, str) and bool(value.strip())
    return has_state or explicit_patient_pass(latest_patient)


def _require_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Invalid " + name + " fields")


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing " + name)


def _verified_patient_quote(round_index, quote, dialogue, name):
    if type(round_index) is not int or not 1 <= round_index <= len(dialogue):
        raise ValueError("Invalid " + name + " patient round")
    _nonempty_string(quote, name + " quote")
    row = dialogue[round_index - 1]
    if not isinstance(row, Mapping) or not isinstance(row.get("patient_response"), str):
        raise ValueError("Invalid dialogue patient response")
    if normalized_quote(quote) not in normalized_quote(row["patient_response"]):
        raise ValueError(name + " quote is not in the referenced patient turn")


def validate_closure_review(value, dialogue):
    """Validate closure evidence, including adopted action and its time/trigger.

    Adoption/future-action semantics are the review model's task. This validator
    verifies field types and that every supplied evidence string is grounded in
    the claimed patient turn, including when ready is false.
    """
    _require_keys(value, {"ready", "insight", "action_plan", "reason"}, "closure review")
    value = copy.deepcopy(value)
    # Some JSON-mode judges express absent evidence as a schema-shaped object
    # of nulls. Canonicalize only that exact equivalent of null; preserve the
    # raw API object and reject partial, extra-field, or nonempty evidence.
    for name, fields in (("insight", {"round", "quote"}),
                         ("action_plan", {"round", "quote", "action", "timing_or_trigger"})):
        component = value[name]
        if (isinstance(component, dict) and set(component) == fields
                and all(item is None for item in component.values())):
            value[name] = None
    if type(value["ready"]) is not bool:
        raise ValueError("Closure ready must be boolean")
    _nonempty_string(value["reason"], "closure reason")
    if value["ready"] and (value["insight"] is None or value["action_plan"] is None):
        raise ValueError("Ready closure requires insight and action-plan evidence")
    if value["insight"] is not None:
        insight = value["insight"]
        _require_keys(insight, {"round", "quote"}, "closure insight")
        _verified_patient_quote(insight["round"], insight["quote"], dialogue, "insight")
    if value["action_plan"] is not None:
        plan = value["action_plan"]
        _require_keys(plan, {"round", "quote", "action", "timing_or_trigger"}, "closure action plan")
        _verified_patient_quote(plan["round"], plan["quote"], dialogue, "action plan")
        normalized_plan_quote = normalized_quote(plan["quote"])
        for key in ("action", "timing_or_trigger"):
            _nonempty_string(plan[key], "action-plan " + key)
            if normalized_quote(plan[key]) not in normalized_plan_quote:
                raise ValueError("Action-plan " + key + " is not in its evidence quote")
    return value


def validate_judge_result(value, dialogue):
    """Validate all judge fields; derive success timing from verified evidence.

    The returned fields match the original judge schema. The caller can save the
    raw value separately: first_success_round is normalized here to the later of
    the independently verified insight and action-plan patient rounds.
    """
    fields = {"success", "first_success_round", "ctr", "empathy", "coherence", "helpfulness",
              "safety", "success_evidence", "metric_evidence", "overall_comment"}
    _require_keys(value, fields, "judgment")
    if type(value["success"]) is not bool:
        raise ValueError("Judgment success must be boolean")
    for key, low, high in (("ctr", 0, 10), ("empathy", 1, 5), ("coherence", 1, 5),
                           ("helpfulness", 1, 5), ("safety", 1, 5)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError("Invalid judgment scale: " + key)
    evidence = value["success_evidence"]
    _require_keys(evidence, {"insight_quote", "insight_round", "action_plan_quote", "action_plan_round"},
                  "success evidence")
    metrics = value["metric_evidence"]
    _require_keys(metrics, {"ctr", "empathy", "coherence", "helpfulness", "safety"}, "metric evidence")
    for key, explanation in metrics.items():
        _nonempty_string(explanation, key + " metric explanation")
    _nonempty_string(value["overall_comment"], "overall comment")
    result = copy.deepcopy(value)
    if value["success"]:
        if type(value["first_success_round"]) is not int or not 1 <= value["first_success_round"] <= len(dialogue):
            raise ValueError("Invalid declared first success round")
        for prefix in ("insight", "action_plan"):
            _verified_patient_quote(evidence[prefix + "_round"], evidence[prefix + "_quote"],
                                    dialogue, prefix)
        result["first_success_round"] = max(evidence["insight_round"], evidence["action_plan_round"])
    elif value["first_success_round"] is not None or any(item is not None for item in evidence.values()):
        raise ValueError("Unsuccessful sessions cannot contain success timing or evidence")
    return result
