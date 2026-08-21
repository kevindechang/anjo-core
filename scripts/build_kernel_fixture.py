#!/usr/bin/env python3
"""Build the public parity fixture from the private production corpus.

Only explicitly named, generalizable deterministic sections are copied. Product
prompt text and prose-producing turn-shape directives never enter the output.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

PUBLIC_GROUPS = {
    "affect_control": (
        "mood_octant",
        "decoding_params",
        "length_factor",
        "apply_length_factor",
        "is_ambivalent",
    ),
    "retrieval": (
        "recency_weight",
        "mood_congruence_factor",
        "candidate_score",
    ),
    "appraisal": (
        "mood_inertia",
        "stage_int",
        "baseline_weight",
        "decay_mood",
        "appraise_input",
        "expectation_emotions",
        "occ_carry_decay",
        "state_emotions",
        "appraise_turn",
    ),
    "surfacing": (
        "clean_text",
        "presence_line",
        "presence_vector",
    ),
}
PUBLIC_SECTIONS = tuple(PUBLIC_GROUPS)
EXPECTED_CASES = 225
EXPECTED_GROUP_COUNTS = {
    "affect_control": {
        "mood_octant": 15,
        "decoding_params": 16,
        "length_factor": 16,
        "apply_length_factor": 45,
        "is_ambivalent": 9,
    },
    "retrieval": {
        "recency_weight": 6,
        "mood_congruence_factor": 5,
        "candidate_score": 5,
    },
    "appraisal": {
        "mood_inertia": 8,
        "stage_int": 6,
        "baseline_weight": 7,
        "decay_mood": 9,
        "appraise_input": 11,
        "expectation_emotions": 11,
        "occ_carry_decay": 6,
        "state_emotions": 7,
        "appraise_turn": 4,
    },
    "surfacing": {
        "clean_text": 15,
        "presence_line": 16,
        "presence_vector": 8,
    },
}
EXPECTED_META = {
    "purpose": "Cross-runtime deterministic companion-kernel contract",
    "source": "Production-derived synthetic vectors; product prose removed",
    "cases": EXPECTED_CASES,
    "float_round": 9,
    "affect_dynamics": True,
    "affect_habituation": False,
    "excludes": [
        "turn-shape prose",
        "prompt text",
        "gate",
        "generation",
        "policy",
        "persistence",
        "reflection",
    ],
}
PAD_FIELDS = frozenset({"valence", "arousal", "dominance"})
PERSONALITY_FIELDS = frozenset({"O", "C", "E", "A", "N"})
EMOTION_FIELDS = frozenset(
    {
        "admiration",
        "anger",
        "disappointment",
        "distress",
        "fatigue",
        "fear",
        "gratitude",
        "hope",
        "joy",
        "longing",
        "relief",
        "reproach",
        "sadness",
        "satisfaction",
        "surprise",
        "unease",
    }
)
INPUT_FIELDS: dict[tuple[str, str], tuple[frozenset[str], ...]] = {
    ("affect_control", "mood_octant"): (PAD_FIELDS,),
    ("affect_control", "decoding_params"): (PAD_FIELDS,),
    ("affect_control", "length_factor"): (PAD_FIELDS,),
    ("affect_control", "apply_length_factor"): (PAD_FIELDS | {"base_tokens"},),
    ("retrieval", "recency_weight"): (
        frozenset({"days_ago"}),
        frozenset({"timestamp"}),
    ),
    ("retrieval", "mood_congruence_factor"): (
        frozenset({"mem_valence", "mood_valence", "congruence_on"}),
    ),
    ("retrieval", "candidate_score"): (
        frozenset({"distance", "days_ago", "episode", "significance", "recall_count"}),
    ),
    ("appraisal", "mood_inertia"): (PERSONALITY_FIELDS,),
    ("appraisal", "stage_int"): (frozenset({"stage"}),),
    ("appraisal", "baseline_weight"): (frozenset({"stage_int"}),),
    ("appraisal", "decay_mood"): (
        PAD_FIELDS | PERSONALITY_FIELDS | {"stage_int", "baseline_valence"},
    ),
    ("appraisal", "appraise_input"): (
        PAD_FIELDS
        | {
            "autonomy",
            "baseline_valence",
            "honesty",
            "intellectual",
            "intent",
            "rapport",
            "respect",
        },
    ),
    ("appraisal", "expectation_emotions"): (frozenset({"expectation", "message"}),),
    ("appraisal", "state_emotions"): (frozenset({"valence", "arousal", "longing"}),),
    ("appraisal", "appraise_turn"): (
        PAD_FIELDS
        | PERSONALITY_FIELDS
        | {
            "baseline_valence",
            "expectation",
            "intent",
            "longing",
            "message",
            "occ_carry",
            "stage_int",
        },
    ),
    ("surfacing", "clean_text"): (frozenset({"value", "max_len"}),),
    ("surfacing", "presence_line"): (
        frozenset(
            {"reflection_pending", "due_intention", "carried_thought", "open_thread"}
        ),
    ),
    ("surfacing", "presence_vector"): (
        PAD_FIELDS
        | {
            "carried_thought",
            "cognition",
            "comfort",
            "longing",
            "stage",
            "trust_score",
        },
    ),
}
DICT_OUTPUT_FIELDS: dict[tuple[str, str], frozenset[str]] = {
    ("affect_control", "decoding_params"): frozenset({"temperature", "top_p"}),
    ("appraisal", "decay_mood"): PAD_FIELDS,
    ("appraisal", "appraise_input"): frozenset(
        {"emotions", "mood", "baseline_valence"}
    ),
    ("appraisal", "appraise_turn"): frozenset(
        {"mood", "active_emotions", "occ_carry", "baseline_valence"}
    ),
    ("surfacing", "presence_vector"): frozenset(
        {
            "trust",
            "valence",
            "arousal",
            "longing",
            "awaiting",
            "mode",
            "line",
            "affect",
            "relationship",
            "cognition",
            "source",
        }
    ),
}
DYNAMIC_INPUT_GROUPS = {
    ("affect_control", "is_ambivalent"),
    ("appraisal", "occ_carry_decay"),
}
DYNAMIC_OUTPUT_GROUPS = {
    ("appraisal", "expectation_emotions"),
    ("appraisal", "occ_carry_decay"),
    ("appraisal", "state_emotions"),
}
NULLABLE_INPUT_GROUPS = {
    ("affect_control", "decoding_params"),
    ("affect_control", "length_factor"),
}
INTENTS = frozenset(
    {
        "ABUSE",
        "APOLOGY",
        "VULNERABILITY",
        "CURIOSITY",
        "CHALLENGE",
        "NEGLECT",
        "CASUAL",
        "GREETING",
    }
)
STAGES = frozenset(
    {"stranger", "acquaintance", "friend", "close", "intimate", "unknown"}
)
MOOD_LABELS = frozenset(
    {
        "neutral",
        "exuberant",
        "dependent",
        "relaxed",
        "docile",
        "hostile",
        "anxious",
        "disdainful",
        "bored",
    }
)
PRESENCE_LINES = frozenset(
    {
        "here with you",
        "holding a pattern",
        "carrying a thread",
        "holding a follow-up",
        "still reflecting",
    }
)
EXPECTATION_TEXT_CASES = frozenset(
    {
        ("", "anything"),
        ("she was angry with me", "i feel better now"),
        ("upset about the argument", "totally fine"),
        ("please answer me", "it got worse"),
        ("come back and tell me", "it's resolved"),
        ("will you apolog", "you hurt me"),
        ("define the boundary", "the cup is broken"),
        ("nothing in particular", "ok"),
        ("she was caféangry with me", "i feel better now"),
        ("please answer me", "٤ok"),
        ("apologé later", "things got worse"),
    }
)
APPRAISE_TURN_TEXT_CASES = frozenset(
    {
        ("CURIOSITY", "", "tell me about the stars"),
        ("ABUSE", "", "you're useless"),
        ("VULNERABILITY", "she was upset", "actually things are better now"),
        ("CASUAL", "", "hey"),
    }
)
CARRIED_THOUGHTS = frozenset(
    {None, "i keep thinking about it", "something", "a thread", '  "  "  '}
)
CLEAN_TEXT_CASES = frozenset(
    {
        (None, 300, None),
        (123, 300, None),
        ("plain", 300, "plain"),
        ("  hello   world  ", 300, "hello world"),
        ('"quoted"', 300, "quoted"),
        ('""double""', 300, "double"),
        ('  "  spaced quotes  "  ', 300, "spaced quotes"),
        ("   ", 300, None),
        ('""', 300, None),
        ("tab\tand\nnewline", 300, "tab and newline"),
        ("truncate me please", 5, "trunc"),
        ("nel\u0085sep", 300, "nel sep"),
        ("fs\u001cgs\u001d", 300, "fs gs"),
        ("bom\ufeffmark", 300, "bom\ufeffmark"),
        ("\u3000\u3000", 300, None),
    }
)
NUMBER_RANGES = {
    "A": (0.0, 1.0),
    "C": (0.0, 1.0),
    "E": (0.0, 1.0),
    "N": (0.0, 1.0),
    "O": (0.0, 1.0),
    "arousal": (-1.0, 1.0),
    "autonomy": (0.0, 1.0),
    "baseline_valence": (-1.0, 1.0),
    "comfort": (0.0, 1.0),
    "days_ago": (0.0, 10_000.0),
    "distance": (0.0, 2.0),
    "dominance": (-1.0, 1.0),
    "honesty": (0.0, 1.0),
    "intellectual": (0.0, 1.0),
    "longing": (0.0, 1.0),
    "mem_valence": (-1.0, 1.0),
    "mood_valence": (-1.0, 1.0),
    "rapport": (0.0, 1.0),
    "respect": (0.0, 1.0),
    "significance": (0.0, 1.0),
    "temperature": (0.0, 5.0),
    "top_p": (0.0, 1.0),
    "trust": (0.0, 1.0),
    "trust_score": (0.0, 1.0),
    "valence": (-1.0, 1.0),
}
INTEGER_RANGES = {
    "base_tokens": (0, 100_000),
    "max_len": (0, 10_000),
    "recall_count": (0, 100_000),
    "stage_int": (0, 100),
}
BOOLEAN_FIELDS = frozenset(
    {
        "awaiting",
        "congruence_on",
        "curiosity",
        "due_intention",
        "episode",
        "intentionality",
        "open_thread",
        "reflection_pending",
    }
)
STRING_FIELDS = frozenset(
    {"expectation", "intent", "line", "message", "mode", "source", "stage", "timestamp"}
)
NUMBER_OUTPUT_GROUPS = frozenset(
    {
        ("affect_control", "length_factor"),
        ("retrieval", "recency_weight"),
        ("retrieval", "mood_congruence_factor"),
        ("retrieval", "candidate_score"),
        ("appraisal", "mood_inertia"),
        ("appraisal", "baseline_weight"),
    }
)
INTEGER_OUTPUT_GROUPS = frozenset(
    {
        ("affect_control", "apply_length_factor"),
        ("appraisal", "stage_int"),
    }
)


def _require_keys(
    value: Any,
    expected: frozenset[str] | tuple[frozenset[str], ...],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    variants = expected if isinstance(expected, tuple) else (expected,)
    keys = frozenset(value)
    if keys not in variants:
        rendered = " or ".join(", ".join(sorted(fields)) for fields in variants)
        raise ValueError(f"{label} has fields {sorted(keys)}; expected {rendered}")
    return value


def _require_number(
    value: Any,
    label: str,
    low: float,
    high: float,
    *,
    integer: bool = False,
) -> int | float:
    expected_type = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected_type):
        kind = "integer" if integer else "number"
        raise TypeError(f"{label} must be a {kind}")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be finite and within [{low}, {high}]")
    return value


def _require_emotions(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value) <= EMOTION_FIELDS:
        raise ValueError(f"{label} contains an unexpected emotion field")
    for name, intensity in value.items():
        _require_number(intensity, f"{label}.{name}", 0.0, 1.0)
    return value


def _validate_named_values(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        child_label = f"{label}.{key}"
        if key in NUMBER_RANGES:
            low, high = NUMBER_RANGES[key]
            if key == "top_p" and child is None:
                continue
            _require_number(child, child_label, low, high)
        elif key in INTEGER_RANGES:
            low, high = INTEGER_RANGES[key]
            _require_number(child, child_label, low, high, integer=True)
        elif key in BOOLEAN_FIELDS:
            if not isinstance(child, bool):
                raise TypeError(f"{child_label} must be a boolean")
        elif key in STRING_FIELDS and not isinstance(child, str):
            raise TypeError(f"{child_label} must be a string")
        elif (
            key == "carried_thought"
            and child is not None
            and not isinstance(child, (bool, str))
        ):
            raise TypeError(f"{child_label} must be a boolean, string, or null")
        if isinstance(child, dict):
            _validate_named_values(child, child_label)


def _validate_value_tree(value: Any, label: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and len(value) > 200:
            raise ValueError(f"{label} contains an oversized string")
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        if abs(value) > 10_000:
            raise ValueError(f"{label} contains an out-of-range number")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} contains a non-string field name")
            _validate_value_tree(child, f"{label}.{key}")
        return
    raise TypeError(f"{label} contains unsupported JSON value {type(value).__name__}")


def _validate_case_content(
    section: str,
    group: str,
    input_value: Any,
    output_value: Any,
    *,
    source_label: str,
) -> None:
    pair = (section, group)
    _validate_value_tree(input_value, f"{section}.{group}.in")
    _validate_value_tree(output_value, f"{section}.{group}.out")
    _validate_named_values(input_value, f"{section}.{group}.in")
    _validate_named_values(output_value, f"{section}.{group}.out")

    if pair in NUMBER_OUTPUT_GROUPS:
        _require_number(output_value, f"{section}.{group}.out", -10.0, 10.0)
    if pair in INTEGER_OUTPUT_GROUPS:
        _require_number(
            output_value,
            f"{section}.{group}.out",
            0,
            100_000,
            integer=True,
        )
    if pair == ("affect_control", "is_ambivalent") and not isinstance(
        output_value, bool
    ):
        raise TypeError("is_ambivalent output must be a boolean")

    if pair == ("affect_control", "mood_octant") and output_value not in MOOD_LABELS:
        raise ValueError("mood_octant contains an unexpected public label")
    if (
        pair == ("retrieval", "recency_weight")
        and "timestamp" in input_value
        and input_value["timestamp"] != "not-a-date"
    ):
        raise ValueError("recency_weight contains unexpected timestamp prose")
    if pair == ("appraisal", "stage_int") and input_value["stage"] not in STAGES:
        raise ValueError("stage_int contains an unexpected stage label")
    if pair == ("appraisal", "appraise_input") and input_value["intent"] not in INTENTS:
        raise ValueError("appraise_input contains an unexpected intent label")
    if pair == ("appraisal", "expectation_emotions"):
        texts = (input_value["expectation"], input_value["message"])
        if texts not in EXPECTATION_TEXT_CASES:
            raise ValueError("expectation_emotions contains non-public prose")
    if pair == ("appraisal", "appraise_turn"):
        texts = (
            input_value["intent"],
            input_value["expectation"],
            input_value["message"],
        )
        if texts not in APPRAISE_TURN_TEXT_CASES:
            raise ValueError("appraise_turn contains non-public prose")
    if pair == ("surfacing", "clean_text"):
        vector = (input_value["value"], input_value["max_len"], output_value)
        if vector not in CLEAN_TEXT_CASES:
            raise ValueError("clean_text contains an unapproved text vector")
    if pair == ("surfacing", "presence_line") and output_value not in PRESENCE_LINES:
        raise ValueError("presence_line contains an unexpected public label")
    if pair == ("surfacing", "presence_vector"):
        if input_value["stage"] not in STAGES - {"unknown"}:
            raise ValueError("presence_vector contains an unexpected stage label")
        if input_value["carried_thought"] not in CARRIED_THOUGHTS:
            raise ValueError(
                "presence_vector contains non-public carried-thought prose"
            )
        if output_value["mode"] not in {"quiet", "reflecting"}:
            raise ValueError("presence_vector contains an unexpected mode label")
        if output_value["line"] not in PRESENCE_LINES:
            raise ValueError("presence_vector contains an unexpected line label")
        if output_value["relationship"]["stage"] != input_value["stage"]:
            raise ValueError("presence_vector relationship stage is inconsistent")
        if output_value["source"] != source_label:
            raise ValueError("presence_vector contains an unexpected source label")


def _sanitize_case(
    section: str,
    group: str,
    case: Any,
    *,
    source_label: str,
) -> dict[str, Any]:
    label = f"{section}.{group} case"
    case = _require_keys(case, frozenset({"in", "out"}), label)
    pair = (section, group)
    input_value = case["in"]
    output_value = case["out"]

    if pair in DYNAMIC_INPUT_GROUPS:
        _require_emotions(input_value, f"{label}.in")
    elif pair in INPUT_FIELDS:
        if input_value is None:
            if pair not in NULLABLE_INPUT_GROUPS:
                raise TypeError(f"{label}.in must be an object")
        else:
            _require_keys(input_value, INPUT_FIELDS[pair], f"{label}.in")
    else:
        raise ValueError(f"no public input schema for {section}.{group}")

    if pair in DYNAMIC_OUTPUT_GROUPS:
        _require_emotions(output_value, f"{label}.out")
    elif pair in DICT_OUTPUT_FIELDS:
        output = _require_keys(output_value, DICT_OUTPUT_FIELDS[pair], f"{label}.out")
        if pair == ("appraisal", "decay_mood"):
            _require_keys(output, PAD_FIELDS, f"{label}.out")
        elif pair == ("appraisal", "appraise_input"):
            _require_emotions(output["emotions"], f"{label}.out.emotions")
            _require_keys(output["mood"], PAD_FIELDS, f"{label}.out.mood")
        elif pair == ("appraisal", "appraise_turn"):
            _require_keys(output["mood"], PAD_FIELDS, f"{label}.out.mood")
            _require_emotions(output["active_emotions"], f"{label}.out.active_emotions")
            _require_emotions(output["occ_carry"], f"{label}.out.occ_carry")
        elif pair == ("surfacing", "presence_vector"):
            _require_keys(output["affect"], PAD_FIELDS, f"{label}.out.affect")
            _require_keys(
                output["relationship"],
                frozenset({"stage", "trust", "longing", "comfort"}),
                f"{label}.out.relationship",
            )
            _require_keys(
                output["cognition"],
                frozenset(
                    {
                        "reflection_pending",
                        "carried_thought",
                        "due_intention",
                        "open_thread",
                        "intentionality",
                        "curiosity",
                    }
                ),
                f"{label}.out.cognition",
            )
    elif isinstance(output_value, (dict, list)):
        raise ValueError(f"{label}.out must be a scalar")

    if pair == ("appraisal", "appraise_turn"):
        _require_emotions(input_value["occ_carry"], f"{label}.in.occ_carry")
    elif pair == ("surfacing", "presence_vector"):
        _require_keys(
            input_value["cognition"],
            frozenset(
                {
                    "reflection_pending",
                    "due_intention",
                    "open_thread",
                    "intentionality",
                    "curiosity",
                }
            ),
            f"{label}.in.cognition",
        )

    _validate_case_content(
        section,
        group,
        input_value,
        output_value,
        source_label=source_label,
    )
    return {"in": deepcopy(input_value), "out": deepcopy(output_value)}


def _case_count(payload: dict[str, Any]) -> int:
    return sum(
        len(value)
        for section in PUBLIC_SECTIONS
        for key, value in payload[section].items()
        if not key.startswith("_") and isinstance(value, list)
    )


def validate_public_fixture(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise TypeError("public fixture must be an object")
    if set(payload) != {"_meta", *PUBLIC_SECTIONS}:
        raise ValueError("public fixture contains an unexpected section")
    if payload["_meta"] != EXPECTED_META:
        raise ValueError("public fixture metadata does not match the release contract")

    for section, groups in PUBLIC_GROUPS.items():
        section_value = _require_keys(
            payload[section],
            frozenset(groups),
            f"public fixture {section}",
        )
        for group in groups:
            cases = section_value[group]
            if not isinstance(cases, list):
                raise TypeError(f"public fixture {section}.{group} must be a list")
            expected_count = EXPECTED_GROUP_COUNTS[section][group]
            if len(cases) != expected_count:
                raise ValueError(
                    f"public fixture {section}.{group} has {len(cases)} cases; "
                    f"expected {expected_count}"
                )
            for case in cases:
                _sanitize_case(
                    section,
                    group,
                    case,
                    source_label="affect_state",
                )

    if _case_count(payload) != EXPECTED_CASES:
        raise ValueError(f"public fixture must contain exactly {EXPECTED_CASES} cases")


def build_fixture(source: Path) -> dict[str, Any]:
    raw = json.loads(source.read_text(encoding="utf-8"))
    missing = [section for section in PUBLIC_SECTIONS if section not in raw]
    if missing:
        raise ValueError(f"source corpus is missing sections: {', '.join(missing)}")

    public_sections: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for section, groups in PUBLIC_GROUPS.items():
        public_sections[section] = {}
        for group in groups:
            cases = raw[section].get(group)
            if not isinstance(cases, list):
                raise TypeError(f"source corpus group {section}.{group} must be a list")
            public_sections[section][group] = [
                _sanitize_case(
                    section,
                    group,
                    case,
                    source_label="self_core",
                )
                for case in cases
            ]
    for case in public_sections["surfacing"]["presence_vector"]:
        case["out"]["source"] = "affect_state"

    payload = {
        "_meta": {
            **deepcopy(EXPECTED_META),
        },
        **public_sections,
    }
    count = _case_count(payload)
    if count != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} public cases, found {count}")
    validate_public_fixture(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "shared/golden/kernel_golden.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = (
        json.dumps(build_fixture(args.source), ensure_ascii=False, indent=2) + "\n"
    )
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            print(f"STALE: {args.output}")
            return 1
        print(f"OK: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output} ({EXPECTED_CASES} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
