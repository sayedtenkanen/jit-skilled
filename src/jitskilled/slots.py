"""Load, render, and patch the editable slot library YAML."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_CATEGORIES = ("input", "output")
_OPERATIONS = ("add_slot", "delete_slot", "modify_slot")


def load_slots(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("slots", {}).setdefault("input", [])
    data["slots"].setdefault("output", [])
    return data


def save_slots(path: str | Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def render_slots(data: dict[str, Any]) -> str:
    lines = []
    for category in _CATEGORIES:
        slots = data["slots"].get(category, [])
        if not slots:
            continue
        lines.append(f"{category.upper()} SLOTS:")
        for slot in slots:
            lines.append(f"  [{slot['id']}] {slot['text']}")
    return "\n".join(lines)


def apply_patch(data: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply an ordered list of add_slot/delete_slot/modify_slot operations.
    Returns a new dict; does not mutate the input.

    Raises ValueError with a specific message on any malformed operation
    (missing key, unknown operation, unknown category, or missing text for
    add/modify) rather than letting a bad LLM response silently corrupt or
    no-op the slot library.
    """
    new_data = copy.deepcopy(data)
    for i, op in enumerate(operations):
        missing = [k for k in ("operation", "category", "slot_id") if k not in op]
        if missing:
            raise ValueError(f"operation #{i} missing required key(s): {missing} ({op!r})")
        if op["operation"] not in _OPERATIONS:
            raise ValueError(f"operation #{i} has unknown operation {op['operation']!r}")
        if op["category"] not in _CATEGORIES:
            raise ValueError(
                f"operation #{i} has unknown category {op['category']!r}; "
                f"must be one of {_CATEGORIES}"
            )

        slot_list = new_data["slots"].setdefault(op["category"], [])
        slot_id = op["slot_id"]
        existing_idx = next(
            (j for j, s in enumerate(slot_list) if s["id"] == slot_id), None
        )
        if op["operation"] == "delete_slot":
            if existing_idx is not None:
                slot_list.pop(existing_idx)
        else:  # add_slot or modify_slot
            if not op.get("text"):
                raise ValueError(f"operation #{i} ({op['operation']}) requires non-empty 'text'")
            entry = {"id": slot_id, "text": op["text"]}
            if existing_idx is not None:
                slot_list[existing_idx] = entry
            else:
                slot_list.append(entry)
    return new_data
