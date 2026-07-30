import pytest

from jitskilled.slots import apply_patch, load_slots, render_slots, save_slots

BASE = {
    "slots": {
        "input": [{"id": "a", "text": "Slot A text."}],
        "output": [{"id": "b", "text": "Slot B text."}],
    }
}


def test_load_slots_roundtrip(tmp_path):
    path = tmp_path / "slots.yaml"
    save_slots(path, BASE)
    loaded = load_slots(path)
    assert loaded == BASE


def test_load_slots_handles_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    loaded = load_slots(path)
    assert loaded == {"slots": {"input": [], "output": []}}


def test_render_slots_includes_ids_and_text():
    rendered = render_slots(BASE)
    assert "[a]" in rendered
    assert "Slot A text." in rendered
    assert "[b]" in rendered


def test_apply_patch_add_slot_input_category():
    patch = [{"operation": "add_slot", "category": "input", "slot_id": "c", "text": "New."}]
    result = apply_patch(BASE, patch)
    ids = [s["id"] for s in result["slots"]["input"]]
    assert "c" in ids
    # original untouched
    assert [s["id"] for s in BASE["slots"]["input"]] == ["a"]


def test_apply_patch_modify_slot_output_category():
    patch = [{"operation": "modify_slot", "category": "output", "slot_id": "b", "text": "Updated."}]
    result = apply_patch(BASE, patch)
    b = next(s for s in result["slots"]["output"] if s["id"] == "b")
    assert b["text"] == "Updated."


def test_apply_patch_delete_slot():
    patch = [{"operation": "delete_slot", "category": "input", "slot_id": "a", "text": None}]
    result = apply_patch(BASE, patch)
    assert [s["id"] for s in result["slots"]["input"]] == []


def test_apply_patch_delete_missing_slot_is_noop():
    patch = [{"operation": "delete_slot", "category": "input", "slot_id": "does_not_exist"}]
    result = apply_patch(BASE, patch)
    assert result == BASE


def test_apply_patch_rejects_unknown_category():
    patch = [{"operation": "add_slot", "category": "middleware", "slot_id": "x", "text": "x"}]
    with pytest.raises(ValueError, match="unknown category"):
        apply_patch(BASE, patch)


def test_apply_patch_rejects_unknown_operation():
    patch = [{"operation": "rename_slot", "category": "input", "slot_id": "a", "text": "x"}]
    with pytest.raises(ValueError, match="unknown operation"):
        apply_patch(BASE, patch)


def test_apply_patch_rejects_missing_key():
    patch = [{"operation": "add_slot", "category": "input"}]  # missing slot_id
    with pytest.raises(ValueError, match="missing required key"):
        apply_patch(BASE, patch)


def test_apply_patch_rejects_add_without_text():
    patch = [{"operation": "add_slot", "category": "input", "slot_id": "c", "text": ""}]
    with pytest.raises(ValueError, match="non-empty 'text'"):
        apply_patch(BASE, patch)
