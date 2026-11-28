"""Sec. 3.1: edit scope, issue ledger with candidate-bound evidence, source-preservation gate."""
import numpy as np
import pytest

from worldsmith.editing import EditState, IssueLedger, source_preservation_gate


def test_objects_outside_the_declared_scope_are_protected():
    state = EditState(source_objects=["body", "hat", "wing"])
    assert not state.declared
    state.declare_scope(["hat"], ["halo"])
    assert state.declared and state.protected == ["body", "wing"] and state.new_objects == ["halo"]


def test_scope_rejects_unknown_objects_and_reused_names():
    state = EditState(source_objects=["body", "hat"])
    with pytest.raises(ValueError):
        state.declare_scope(["tail"])
    with pytest.raises(ValueError):
        state.declare_scope(["hat"], ["body"])


def test_ledger_puts_the_highest_priority_open_issue_first():
    ledger = IssueLedger()
    ledger.observe_candidate("scene.blend", "r1")
    evidence = ledger.register_evidence("render")
    ledger.update([{"id": "colour", "category": "material", "priority": 3, "evidence_ids": [evidence]},
                   {"id": "floating", "category": "placement_contact", "priority": 5}])
    assert ledger.summary()["largest_issue"]["id"] == "floating"
    assert ledger.summary()["other_open"] == ["colour"]
    ledger.update([{"id": "floating", "category": "placement_contact", "priority": 5, "status": "resolved"}])
    assert ledger.summary()["largest_issue"]["id"] == "colour"


def test_evidence_goes_stale_when_the_candidate_changes():
    ledger = IssueLedger()
    ledger.observe_candidate("scene.blend", "r1")
    evidence = ledger.register_evidence("render")
    ledger.update([{"id": "colour", "category": "material", "priority": 3, "evidence_ids": [evidence]}])
    current = ledger.summary()["largest_issue"]
    assert current["assessment_is_current"] and current["evidence"][0]["freshness"] == "current"

    assert ledger.observe_candidate("scene.blend", "r1")["id"] == current["assessed_candidate_id"]   # unchanged asset
    ledger.observe_candidate("scene.blend", "r2")                                                     # asset edited
    stale = ledger.summary()["largest_issue"]
    assert not stale["assessment_is_current"]
    assert stale["evidence"][0]["freshness"] == "stale_or_other_candidate"


def test_ledger_rejects_malformed_updates():
    ledger = IssueLedger()
    with pytest.raises(ValueError):
        ledger.register_evidence("render")                                # no candidate bound yet
    with pytest.raises(ValueError):
        ledger.update([{"id": "x", "category": "not_a_category", "priority": 3}])
    with pytest.raises(ValueError):
        ledger.update([{"id": "x", "category": "other", "priority": 9}])
    with pytest.raises(ValueError):
        ledger.update([{"id": "x", "category": "other", "priority": 3, "evidence_ids": ["evidence:made_up"]}])
    ledger.update([{"id": "low", "category": "other", "priority": 1}, {"id": "high", "category": "other", "priority": 4}])
    with pytest.raises(ValueError):
        ledger.update([], largest_issue_id="low")                         # not the highest open priority


def test_empty_ledger_is_not_an_acceptance():
    summary = IssueLedger().summary()
    assert summary["largest_issue"] is None and "not an acceptance" in summary["status"]


def test_preservation_gate():
    source = {"body": np.zeros((4, 3)), "wing": np.ones((4, 3))}
    same = {"body": np.zeros((4, 3)), "wing": np.ones((4, 3)), "halo": np.ones((9, 3))}
    assert source_preservation_gate(source, same, ["body", "wing"]) == {"passed": True, "errors": []}

    moved = {"body": np.zeros((4, 3)) + 1e-6}
    result = source_preservation_gate(source, moved, ["body", "wing"])
    assert not result["passed"] and len(result["errors"]) == 2            # body changed, wing missing

    retopologised = {"body": np.zeros((5, 3))}
    assert not source_preservation_gate(source, retopologised, ["body"])["passed"]
    assert source_preservation_gate(source, moved, ["body"], tol=1e-5)["passed"]
