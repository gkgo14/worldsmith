"""Sec. 3: the shared loop, driven by a scripted model and a stand-in Blender (no external model needed)."""
import json

import numpy as np
import pytest

from worldsmith import WorldSmith


def _tool_messages(loop_messages):
    return [json.loads(m["content"]) for m in loop_messages if m.get("role") == "tool"]


def _run(make_toolbox, tmp_path, task, calls, **inputs):
    """Run one task and return (result, model, parsed tool messages)."""
    from worldsmith import AgentLoop
    recorded = []
    original = AgentLoop.run

    def spy(self):
        out = original(self)
        recorded.extend(self.messages)
        return out

    toolbox = make_toolbox(calls)
    AgentLoop.run = spy
    try:
        result = WorldSmith(toolbox, tmp_path).run(task, **inputs)
    finally:
        AgentLoop.run = original
    return result, toolbox.vlm, _tool_messages(recorded)


def test_edit_run_ends_at_the_preservation_gate(make_toolbox, tmp_path):
    body = np.zeros((4, 3))
    calls = [{"name": "update_edit_scope", "arguments": {"editable": ["hat"]}},
             {"name": "run_script", "arguments": {"code": "pass"}},
             {"name": "inspect_edit_candidate"}]
    result, model, tools = _run(make_toolbox, tmp_path, "edit", calls, source_objects=["body", "hat"],
                                source_vertices={"body": body}, delivered_vertices={"body": body.copy()})
    assert model.requests == 4                                  # three tool calls, then the model stops
    assert [t["ok"] for t in tools] == [True, True, True]
    assert result["check"] == {"passed": True, "errors": []}
    assert result["delivery"].endswith("delivery.glb")


def test_gate_fails_when_a_protected_object_changed(make_toolbox, tmp_path):
    calls = [{"name": "update_edit_scope", "arguments": {"editable": ["hat"]}}]
    result, _, _ = _run(make_toolbox, tmp_path, "edit", calls, source_objects=["body", "hat"],
                        source_vertices={"body": np.zeros((4, 3))}, delivered_vertices={"body": np.ones((4, 3))})
    assert result["check"]["passed"] is False


def test_rejected_calls_are_reported_to_the_model_instead_of_ending_the_run(make_toolbox, tmp_path):
    calls = [{"name": "update_edit_issues", "arguments": {"issues": [{"id": "x", "category": "nope", "priority": 1}]}},
             {"name": "no_such_tool"},
             {"name": "run_script", "arguments": {"wrong_argument": 1}},
             {"name": "run_script", "arguments": {"code": "pass"}}]
    _, model, tools = _run(make_toolbox, tmp_path, "edit", calls, source_objects=["body"])
    assert model.requests == 5
    assert [t["ok"] for t in tools] == [False, False, False, True]
    assert "unknown issue category" in tools[0]["error"]
    assert "no_such_tool" in tools[1]["error"] and "run_script" in tools[1]["error"]
    assert "TypeError" in tools[2]["error"]


def test_plain_return_values_are_wrapped(make_toolbox, tmp_path):
    calls = [{"name": "run_script", "arguments": {"code": "pass"}}, {"name": "save_candidate"}]
    _, _, tools = _run(make_toolbox, tmp_path, "animate", calls, rigged="rig.glb", video="clip.mp4")
    assert tools[1]["ok"] and tools[1]["data"]["result"].startswith("cand:")


def test_animation_cannot_finish_before_four_candidates_are_compared(make_toolbox, tmp_path):
    calls = [{"name": "save_candidate"}, {"name": "finish_animation", "arguments": {"candidate": "cand:unknown"}}]
    _, _, tools = _run(make_toolbox, tmp_path, "animate", calls, rigged="rig.glb", video="clip.mp4")
    assert tools[1]["ok"] is False and "compare >= 4" in tools[1]["error"]


def test_emission_view_budget(make_toolbox, tmp_path):
    calls = [{"name": "capture_views"}] + [{"name": "add_view", "arguments": {"azimuth": 0, "elevation": 80}}] * 9
    _, _, tools = _run(make_toolbox, tmp_path, "emission", calls, mesh="lamp.glb")
    assert [t["ok"] for t in tools] == [True] * 9 + [False]     # 4 default views + 8 extra = 12, the ninth is refused
    assert "at most 12 views" in tools[-1]["error"]


def test_the_loop_stops_at_max_turns(make_toolbox, tmp_path):
    calls = [{"name": "capture_views"}] * 100
    _, model, _ = _run(make_toolbox, tmp_path, "emission", calls, mesh="lamp.glb")
    assert model.requests == 30                                 # at most 30 model requests per shape


def test_array_inputs_do_not_break_the_first_message(make_toolbox, tmp_path):
    result, _, _ = _run(make_toolbox, tmp_path, "edit", [], source_objects=["body"],
                        source_vertices={"body": np.zeros((4, 3))}, delivered_vertices={"body": np.zeros((4, 3))})
    assert result["check"]["passed"]


def test_scene3d_layout_stage_runs_with_no_objects(make_toolbox, tmp_path):
    result, model, _ = _run(make_toolbox, tmp_path, "scene3d", [{"name": "render_source_view"}],
                            image="room.png", camera={})
    assert result["objects"] == [] and result["scene"].endswith("scene.glb") and model.requests == 2


def test_unknown_task_is_rejected(make_toolbox, tmp_path):
    with pytest.raises(ValueError, match="expected one of"):
        WorldSmith(make_toolbox([]), tmp_path).run("sculpt")
