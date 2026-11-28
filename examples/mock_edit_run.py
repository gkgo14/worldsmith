"""Run the shared agent loop on the editing task with stand-in backends.

Nothing here calls a real model or Blender. A scripted "model" replays six tool calls and a stub "Blender" reports
success, so you can watch what the host side does: the edit scope, the issue ledger, an assessment that goes stale
when the asset changes, and the source-preservation gate at delivery.

    python examples/mock_edit_run.py

To run WorldSmith for real, replace `ScriptedModel` and `StubBlender` with your own backends (see worldsmith/tools.py).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldsmith import AgentLoop, Toolbox, ToolResult, Workspace, editing  # noqa: E402

PLAN = [
    ("update_edit_scope", {"editable": ["hat"], "new_objects": ["halo"]}),
    ("run_script", {"code": "# add a halo above the hat\nimport bpy"}),
    ("inspect_edit_candidate", {}),
    # an issue that cites evidence the host never registered is refused
    ("update_edit_issues", {"issues": [{"id": "halo_too_low", "category": "placement_contact", "priority": 4,
                                        "evidence_ids": ["evidence:made_up"]}]}),
    ("update_edit_issues", {"issues": [{"id": "halo_too_low", "category": "placement_contact", "priority": 4}]}),
    # changing the asset makes the assessment above out of date
    ("run_script", {"code": "# raise the halo\nimport bpy"}),
]


class StubBlender:
    def run_script(self, code, scene):
        return ToolResult(True, [Path(scene)], {"stdout": f"ran {len(code.splitlines())} line(s)"})

    def render(self, scene, cameras, out, **opts):
        return ToolResult(True, [Path(out) / "render.png"])

    def export(self, scene, out, fmt="glb"):
        return ToolResult(True, [Path(out)])


class ScriptedModel:
    """Replays PLAN. A real backend would read `messages` and decide the next call itself."""

    def __init__(self, verbose=True):
        self.calls, self.verbose = list(PLAN), verbose

    def step(self, messages, tools, images=()):
        issue = json.loads(messages[-1]["content"])["context"].get("largest_issue")
        if self.verbose:
            seen = "no open issue" if issue is None else \
                f"largest issue = {issue['id']}, assessed on the current asset = {issue['assessment_is_current']}"
            print(f"\nmodel sees  : {seen}")
        if not self.calls:
            return {"role": "assistant", "content": "done"}
        name, arguments = self.calls.pop(0)
        if self.verbose:
            print(f"model calls : {name}({json.dumps(arguments)[:96]})")
        return {"role": "assistant", "tool_call": {"name": name, "arguments": arguments}}


class PrintingLoop(AgentLoop):
    def call_tool(self, call):
        result = super().call_tool(call)
        print(f"host answers: ok={result.ok}" + (f", {result.error}" if result.error else ""))
        return result


def run_edit(root, delivered_body, source_body, loop_cls, verbose):
    toolbox = Toolbox(vlm=ScriptedModel(verbose), blender=StubBlender())
    ws = Workspace(Path(root) / "edit", {"instruction": "add a halo above the hat", "source_objects": ["body", "hat"],
                                         "source_vertices": {"body": source_body},
                                         "delivered_vertices": {"body": delivered_body}})
    return loop_cls(toolbox, editing.harness(toolbox, ws), ws).run()


def main():
    body = np.random.default_rng(0).normal(size=(200, 3))
    with tempfile.TemporaryDirectory() as root:
        kept = run_edit(root, body.copy(), body, PrintingLoop, verbose=True)
        moved = run_edit(root, body + 0.001, body, AgentLoop, verbose=False)
    print("\ndelivery check, protected object untouched :", kept["check"])
    print("delivery check, protected object moved 1 mm:", moved["check"])


if __name__ == "__main__":
    main()
