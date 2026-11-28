"""WorldSmith: one agent loop for five 3D/4D creation tasks (Sec. 3, Fig. 2).

    task inputs -> VLM observes and decides -> writes Blender Python / calls a specialist tool -> executes in Blender
    -> renders and inspects its own result -> revises -> host checks -> delivery

The loop is shared. Each task plugs in a *harness*: the task prompt, the tools it may call, the host-side state it
keeps (issue ledger, candidate pool, ...) and the delivery check. The agent never sees ground-truth geometry, motion
or evaluation scores; its only observations are the task inputs and renders/measurements of its own candidates.

`WorldSmith.run(task, **inputs)` is the single entry point; 4D scene generation is a composition of the 3D scene
harness and the animation harness (Sec. 3.5).  Model calls are pseudocode (see tools.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .tools import Toolbox, ToolResult


@dataclass
class Harness:
    """Task-specific part of the loop."""
    name: str
    system_prompt: str
    tools: dict[str, Callable[..., ToolResult]]           # name -> host function exposed to the model
    on_tool_result: Callable[[str, ToolResult], None] = lambda name, r: None   # e.g. bind evidence to candidate
    context: Callable[[], dict] = lambda: {}               # bounded live context added to every request
    deliver: Callable[[Path], dict] = lambda p: {"passed": True}               # host delivery check
    max_turns: int = 60


@dataclass
class Workspace:
    root: Path
    inputs: dict[str, Any] = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _brief(value: Any) -> str:
    """Text form of a value that JSON cannot hold: arrays are described by shape and type, the rest by str()."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"<array shape={tuple(value.shape)} dtype={value.dtype}>"
    return str(value)


class AgentLoop:
    """Shared observe -> act -> execute -> inspect -> revise loop (pseudocode)."""

    def __init__(self, toolbox: Toolbox, harness: Harness, workspace: Workspace):
        self.tb, self.h, self.ws = toolbox, harness, workspace
        self.messages: list[dict] = [
            {"role": "system", "content": harness.system_prompt},
            {"role": "user", "content": json.dumps({"task_inputs": workspace.inputs}, default=_brief)},
        ]

    def tool_specs(self) -> list[dict]:
        return [{"name": n, "callable": f.__doc__ or ""} for n, f in self.h.tools.items()]

    def call_tool(self, call: dict) -> ToolResult:
        """Run one tool call and always return a ToolResult.

        A call the host rejects (unknown tool, wrong arguments, a check that raises ValueError) goes back to the
        model as an error it can act on. A plain return value (None, an ID, a list of masks) is wrapped.
        """
        fn = self.h.tools.get(call["name"])
        if fn is None:
            return ToolResult(False, error=f"unknown tool {call['name']!r}; available: {sorted(self.h.tools)}")
        try:
            out = fn(**call.get("arguments", {}))
        except (ValueError, TypeError) as e:
            return ToolResult(False, error=f"{type(e).__name__}: {e}")
        if isinstance(out, ToolResult):
            return out
        return ToolResult(True, data={} if out is None else {"result": out})

    def run(self) -> dict:
        for _ in range(self.h.max_turns):
            # 1. the model sees the inputs, its previous actions and the *current* harness context
            self.messages.append({"role": "user", "content": json.dumps({"context": self.h.context()}, default=_brief)})
            reply = self.tb.vlm.step(self.messages, self.tool_specs())
            self.messages.append(reply)
            call = reply.get("tool_call")
            if call is None:                      # the agent decides when the result is good enough
                break
            # 2. execute: Blender Python (run_script) or a specialist tool; outputs land in the workspace
            result = self.call_tool(call)
            # 3. host bookkeeping (e.g. bind renders as evidence of the candidate that produced them)
            self.h.on_tool_result(call["name"], result)
            self.messages.append({"role": "tool", "name": call["name"],
                                  "content": json.dumps({"ok": result.ok, "data": result.data,
                                                         "files": [str(p) for p in result.files],
                                                         "error": result.error}, default=_brief)})
        delivery = self.ws.path("out", "delivery.glb")
        check = self.h.deliver(delivery)         # e.g. source-preservation gate, native-vs-GLB frame check
        return {"delivery": str(delivery), "check": check}


class WorldSmith:
    """Unified entry point for the five tasks of the paper."""

    TASKS = ("edit", "emission", "animate", "scene3d", "scene4d")

    def __init__(self, toolbox: Toolbox, root: Path):
        self.tb, self.root = toolbox, Path(root)

    def run(self, task: str, **inputs) -> dict:
        from . import animation, editing, emission, scene3d, scene4d
        ws = Workspace(self.root / task, inputs)
        if task == "edit":        # Sec. 3.1: previous output M_{t-1}, instruction I_t, condition image C_t, camera K
            return AgentLoop(self.tb, editing.harness(self.tb, ws), ws).run()
        if task == "emission":    # Sec. 3.3: mesh, non-emissive base colour, thumbnail
            return AgentLoop(self.tb, emission.harness(self.tb, ws), ws).run()
        if task == "animate":     # Sec. 3.2: rigged rest mesh + video
            return AgentLoop(self.tb, animation.harness(self.tb, ws), ws).run()
        if task == "scene3d":     # Sec. 3.4: image + camera
            return scene3d.run(self.tb, ws, AgentLoop)
        if task == "scene4d":     # Sec. 3.5: video + camera = scene3d(frame 1) + track + rig + animate + anchor
            return scene4d.run(self.tb, ws, AgentLoop, self.run)
        raise ValueError(f"unknown task {task!r}; expected one of {self.TASKS}")
