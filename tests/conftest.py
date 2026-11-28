"""Shared test helpers: a look-at camera and stand-in backends that need no external model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldsmith import Toolbox, ToolResult


def look_at(eye, target=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Camera-to-world matrix of a Blender camera (looks along -Z, +Y up) placed at `eye` and aimed at `target`."""
    eye = np.asarray(eye, float)
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -forward, eye
    return m


class ScriptedModel:
    """Stands in for the vision-language model: replays a fixed list of tool calls, then stops."""

    def __init__(self, calls):
        self.calls = list(calls)
        self.requests = 0

    def step(self, messages, tools, images=()):
        self.requests += 1
        if self.calls:
            return {"role": "assistant", "tool_call": self.calls.pop(0)}
        return {"role": "assistant", "content": "done"}


class StubBlender:
    """Stands in for Blender: reports success and names the files a real run would write."""

    def run_script(self, code, scene):
        return ToolResult(True, [Path(scene)], {"stdout": ""})

    def render(self, scene, cameras, out, **opts):
        return ToolResult(True, [Path(out) / "render.png"])

    def export(self, scene, out, fmt="glb"):
        return ToolResult(True, [Path(out)])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def make_toolbox():
    def make(calls):
        return Toolbox(vlm=ScriptedModel(calls), blender=StubBlender())
    return make
