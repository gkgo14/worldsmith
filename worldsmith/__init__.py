"""WorldSmith core code.

    from worldsmith.agent import WorldSmith
    WorldSmith(toolbox, root).run("edit" | "emission" | "animate" | "scene3d" | "scene4d", **inputs)
"""
from .agent import AgentLoop, Harness, WorldSmith, Workspace  # noqa: F401
from .tools import Toolbox, ToolResult  # noqa: F401

__version__ = "0.1.0"
