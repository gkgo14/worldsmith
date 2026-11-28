"""Tool interfaces used by the WorldSmith agent (Sec. 3).

Every specialist model is a *tool call* that writes files into the task workspace, where the agent can import,
render and modify them. Only the interfaces are given here; the backends are the public models named in the paper
(Sec. 4.1): a vision-language model (GPT-6 Astra / GPT-6.1 Sol), SAM 2 / SAM 3, gpt-image-2, BiRefNet, TRELLIS.2,
SkinTokens, Puppeteer and Blender (bpy).  This file is interface/pseudocode, not a runnable backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass
class ToolResult:
    ok: bool
    files: list[Path] = field(default_factory=list)   # outputs written into the workspace
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class VLM(Protocol):
    """Vision-language model that plans, writes Blender Python and calls tools (function calling)."""
    def step(self, messages: list[dict], tools: list[dict], images: Sequence[Path] = ()) -> dict: ...


class BlenderExecutor(Protocol):
    """Runs agent-written Blender Python in a sandboxed workspace that only contains the task inputs."""
    def run_script(self, code: str, scene: Path) -> ToolResult: ...
    def render(self, scene: Path, cameras: Sequence[dict], out: Path, **opts) -> ToolResult: ...
    def export(self, scene: Path, out: Path, fmt: str = "glb") -> ToolResult: ...


class Segmenter(Protocol):          # SAM 2 (points/boxes, video tracking) and SAM 3 (text prompts)
    def segment(self, image: Path, *, text: str | None = None, points=None, box=None) -> list[Any]: ...
    def track(self, frames_dir: Path, first_frame_masks: dict[int, Any]) -> Any: ...


class ImageEditor(Protocol):        # gpt-image-2: completes an occluded object crop into a clean reference
    def edit(self, image: Path, prompt: str) -> Path: ...


class Matting(Protocol):            # BiRefNet foreground mask
    def alpha(self, image: Path) -> Any: ...


class MeshGenerator(Protocol):      # TRELLIS.2 image-to-3D
    def generate(self, reference: Path, mask: Path, seed: int = 0) -> Path: ...


class Rigger(Protocol):             # SkinTokens skeleton + Puppeteer skinning weights
    def rig(self, mesh: Path) -> Path: ...


@dataclass
class Toolbox:
    """Everything the agent may call. A tool that is absent for a task is simply not offered to the model."""
    vlm: VLM
    blender: BlenderExecutor
    segmenter: Segmenter | None = None
    image_editor: ImageEditor | None = None
    matting: Matting | None = None
    mesh_generator: MeshGenerator | None = None
    rigger: Rigger | None = None
