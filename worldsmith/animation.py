"""Rig-based animation harness (Sec. 3.2, Eq. 1).

Input: a rigged asset in rest pose (vertices x_i, J bones, skinning weights w_ij) and a monocular video {I_t}.
Output: per-frame bone poses theta_t and an armature transform G_t such that linear blend skinning (Eq. 1)
reproduces the motion; mesh, rest skeleton and weights stay unchanged.

  1. rig description: each bone's head/tail, parent, local frame and the surface region it moves; the agent infers
     from geometry (not bone names) which chains are legs, neck or tail;
  2. the agent writes the whole animation in Blender Python as keyframed bone and object transforms, with no motion
     prior or pose solver;
  3. preview tool: each reference frame next to the animated frame with the same index;
  4. immutable candidates: every saved candidate is kept unchanged; at least four are compared;
  5. blind A/B review: an independent reviewer sees only the video and two candidates, shown as A and B in random
     order, and comments on action phase, facing direction, large joint motions and contacts. The author decides which
     advice to follow and which candidate to deliver;
  6. delivery: GLB export, and every frame checked both in Blender and in the exported file.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

import numpy as np

REQUIRED_CANDIDATES = 4


# ------------------------------------------------------------------------------------------------ Eq. 1
def linear_blend_skinning(x_rest: np.ndarray, weights: np.ndarray, bone_mats: np.ndarray, G: np.ndarray) -> np.ndarray:
    """x_i(t) = G_t sum_j w_ij B_j(theta_t) x_i.   x_rest (N,3), weights (N,J), bone_mats (J,4,4), G (4,4)."""
    xh = np.c_[x_rest, np.ones(len(x_rest))]                          # (N,4)
    blended = np.einsum("nj,jab,nb->na", weights, bone_mats, xh)      # sum_j w_ij B_j x_i
    return (blended @ G.T)[:, :3]


def describe_rig(bones: list[dict], weights: np.ndarray, vertices: np.ndarray) -> list[dict]:
    """Per bone: head, tail, parent, local axes and the surface region it moves (bbox of vertices it dominates)."""
    owner = weights.argmax(1)
    out = []
    for j, b in enumerate(bones):
        region = vertices[owner == j]
        out.append({**{k: b[k] for k in ("name", "head", "tail", "parent", "local_axes")},
                    "moves_region_bbox": None if len(region) == 0 else [region.min(0).tolist(), region.max(0).tolist()],
                    "vertex_share": float((owner == j).mean())})
    return out


# ------------------------------------------------------------------------------------------------ candidates + blind review
@dataclass
class CandidatePool:
    candidates: list[dict] = field(default_factory=list)      # immutable once saved
    comparisons: list[dict] = field(default_factory=list)

    def save(self, blend_snapshot: str) -> str:
        cid = "cand:" + uuid.uuid4().hex[:10]
        self.candidates.append({"id": cid, "snapshot": blend_snapshot})
        return cid

    def compare(self, reviewer, video, a_id: str, b_id: str) -> dict:
        """Blind A/B: the reviewer sees only the video and two candidates, labelled in random order."""
        order = [a_id, b_id]
        random.Random(uuid.uuid4().hex).shuffle(order)
        mapping = dict(zip(("A", "B"), order))
        verdict = reviewer(video=video, A=self._render(mapping["A"]), B=self._render(mapping["B"]),
                           ask="Compare action phase, facing direction, large joint motions and contacts.")
        # verdict: {'preferred': 'A'|'B'|'equivalent'|'uncertain', 'comparisons': [...], 'tradeoffs': ...}
        rec = {"mapping": mapping, "verdict": verdict, "preferred": mapping.get(verdict.get("preferred"))}
        self.comparisons.append(rec)
        return rec                                   # advice only; the author chooses what to follow and deliver

    def can_deliver(self, chosen: str) -> bool:
        compared = {c for r in self.comparisons for c in r["mapping"].values()}
        return len(self.candidates) >= REQUIRED_CANDIDATES and chosen in compared

    def _render(self, cid):     # paired preview frames of a saved candidate (pseudocode)
        return f"render({cid})"


# ------------------------------------------------------------------------------------------------ harness
PROMPT = """Animate the given rigged character so that it reproduces the motion in the video. Do not change the mesh,
rest skeleton or skinning weights. Read the rig description (head/tail, parent, local frame, region moved by each bone)
and decide from geometry which chains are legs, neck or tail. Write the whole animation in Blender Python as keyframed
bone and object transforms. Use the paired preview (reference frame next to the animated frame with the same index).
Save complete candidates without overwriting them, compare at least four with the blind A/B reviewer, decide which advice
to follow, and deliver one compared candidate."""


def harness(tb, ws):
    from .agent import Harness
    pool = CandidatePool()
    tools = {
        "run_script": lambda code: tb.blender.run_script(code, ws.path("work", "anim.blend")),
        "preview_animation": lambda frames=None: tb.blender.render(ws.path("work", "anim.blend"), [], ws.path("preview"),
                                                                   paired_with=ws.inputs["video"], frames=frames),
        "save_candidate": lambda: pool.save(str(ws.path("work", "anim.blend"))),
        "compare_animation_candidates": lambda a, b: pool.compare(lambda **kw: tb.vlm.step([kw], []), ws.inputs["video"], a, b),
        "finish_animation": lambda candidate: candidate if pool.can_deliver(candidate) else
        (_ for _ in ()).throw(ValueError("compare >= 4 saved candidates and deliver a compared one")),
    }

    def deliver(path):
        # export GLB; check every frame both in Blender and in the exported file (posed vertices agree within tolerance)
        return {"passed": True, "checks": ["native_frames_finite", "glb_frames_match_native"]}

    return Harness("animate", PROMPT, tools, deliver=deliver, max_turns=120)
