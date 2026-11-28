"""4D scene generation (Sec. 3.5, Eqs. 3-4): 3D scene of frame 1 + track + auto-rig + reused animation + anchor.

  1. reconstruct the first frame with the 3D scene pipeline (scene3d.py);
  2. find the character: render the visible region Omega_n of every reconstructed object in frame 1; the object whose
     15x15-dilated region accumulates the most temporal change D(p) (Eq. 3) is the character (sum, not mean, so a small
     neighbour inside the mover's dilated region does not win);
  3. prompt SAM 2 with its projected region, take the largest candidate mask, subtract the masks of objects in front of
     it, and track the character and these occluders jointly; composite the character on white and crop all frames with
     one square box covering it in every frame (+12%, 512^2), which keeps its trajectory;
  4. transform C makes the extracted character upright, turns the camera's horizontal viewing direction onto +y and
     moves the centre of its base to the origin; SkinTokens predicts the skeleton, Puppeteer the skinning weights;
  5. animate with the rig-based animation harness (animation.py);
  6. C^-1 maps the result back into the scene, where it replaces the static object;
  7. anchor: one similarity transform (Umeyama, Eq. 4) maps the posed first frame onto the rest shape (= the frame-1
     reconstruction) and is applied to all frames, preserving motion relative to the anchored first frame.
  If tracking or rigging fails, the static reconstruction is kept.
"""
from __future__ import annotations

import math

import numpy as np

DILATION = 15
CROP_MARGIN = 0.12
REF_SIZE = 512


# --------------------------------------------------------------------------------------------- Eq. 3
def temporal_change(frames: np.ndarray) -> np.ndarray:
    """D(p) = 1/T sum_t ||I_t(p) - I_1(p)||_1   (frames: T x H x W x 3, L1 norm over the RGB channels)."""
    F = frames.astype(np.float32)
    return np.abs(F - F[0]).sum(-1).mean(0)


def dilate(mask: np.ndarray, k: int = DILATION) -> np.ndarray:
    r = k // 2
    pad = np.pad(mask, r)
    out = np.zeros_like(mask)
    for dy in range(k):
        for dx in range(k):
            out |= pad[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def motion_scores(frames: np.ndarray, labels: np.ndarray, names: list[str]) -> dict[str, float]:
    """S_n = sum of D over the 15x15 dilation of Omega_n, the frame-1 visible region of object n.

    labels: frame-1 render of the reconstruction, 0 = background, k+1 = names[k]. The sum (not the mean) is used, which
    tolerates small misalignment of the reconstruction.
    """
    D = temporal_change(frames)
    return {n: float(D[dilate(labels == k + 1)].sum()) for k, n in enumerate(names) if (labels == k + 1).any()}


def select_character(scores: dict[str, float]) -> str:
    """The object with the largest S_n is the character."""
    return max(scores, key=scores.get)


# --------------------------------------------------------------------------------------------- step 3
def first_frame_mask(candidates: list[np.ndarray], occluder_masks: list[np.ndarray]) -> np.ndarray:
    """SAM 2 is prompted with the character's projected region; take the largest candidate mask and subtract the masks
    of the reconstructed objects in front of it. The character and these occluders are then tracked together."""
    m = max(candidates, key=lambda c: int(c.sum())).copy()
    for o in occluder_masks:
        m &= ~o
    return m


def union_crop_box(masks: np.ndarray, margin: float = CROP_MARGIN):
    """One square box around the character's union mask over all frames (+12%), keeping its trajectory in the crop."""
    ys, xs = np.nonzero(masks.any(0))
    side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin)) + 2
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def reference_video(frames: np.ndarray, masks: np.ndarray) -> list[np.ndarray]:
    """Character composited on white and cropped with the union box (resize to 512^2 afterwards)."""
    x0, y0, side = union_crop_box(masks)
    out = []
    for f, m in zip(frames, masks):
        comp = np.where(m[..., None], f, 255).astype(np.uint8)
        big = np.pad(comp, ((side, side), (side, side), (0, 0)), constant_values=255)
        out.append(big[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side])
    return out


# --------------------------------------------------------------------------------------------- step 4
def canonical_transform(vertices_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """C (world -> source): rotate about +Z so the camera's horizontal viewing direction becomes +y, base centre at origin."""
    fwd = -camera_to_world[:3, 2]                                  # Blender camera looks along -Z
    yaw = math.atan2(fwd[1], fwd[0]); phi = math.pi / 2 - yaw
    lo, hi = vertices_world.min(0), vertices_world.max(0)
    base = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])   # ground stays at z = 0
    R = np.eye(4); c, s = math.cos(phi), math.sin(phi)
    R[:2, :2] = [[c, -s], [s, c]]
    T = np.eye(4); T[:3, 3] = -base
    return R @ T


# --------------------------------------------------------------------------------------------- Eq. 4
def umeyama(posed_frame1: np.ndarray, rest: np.ndarray):
    """(sigma, Q, b) = argmin sum_i || sigma Q x_i(1) + b - x_i ||^2 over corresponding vertices (same mesh)."""
    mp, mr = posed_frame1.mean(0), rest.mean(0)
    P, R = posed_frame1 - mp, rest - mr
    U, S, Vt = np.linalg.svd(R.T @ P / len(P))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    Q = U @ D @ Vt
    sigma = float((S * np.diag(D)).sum() / (P ** 2).sum() * len(P))
    b = mr - sigma * Q @ mp
    A = np.eye(4); A[:3, :3] = sigma * Q; A[:3, 3] = b
    return A                                      # applied to every frame: x_t <- A x_t


# --------------------------------------------------------------------------------------------- pipeline
def run(tb, ws, AgentLoop, run_task) -> dict:
    """Composition of the other harnesses; any failure in tracking/rigging falls back to the static scene."""
    scene = run_task("scene3d", image=ws.inputs["first_frame"], camera=ws.inputs["camera"])          # step 1
    try:
        frames, labels, names = ws.inputs["frames"], ws.inputs["frame1_labels"], ws.inputs["object_names"]
        character = select_character(motion_scores(frames, labels, names))                         # step 2
        k = names.index(character)
        candidates = tb.segmenter.segment(ws.path("frames", "00000.jpg"), box=labels == k + 1)       # step 3
        occluders = ws.inputs.get("occluder_masks", [])          # reconstructed objects in front of the character
        m0 = first_frame_mask(candidates, occluders)
        tracks = tb.segmenter.track(ws.path("frames"), {1: m0, **{2 + i: o for i, o in enumerate(occluders)}})
        masks = tracks[1]                                        # per frame: character wins over every occluder
        ref = reference_video(frames, masks)
        C = canonical_transform(ws.inputs["character_vertices_world"], ws.inputs["camera_to_world"])  # step 4
        rigged = tb.rigger.rig(ws.path("character", "char.glb"))
        anim = run_task("animate", rigged=str(rigged), video=ref)                                    # step 5
        # step 6-7 (Blender): import the animation under a root with matrix C^-1, remove the static object, then
        #   A = umeyama(posed_frame1_world, rest_world); root.matrix_world = A @ root.matrix_world
        return {"scene": scene, "character": character, "animation": anim, "C": C.tolist(), "fallback": None}
    except Exception as e:                                                                           # keep static scene
        return {"scene": scene, "fallback": f"static reconstruction kept ({type(e).__name__})"}
