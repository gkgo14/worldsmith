"""3D scene generation (Sec. 3.4): layout first, then one detailed mesh per object.

  1. low-poly layout (agent loop): starting from a seed scene with the source camera, the ground z=0 and an empty
     `Objects` node, the agent builds each object under that node as a low-poly proxy from primitives; it places ground
     contacts by intersecting the camera ray through a pixel with z=0, renders from the source camera and adjusts
     position, size and orientation until the render matches the input. Later stages change geometry, not layout;
  2. reference: crop the input around the visible part of each proxy silhouette (12% margin, 512^2) and fill occluded
     pixels with a render of the isolated proxy;
  3. completion: gpt-image-2 completes the crop into a pixel-aligned reference of the object alone; accepted only if
     its BiRefNet mask has IoU >= 0.55 with the original crop mask, otherwise the proxy-filled crop is used;
  4. TRELLIS.2 generates a mesh per object from its reference;
  5. fit: uniform scale, rotation R_z(psi) about the vertical axis and a translation that minimise the silhouette
     difference to the proxy in the source camera; the mesh inherits the proxy's placement and stays upright;
  6. review: a reviewer keeps either the generated mesh or the proxy for each object.
"""
from __future__ import annotations

import numpy as np

CROP_MARGIN = 0.12
REF_SIZE = 512
COMPLETION_MIN_IOU = 0.55


# --------------------------------------------------------------------------------------------- step 1 helper
def ground_point(pixel_xy, K: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """Intersect the camera ray through `pixel_xy` with the ground plane z = 0 (Blender camera looks along -Z)."""
    u, v = pixel_xy
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -1.0])
    o, d = cam_to_world[:3, 3], cam_to_world[:3, :3] @ d_cam
    if abs(d[2]) < 1e-9 or -o[2] / d[2] <= 0:
        raise ValueError("ray does not hit the ground in front of the camera")
    return o + (-o[2] / d[2]) * d


# --------------------------------------------------------------------------------------------- step 2
def reference_crop_box(visible_mask: np.ndarray, margin: float = CROP_MARGIN):
    """Square box around the visible proxy silhouette, enlarged by `margin` (resized to 512^2 afterwards)."""
    ys, xs = np.nonzero(visible_mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin)
    return int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))


def fill_occluded(crop_rgb: np.ndarray, visible: np.ndarray, isolated_proxy_render: np.ndarray,
                  isolated_silhouette: np.ndarray) -> np.ndarray:
    """Pixels of the object hidden by other objects are replaced by the render of the isolated proxy."""
    out = crop_rgb.copy()
    hidden = isolated_silhouette & ~visible
    out[hidden] = isolated_proxy_render[hidden]
    return out


# --------------------------------------------------------------------------------------------- step 3
def accept_completion(new_alpha: np.ndarray, old_mask: np.ndarray, min_iou: float = COMPLETION_MIN_IOU):
    """Keep the completed reference only if it stays aligned with the scene (BiRefNet mask IoU >= 0.55)."""
    new = new_alpha > 0.5
    iou = float((new & old_mask).sum() / max(1, (new | old_mask).sum()))
    return iou >= min_iou, iou


# --------------------------------------------------------------------------------------------- step 5
def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def fit_to_proxy(render_silhouette, proxy_silhouette: np.ndarray, proxy_bbox, mesh_bbox,
                 yaw_candidates_deg=range(0, 360, 15), refine=True):
    """Uniform scale s, yaw psi about +Z and translation (x, y) minimising silhouette difference in the source camera.

    render_silhouette(s, psi, txy) -> boolean mask of the transformed mesh rendered from the source camera (Blender).
    The initial uniform scale matches the proxy's bounding box; the mesh is placed at the proxy, its base stays on
    z = 0 and it is only rotated about the vertical axis, so it inherits the proxy's placement and stays upright.
    """
    s0 = float(np.cbrt(np.prod(np.asarray(proxy_bbox)) / max(1e-9, np.prod(np.asarray(mesh_bbox)))))
    best = (-1.0, s0, 0.0, np.zeros(2))
    for psi in yaw_candidates_deg:                     # coarse yaw search
        iou = silhouette_iou(render_silhouette(s0, np.radians(psi), np.zeros(2)), proxy_silhouette)
        if iou > best[0]:
            best = (iou, s0, float(np.radians(psi)), np.zeros(2))
    if refine:                                         # local refinement of (s, psi, t)
        iou, s, psi, t = best
        for step in (0.1, 0.05, 0.02):
            for ds, dpsi, dt in [(1 + step, 0, (0, 0)), (1 - step, 0, (0, 0)), (1, step, (0, 0)), (1, -step, (0, 0)),
                                 (1, 0, (step, 0)), (1, 0, (-step, 0)), (1, 0, (0, step)), (1, 0, (0, -step))]:
                cand = (s * ds, psi + dpsi, t + np.asarray(dt) * s)
                c_iou = silhouette_iou(render_silhouette(*cand), proxy_silhouette)
                if c_iou > iou:
                    iou, s, psi, t = c_iou, *cand
        best = (iou, s, psi, t)
    iou, s, psi, t = best
    return {"scale": s, "yaw": psi, "translation_xy": t.tolist(), "silhouette_iou": iou}


# --------------------------------------------------------------------------------------------- pipeline
LAYOUT_PROMPT = """Rebuild the scene in the image as low-poly proxies. Edit the given Blender script: under the empty
`Objects` node add one child per object, built from primitives. Place each object's ground contact by intersecting the
camera ray through its contact pixel with z=0. Render from the source camera and adjust position, size and orientation
until the render matches the input."""

REVIEW_PROMPT = """For this object, compare the generated high-poly mesh and the low-poly proxy, both rendered in the
scene from the source camera, against the input crop. Keep the high-poly mesh only if it is the object and has no
missing surfaces, floating fragments or distorted parts; otherwise keep the proxy. Answer {"use": "high"|"low", "reason"}."""


def run(tb, ws, AgentLoop) -> dict:
    from .agent import Harness
    # 1. layout
    layout = Harness("scene3d_layout", LAYOUT_PROMPT,
                     {"run_script": lambda code: tb.blender.run_script(code, ws.path("work", "scene.blend")),
                      "render_source_view": lambda: tb.blender.render(ws.path("work", "scene.blend"),
                                                                      [ws.inputs["camera"]], ws.path("work", "render"))},
                     max_turns=10)
    AgentLoop(tb, layout, ws).run()
    # pseudocode below: images are loaded from / written to the workspace by helper functions not shown
    objects = ws.inputs.get("proxies", [])            # names of the children of `Objects` after layout
    decisions = []
    for obj in objects:
        # 2. reference: 12%-margin crop around the visible proxy silhouette, occluded pixels filled with the proxy render
        crop, crop_mask = ws.path("refs", f"{obj}.png"), ws.path("refs", f"{obj}_mask.png")
        # 3. completion, kept only if its BiRefNet mask stays aligned with the original crop mask
        completed = tb.image_editor.edit(crop, f"Only the {obj}: same position, size and viewpoint, occluded parts "
                                               f"completed, other objects removed, white background")
        ok, iou = accept_completion(tb.matting.alpha(completed), load_mask(crop_mask))
        reference = completed if ok else crop
        # 4. one mesh per object
        mesh = tb.mesh_generator.generate(reference, crop_mask)
        # 5. fit to the proxy silhouette in the source camera; the layout is not changed
        fit = fit_to_proxy(silhouette_renderer(mesh, ws.inputs["camera"]), proxy_silhouette(obj),
                           proxy_bbox(obj), mesh_bbox(mesh))
        # 6. review: keep the generated mesh or the proxy
        answer = tb.vlm.step([{"role": "user", "content": REVIEW_PROMPT}], [],
                             images=[crop, render_in_scene(mesh, fit), render_in_scene(obj, None)])
        decisions.append({"object": obj, "completion_iou": iou, "fit": fit, "use": answer.get("use", "low")})
    return {"scene": str(ws.path("out", "scene.glb")), "objects": decisions}


# helper stubs used by the pseudocode above (Blender / image I/O)
def load_mask(path): ...
def silhouette_renderer(mesh, camera): ...
def proxy_silhouette(obj): ...
def proxy_bbox(obj): ...
def mesh_bbox(mesh): ...
def render_in_scene(obj_or_mesh, fit): ...
