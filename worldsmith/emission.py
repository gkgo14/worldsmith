"""Surface emission harness (Sec. 3.3, Eq. 2).

The harness unwraps one non-overlapping atlas, renders V=4 unlit base-colour views around the object (azimuth
25/115/205/295 deg, elevation 15 deg) and records, for every pixel, the first texel hit by ray casting, so occluded
surfaces receive no projection. In each view the agent segments the emissive parts (SAM 3 text, SAM 2 points/boxes or
CIE76 colour thresholding with tau_c = 15), checks a zoomed view of the boundary and refines the prompt or adds an
exclusion mask until it accepts the mask m_v. Surfaces that no view sees get extra cameras, up to 12 views in total.

The numerical parts below (reliability weight, weighted fusion, CIE76 selection, emission) run with NumPy.
"""
from __future__ import annotations

import numpy as np

DEFAULT_VIEWS = [{"azimuth": a, "elevation": 15} for a in (25, 115, 205, 295)]
MAX_VIEWS = 12
CIE76_TOLERANCE = 15.0
SATURATION_SAMPLES = 4          # N(v,t)/4 in Eq. 2


# ------------------------------------------------------------------------------------------- Eq. 2, first factor
def atlas_weights(texel_uv: np.ndarray, incidence_cos: np.ndarray, valid: np.ndarray, size: int):
    """w(v,t) = mean(|n . d_v|^2) * min(N(v,t)/4, 1) for one view.

    texel_uv:      (H, W, 2) atlas coordinate of the first surface hit at every pixel
    incidence_cos: (H, W)    |n . d_v| at that hit
    valid:         (H, W)    pixel sees the object
    N(v,t) counts visible pixels whose 2x2 texel footprint covers t (same footprint as the UV projection).
    """
    q, c = texel_uv[valid], incidence_cos[valid].astype(np.float64)
    x = np.clip(np.floor(q[:, 0] * size).astype(int), 0, size - 1)
    y = np.clip(np.floor((1 - q[:, 1]) * size).astype(int), 0, size - 1)
    count = np.zeros((size, size)); angle = np.zeros((size, size))
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = x + dx, y + dy
        inside = (xx < size) & (yy < size)
        np.add.at(count, (yy[inside], xx[inside]), 1)
        np.add.at(angle, (yy[inside], xx[inside]), c[inside] ** 2)
    mean_cos2 = np.divide(angle, count, out=np.zeros_like(angle), where=count > 0)
    return mean_cos2 * np.minimum(count / SATURATION_SAMPLES, 1.0)


# ------------------------------------------------------------------------------------------- Eq. 2, fusion
def fuse_views(per_view: list[dict], size: int):
    """P(t) = sum_v w m_v,  Q(t) = sum_v w (1 - m_v),  s(t) = 1[P > Q] (ties excluded).

    per_view: dicts with 'weight' (size x size, from atlas_weights), 'mask' (selected texels of the accepted m_v)
              and 'seen' (texels this view observes). A view that does not see t has zero weight there, so it cannot
              vote against a selection made in other views.
    """
    P = np.zeros((size, size)); Q = np.zeros((size, size))
    for v in per_view:
        w = v["weight"] * v["seen"]
        P += w * v["mask"]
        Q += w * ~v["mask"]
    selected = (P > Q) & ~np.isclose(P, Q, rtol=1e-6, atol=1e-8)
    return selected, {"observed": (P + Q) > 0, "conflict": (P > 0) & (Q > 0)}


def emission_map(selected: np.ndarray, base_color_linear: np.ndarray) -> np.ndarray:
    """E(t) = s(t) a(t) in linear space: a selected texel emits its own base colour, all others emit nothing."""
    return selected[..., None] * base_color_linear


# ------------------------------------------------------------------------------------------- CIE76 selection tool
def srgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750],
                          [.0193339, .1191920, .9503041]]).T / np.array([.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def select_colors(unlit_render: np.ndarray, visible: np.ndarray, colors, tolerance=CIE76_TOLERANCE, roi=None):
    """Pixels within CIE76 distance `tolerance` of any sampled colour (on the unlit base-colour render)."""
    lab = srgb_to_lab(unlit_render)
    dist = np.min([np.linalg.norm(lab - c, axis=-1) for c in srgb_to_lab(np.asarray(colors, float))], axis=0)
    return (dist <= tolerance) & visible & (np.ones_like(visible) if roi is None else roi)


# ------------------------------------------------------------------------------------------- harness
PROMPT = """Predict which surfaces of the shape emit light. For every view: segment the emissive parts with SAM 3 (text),
SAM 2 (points/boxes) or CIE76 colour selection, inspect a zoomed view of the mask boundary, refine the prompt or add an
exclusion mask until you accept it (an empty mask is a valid decision). If an emissive surface is not visible in any
view, add a camera (at most 12 views). Then fuse, apply and review the exported asset."""


def harness(tb, ws):
    from .agent import Harness
    views = list(DEFAULT_VIEWS)
    tools = {
        "capture_views": lambda: tb.blender.render(ws.inputs["mesh"], views, ws.path("views"), mode="unlit_first_hit"),
        "segment_view": lambda view, text=None, points=None, box=None: tb.segmenter.segment(
            ws.path("views", str(view), "render.png"), text=text, points=points, box=box),
        "add_view": lambda azimuth, elevation: views.append({"azimuth": azimuth, "elevation": elevation})
        if len(views) < MAX_VIEWS else (_ for _ in ()).throw(ValueError("at most 12 views")),
        "fuse_and_apply": lambda: None,     # host: atlas_weights per view -> fuse_views -> emission_map -> GLB
        "run_script": lambda code: tb.blender.run_script(code, ws.path("work", "scene.blend")),
    }
    return Harness("emission", PROMPT, tools, max_turns=30)   # at most 30 model requests per shape (Sec. 4.1)
