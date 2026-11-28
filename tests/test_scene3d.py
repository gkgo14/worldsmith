"""Sec. 3.4: ground-ray placement, reference crop, occlusion fill, completion gate, silhouette fit."""
import numpy as np
import pytest
from conftest import look_at

from worldsmith.scene3d import accept_completion, fill_occluded, fit_to_proxy, ground_point, reference_crop_box

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def _project(point, cam_to_world):
    x, y, z, _ = np.linalg.inv(cam_to_world) @ np.r_[point, 1.0]
    return K[0, 0] * x / -z + K[0, 2], -K[1, 1] * y / -z + K[1, 2]


def test_principal_ray_hits_the_look_at_target():
    cam = look_at([0.0, -5.0, 3.0])
    assert np.allclose(ground_point((320, 240), K, cam), [0.0, 0.0, 0.0], atol=1e-9)


def test_ground_point_inverts_projection():
    cam = look_at([2.0, -5.0, 3.0])
    point = np.array([1.2, 0.7, 0.0])
    assert np.allclose(ground_point(_project(point, cam), K, cam), point)


def test_ray_above_the_horizon_is_rejected():
    with pytest.raises(ValueError):
        ground_point((320, -5000), K, look_at([0.0, -5.0, 3.0]))


def test_reference_crop_is_a_square_with_margin_around_the_silhouette():
    mask = np.zeros((100, 100), bool)
    mask[20:60, 30:50] = True                       # 39 px tall, 19 px wide (max - min)
    x0, y0, side = reference_crop_box(mask)
    assert side == round(39 * 1.12)
    assert np.isclose(x0 + side / 2, 39.5, atol=1) and np.isclose(y0 + side / 2, 39.5, atol=1)


def test_fill_occluded_only_touches_hidden_object_pixels(rng):
    crop = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    visible = np.zeros((100, 100), bool)
    visible[20:60, 30:50] = True
    silhouette = np.zeros((100, 100), bool)
    silhouette[20:80, 30:50] = True                 # rows 60-79 are hidden behind another object
    proxy = np.full_like(crop, 7)
    out = fill_occluded(crop, visible, proxy, silhouette)
    hidden = silhouette & ~visible
    assert (out[hidden] == 7).all() and (out[~hidden] == crop[~hidden]).all()


def test_completion_gate_threshold():
    old = np.zeros((100, 100), bool)
    old[:, :60] = True
    aligned = np.zeros((100, 100))
    aligned[:, :40] = 1.0                           # IoU 40/60
    shifted = np.zeros((100, 100))
    shifted[:, 40:100] = 1.0                        # IoU 20/100
    ok, iou = accept_completion(aligned, old)
    assert ok and np.isclose(iou, 2 / 3)
    ok, iou = accept_completion(shifted, old)
    assert not ok and np.isclose(iou, 0.2)


def test_fit_recovers_scale_yaw_and_shift_of_an_ellipse():
    yy, xx = np.mgrid[:64, :64]

    def render(s, psi, t):
        cx, cy = 32 + 10 * t[0], 32 + 10 * t[1]
        x = (xx - cx) * np.cos(psi) + (yy - cy) * np.sin(psi)
        y = -(xx - cx) * np.sin(psi) + (yy - cy) * np.cos(psi)
        return (x / (12 * s)) ** 2 + (y / (5 * s)) ** 2 <= 1

    target = render(1.3, np.radians(50), np.array([0.2, -0.1]))
    fit = fit_to_proxy(render, target, proxy_bbox=(1.3, 1.3, 1.3), mesh_bbox=(1.0, 1.0, 1.0))
    assert fit["silhouette_iou"] > 0.9
    assert abs(fit["scale"] - 1.3) < 0.1
    assert abs(np.degrees(fit["yaw"]) - 50) < 5
    assert np.allclose(fit["translation_xy"], [0.2, -0.1], atol=0.1)
