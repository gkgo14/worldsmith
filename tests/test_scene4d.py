"""Sec. 3.5: motion score (Eq. 3), tracking mask, union crop, canonical transform, Umeyama anchor (Eq. 4)."""
import numpy as np
from conftest import look_at

from worldsmith.scene4d import (canonical_transform, dilate, first_frame_mask, motion_scores, reference_video,
                                select_character, temporal_change, umeyama, union_crop_box)


def _clip():
    """Six frames: one bright square moves right, a second object stays still."""
    frames = np.full((6, 40, 40, 3), 50, np.uint8)
    for t in range(6):
        frames[t, 10:16, 5 + 3 * t:11 + 3 * t] = 200
    labels = np.zeros((40, 40), int)
    labels[10:16, 5:11] = 1
    labels[30:36, 30:36] = 2
    return frames, labels


def test_temporal_change_is_zero_for_a_static_clip():
    frames = np.full((4, 8, 8, 3), 90, np.uint8)
    assert (temporal_change(frames) == 0).all()


def test_dilation_grows_one_pixel_to_a_15x15_block():
    mask = np.zeros((31, 31), bool)
    mask[15, 15] = True
    grown = dilate(mask)
    assert grown.sum() == 15 * 15 and grown[8:23, 8:23].all()


def test_moving_object_is_selected_as_the_character():
    frames, labels = _clip()
    scores = motion_scores(frames, labels, ["mover", "still", "not_in_frame"])
    assert set(scores) == {"mover", "still"}        # an object with no visible region gets no score
    assert scores["mover"] > 0 and scores["still"] == 0
    assert select_character(scores) == "mover"


def test_first_frame_mask_takes_largest_candidate_minus_occluders():
    small, large = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    small[:2] = True
    large[:6] = True
    occluder = np.zeros((10, 10), bool)
    occluder[:, :3] = True
    mask = first_frame_mask([small, large], [occluder])
    assert (mask == (large & ~occluder)).all()
    assert large[:6].all()                          # the candidate itself is not modified


def test_union_crop_covers_the_character_in_every_frame(rng):
    masks = np.zeros((4, 50, 60), bool)
    for t in range(4):
        masks[t, 10:20, 2 + 10 * t:12 + 10 * t] = True
    x0, y0, side = union_crop_box(masks)
    ys, xs = np.nonzero(masks.any(0))
    assert x0 <= xs.min() and y0 <= ys.min() and x0 + side > xs.max() and y0 + side > ys.max()

    frames = rng.integers(0, 255, (4, 50, 60, 3), dtype=np.uint8)
    video = reference_video(frames, masks)
    assert all(f.shape == (side, side, 3) and f.dtype == np.uint8 for f in video)
    for t, crop in enumerate(video):                # character pixels kept, everything else white
        inside = crop[10 - y0:20 - y0, 2 + 10 * t - x0:12 + 10 * t - x0]
        assert (inside == frames[t, 10:20, 2 + 10 * t:12 + 10 * t]).all()
        assert (crop == 255).all(-1).sum() >= side * side - 100


def test_canonical_transform_faces_plus_y_with_base_at_origin(rng):
    cam = look_at([3.0, -4.0, 2.0])
    vertices = rng.normal(size=(40, 3)) + [2.0, 1.0, 1.0]
    vertices[:, 2] -= vertices[:, 2].min()          # the character stands on the ground
    C = canonical_transform(vertices, cam)
    view_dir = -cam[:3, 2] * [1.0, 1.0, 0.0]
    turned = C[:3, :3] @ (view_dir / np.linalg.norm(view_dir))
    assert np.allclose(turned, [0.0, 1.0, 0.0], atol=1e-9)
    base = np.r_[(vertices.min(0) + vertices.max(0))[:2] / 2, 0.0, 1.0]     # centre of the bounding box, on the ground
    assert np.allclose(C @ base, [0.0, 0.0, 0.0, 1.0], atol=1e-9)
    moved = (np.c_[vertices, np.ones(40)] @ C.T)[:, :3]
    assert np.allclose(moved[:, 2], vertices[:, 2])                         # heights unchanged, ground stays z = 0
    assert np.isclose(np.linalg.det(C[:3, :3]), 1.0)


def test_umeyama_recovers_a_known_similarity(rng):
    posed = rng.normal(size=(200, 3))
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(0.9) * k + (1 - np.cos(0.9)) * k @ k
    scale, shift = 1.7, np.array([0.3, -2.0, 5.0])
    rest = scale * posed @ rot.T + shift
    A = umeyama(posed, rest)
    assert np.allclose(A[:3, :3], scale * rot) and np.allclose(A[:3, 3], shift)
    assert np.allclose((np.c_[posed, np.ones(200)] @ A.T)[:, :3], rest)


def test_umeyama_never_returns_a_reflection(rng):
    flat = rng.normal(size=(100, 3))
    flat[:, 2] = 0.0                                # planar points: a mirror image fits equally well
    mirrored = flat * [1.0, -1.0, 1.0]
    assert np.linalg.det(umeyama(flat, mirrored)[:3, :3]) > 0
