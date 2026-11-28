"""Sec. 3.3: reliability weight and weighted UV fusion (Eq. 2), CIE76 colour selection, emission map."""
import numpy as np

from worldsmith.emission import atlas_weights, emission_map, fuse_views, select_colors, srgb_to_lab


def _full_view(size=8, res=16, cos=0.8):
    """A view whose pixels cover the whole atlas evenly at one incidence angle."""
    u, v = np.meshgrid((np.arange(res) + 0.5) / res, (np.arange(res) + 0.5) / res)
    return atlas_weights(np.stack([u, v], -1), np.full((res, res), cos), np.ones((res, res), bool), size)


def test_weight_is_squared_cosine_once_enough_pixels_hit_a_texel():
    w = _full_view(cos=0.8)
    assert w.shape == (8, 8)
    assert np.allclose(w, 0.8 ** 2)                 # every texel has >= 4 samples, so the count term saturates at 1


def test_weight_is_zero_where_the_view_sees_nothing():
    res, size = 16, 8
    u, v = np.meshgrid((np.arange(res) + 0.5) / res, (np.arange(res) + 0.5) / res)
    valid = np.zeros((res, res), bool)
    valid[:, : res // 4] = True                     # only the left quarter of the atlas is hit
    w = atlas_weights(np.stack([u, v], -1), np.ones((res, res)), valid, size)
    assert (w[:, 4:] == 0).all() and (w[:, :2] > 0).all()


def test_fusion_follows_the_more_reliable_view():
    w = _full_view()
    mask = np.zeros((8, 8), bool)
    mask[:4] = True
    seen = w > 0
    views = [{"weight": w, "mask": mask, "seen": seen}, {"weight": 0.5 * w, "mask": ~mask, "seen": seen}]
    selected, info = fuse_views(views, 8)
    assert (selected == mask).all()                 # the half-weight view is outvoted everywhere
    assert info["conflict"].all() and info["observed"].all()


def test_unseen_view_cannot_vote_and_ties_are_excluded():
    w = np.ones((4, 4))
    mask = np.ones((4, 4), bool)
    blind = np.zeros((4, 4), bool)
    selected, _ = fuse_views([{"weight": w, "mask": mask, "seen": ~blind},
                              {"weight": 10 * w, "mask": ~mask, "seen": blind}], 4)
    assert selected.all()
    tied, _ = fuse_views([{"weight": w, "mask": mask, "seen": ~blind},
                          {"weight": w, "mask": ~mask, "seen": ~blind}], 4)
    assert not tied.any()


def test_emission_is_base_colour_on_selected_texels_only(rng):
    selected = np.zeros((8, 8), bool)
    selected[2:5, 3:6] = True
    base = rng.random((8, 8, 3))
    e = emission_map(selected, base)
    assert np.allclose(e[selected], base[selected]) and (e[~selected] == 0).all()


def test_srgb_to_lab_reference_values():
    lab = srgb_to_lab([[255, 255, 255], [0, 0, 0], [255, 0, 0]])
    assert np.allclose(lab[0], [100.0, 0.0, 0.0], atol=1e-2)
    assert np.allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(lab[2], [53.24, 80.09, 67.20], atol=2e-2)


def test_colour_selection_keeps_near_colours_inside_visible_region_and_roi():
    image = np.zeros((4, 4, 3), np.uint8)
    image[:2] = [255, 0, 0]
    image[2:] = [0, 0, 255]
    visible = np.ones((4, 4), bool)
    picked = select_colors(image, visible, [[250, 5, 5]])
    assert picked[:2].all() and not picked[2:].any()
    roi = np.zeros((4, 4), bool)
    roi[:, :1] = True
    assert select_colors(image, visible, [[250, 5, 5]], roi=roi).sum() == 2
    visible[0] = False
    assert select_colors(image, visible, [[250, 5, 5]]).sum() == 4
