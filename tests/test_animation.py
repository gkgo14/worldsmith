"""Sec. 3.2: linear blend skinning (Eq. 1), rig description, candidate pool with blind A/B review."""
import numpy as np

from worldsmith.animation import REQUIRED_CANDIDATES, CandidatePool, describe_rig, linear_blend_skinning


def _weights(rng, n, j):
    w = rng.random((n, j))
    return w / w.sum(1, keepdims=True)


def test_lbs_identity_leaves_rest_pose(rng):
    x, w = rng.normal(size=(50, 3)), _weights(rng, 50, 3)
    out = linear_blend_skinning(x, w, np.stack([np.eye(4)] * 3), np.eye(4))
    assert np.allclose(out, x)


def test_lbs_same_translation_on_bones_or_armature(rng):
    x, w = rng.normal(size=(50, 3)), _weights(rng, 50, 3)
    shift = np.eye(4)
    shift[:3, 3] = [1.0, 2.0, 3.0]
    identity = np.stack([np.eye(4)] * 3)
    by_bones = linear_blend_skinning(x, w, np.stack([shift] * 3), np.eye(4))
    by_armature = linear_blend_skinning(x, w, identity, shift)
    assert np.allclose(by_bones, x + [1.0, 2.0, 3.0])
    assert np.allclose(by_armature, x + [1.0, 2.0, 3.0])


def test_lbs_one_bone_rotates_its_vertices(rng):
    x = rng.normal(size=(50, 3))
    w = np.zeros((50, 3))
    w[:, 1] = 1.0
    c, s = np.cos(0.7), np.sin(0.7)
    rot = np.eye(4)
    rot[:2, :2] = [[c, -s], [s, c]]
    out = linear_blend_skinning(x, w, np.stack([np.eye(4), rot, np.eye(4)]), np.eye(4))
    assert np.allclose(out, x @ rot[:3, :3].T)


def test_describe_rig_reports_region_per_bone(rng):
    bones = [{"name": f"b{j}", "head": [0, 0, j], "tail": [0, 0, j + 1], "parent": None if j == 0 else f"b{j - 1}",
              "local_axes": np.eye(3).tolist()} for j in range(3)]
    vertices = rng.normal(size=(30, 3))
    w = np.zeros((30, 3))
    w[:20, 0] = 1.0
    w[20:, 1] = 1.0
    rig = describe_rig(bones, w, vertices)
    assert [b["name"] for b in rig] == ["b0", "b1", "b2"]
    assert np.isclose(rig[0]["vertex_share"], 20 / 30) and np.isclose(rig[1]["vertex_share"], 10 / 30)
    assert rig[2]["vertex_share"] == 0.0 and rig[2]["moves_region_bbox"] is None
    assert np.allclose(rig[0]["moves_region_bbox"], [vertices[:20].min(0), vertices[:20].max(0)])


def test_delivery_needs_four_candidates_and_a_compared_one():
    pool = CandidatePool()
    ids = [pool.save(f"snapshot_{i}") for i in range(REQUIRED_CANDIDATES - 1)]
    pool.compare(lambda **kw: {"preferred": "A"}, "video.mp4", ids[0], ids[1])
    assert not pool.can_deliver(ids[0])            # only three candidates saved
    ids.append(pool.save("snapshot_last"))
    assert pool.can_deliver(ids[0])
    assert not pool.can_deliver(ids[-1])           # saved but never compared


def test_blind_review_maps_the_verdict_back_to_a_candidate():
    pool = CandidatePool()
    a, b = pool.save("a"), pool.save("b")
    seen = {}

    def reviewer(**kw):
        seen.update(kw)
        return {"preferred": "B"}

    record = pool.compare(reviewer, "video.mp4", a, b)
    assert set(record["mapping"]) == {"A", "B"} and set(record["mapping"].values()) == {a, b}
    assert record["preferred"] == record["mapping"]["B"]
    assert set(seen) == {"video", "A", "B", "ask"}        # the reviewer gets the video and two unnamed candidates
    assert pool.compare(lambda **kw: {"preferred": "uncertain"}, "video.mp4", a, b)["preferred"] is None
