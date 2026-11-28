"""Cover native mesh edits and scope rejection on a disposable Blender scene."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from worldsmith.asset_edit import apply_edit, export_mesh, inspect_asset


def test_blender_edits_and_checks_indirect_changes(tmp_path):
    blender = shutil.which(os.environ.get("BLENDER_BIN", "blender"))
    if not blender:
        pytest.skip("requires Blender 4.2 on PATH or BLENDER_BIN")

    source = tmp_path / "source.blend"
    setup = tmp_path / "setup.py"
    setup.write_text(
        "import bpy\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        "bpy.ops.mesh.primitive_cube_add(size=1)\n"
        "parent = bpy.context.object\n"
        "parent.name = 'Target'\n"
        "child = parent.copy()\n"
        "child.name = 'Protected'\n"
        "bpy.context.scene.collection.objects.link(child)\n"
        "child.parent = parent\n"
        "child.location.x = 2\n"
        f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r})\n",
        encoding="utf-8",
    )
    subprocess.run([blender, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(setup)],
                   check=True, capture_output=True, text=True, timeout=120)

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "editable": ["Target"], "new_objects": ["Indicator"], "operations": [
            {"op": "transform_part", "object": "Target", "face_mask": [0, 1], "translate": [0, 0, 0.25]},
            {"op": "add_object", "name": "Indicator", "primitive": "cube", "location": [0, 0, 1], "scale": 0.1},
        ],
    }), encoding="utf-8")
    edited = apply_edit(source, plan, tmp_path / "accepted", blender=blender)
    assert edited["check"] == {"passed": True, "errors": []}
    assert edited["protected"] == ["Protected"]
    meshes = {obj["name"]: obj for obj in inspect_asset(edited["delivery"], blender=blender)["objects"]}
    assert set(meshes) == {"Target", "Protected", "Indicator"}
    assert meshes["Target"]["dimensions"][2] == 1.25
    assert meshes["Protected"]["dimensions"][2] == 1.0

    mesh = tmp_path / "mesh.npz"
    export_mesh(edited["blend"], "Protected", mesh, blender=blender)
    with np.load(mesh, allow_pickle=False) as arrays:
        assert arrays["verts"].shape == (8, 3)
        assert arrays["tris"].shape == (12, 3)

    plan.write_text(json.dumps({
        "editable": ["Target"], "new_objects": [], "operations": [
            {"op": "transform_object", "object": "Target", "translate": [0, 0, 1]},
        ],
    }), encoding="utf-8")
    rejected = apply_edit(source, plan, tmp_path / "rejected", blender=blender)
    assert rejected["check"] == {"passed": False, "errors": ["Protected: protected vertices changed"]}
    assert not (tmp_path / "rejected" / "delivery.glb").exists()
    assert (tmp_path / "rejected" / "report.json").is_file()
    assert Path(edited["blend"]).is_file()
