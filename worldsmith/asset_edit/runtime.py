"""Import a source asset, apply ordered operations and check the delivery."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..editing import EditState, source_preservation_gate


_TRANSFORM = {"location", "rotation_deg", "scale"}
# Required and optional parameters for the public edit operations. Output paths
# belong to the runner so each step writes a separate Blender candidate.
_OPERATIONS = {
    "add_object": ({"name"}, {"primitive", "source"} | _TRANSFORM),
    "duplicate_object": ({"object", "new_name"}, _TRANSFORM),
    "delete_object": ({"object"}, set()),
    "transform_object": ({"object"}, {"translate", "rotation_deg", "scale", "apply"}),
    "scale_object": ({"object"}, {"factor", "dimensions", "apply"}),
    "set_material": ({"object"}, {"material_name", "base_color", "roughness", "metallic"}),
    "delete_part": ({"object", "face_mask"}, {"fill_holes", "fill_sides"}),
    "transform_part": ({"object", "face_mask"}, {"translate", "rotate_deg", "scale"}),
    "add_part": ({"object", "asset_path"}, _TRANSFORM),
}


class BlenderError(RuntimeError):
    """A Blender operation failed. Its logs are available in the run directory."""


class _Blender:
    def __init__(self, directory: Path, executable: str | Path | None):
        binary = shutil.which(str(executable or os.environ.get("BLENDER_BIN") or "blender"))
        if binary is None:
            raise ValueError("Blender executable not found. Set BLENDER_BIN or pass --blender.")
        self.executable = str(Path(binary).resolve())
        self.logs = directory / "ops"
        self.logs.mkdir()
        self.count = 0

    def run(self, op: str, scene: Path | None = None, **params) -> dict:
        self.count += 1
        job = self.logs / f"{self.count:03d}_{op}"
        job.mkdir()
        request, reply = job / "params.json", job / "result.json"
        request.write_text(json.dumps(params, allow_nan=False), encoding="utf-8")
        command = [self.executable, "--background", "--factory-startup", "--disable-autoexec"]
        if scene is not None:
            command.append(str(scene))
        script = Path(__file__).with_name("blender_ops.py")
        command += ["--python-exit-code", "1", "--python", str(script), "--", op, str(request), str(reply)]
        with (job / "blender.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(command, check=False, timeout=600, stdout=log, stderr=subprocess.STDOUT)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BlenderError(f"{op}: {exc}. See {job / 'blender.log'}") from exc
        if process.returncode or not reply.is_file():
            raise BlenderError(f"{op}: Blender exited with status {process.returncode}. See {job / 'blender.log'}")
        result = json.loads(reply.read_text(encoding="utf-8"))
        if not result.pop("ok"):
            error = result["error"].strip().splitlines()[-1]
            raise BlenderError(f"{op}: {error}. See {reply}")
        return result


def _file(path: str | Path, extensions: tuple[str, ...]) -> Path:
    path = Path(path).resolve()
    if path.suffix.lower() not in extensions or not path.is_file():
        raise ValueError(f"expected an existing {' or '.join(extensions)} file: {path}")
    return path


def _source_scene(source: Path, directory: Path, blender: _Blender) -> Path:
    if source.suffix.lower() == ".blend":
        return source
    scene = directory / "source.blend"
    blender.run("import_glb", path=str(source), out_blend=str(scene))
    return scene


def inspect_asset(source: str | Path, *, blender: str | Path | None = None) -> dict:
    """List mesh names, triangle counts and world-space bounds for a .blend or .glb."""
    source = _file(source, (".blend", ".glb"))
    with tempfile.TemporaryDirectory(prefix="worldsmith-inspect-") as temp:
        directory = Path(temp)
        runner = _Blender(directory, blender)
        scene = _source_scene(source, directory, runner)
        return runner.run("inspect", scene)


def export_mesh(source: str | Path, name: str, output: str | Path, *, blender: str | Path | None = None) -> dict:
    """Write world-space verts and indexed tris for one Blender mesh object to an NPZ."""
    source = _file(source, (".blend", ".glb"))
    output = Path(output).resolve()
    if output.suffix != ".npz" or output.exists():
        raise ValueError("mesh output must be a new .npz file")
    with tempfile.TemporaryDirectory(prefix="worldsmith-mesh-") as temp:
        directory = Path(temp)
        runner = _Blender(directory, blender)
        scene = _source_scene(source, directory, runner)
        result = runner.run("export_mesh", scene, object=name, out_npz=str(directory / "mesh.npz"))
        with output.open("xb") as out, (directory / "mesh.npz").open("rb") as mesh:
            shutil.copyfileobj(mesh, out)
        return {**result, "out_npz": str(output)}


def _names(values, key: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(name, str) or not name.strip() for name in values):
        raise ValueError(f"{key} must be a list of nonempty object names")
    if len(values) != len(set(values)):
        raise ValueError(f"{key} contains duplicate names")
    return values


def _read_plan(path: Path) -> tuple[list[str], list[str], list[tuple[str, dict]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"editable", "new_objects", "operations"}:
        raise ValueError("edit plan must have editable, new_objects and operations")
    editable = _names(raw["editable"], "editable")
    new_objects = _names(raw["new_objects"], "new_objects")
    if not isinstance(raw["operations"], list) or not raw["operations"]:
        raise ValueError("operations must be a nonempty list")
    steps = []
    for index, step in enumerate(raw["operations"], 1):
        if not isinstance(step, dict) or not isinstance(step.get("op"), str) or step["op"] not in _OPERATIONS:
            raise ValueError(f"operation {index} has an unknown op")
        op = step["op"]
        params = {key: value for key, value in step.items() if key != "op"}
        required, optional = _OPERATIONS[op]
        missing, extra = required - params.keys(), params.keys() - required - optional
        if missing or extra:
            raise ValueError(f"operation {index} ({op}): missing {sorted(missing)}, unknown {sorted(extra)}")
        for key in params.keys() & {"name", "new_name", "object"}:
            if not isinstance(params[key], str) or not params[key].strip():
                raise ValueError(f"operation {index} ({op}) needs a nonempty {key}")
        if op in {"add_object", "duplicate_object"}:
            name = params.get("name") if op == "add_object" else params["new_name"]
            if name not in new_objects:
                raise ValueError(f"operation {index} ({op}) must use a name declared in new_objects")
        elif params["object"] not in editable and params["object"] not in new_objects:
            raise ValueError(f"operation {index} ({op}) targets a mesh outside the edit scope")
        for key in params.keys() & {"source", "asset_path"}:
            params[key] = str(_file(path.parent / params[key], (".glb",)))
        json.dumps(params, allow_nan=False)
        steps.append((op, params))
    return editable, new_objects, steps


def _snapshot(runner: _Blender, scene: Path, path: Path) -> tuple[dict, dict]:
    names = runner.run("snapshot", scene, path=str(path))["names"]
    with np.load(path, allow_pickle=False) as snapshot:
        verts = {name: snapshot[f"verts_{index}"] for index, name in enumerate(names)}
        faces = {name: snapshot[f"tris_{index}"] for index, name in enumerate(names)}
    if any(not np.isfinite(array).all() for array in verts.values()):
        raise BlenderError(f"non-finite vertex coordinates in {scene}")
    return verts, faces


def _check_scope(state: EditState, source: tuple[dict, dict], edited: tuple[dict, dict]) -> dict:
    source_verts, source_faces = source
    edited_verts, edited_faces = edited
    errors = source_preservation_gate(source_verts, edited_verts, state.protected)["errors"]
    for name in state.protected:
        if name in edited_faces and not np.array_equal(source_faces[name], edited_faces[name]):
            errors.append(f"{name}: protected triangles changed")
    additions = edited_verts.keys() - source_verts.keys()
    errors.extend(f"{name}: undeclared new mesh" for name in sorted(additions - set(state.new_objects)))
    errors.extend(f"{name}: declared mesh was not created" for name in sorted(set(state.new_objects) - additions))
    if not any(len(faces) for faces in edited_faces.values()):
        errors.append("the delivery has no mesh triangles")
    return {"passed": not errors, "errors": errors}


def _bounds(vertices: dict) -> list:
    arrays = [array for array in vertices.values() if len(array)]
    return [np.min([array.min(axis=0) for array in arrays], axis=0).tolist(),
            np.max([array.max(axis=0) for array in arrays], axis=0).tolist()]


def apply_edit(source: str | Path, plan: str | Path, output: str | Path, *,
               blender: str | Path | None = None, render: bool = False) -> dict:
    """Apply a JSON plan in a new output directory and export accepted geometry."""
    source = _file(source, (".blend", ".glb"))
    editable, new_objects, steps = _read_plan(Path(plan).resolve())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner = _Blender(output, blender)
    original = _source_scene(source, output, runner)
    before = _snapshot(runner, original, output / "source_meshes.npz")
    if not any(len(faces) for faces in before[1].values()):
        raise ValueError("source asset has no mesh triangles")
    state = EditState(list(before[0]))
    state.declare_scope(editable, new_objects)
    candidates = output / "candidates"
    candidates.mkdir()
    scene = original
    results = []
    for index, (op, params) in enumerate(steps, 1):
        candidate = candidates / f"{index:03d}.blend"
        results.append({"op": op, **runner.run(op, scene, **params, out_blend=str(candidate))})
        if not candidate.is_file():
            raise BlenderError(f"{op} did not save {candidate}")
        scene = candidate
    after = _snapshot(runner, scene, output / "edited_meshes.npz")
    check = _check_scope(state, before, after)
    result = {"check": check, "protected": state.protected, "operations": results}
    if check["passed"]:
        if render:
            bounds = _bounds(before[0])
            for label, asset in (("before", original), ("after", scene)):
                result[label] = runner.run("render_views", asset, bounds=bounds, resolution=384,
                                           out_dir=str(output / "renders" / label))["renders"]
        with tempfile.TemporaryDirectory(dir=output) as temp:
            delivery = Path(temp) / "delivery.glb"
            runner.run("export_glb", scene, path=str(delivery))
            if not delivery.is_file() or delivery.stat().st_size == 0:
                raise BlenderError("Blender did not write the GLB delivery")
            delivery.replace(output / "delivery.glb")
        result["delivery"] = str(output / "delivery.glb")
        result["blend"] = str(scene)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
