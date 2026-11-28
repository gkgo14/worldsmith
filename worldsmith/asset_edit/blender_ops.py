"""Headless Blender operations adapted from the WorldSmith 3D editing backup.

Blender runs this worker in a separate process for each operation. Face masks
index the target mesh's Blender loop triangles in their current order.
"""

import json
import sys
import traceback
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector


def _argv():
    argv = sys.argv
    return argv[argv.index("--") + 1:]


def _load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_obj(name):
    obj = bpy.context.scene.objects.get(name)
    if obj is None:
        raise ValueError(f"object not found: {name}")
    return obj


def _mesh_arrays(obj):
    """World-space vertices and triangle indices for a mesh object."""
    if obj.type != 'MESH':
        raise ValueError(f"{obj.name} is not a mesh")
    mesh = obj.data
    mesh.calc_loop_triangles()
    mw = obj.matrix_world
    verts = np.array([tuple(mw @ v.co) for v in mesh.vertices], dtype=np.float64).reshape(-1, 3)
    tris = np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int64).reshape(-1, 3)
    return verts, tris


def op_inspect(params):
    objects = []
    for o in bpy.context.scene.objects:
        entry = {"name": o.name, "type": o.type,
                 "parent": o.parent.name if o.parent else None}
        if o.type == "MESH":
            o.data.calc_loop_triangles()
            bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
            entry.update({
                "verts": len(o.data.vertices),
                "faces": len(o.data.polygons),
                "tris": len(o.data.loop_triangles),
                "dimensions": [round(d, 5) for d in o.dimensions],
                "bbox_min": [round(min(v[i] for v in bb), 5) for i in range(3)],
                "bbox_max": [round(max(v[i] for v in bb), 5) for i in range(3)],
                "materials": [m.name for m in o.data.materials if m],
            })
        objects.append(entry)
    return {"objects": objects, "up_axis": "Z"}


def op_export_mesh(params):
    obj = _get_obj(params["object"])
    verts, tris = _mesh_arrays(obj)
    np.savez(params["out_npz"], verts=verts, tris=tris)
    return {"object": obj.name, "out_npz": params["out_npz"],
            "num_verts": int(len(verts)), "num_tris": int(len(tris))}


def op_snapshot(params):
    meshes = sorted((obj for obj in bpy.context.scene.objects if obj.type == 'MESH'), key=lambda obj: obj.name)
    arrays = {}
    for index, obj in enumerate(meshes):
        arrays[f"verts_{index}"], arrays[f"tris_{index}"] = _mesh_arrays(obj)
    np.savez(params["path"], **arrays)
    return {"names": [obj.name for obj in meshes]}


def _look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def op_render_views(params):
    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res = params["resolution"]
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 12
    scene.cycles.device = 'CPU'
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("Edit Preview World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.35, 0.35, 1)
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True
    lo, hi = map(Vector, params["bounds"])
    center = (lo + hi) / 2
    size = max((hi - lo).length, 1e-3)

    cam = bpy.data.objects.new("Edit Preview Camera", bpy.data.cameras.new("Edit Preview Camera"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.clip_start = size * 0.001
    cam.data.clip_end = size * 100
    written = []
    for name, yaw, pitch in (("front", 0, 12), ("threequarter", 45, 20), ("back", 180, 12)):
        yaw_r, pitch_r = np.radians(yaw), np.radians(pitch)
        r = size * 1.55
        offset = Vector((r * np.cos(pitch_r) * np.sin(yaw_r),
                         -r * np.cos(pitch_r) * np.cos(yaw_r),
                         r * np.sin(pitch_r)))
        cam.location = center + offset
        _look_at(cam, center)
        path = str(out_dir / f"{name}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        written.append(path)
    return {"renders": written}


def op_export_glb(params):
    result = bpy.ops.export_scene.gltf(filepath=params["path"], export_format='GLB')
    if 'FINISHED' not in result:
        raise RuntimeError("Blender could not export the GLB")
    return {"path": params["path"]}


def _save_blend(path):
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=path):
        raise RuntimeError("Blender could not save the candidate")


def op_import_glb(params):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if 'FINISHED' not in bpy.ops.import_scene.gltf(filepath=params["path"]):
        raise RuntimeError("Blender could not import the GLB")
    return {"source": params["path"]}


def _editable_mesh(obj):
    if obj.type != 'MESH':
        raise ValueError(f"{obj.name} is not a mesh")
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    return obj.data


def _import_single_mesh(path):
    before = set(bpy.context.scene.objects)
    if 'FINISHED' not in bpy.ops.import_scene.gltf(filepath=path):
        raise RuntimeError("Blender could not import the replacement GLB")
    meshes = [obj for obj in bpy.context.scene.objects if obj not in before]
    if len(meshes) != 1 or meshes[0].type != 'MESH':
        raise ValueError("replacement GLB must contain exactly one mesh object")
    return meshes[0]


def _apply_trs(obj, params):
    """Shared location/rotation_deg/scale handling for creation-style ops."""
    if params.get("location") is not None:
        obj.location = Vector(params["location"])
    if params.get("rotation_deg") is not None:
        angles = [np.radians(float(a)) for a in params["rotation_deg"]]
        if len(angles) != 3:
            raise ValueError("rotation_deg must have 3 components")
        obj.rotation_euler = angles
    if params.get("scale") is not None:
        s = params["scale"]
        obj.scale = (float(s),) * 3 if isinstance(s, (int, float)) else Vector(s)


def _select_only(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def op_add_object(params):
    """Add a mesh primitive or import one mesh object from a GLB."""
    source, primitive = params.get("source"), params.get("primitive")
    if bool(source) == bool(primitive):
        raise ValueError("add_object requires exactly one of source or primitive")
    if source:
        obj = _import_single_mesh(source)
    else:
        primitives = {"cube": bpy.ops.mesh.primitive_cube_add,
                      "uv_sphere": bpy.ops.mesh.primitive_uv_sphere_add,
                      "cylinder": bpy.ops.mesh.primitive_cylinder_add,
                      "cone": bpy.ops.mesh.primitive_cone_add,
                      "plane": bpy.ops.mesh.primitive_plane_add,
                      "torus": bpy.ops.mesh.primitive_torus_add,
                      "monkey": bpy.ops.mesh.primitive_monkey_add}
        if primitive not in primitives:
            raise ValueError(f"unknown primitive {primitive!r}; "
                             f"available: {sorted(primitives)}")
        before = set(bpy.data.objects)
        primitives[primitive]()
        obj = next(o for o in bpy.data.objects if o not in before)
    obj.name = params["name"]
    _apply_trs(obj, params)
    bpy.context.view_layer.update()
    return {"object": obj.name, "type": obj.type,
            "dimensions": [round(d, 5) for d in obj.dimensions]}


def op_delete_object(params):
    obj = _get_obj(params["object"])
    name = obj.name
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
        bpy.data.meshes.remove(data)
    return {"deleted": name}


def op_duplicate_object(params):
    obj = _get_obj(params["object"])
    if obj.type != 'MESH':
        raise ValueError(f"{obj.name} is not a mesh")
    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.name = params["new_name"]
    bpy.context.scene.collection.objects.link(dup)
    _apply_trs(dup, params)
    return {"object": dup.name, "source": obj.name}


def op_transform_object(params):
    """Translate (world-space offset) / rotation_deg (absolute) / scale (absolute)
    an existing object; optionally bake the transform into the mesh data."""
    obj = _get_obj(params["object"])
    if params.get("translate") is not None:
        matrix = obj.matrix_world.copy()
        matrix.translation += Vector(params["translate"])
        obj.matrix_world = matrix
    if params.get("rotation_deg") is not None:
        angles = [np.radians(float(a)) for a in params["rotation_deg"]]
        if len(angles) != 3:
            raise ValueError("rotation_deg must have 3 components")
        obj.rotation_euler = angles
    if params.get("scale") is not None:
        s = params["scale"]
        obj.scale = (float(s),) * 3 if isinstance(s, (int, float)) else Vector(s)
    if params.get("apply"):
        _editable_mesh(obj)
        _select_only(obj)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.view_layer.update()
    return {"object": obj.name, "location": [round(v, 5) for v in obj.matrix_world.translation],
            "dimensions": [round(d, 5) for d in obj.dimensions],
            "applied": bool(params.get("apply"))}


def op_set_material(params):
    """Create a Principled material and assign it to every slot of the object."""
    obj = _get_obj(params["object"])
    mesh = _editable_mesh(obj)
    mat = bpy.data.materials.new(params.get("material_name", "edit_mat"))
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if params.get("base_color") is not None:
        c = [float(x) for x in params["base_color"]]
        if len(c) not in (3, 4):
            raise ValueError("base_color must have 3 or 4 components")
        bsdf.inputs["Base Color"].default_value = (c + [1.0])[:4]
    if params.get("roughness") is not None:
        bsdf.inputs["Roughness"].default_value = float(params["roughness"])
    if params.get("metallic") is not None:
        bsdf.inputs["Metallic"].default_value = float(params["metallic"])
    mesh.materials.clear()
    mesh.materials.append(mat)
    for polygon in mesh.polygons:
        polygon.material_index = 0
    return {"object": obj.name, "material": mat.name}


def op_delete_part(params):
    """Part-level deletion: remove masked triangles, drop orphaned verts,
    optionally fill the boundary holes left behind."""
    obj = _get_obj(params["object"])
    mesh = _editable_mesh(obj)
    mesh.calc_loop_triangles()
    n_tris = len(mesh.loop_triangles)
    ids = _face_indices(params["face_mask"], n_tris)
    polys = sorted({int(mesh.loop_triangles[i].polygon_index) for i in ids})
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    doomed = [bm.faces[i] for i in polys]
    cut_edges = {edge for face in doomed for edge in face.edges if not edge.is_boundary}
    initial_verts = len(bm.verts)
    bmesh.ops.delete(bm, geom=doomed, context='FACES')
    filled = 0
    if params.get("fill_holes"):
        boundary = [edge for edge in cut_edges if edge.is_valid and edge.is_boundary]
        if boundary:
            res = bmesh.ops.holes_fill(bm, edges=boundary,
                                       sides=int(params.get("fill_sides", 0)))
            filled = len(res.get("faces", []))
    removed_verts = initial_verts - len(bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return {"object": obj.name, "removed_faces": len(polys),
            "removed_verts": removed_verts, "filled_faces": filled}


def op_transform_part(params):
    """Transform only the vertices referenced by the masked triangles, around
    the selection centroid. Coordinates are object-local (equal to world space
    when the object transform is identity). Verts shared with unmasked faces
    move along — that is how meshes stay connected."""
    obj = _get_obj(params["object"])
    mesh = _editable_mesh(obj)
    mesh.calc_loop_triangles()
    tris = np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int64)
    ids = _face_indices(params["face_mask"], len(tris))
    vert_ids = np.unique(tris[ids].ravel())
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[int(i)] for i in vert_ids]
    center = Vector(np.mean([tuple(v.co) for v in verts], axis=0))
    rot = Matrix.Identity(4)
    if params.get("rotate_deg") is not None:
        rx, ry, rz = [np.radians(float(a)) for a in params["rotate_deg"]]
        rot = (Matrix.Rotation(rz, 4, 'Z') @ Matrix.Rotation(ry, 4, 'Y')
               @ Matrix.Rotation(rx, 4, 'X'))
    s = params.get("scale", 1.0)
    s = (float(s),) * 3 if isinstance(s, (int, float)) else tuple(s)
    sc = Matrix.Diagonal((*s, 1.0))
    t = Vector(params.get("translate", (0.0, 0.0, 0.0)))
    mat = Matrix.Translation(center + t) @ rot @ sc @ Matrix.Translation(-center)
    for v in verts:
        v.co = mat @ v.co
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return {"object": obj.name, "moved_verts": len(verts),
            "centroid": [round(c, 5) for c in center]}


def op_add_part(params):
    """Import a single-mesh GLB, place it, and join it into the target mesh."""
    target = _get_obj(params["object"])
    _editable_mesh(target)
    part = _import_single_mesh(params["asset_path"])
    _apply_trs(part, params)
    bpy.ops.object.select_all(action='DESELECT')
    part.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    if 'FINISHED' not in bpy.ops.object.join():
        raise RuntimeError("Blender could not join the replacement mesh")
    return {"object": target.name}


def op_scale_object(params):
    """Uniform factor or exact target dimensions for one object. The object's
    world position is kept; only scale changes (baked by default)."""
    obj = _get_obj(params["object"])
    before_dims = [float(d) for d in obj.dimensions]
    if (params.get("factor") is None) == (params.get("dimensions") is None):
        raise ValueError("scale_object requires exactly one of factor or dimensions")
    if params.get("factor") is not None:
        f = float(params["factor"])
        if f <= 0:
            raise ValueError("factor must be positive")
        factors = Vector((f, f, f))
    else:
        target = [float(d) for d in params["dimensions"]]
        if len(target) != 3 or min(target) <= 0:
            raise ValueError("dimensions must be 3 positive numbers")
        if min(before_dims) == 0:
            raise ValueError("cannot scale a flat mesh to positive dimensions on all axes")
        factors = Vector(wanted / current for wanted, current in zip(target, before_dims))
    obj.scale = Vector(obj.scale) * factors
    if params.get("apply", True):
        _editable_mesh(obj)
        _select_only(obj)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()
    return {"object": obj.name, "applied_scale": [round(f, 6) for f in factors],
            "dimensions_before": [round(d, 5) for d in before_dims],
            "dimensions_after": [round(d, 5) for d in obj.dimensions]}


def _face_indices(values, count):
    if not isinstance(values, list) or not values or any(type(i) is not int or not 0 <= i < count for i in values):
        raise ValueError(f"face_mask must contain triangle indices in 0..{count - 1}")
    return values


OPS = {
    "inspect": op_inspect,
    "export_mesh": op_export_mesh,
    "snapshot": op_snapshot,
    "render_views": op_render_views,
    "export_glb": op_export_glb,
    "import_glb": op_import_glb,
    "add_object": op_add_object,
    "delete_object": op_delete_object,
    "duplicate_object": op_duplicate_object,
    "transform_object": op_transform_object,
    "set_material": op_set_material,
    "delete_part": op_delete_part,
    "transform_part": op_transform_part,
    "add_part": op_add_part,
    "scale_object": op_scale_object,
}


def main():
    op, params_path, out_path = _argv()
    params = _load_params(params_path)
    try:
        out_blend = params.pop("out_blend", None)
        result = OPS[op](params)
        if out_blend is not None:
            _save_blend(out_blend)
            result["out_blend"] = out_blend
        result["ok"] = True
    except Exception:
        result = {"ok": False, "error": traceback.format_exc()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
