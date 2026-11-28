# 3D asset editing with Blender

`worldsmith-edit` uses headless Blender to apply ordered object and mesh-part operations to a `.blend` or `.glb`
asset. It comes from the local WorldSmith editing harness backup dated September 17, 2026. This entry point runs
explicit JSON plans and needs no model credentials, GPU or downloaded checkpoints.

## Quick start

Install Python 3.10 or newer and Blender 4.2 LTS, then run `pip install -e .` in the repository.

```bash
mkdir -p runs
blender --background --factory-startup --python-expr 'import bpy; bpy.ops.wm.save_as_mainfile(filepath="runs/cube.blend")'
worldsmith-edit inspect runs/cube.blend
worldsmith-edit apply runs/cube.blend examples/asset_edit_plan.json runs/cube-edit --render
```

On macOS, Blender's executable is typically `/Applications/Blender.app/Contents/MacOS/Blender`. Set `BLENDER_BIN`
if it is outside `PATH`. To run without the installed console script, use `python -m worldsmith.asset_edit` in place of
`worldsmith-edit`.

A successful run writes `delivery.glb`, `report.json`, Blender logs under `ops/` and one `.blend` per operation in
`candidates/`. With `--render`, `renders/before/` and `renders/after/` use the same three cameras framed around the
source. For a second edit, use the last candidate `.blend` as the next input to keep its native mesh data.

## Edit plans

`editable` names source meshes the plan can change or delete. Every other source mesh is protected. `new_objects`
names the meshes the plan must create. `operations` run in order. Use `inspect` to see the source's object names.

| `op` | Required fields | Optional fields |
|---|---|---|
| `add_object` | `name`, exactly one of `primitive` or `source` | `location`, `rotation_deg`, `scale` |
| `duplicate_object` | `object`, `new_name` | `location`, `rotation_deg`, `scale` |
| `delete_object` | `object` | |
| `transform_object` | `object` | `translate`, `rotation_deg`, `scale`, `apply` |
| `scale_object` | `object`, `factor` or `dimensions` | `apply` (defaults to `true`) |
| `set_material` | `object` | `material_name`, `base_color`, `roughness`, `metallic` |
| `delete_part` | `object`, `face_mask` | `fill_holes`, `fill_sides` |
| `transform_part` | `object`, `face_mask` | `translate`, `rotate_deg`, `scale` |
| `add_part` | `object`, `asset_path` | `location`, `rotation_deg`, `scale` |

- `primitive` accepts `cube`, `uv_sphere`, `cylinder`, `cone`, `plane`, `torus` or `monkey`. `source` and `asset_path`
  point to a GLB with one mesh object and no other scene nodes. Relative paths are resolved from the plan file.
- Transforms use Blender units, Z-up coordinates and degrees. Whole-object `translate` is a world-space offset.
  `transform_part` happens in object-local coordinates around the selected vertices' centroid. Vertices shared with
  neighbouring faces move with the selection. `add_part` places and joins the mesh to the target object. It does
  not weld the meshes along a seam.
- `set_material` replaces the target mesh's material slots with a Principled material. `base_color` is RGB or RGBA
  in linear colour space, with components in `[0, 1]`.
- `face_mask` lists zero-based Blender loop-triangle indices in the current candidate. To retrieve the indices,
  run `worldsmith-edit mesh input.blend Object faces.npz`. The archive contains `verts` in world space and indexed
  `tris`. `delete_part` removes the full Blender polygon for each selected triangle. After an operation changes
  topology, export the mesh again from that candidate before choosing indices for the next step.

## Delivery checks and coverage

Before exporting the GLB, the runner compares the protected meshes in the source and final `.blend` by name,
world-space vertex coordinates and triangle indices. It checks that every declared addition exists and that the
candidate still contains mesh triangles. The check measures base meshes. Evaluated modifiers, shape keys, UVs,
materials, rigs and exact GLB equivalence are outside this check. Exit status is `0` for success, `1` for a rejected
candidate and `2` for invalid input or a runtime error. A rejected candidate keeps its report and intermediate
`.blend` files for inspection.

The research agent's automatic mask localisation, model-driven repair and visual judging need the research runtime.
The public editor exposes the underlying Blender operations for explicit plans.
