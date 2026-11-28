"""Command-line access to the Blender asset editor."""

import argparse
import json
import sys

from .runtime import BlenderError, apply_edit, export_mesh, inspect_asset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="worldsmith-edit", description=__doc__)
    parser.add_argument("--blender", help="path to Blender, or set BLENDER_BIN")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="list objects in a .blend or .glb")
    inspect.add_argument("source")
    mesh = commands.add_parser("mesh", help="write triangle indices and vertex positions to an NPZ")
    mesh.add_argument("source")
    mesh.add_argument("name")
    mesh.add_argument("output")
    apply = commands.add_parser("apply", help="apply an edit plan and export a checked GLB")
    apply.add_argument("source")
    apply.add_argument("plan")
    apply.add_argument("output", help="a new run directory")
    apply.add_argument("--render", action="store_true", help="render fixed-camera before/after previews")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_asset(args.source, blender=args.blender)
        elif args.command == "mesh":
            result = export_mesh(args.source, args.name, args.output, blender=args.blender)
        else:
            result = apply_edit(args.source, args.plan, args.output, blender=args.blender, render=args.render)
    except (BlenderError, OSError, ValueError, TypeError) as exc:
        print(f"worldsmith-edit: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(args.command == "apply" and not result["check"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
