"""3D asset editing harness (Sec. 3.1).

Host-side mechanisms:
  * edit scope: before editing, the agent declares which source objects it may change and which new objects it adds;
    every other source object is protected (EditState);
  * issue ledger: the agent records what still differs from the instruction, each issue with a category, a 1-5
    priority and the renders/measurements that show it, and works on the most serious open issue (IssueLedger);
  * candidate-bound evidence: renders/measurements are bound to the candidate version that produced them; once the
    asset changes, earlier renders no longer count as evidence;
  * source-preservation gate: the delivery is marked incomplete if any protected vertex differs from the source.
Local editing itself (face localisation with PartField / multi-view face voting / SAM, deletion, repair with new
faces, retexture) is written by the agent in Blender Python and is not reproduced here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

CATEGORIES = ("silhouette_shape", "depth_orientation", "placement_contact", "material", "inherited_occlusion", "other")
STATUSES = ("open", "resolved", "deferred")


# ----------------------------------------------------------------------------------------------- edit scope
@dataclass
class EditState:
    source_objects: list[str]
    editable: list[str] = field(default_factory=list)
    new_objects: list[str] = field(default_factory=list)
    declared: bool = False

    def declare_scope(self, editable: list[str], new_objects: list[str] = ()):  # tool: update_edit_scope
        unknown = set(editable) - set(self.source_objects)
        if unknown:
            raise ValueError(f"unknown source objects {sorted(unknown)}")
        if set(new_objects) & set(self.source_objects):
            raise ValueError("new objects cannot reuse source object names")
        self.editable, self.new_objects, self.declared = sorted(editable), sorted(new_objects), True

    @property
    def protected(self) -> list[str]:
        return sorted(set(self.source_objects) - set(self.editable))


# ----------------------------------------------------------------------------------------------- issue ledger
@dataclass
class IssueLedger:
    candidate: dict | None = None                 # {'id', 'revision'} of the asset currently being inspected
    issues: dict[str, dict] = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    largest: str | None = None

    # host hook: a new or changed candidate gets a new ID, so old evidence becomes stale
    def observe_candidate(self, path: str, revision: str) -> dict:
        if self.candidate and self.candidate["path"] == path and self.candidate["revision"] == revision:
            return self.candidate
        self.candidate = {"id": "candidate:" + uuid.uuid4().hex[:12], "path": path, "revision": revision}
        return self.candidate

    # host hook: only after an actual observation (render, projection, contact ...) of a bound candidate
    def register_evidence(self, kind: str, detail: str = "") -> str:
        if self.candidate is None:
            raise ValueError("bind a candidate before registering evidence")
        eid = "evidence:" + uuid.uuid4().hex[:12]
        self.evidence.append({"id": eid, "candidate_id": self.candidate["id"], "kind": kind, "detail": detail})
        return eid

    # tool for the agent: update_edit_issues
    def update(self, issues: list[dict], largest_issue_id: str | None = None):
        for it in issues:
            if it["category"] not in CATEGORIES:
                raise ValueError("unknown issue category")
            if not (isinstance(it["priority"], int) and 1 <= it["priority"] <= 5):
                raise ValueError("priority must be an integer 1..5 (5 = largest)")
            known = {e["id"] for e in self.evidence}
            if set(it.get("evidence_ids", [])) - known:
                raise ValueError("issues may only cite registered evidence")
            self.issues[it["id"]] = {**self.issues.get(it["id"], {}), **it,
                                     "assessed_candidate_id": (self.candidate or {}).get("id")}
        open_ = sorted((i for i in self.issues.values() if i.get("status", "open") == "open"),
                       key=lambda i: (-i["priority"], i["id"]))
        top = {i["id"] for i in open_ if open_ and i["priority"] == open_[0]["priority"]}
        if largest_issue_id is not None and largest_issue_id not in top:
            raise ValueError("largest_issue_id must be open and have the highest declared priority")
        self.largest = largest_issue_id or (self.largest if self.largest in top else (open_[0]["id"] if open_ else None))

    # bounded live context added to every model request: largest issue first, with evidence freshness
    def summary(self) -> dict:
        cid = (self.candidate or {}).get("id")
        idx = {e["id"]: e for e in self.evidence}
        issue = self.issues.get(self.largest)
        if issue is None:
            return {"largest_issue": None, "status": "no_open_assessment (not an acceptance)"}
        ev = [{"id": r, "freshness": "current" if idx[r]["candidate_id"] == cid else "stale_or_other_candidate"}
              for r in issue.get("evidence_ids", [])]
        return {"largest_issue": {**issue, "evidence": ev,
                                  "assessment_is_current": issue.get("assessed_candidate_id") == cid},
                "other_open": [i["id"] for i in self.issues.values()
                               if i.get("status", "open") == "open" and i["id"] != self.largest][:3]}


# ----------------------------------------------------------------------------------------------- delivery gate
def source_preservation_gate(source_vertices: dict[str, np.ndarray], delivered_vertices: dict[str, np.ndarray],
                             protected: list[str], tol: float = 0.0) -> dict:
    """Delivery is incomplete if any protected object is missing or any of its vertices differs from the source."""
    errors = []
    for name in protected:
        if name not in delivered_vertices:
            errors.append(f"{name}: missing")
            continue
        a, b = source_vertices[name], delivered_vertices[name]
        if a.shape != b.shape or float(np.abs(a - b).max(initial=0.0)) > tol:
            errors.append(f"{name}: protected vertices changed")
    return {"passed": not errors, "errors": errors}


# ----------------------------------------------------------------------------------------------- harness
PROMPT = """You edit the given 3D asset with Blender Python. Turn t edits the previous output.
Declare the objects you may change or add before editing; all others are protected.
Select target faces (PartField parts, multi-view face voting or SAM masks), delete/replace/retexture only them and repair
the surface with new faces; keep all other vertices, connectivity, UVs and materials.
Render the current candidate from the fixed camera, compare it with the conditioning image, measure projection/contact
when placement is unclear, record what still differs in the issue ledger and work on the most serious open issue.
Renders of an earlier candidate are not evidence for the current one."""


def harness(tb, ws):
    from .agent import Harness
    state = EditState(source_objects=ws.inputs.get("source_objects", []))
    ledger = IssueLedger()
    tools = {
        "run_script": lambda code: tb.blender.run_script(code, ws.path("work", "scene.blend")),
        "update_edit_scope": lambda editable, new_objects=(): state.declare_scope(editable, list(new_objects)),
        "update_edit_issues": lambda issues, largest_issue_id=None: ledger.update(issues, largest_issue_id),
        "inspect_edit_candidate": lambda: tb.blender.render(ws.path("work", "scene.blend"),
                                                            [ws.inputs.get("camera", {})], ws.path("work", "inspect")),
    }

    def on_tool_result(name, result):
        if name == "run_script":                   # any mutation: new candidate version
            ledger.observe_candidate(str(ws.path("work", "scene.blend")), revision=uuid.uuid4().hex)
        if name == "inspect_edit_candidate" and result.ok:
            ledger.register_evidence("condition_comparison")

    def deliver(path):
        src, out = ws.inputs.get("source_vertices", {}), ws.inputs.get("delivered_vertices", {})
        return source_preservation_gate(src, out, state.protected)

    return Harness("edit", PROMPT, tools, on_tool_result, ledger.summary, deliver)
