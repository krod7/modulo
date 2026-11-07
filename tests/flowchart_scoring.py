"""Structural scoring of recognised flowcharts against hand-annotated ground truth.

Recognised node IDs are assigned spatially, so expected nodes are matched by
geometry (the annotated centre must fall inside a recognised bounding box), not
by ID. Each pipeline stage is then scored independently:

* detection  - expected nodes found, plus spurious extra nodes
* shape      - recognised type of every matched node
* OCR        - normalised text similarity of every matched node
* graph      - directed edges (missing, extra, reversed) and branch labels

A sample "passes" only when its interpreted structure is correct: every node is
found with the right type and readable text, no extra nodes or edges exist,
every edge has the right direction and every branch label is right.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = Path(__file__).parent / "fixtures" / "samples_ground_truth.json"
SAMPLES_DIR = ROOT / "samples"
TEXT_THRESHOLD = 0.75

_VERBS = r"^(?:output|display|print|show|input|read|enter|if)\b\s*:?\s*"


def normalise_text(text: str) -> str:
    """Normalise surface variation that does not change a label's meaning."""
    value = (text or "").casefold().replace("ρ", "p")
    value = value.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    value = re.sub(_VERBS, "", value.strip())
    value = re.sub(r"=\s*=", "=", value)
    return re.sub(r"[\s\"'.,:;!?]", "", value)


def text_similarity(actual: str, expected: str) -> float:
    a, b = normalise_text(actual), normalise_text(expected)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def canonical_label(label: str | None) -> str | None:
    from backend.notation import normalize_label
    return normalize_label(label) if label else None


@dataclass
class SampleScore:
    name: str
    expected_nodes: int = 0
    matched_nodes: int = 0
    extra_nodes: list[str] = field(default_factory=list)
    missing_nodes: list[str] = field(default_factory=list)
    type_errors: list[str] = field(default_factory=list)
    text_errors: list[str] = field(default_factory=list)
    text_similarity: float = 0.0
    expected_edges: int = 0
    correct_edges: int = 0
    missing_edges: list[str] = field(default_factory=list)
    extra_edges: list[str] = field(default_factory=list)
    reversed_edges: list[str] = field(default_factory=list)
    label_errors: list[str] = field(default_factory=list)
    pipeline_status: str = ""
    stages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notation_supported: bool = False

    @property
    def structure_ok(self) -> bool:
        return not (self.missing_nodes or self.extra_nodes or self.type_errors or self.missing_edges
                    or self.extra_edges or self.label_errors)

    @property
    def passed(self) -> bool:
        if not self.structure_ok or self.text_errors:
            return False
        # Executable-notation charts must also complete logic recovery.
        return "pseudocode" in self.stages if self.notation_supported else True

    def summary(self) -> str:
        parts = [f"nodes {self.matched_nodes}/{self.expected_nodes}",
                 f"edges {self.correct_edges}/{self.expected_edges}",
                 f"text {self.text_similarity:.2f}"]
        for name in ("extra_nodes", "missing_nodes", "type_errors", "text_errors", "missing_edges",
                     "extra_edges", "reversed_edges", "label_errors"):
            values = getattr(self, name)
            if values:
                parts.append(f"{name}={values}")
        if self.errors:
            parts.append(f"errors={sorted(set(self.errors))}")
        return "; ".join(parts)


def score_context(name: str, spec: dict, context, original_size: tuple[int, int]) -> SampleScore:
    """Compare a finished PipelineContext with one ground-truth specification."""
    score = SampleScore(name, notation_supported=bool(spec.get("notation_supported")))
    width, height = original_size
    page_h, page_w = context.page_image.shape[:2] if context.page_image is not None else (height, width)
    sx, sy = page_w / width, page_h / height
    nodes = list(context.graph.nodes if context.graph else context.elements)
    matched: dict[str, object] = {}
    used = set()
    for key, expected in spec["nodes"].items():
        cx, cy = expected["center"][0] * sx, expected["center"][1] * sy
        hits = [node for node in nodes if node.id not in used
                and node.bbox.x <= cx <= node.bbox.x + node.bbox.width
                and node.bbox.y <= cy <= node.bbox.y + node.bbox.height]
        if hits:
            best = min(hits, key=lambda node: (node.bbox.center[0] - cx) ** 2 + (node.bbox.center[1] - cy) ** 2)
            matched[key] = best
            used.add(best.id)
    score.expected_nodes = len(spec["nodes"])
    score.matched_nodes = len(matched)
    score.missing_nodes = [key for key in spec["nodes"] if key not in matched]
    score.extra_nodes = [f"{node.id}:{node.type}:{node.text!r}" for node in nodes if node.id not in used]
    sims = []
    for key, node in matched.items():
        expected = spec["nodes"][key]
        if node.type != expected["type"]:
            score.type_errors.append(f"{key}:{node.type}!={expected['type']}")
        similarity = text_similarity(node.text, expected["text"])
        sims.append(similarity)
        if similarity < TEXT_THRESHOLD:
            score.text_errors.append(f"{key}:{node.text!r}")
    score.text_similarity = sum(sims) / len(sims) if sims else 0.0

    by_id = {node.id: key for key, node in matched.items()}
    actual = []
    for edge in (context.graph.edges if context.graph else []):
        if edge.structural:
            continue
        actual.append((by_id.get(edge.source, edge.source), by_id.get(edge.target, edge.target),
                       canonical_label(edge.label)))
    expected_edges = [(s, t, label) for s, t, label in spec["edges"]]
    score.expected_edges = len(expected_edges)
    remaining = list(actual)
    for source, target, label in expected_edges:
        hit = next((item for item in remaining if item[0] == source and item[1] == target), None)
        if hit is None:
            score.missing_edges.append(f"{source}->{target}")
            if any(item[0] == target and item[1] == source for item in remaining):
                score.reversed_edges.append(f"{source}->{target}")
            continue
        remaining.remove(hit)
        score.correct_edges += 1
        if label is not None and hit[2] != label:
            score.label_errors.append(f"{source}->{target}:{hit[2]}!={label}")
    score.extra_edges = [f"{s}->{t}" for s, t, _ in remaining]
    score.stages = list(context.completed_stages)
    score.errors = [d.code for d in context.diagnostics if d.severity == "error"]
    score.pipeline_status = "ready" if context.ir is not None and not context.has_errors else (
        "failed" if context.has_errors else "analyzed")
    return score


def load_ground_truth(path: Path | None = None) -> dict:
    data = json.loads(Path(path or GROUND_TRUTH).read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def evaluate_sample(name: str, spec: dict | None = None, settings=None,
                    image_dir: Path | None = None) -> tuple[SampleScore, object]:
    import cv2
    from backend.pipeline import analyze
    spec = spec or load_ground_truth()[name]
    path = Path(image_dir or SAMPLES_DIR) / spec["image"]
    data = path.read_bytes()
    image = cv2.imread(str(path))
    context = analyze(data, filename=path.name, settings=settings)
    return score_context(name, spec, context, (image.shape[1], image.shape[0])), context
