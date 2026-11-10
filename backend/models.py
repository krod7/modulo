"""Shared contracts; no HTTP, persistence, or recognition logic."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np

@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    def to_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]
    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

@dataclass
class OCRWord:
    text: str
    confidence: float
    bbox: BoundingBox
    def to_dict(self) -> dict:
        return {"text": self.text, "confidence": round(self.confidence, 4), "bbox": self.bbox.to_list()}

@dataclass
class Node:
    id: str
    type: str
    text: str
    bbox: BoundingBox
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)
    polygon: list[tuple[int, int]] = field(default_factory=list)
    words: list[OCRWord] = field(default_factory=list)
    ocr_candidates: list[dict[str, Any]] = field(default_factory=list)
    original_ocr_text: str | None = None
    def __post_init__(self) -> None:
        if self.original_ocr_text is None:
            self.original_ocr_text = self.text
    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "text": self.text, "original_ocr_text": self.original_ocr_text, "bbox": self.bbox.to_list(),
                "confidence": round(self.confidence, 4), "flags": self.flags,
                "polygon": self.polygon, "words": [word.to_dict() for word in self.words],
                "ocr_candidates": self.ocr_candidates}

@dataclass
class Arrow:
    id: str
    points: list[tuple[int, int]]
    tip: tuple[int, int] | None
    label: str | None = None
    source_hint: str | None = None
    target_hint: str | None = None

@dataclass
class Edge:
    id: str
    source: str
    target: str
    label: str | None = None
    points: list[tuple[int, int]] = field(default_factory=list)
    structural: bool = False
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class FlowGraph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    start_id: str | None = None
    end_ids: list[str] = field(default_factory=list)

@dataclass
class Diagnostic:
    severity: str
    code: str
    message: str
    hint: str
    element_ids: list[str] = field(default_factory=list)
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class SourceLine:
    line: int
    indent: int
    text: str
    source_ids: list[str] = field(default_factory=list)

@dataclass
class IRNode:
    kind: str
    source_ids: list[str] = field(default_factory=list)
    text: str = ""
    condition: str = ""
    children: list["IRNode"] = field(default_factory=list)
    otherwise: list["IRNode"] = field(default_factory=list)
    variable: str = ""
    expression: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "source_ids": self.source_ids, "text": self.text,
                "condition": self.condition, "children": [child.to_dict() for child in self.children],
                "otherwise": [child.to_dict() for child in self.otherwise],
                "variable": self.variable, "expression": self.expression}

@dataclass
class PipelineContext:
    filename: str = ""
    uploaded_bytes: bytes = b""
    settings: Any = None
    page_image: np.ndarray | None = None
    grayscale: np.ndarray | None = None
    binary: np.ndarray | None = None
    elements: list[Node] = field(default_factory=list)
    arrows: list[Arrow] = field(default_factory=list)
    edge_words: list[OCRWord] = field(default_factory=list)
    stroke_mask: np.ndarray | None = None
    graph: FlowGraph | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    completed_stages: list[str] = field(default_factory=list)
    ir: Any = None
    pseudocode: list[SourceLine] = field(default_factory=list)
    code: list[SourceLine] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    approved: bool = False
    @property
    def has_errors(self) -> bool:
        return any(item.severity == "error" for item in self.diagnostics)
    def diagnose(self, severity: str, code: str, message: str, hint: str, ids: list[str] | None = None) -> None:
        self.diagnostics.append(Diagnostic(severity, code, message, hint, ids or []))

@dataclass
class AnalysisResponse:
    context: PipelineContext
    analysis_id: str | None = None
    expires_at: str | None = None
    def to_dict(self) -> dict:
        ctx = self.context
        from backend.services.confidence_service import calculate_confidence
        graph = ctx.graph or FlowGraph(nodes=ctx.elements)
        page = None
        if ctx.page_image is not None:
            height, width = ctx.page_image.shape[:2]
            page = {"width": width, "height": height, "image_url": f"/api/analysis/{self.analysis_id}/image" if self.analysis_id else None}
        status = "failed" if ctx.has_errors else "ready" if ctx.ir is not None and ctx.pseudocode else "analyzed"
        nodes = graph.nodes
        confidence = calculate_confidence(nodes, graph, ctx.diagnostics)
        geometry_score = round((confidence["shape"] + confidence["graph"]) / 2)
        geometry_issues = sum(1 for item in ctx.diagnostics
                              if item.code in {"UNSUPPORTED_SHAPE", "AMBIGUOUS_CONNECTOR",
                                               "DANGLING_ARROW", "AMBIGUOUS_ARROW_DIRECTION"})
        return {"status": status, "analysis_id": self.analysis_id,
                "expires_at": self.expires_at, "page": page,
                "nodes": [node.to_dict() for node in graph.nodes], "edges": [edge.to_dict() for edge in graph.edges],
                "start_id": graph.start_id, "end_ids": graph.end_ids,
                "pseudocode": [asdict(line) for line in ctx.pseudocode],
                "ir": ctx.ir.to_dict() if hasattr(ctx.ir, "to_dict") else ctx.ir,
                "approved": ctx.approved,
                "diagnostics": [item.to_dict() for item in ctx.diagnostics],
                "completed_stages": ctx.completed_stages, "generation_available": bool(ctx.ir and not ctx.has_errors),
                "recognition_score": {**confidence, "geometry": geometry_score,
                                      "geometry_issues": geometry_issues}}
