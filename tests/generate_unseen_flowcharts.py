"""Generate unseen-style flowcharts with ground truth for generalisation testing.

None of these layouts, styles or labels are used to tune the recogniser; they
exercise the same failure categories found in the samples (filled shapes,
merging connectors, loop-backs, labels on lines, tinted pages, thick/thin
strokes, open/filled/barbed arrowheads) plus image degradations (low
resolution, JPEG/noise, slight rotation).

Usage:
    python tests/generate_unseen_flowcharts.py OUTPUT_DIR
writes OUTPUT_DIR/*.png and OUTPUT_DIR/ground_truth.json (scorer format).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
SS = 3  # supersampling factor for anti-aliased drawing

STYLES = {
    "outline": dict(bg=(255, 255, 255), line=(20, 20, 20), width=2, head="open", fill=None,
                    outline=(20, 20, 20), text=(20, 20, 20), terminal="stadium"),
    "filled": dict(bg=(246, 247, 251), line=(90, 100, 115), width=2, head="filled",
                   fill={"terminal": (120, 200, 160), "process": (160, 200, 240), "decision": (250, 210, 120),
                         "input": (230, 170, 220), "output": (230, 170, 220)},
                   outline=None, text=(25, 25, 40), terminal="stadium"),
    "bold": dict(bg=(255, 255, 255), line=(40, 40, 160), width=4, head="barbed",
                 fill={"terminal": (255, 230, 230), "process": (235, 245, 255), "decision": (255, 250, 220),
                       "input": (235, 255, 235), "output": (235, 255, 235)},
                 outline=(40, 40, 160), text=(10, 10, 10), terminal="ellipse"),
}


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size * SS)
    return ImageFont.load_default()


def _shape_polygon(kind: str, cx: float, cy: float, w: float, h: float, terminal: str):
    if kind == "decision":
        return [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    if kind in ("input", "output"):
        skew = h * .3
        return [(cx - w / 2 + skew, cy - h / 2), (cx + w / 2, cy - h / 2),
                (cx + w / 2 - skew, cy + h / 2), (cx - w / 2, cy + h / 2)]
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def _arrowhead(draw, tip, previous, style, scale):
    size = (11 + 2 * style["width"]) * scale
    angle = math.atan2(tip[1] - previous[1], tip[0] - previous[0])
    left = (tip[0] - size * math.cos(angle - .45), tip[1] - size * math.sin(angle - .45))
    right = (tip[0] - size * math.cos(angle + .45), tip[1] - size * math.sin(angle + .45))
    width = style["width"] * scale
    if style["head"] == "filled":
        draw.polygon([tip, left, right], fill=style["line"])
    elif style["head"] == "barbed":
        notch = (tip[0] - size * .6 * math.cos(angle), tip[1] - size * .6 * math.sin(angle))
        draw.polygon([tip, left, notch, right], fill=style["line"])
    else:
        draw.line([left, tip, right], fill=style["line"], width=width)


def render(spec: dict, style_name: str, degrade: str | None = None) -> tuple[np.ndarray, dict]:
    style = STYLES[style_name]
    width, height = spec["size"]
    image = Image.new("RGB", (width * SS, height * SS), style["bg"])
    draw = ImageDraw.Draw(image)
    font = _font(spec.get("font", 15))
    label_font = _font(spec.get("label_font", 13))
    nodes = spec["nodes"]
    lw = style["width"] * SS
    # Connectors first so shapes cover their ends.
    for edge in spec["edges"]:
        points = [(x * SS, y * SS) for x, y in edge["route"]]
        draw.line(points, fill=style["line"], width=lw, joint="curve")
        if edge.get("head", True):
            _arrowhead(draw, points[-1], points[-2], style, SS)
    for key, node in nodes.items():
        cx, cy = node["center"]
        w, h = node["size"]
        kind = node["type"]
        fill = style["fill"].get(kind) if style["fill"] else None
        outline = style["outline"]
        box = [(cx - w / 2) * SS, (cy - h / 2) * SS, (cx + w / 2) * SS, (cy + h / 2) * SS]
        if kind == "terminal" and style["terminal"] == "ellipse":
            draw.ellipse(box, fill=fill or style["bg"], outline=outline, width=lw)
        elif kind == "terminal":
            draw.rounded_rectangle(box, radius=h / 2 * SS, fill=fill or style["bg"], outline=outline, width=lw)
        else:
            polygon = [(x * SS, y * SS) for x, y in _shape_polygon(kind, cx, cy, w, h, style["terminal"])]
            draw.polygon(polygon, fill=fill or style["bg"], outline=outline, width=lw if outline else 0)
        draw.multiline_text((cx * SS, cy * SS), node["text"], fill=style["text"], font=font, anchor="mm",
                            align="center", spacing=4 * SS)
    for edge in spec["edges"]:
        if not edge.get("label"):
            continue
        lx, ly = edge["label_at"]
        bbox = draw.textbbox((lx * SS, ly * SS), edge["label"], font=label_font, anchor="mm")
        if edge.get("label_on_line"):
            pad = 3 * SS
            draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=style["bg"])
        draw.text((lx * SS, ly * SS), edge["label"], fill=style["text"], font=label_font, anchor="mm")
    image = image.resize((width, height), Image.LANCZOS)
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    truth = {key: {"center": list(node["center"]), "type": node["type"], "text": node["text"].replace("\n", " ")}
             for key, node in nodes.items()}
    if degrade == "lowres":
        array = cv2.resize(array, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        truth = {k: {**v, "center": [c * .5 for c in v["center"]]} for k, v in truth.items()}
    elif degrade == "jpeg_noise":
        rng = np.random.default_rng(7)
        noisy = np.clip(array.astype(np.int16) + rng.normal(0, 6, array.shape).astype(np.int16), 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 35])
        array = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif degrade == "rotated":
        angle = 1.5
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        array = cv2.warpAffine(array, matrix, (width, height), borderValue=style["bg"][::-1])
        for value in truth.values():
            x, y = value["center"]
            value["center"] = [float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
                               float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])]
    edges = [[e["source"], e["target"], e.get("truth_label")] for e in spec["edges"] if e.get("head", True)]
    return array, {"nodes": truth, "edges": edges}


def _v(x0, y0, x1, y1):
    return [(x0, y0), (x1, y1)]


def structures() -> dict[str, dict]:
    """Five chart structures not present among the samples or fixtures."""
    specs = {}
    # 1. Linear sequence with I/O.
    specs["linear"] = {"size": (520, 760), "nodes": {
        "start": {"center": (260, 60), "size": (160, 56), "type": "terminal", "text": "Start"},
        "in": {"center": (260, 200), "size": (220, 70), "type": "input", "text": "Input x"},
        "calc": {"center": (260, 340), "size": (220, 70), "type": "process", "text": "y = x * 2"},
        "out": {"center": (260, 480), "size": (220, 70), "type": "output", "text": "Output y"},
        "end": {"center": (260, 620), "size": (160, 56), "type": "terminal", "text": "End"}},
        "edges": [{"source": "start", "target": "in", "route": _v(260, 88, 260, 165)},
                  {"source": "in", "target": "calc", "route": _v(260, 235, 260, 305)},
                  {"source": "calc", "target": "out", "route": _v(260, 375, 260, 445)},
                  {"source": "out", "target": "end", "route": _v(260, 515, 260, 592)}]}
    # 2. If/else whose branches merge into one arrowhead (T-junction merge).
    specs["merge"] = {"size": (720, 820), "nodes": {
        "start": {"center": (360, 60), "size": (160, 56), "type": "terminal", "text": "Start"},
        "in": {"center": (360, 180), "size": (220, 70), "type": "input", "text": "Input n"},
        "dec": {"center": (360, 330), "size": (220, 120), "type": "decision", "text": "n > 0"},
        "pos": {"center": (150, 500), "size": (200, 70), "type": "output", "text": "Output 1"},
        "neg": {"center": (570, 500), "size": (200, 70), "type": "output", "text": "Output 0"},
        "end": {"center": (360, 730), "size": (160, 56), "type": "terminal", "text": "End"}},
        "edges": [{"source": "start", "target": "in", "route": _v(360, 88, 360, 145)},
                  {"source": "in", "target": "dec", "route": _v(360, 215, 360, 270)},
                  {"source": "dec", "target": "pos", "route": [(250, 330), (150, 330), (150, 465)],
                   "label": "True", "label_at": (195, 312), "truth_label": "Yes"},
                  {"source": "dec", "target": "neg", "route": [(470, 330), (570, 330), (570, 465)],
                   "label": "False", "label_at": (520, 312), "truth_label": "No"},
                  {"source": "pos", "target": "end", "route": [(150, 535), (150, 620), (360, 620)], "head": False},
                  {"source": "neg", "target": "end", "route": [(570, 535), (570, 620), (360, 620), (360, 702)]}]}
    # Arrowhead-free segments are drawn but are not edges themselves; the
    # junction they join implies these semantic edges in the ground truth.
    specs["merge"]["extra_truth"] = [["pos", "end", None]]
    # 3. Loop-back into the side of a decision, labels on the lines.
    specs["loop"] = {"size": (760, 700), "nodes": {
        "start": {"center": (300, 60), "size": (160, 56), "type": "terminal", "text": "Start"},
        "init": {"center": (300, 180), "size": (200, 70), "type": "process", "text": "i = 0"},
        "dec": {"center": (300, 350), "size": (220, 120), "type": "decision", "text": "i < 5"},
        "inc": {"center": (300, 540), "size": (200, 70), "type": "process", "text": "i = i + 1"},
        "end": {"center": (640, 350), "size": (160, 56), "type": "terminal", "text": "End"}},
        "edges": [{"source": "start", "target": "init", "route": _v(300, 88, 300, 145)},
                  {"source": "init", "target": "dec", "route": _v(300, 215, 300, 290)},
                  {"source": "dec", "target": "inc", "route": _v(300, 410, 300, 505),
                   "label": "Yes", "label_at": (300, 455), "label_on_line": True, "truth_label": "Yes"},
                  {"source": "inc", "target": "dec", "route": [(200, 540), (110, 540), (110, 350), (190, 350)]},
                  {"source": "dec", "target": "end", "route": _v(410, 350, 560, 350),
                   "label": "No", "label_at": (485, 350), "label_on_line": True, "truth_label": "No"}]}
    # 4. Decision tree with three outputs collected by a bus line into End.
    specs["tree"] = {"size": (900, 860), "nodes": {
        "start": {"center": (450, 50), "size": (150, 52), "type": "terminal", "text": "Start"},
        "d1": {"center": (450, 180), "size": (200, 110), "type": "decision", "text": "a > b"},
        "d2": {"center": (230, 360), "size": (200, 110), "type": "decision", "text": "b > c"},
        "o1": {"center": (680, 360), "size": (180, 66), "type": "output", "text": "Output a"},
        "o2": {"center": (110, 560), "size": (170, 66), "type": "output", "text": "Output b"},
        "o3": {"center": (370, 560), "size": (170, 66), "type": "output", "text": "Output c"},
        "end": {"center": (450, 790), "size": (150, 52), "type": "terminal", "text": "End"}},
        "edges": [{"source": "start", "target": "d1", "route": _v(450, 76, 450, 125)},
                  {"source": "d1", "target": "d2", "route": [(350, 180), (230, 180), (230, 305)],
                   "label": "No", "label_at": (290, 165), "truth_label": "No"},
                  {"source": "d1", "target": "o1", "route": [(550, 180), (680, 180), (680, 327)],
                   "label": "Yes", "label_at": (615, 165), "truth_label": "Yes"},
                  {"source": "d2", "target": "o2", "route": [(130, 360), (110, 360), (110, 527)],
                   "label": "Yes", "label_at": (112, 340), "truth_label": "Yes"},
                  {"source": "d2", "target": "o3", "route": [(330, 360), (370, 360), (370, 527)],
                   "label": "No", "label_at": (352, 340), "truth_label": "No"},
                  {"source": "o2", "target": "end", "route": [(110, 593), (110, 690), (450, 690)], "head": False},
                  {"source": "o3", "target": "end", "route": [(370, 593), (370, 690)], "head": False},
                  {"source": "o1", "target": "end", "route": [(680, 393), (680, 690), (450, 690), (450, 764)]}]}
    specs["tree"]["extra_truth"] = [["o2", "end", None], ["o3", "end", None]]
    # 5. Left-to-right flow.
    specs["horizontal"] = {"size": (1100, 300), "font": 14, "nodes": {
        "start": {"center": (90, 150), "size": (130, 56), "type": "terminal", "text": "Start"},
        "read": {"center": (330, 150), "size": (190, 70), "type": "input", "text": "Read name"},
        "greet": {"center": (580, 150), "size": (190, 70), "type": "process", "text": "msg = name"},
        "show": {"center": (830, 150), "size": (190, 70), "type": "output", "text": "Output msg"},
        "end": {"center": (1020, 150), "size": (110, 56), "type": "terminal", "text": "End"}},
        "edges": [{"source": "start", "target": "read", "route": _v(155, 150, 236, 150)},
                  {"source": "read", "target": "greet", "route": _v(425, 150, 485, 150)},
                  {"source": "greet", "target": "show", "route": _v(675, 150, 736, 150)},
                  {"source": "show", "target": "end", "route": _v(925, 150, 965, 150)}]}
    return specs


VARIANTS = [(structure, style, None) for structure in ("linear", "merge", "loop", "tree", "horizontal")
            for style in ("outline", "filled", "bold")] + [
    ("merge", "filled", "lowres"), ("tree", "outline", "jpeg_noise"), ("loop", "bold", "jpeg_noise"),
    ("linear", "outline", "rotated"), ("merge", "outline", "rotated")]


def main(output: str) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    truth = {"_about": "Synthetic unseen-style flowcharts; see tests/generate_unseen_flowcharts.py."}
    specs = structures()
    for structure, style, degrade in VARIANTS:
        spec = specs[structure]
        array, gt = render(spec, style, degrade)
        name = f"{structure}_{style}" + (f"_{degrade}" if degrade else "")
        cv2.imwrite(str(out / f"{name}.png"), array)
        edges = gt["edges"]
        drawn = {(s, t) for s, t, _ in edges}
        edges = edges + [e for e in spec.get("extra_truth", []) if (e[0], e[1]) not in drawn]
        truth[name] = {"image": f"{name}.png", "notation_supported": False, "nodes": gt["nodes"], "edges": edges}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"wrote {len(VARIANTS)} charts to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "unseen_flowcharts")
