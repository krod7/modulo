"""Generate standardized test flowcharts (SVG / PNG / JPG / JPEG / PDF) for Modulo.

Conventions (match the app's "standardized flowchart" assumption):
  rounded rectangle = Start/End, rectangle = process, diamond = decision,
  parallelogram = Input/Output, arrows = flow, decisions labeled Yes/No.
Black on white, ASCII-only text (OCR friendly), no titles or extra text on the page.
"""
import os, html, csv, zipfile, random
import numpy as np
from PIL import Image, ImageFilter
import cairosvg

OUT = "/mnt/user-data/outputs/modulo_test_flowcharts"
SIZES = {"term": (170, 56), "proc": (270, 64), "io": (250, 64), "dec": (290, 120)}
SKEW = 26
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"
MARGIN = 60


class Chart:
    def __init__(s):
        s.nodes, s.edges = {}, []

    def n(s, id, kind, text, x, y):
        w, h = SIZES[kind]
        s.nodes[id] = dict(kind=kind, text=text, x=x, y=y, w=w, h=h)

    def e(s, a, sa, b, sb=None, via=None, label=None, loff=None):
        s.edges.append(dict(a=a, sa=sa, b=b, sb=sb, via=via, label=label, loff=loff))

    def anchor(s, id, side):
        n = s.nodes[id]
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        if side == "t":
            return (x, y - h / 2)
        if side == "b":
            return (x, y + h / 2)
        dx = w / 2 - (SKEW / 2 if n["kind"] == "io" else 0)
        return (x - dx, y) if side == "l" else (x + dx, y)

    def path(s, e):
        a = e["a"] if isinstance(e["a"], tuple) else s.anchor(e["a"], e["sa"])
        arrow = not isinstance(e["b"], tuple)
        b = e["b"] if isinstance(e["b"], tuple) else s.anchor(e["b"], e["sb"])
        via = e["via"]
        if via is None and e["sb"] is None:
            via = []
        if via is None:
            sa, sb = e["sa"], e["sb"]
            if sa == "b" and sb == "t":
                via = [] if a[0] == b[0] else [(a[0], (a[1] + b[1]) / 2), (b[0], (a[1] + b[1]) / 2)]
            elif sa in "lr" and sb in "tb":
                via = [(b[0], a[1])]
            elif sa in "bt" and sb in "lr":
                via = [(a[0], b[1])]
            else:
                via = []
        return [a] + list(via) + [b], arrow


def f(v):
    return f"{v:.1f}"


def render_svg(c):
    paths = [c.path(e) for e in c.edges]
    pts = []
    for n in c.nodes.values():
        pts += [(n["x"] - n["w"] / 2, n["y"] - n["h"] / 2), (n["x"] + n["w"] / 2, n["y"] + n["h"] / 2)]
    for p, _ in paths:
        pts += p
    minx, miny = min(p[0] for p in pts) - MARGIN, min(p[1] for p in pts) - MARGIN
    maxx, maxy = max(p[0] for p in pts) + MARGIN, max(p[1] for p in pts) + MARGIN
    W, H = maxx - minx, maxy - miny
    T = lambda p: (p[0] - minx, p[1] - miny)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{f(W)}" height="{f(H)}" viewBox="0 0 {f(W)} {f(H)}">',
         f'<rect width="100%" height="100%" fill="#ffffff"/>',
         f'<g font-family="{FONT}" font-size="17" fill="#000" text-anchor="middle">']
    st = 'fill="#fff" stroke="#000" stroke-width="2.5" stroke-linejoin="round"'
    for id, n in c.nodes.items():
        x, y = T((n["x"], n["y"]))
        w, h = n["w"], n["h"]
        k = n["kind"]
        if k == "term":
            o.append(f'<rect x="{f(x-w/2)}" y="{f(y-h/2)}" width="{w}" height="{h}" rx="{h/2}" {st}/>')
        elif k == "proc":
            o.append(f'<rect x="{f(x-w/2)}" y="{f(y-h/2)}" width="{w}" height="{h}" {st}/>')
        elif k == "dec":
            pp = [(x, y - h / 2), (x + w / 2, y), (x, y + h / 2), (x - w / 2, y)]
            o.append(f'<polygon points="{" ".join(f(a)+","+f(b) for a,b in pp)}" {st}/>')
        elif k == "io":
            pp = [(x - w / 2 + SKEW, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2 - SKEW, y + h / 2), (x - w / 2, y + h / 2)]
            o.append(f'<polygon points="{" ".join(f(a)+","+f(b) for a,b in pp)}" {st}/>')
        lines = n["text"].split("\n")
        y0 = y - (len(lines) - 1) * 11 + 6
        for i, ln in enumerate(lines):
            o.append(f'<text x="{f(x)}" y="{f(y0 + i*22)}">{html.escape(ln)}</text>')
    for e, (p, arrow) in zip(c.edges, paths):
        q = [T(a) for a in p]
        o.append(f'<polyline points="{" ".join(f(a)+","+f(b) for a,b in q)}" fill="none" stroke="#000" stroke-width="2.5" stroke-linejoin="round"/>')
        if arrow:
            (x1, y1), (x2, y2) = q[-2], q[-1]
            dx, dy = x2 - x1, y2 - y1
            L = (dx * dx + dy * dy) ** 0.5 or 1
            ux, uy = dx / L, dy / L
            bx, by = x2 - ux * 15, y2 - uy * 15
            px, py = -uy * 6.5, ux * 6.5
            o.append(f'<polygon points="{f(x2)},{f(y2)} {f(bx+px)},{f(by+py)} {f(bx-px)},{f(by-py)}" fill="#000"/>')
        if e["label"]:
            ax, ay = q[0]
            sa = e["sa"]
            dx, dy, an = e["loff"] or {"b": (10, 22, "start"), "t": (10, -8, "start"),
                                       "r": (10, -10, "start"), "l": (-10, -10, "end")}.get(sa, (10, -10, "start"))
            o.append(f'<text x="{f(ax+dx)}" y="{f(ay+dy)}" text-anchor="{an}" font-size="16" font-weight="bold">{e["label"]}</text>')
    o.append("</g></svg>")
    return "\n".join(o)


# ----------------------------------------------------------------- chart definitions
def sequence_sum(end=True, orphan=False):
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("a", "io", "Input a", 300, 170)
    c.n("b", "io", "Input b", 300, 280)
    c.n("p", "proc", "sum = a + b", 300, 390)
    c.n("o", "io", "Output sum", 300, 500)
    for x, y in [("s", "a"), ("a", "b"), ("b", "p"), ("p", "o")]:
        c.e(x, "b", y, "t")
    if end:
        c.n("e", "term", "End", 300, 610)
        c.e("o", "b", "e", "t")
    if orphan:
        c.n("x", "proc", "temp = 0", 640, 390)
    return c


def if_else(labels=True):
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("i", "io", "Input n", 300, 170)
    c.n("d", "dec", "n % 2 == 0?", 300, 300)
    c.n("y", "io", "Output Even", 100, 450)
    c.n("n", "io", "Output Odd", 500, 450)
    c.n("e", "term", "End", 300, 570)
    c.e("s", "b", "i", "t"); c.e("i", "b", "d", "t")
    c.e("d", "l", "y", "t", label="Yes" if labels else None)
    c.e("d", "r", "n", "t", label="No" if labels else None)
    c.e("y", "b", "e", "l"); c.e("n", "b", "e", "r")
    return c


def if_only(no_branch=True):
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("i", "io", "Input age", 300, 170)
    c.n("d", "dec", "age >= 18?", 300, 300)
    c.n("o", "io", "Output Adult", 300, 450)
    c.n("e", "term", "End", 300, 560)
    c.e("s", "b", "i", "t"); c.e("i", "b", "d", "t")
    c.e("d", "b", "o", "t", label="Yes")
    c.e("o", "b", "e", "t")
    if no_branch:
        c.e("d", "r", "e", "r", via=[(520, 300), (520, 560)], label="No")
    return c


def else_if_chain():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("i", "io", "Input score", 300, 170)
    c.n("d1", "dec", "score >= 90?", 300, 310)
    c.n("d2", "dec", "score >= 80?", 300, 480)
    c.n("d3", "dec", "score >= 70?", 300, 650)
    c.n("A", "io", "Output A", 640, 310)
    c.n("B", "io", "Output B", 640, 480)
    c.n("C", "io", "Output C", 640, 650)
    c.n("F", "io", "Output F", 300, 800)
    c.n("e", "term", "End", 300, 910)
    c.e("s", "b", "i", "t"); c.e("i", "b", "d1", "t")
    c.e("d1", "b", "d2", "t", label="No"); c.e("d2", "b", "d3", "t", label="No")
    c.e("d3", "b", "F", "t", label="No")
    for d, o, y in [("d1", "A", 310), ("d2", "B", 480), ("d3", "C", 650)]:
        c.e(d, "r", o, "l", label="Yes")
        c.e(o, "r", (800, y))
    c.e("F", "b", "e", "t")
    c.e((800, 310), None, "e", "r", via=[(800, 910)])
    return c


def while_loop():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("p0", "proc", "count = 1", 300, 170)
    c.n("d", "dec", "count <= 5?", 300, 310)
    c.n("o", "io", "Output count", 300, 450)
    c.n("p1", "proc", "count = count + 1", 300, 560)
    c.n("e", "term", "End", 300, 720)
    c.e("s", "b", "p0", "t"); c.e("p0", "b", "d", "t")
    c.e("d", "b", "o", "t", label="Yes"); c.e("o", "b", "p1", "t")
    c.e("p1", "b", "d", "t", via=[(300, 640), (80, 640), (80, 226), (300, 226)])
    c.e("d", "r", "e", "r", via=[(580, 310), (580, 720)], label="No")
    return c


def for_loop_sum():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("p0", "proc", "sum = 0", 300, 170)
    c.n("p1", "proc", "i = 1", 300, 280)
    c.n("d", "dec", "i <= 10?", 300, 420)
    c.n("p2", "proc", "sum = sum + i", 300, 570)
    c.n("p3", "proc", "i = i + 1", 300, 680)
    c.n("o", "io", "Output sum", 300, 830)
    c.n("e", "term", "End", 300, 940)
    for a, b in [("s", "p0"), ("p0", "p1"), ("p1", "d")]:
        c.e(a, "b", b, "t")
    c.e("d", "b", "p2", "t", label="Yes"); c.e("p2", "b", "p3", "t")
    c.e("p3", "b", "d", "t", via=[(300, 750), (80, 750), (80, 336), (300, 336)])
    c.e("d", "r", "o", "r", via=[(580, 420), (580, 830)], label="No")
    c.e("o", "b", "e", "t")
    return c


def do_while():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("p0", "proc", "count = 0", 300, 170)
    c.n("i", "io", "Input guess", 300, 280)
    c.n("p1", "proc", "count = count + 1", 300, 390)
    c.n("d", "dec", "guess != 7?", 300, 530)
    c.n("o", "io", "Output count", 300, 690)
    c.n("e", "term", "End", 300, 800)
    c.e("s", "b", "p0", "t"); c.e("p0", "b", "i", "t"); c.e("i", "b", "p1", "t"); c.e("p1", "b", "d", "t")
    c.e("d", "l", "i", "t", via=[(80, 530), (80, 225), (300, 225)], label="Yes")
    c.e("d", "b", "o", "t", label="No"); c.e("o", "b", "e", "t")
    return c


def nested_loops():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("p0", "proc", "i = 1", 300, 170)
    c.n("d1", "dec", "i <= 3?", 300, 310)
    c.n("p1", "proc", "j = 1", 300, 450)
    c.n("d2", "dec", "j <= 3?", 300, 590)
    c.n("o", "io", "Output i * j", 300, 730)
    c.n("p2", "proc", "j = j + 1", 300, 840)
    c.n("p3", "proc", "i = i + 1", 640, 590)
    c.n("e", "term", "End", 300, 1000)
    c.e("s", "b", "p0", "t"); c.e("p0", "b", "d1", "t")
    c.e("d1", "b", "p1", "t", label="Yes"); c.e("p1", "b", "d2", "t")
    c.e("d2", "b", "o", "t", label="Yes"); c.e("o", "b", "p2", "t")
    c.e("p2", "b", "d2", "t", via=[(300, 905), (120, 905), (120, 520), (300, 520)])
    c.e("d2", "r", "p3", "l", label="No")
    c.e("p3", "t", "d1", "t", via=[(640, 226), (300, 226)])
    c.e("d1", "l", "e", "l", via=[(60, 310), (60, 1000)], label="No")
    return c


def loop_with_if():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("p0", "proc", "sum = 0", 300, 170)
    c.n("p1", "proc", "i = 1", 300, 280)
    c.n("d1", "dec", "i <= 10?", 300, 420)
    c.n("d2", "dec", "i % 2 == 0?", 300, 590)
    c.n("p2", "proc", "sum = sum + i", 300, 730)
    c.n("p3", "proc", "i = i + 1", 300, 840)
    c.n("o", "io", "Output sum", 300, 1000)
    c.n("e", "term", "End", 300, 1110)
    for a, b in [("s", "p0"), ("p0", "p1"), ("p1", "d1")]:
        c.e(a, "b", b, "t")
    c.e("d1", "b", "d2", "t", label="Yes")
    c.e("d2", "b", "p2", "t", label="Yes"); c.e("p2", "b", "p3", "t")
    c.e("d2", "r", "p3", "r", via=[(580, 590), (580, 840)], label="No")
    c.e("p3", "b", "d1", "t", via=[(300, 930), (80, 930), (80, 336), (300, 336)])
    c.e("d1", "r", "o", "r", via=[(700, 420), (700, 1000)], label="No")
    c.e("o", "b", "e", "t")
    return c


def early_exit():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("p0", "proc", "i = 1", 300, 170)
    c.n("d1", "dec", "i <= 100?", 300, 310)
    c.n("d2", "dec", "i % 7 == 0?", 300, 470)
    c.n("o", "io", "Output i", 640, 470)
    c.n("p1", "proc", "i = i + 1", 300, 620)
    c.n("e", "term", "End", 300, 780)
    c.e("s", "b", "p0", "t"); c.e("p0", "b", "d1", "t")
    c.e("d1", "b", "d2", "t", label="Yes")
    c.e("d2", "r", "o", "l", label="Yes")
    c.e("d2", "b", "p1", "t", label="No")
    c.e("p1", "b", "d1", "t", via=[(300, 700), (80, 700), (80, 226), (300, 226)])
    c.e("o", "r", (840, 470))
    c.e("d1", "r", "e", "r", via=[(840, 310), (840, 780)], label="No")
    return c


def factorial():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("i", "io", "Input n", 300, 170)
    c.n("p0", "proc", "fact = 1", 300, 280)
    c.n("p1", "proc", "i = 1", 300, 390)
    c.n("d", "dec", "i <= n?", 300, 530)
    c.n("p2", "proc", "fact = fact * i", 300, 670)
    c.n("p3", "proc", "i = i + 1", 300, 780)
    c.n("o", "io", "Output fact", 300, 950)
    c.n("e", "term", "End", 300, 1060)
    for a, b in [("s", "i"), ("i", "p0"), ("p0", "p1"), ("p1", "d")]:
        c.e(a, "b", b, "t")
    c.e("d", "b", "p2", "t", label="Yes"); c.e("p2", "b", "p3", "t")
    c.e("p3", "b", "d", "t", via=[(300, 860), (80, 860), (80, 446), (300, 446)])
    c.e("d", "r", "o", "r", via=[(580, 530), (580, 950)], label="No")
    c.e("o", "b", "e", "t")
    return c


def max_of_three():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("i", "io", "Input a, b, c", 300, 170)
    c.n("p0", "proc", "max = a", 300, 280)
    c.n("d1", "dec", "b > max?", 300, 420)
    c.n("pb", "proc", "max = b", 640, 420)
    c.n("d2", "dec", "c > max?", 300, 600)
    c.n("pc", "proc", "max = c", 640, 600)
    c.n("o", "io", "Output max", 300, 740)
    c.n("e", "term", "End", 300, 850)
    for a, b in [("s", "i"), ("i", "p0"), ("p0", "d1")]:
        c.e(a, "b", b, "t")
    c.e("d1", "r", "pb", "l", label="Yes")
    c.e("d1", "b", "d2", "t", label="No", loff=(10, 22, "start"))
    c.e("pb", "b", "d2", "t", via=[(640, 515), (300, 515)])
    c.e("d2", "r", "pc", "l", label="Yes")
    c.e("d2", "b", "o", "t", label="No")
    c.e("pc", "b", "o", "r", via=[(640, 740)])
    c.e("o", "b", "e", "t")
    return c


def prime_check():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("i", "io", "Input n", 300, 170)
    c.n("d1", "dec", "n < 2?", 300, 310)
    c.n("np1", "io", "Output Not Prime", 640, 310)
    c.n("p0", "proc", "i = 2", 300, 460)
    c.n("d2", "dec", "i * i <= n?", 300, 600)
    c.n("pr", "io", "Output Prime", 640, 600)
    c.n("d3", "dec", "n % i == 0?", 300, 770)
    c.n("np2", "io", "Output Not Prime", 640, 770)
    c.n("p1", "proc", "i = i + 1", 300, 920)
    c.n("e", "term", "End", 300, 1070)
    c.e("s", "b", "i", "t"); c.e("i", "b", "d1", "t")
    c.e("d1", "r", "np1", "l", label="Yes")
    c.e("d1", "b", "p0", "t", label="No")
    c.e("p0", "b", "d2", "t")
    c.e("d2", "r", "pr", "l", label="No")
    c.e("d2", "b", "d3", "t", label="Yes")
    c.e("d3", "r", "np2", "l", label="Yes")
    c.e("d3", "b", "p1", "t", label="No")
    c.e("p1", "b", "d2", "t", via=[(300, 995), (80, 995), (80, 516), (300, 516)])
    for o, y in [("np1", 310), ("pr", 600), ("np2", 770)]:
        c.e(o, "r", (840, y))
    c.e((840, 310), None, "e", "r", via=[(840, 1070)])
    return c


def fibonacci():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("i", "io", "Input n", 300, 170)
    c.n("p0", "proc", "a = 0", 300, 270)
    c.n("p1", "proc", "b = 1", 300, 370)
    c.n("p2", "proc", "i = 1", 300, 470)
    c.n("d", "dec", "i <= n?", 300, 610)
    c.n("o", "io", "Output a", 300, 750)
    c.n("p3", "proc", "temp = a + b", 300, 850)
    c.n("p4", "proc", "a = b", 300, 950)
    c.n("p5", "proc", "b = temp", 300, 1050)
    c.n("p6", "proc", "i = i + 1", 300, 1150)
    c.n("e", "term", "End", 300, 1310)
    for a, b in [("s", "i"), ("i", "p0"), ("p0", "p1"), ("p1", "p2"), ("p2", "d")]:
        c.e(a, "b", b, "t")
    c.e("d", "b", "o", "t", label="Yes")
    for a, b in [("o", "p3"), ("p3", "p4"), ("p4", "p5"), ("p5", "p6")]:
        c.e(a, "b", b, "t")
    c.e("p6", "b", "d", "t", via=[(300, 1230), (80, 1230), (80, 526), (300, 526)])
    c.e("d", "r", "e", "r", via=[(580, 610), (580, 1310)], label="No")
    return c


def login_retry():
    c = Chart()
    c.n("s", "term", "Start", 300, 60)
    c.n("p0", "proc", "attempts = 0", 300, 170)
    c.n("d1", "dec", "attempts < 3?", 300, 310)
    c.n("lock", "io", "Output Locked Out", 640, 310)
    c.n("i", "io", "Input password", 300, 460)
    c.n("d2", "dec", "password == 1234?", 300, 610)
    c.n("ok", "io", "Output Access Granted", 640, 610)
    c.n("p1", "proc", "attempts = attempts + 1", 300, 760)
    c.n("e", "term", "End", 300, 950)
    c.e("s", "b", "p0", "t"); c.e("p0", "b", "d1", "t")
    c.e("d1", "r", "lock", "l", label="No")
    c.e("d1", "b", "i", "t", label="Yes"); c.e("i", "b", "d2", "t")
    c.e("d2", "r", "ok", "l", label="Yes")
    c.e("d2", "b", "p1", "t", label="No")
    c.e("p1", "b", "d1", "t", via=[(300, 840), (80, 840), (80, 226), (300, 226)])
    c.e("lock", "r", (840, 310)); c.e("ok", "r", (840, 610))
    c.e((840, 310), None, "e", "r", via=[(840, 950)])
    return c


VALID = [
    ("01_sequence_sum", sequence_sum, "Linear sequence: input, assignment, output", "sequence; input; assignment; output"),
    ("02_if_else_even_odd", if_else, "Simple if/else with two branches merging", "if-else; modulo"),
    ("03_if_only_adult", if_only, "If with no else (No arrow skips to End)", "if; skip branch"),
    ("04_else_if_chain_grade", else_if_chain, "Chained else-if ladder with 4 outcomes", "if / elif / elif / else; branch merge"),
    ("05_while_loop_count", while_loop, "Pre-test while loop with counter", "while; increment"),
    ("06_for_loop_sum", for_loop_sum, "Counting for-style loop accumulating a sum", "for-style loop; accumulator"),
    ("07_do_while_guess", do_while, "Post-test (do-while) loop, condition at bottom", "do-while; input inside loop"),
    ("08_nested_loops_table", nested_loops, "Nested loops (outer i, inner j)", "nested while/for; two loop-backs"),
    ("09_loop_with_if_even_sum", loop_with_if, "If inside a loop, false branch rejoins increment", "loop + conditional; branch rejoin"),
    ("10_early_exit_first_multiple", early_exit, "Loop with early exit from inside a decision", "break-style exit; loop + if"),
    ("11_factorial", factorial, "Factorial using a loop", "input; while; multiplication accumulator"),
    ("12_max_of_three", max_of_three, "Sequential if statements updating a variable", "consecutive ifs; no else"),
    ("13_prime_check", prime_check, "Prime test: if + loop + nested if, several exits", "nested decisions in loop; multiple outputs"),
    ("14_fibonacci", fibonacci, "Fibonacci with temp swap in a loop", "loop with multi-statement body"),
    ("15_login_retry", login_retry, "Retry loop with a counter and an inner decision", "while + inner if; two terminal outputs"),
]

INVALID = [
    ("invalid_01_missing_end", lambda: sequence_sum(end=False), "No End terminator after the last Output", "validation: missing End"),
    ("invalid_02_decision_missing_branch", lambda: if_only(no_branch=False), "Decision has only a Yes outlet; No arrow missing", "validation: dangling decision branch"),
    ("invalid_03_orphan_node", lambda: sequence_sum(orphan=True), "A disconnected process box not on any path", "validation: unreachable node"),
    ("invalid_04_unlabeled_branches", lambda: if_else(labels=False), "Decision outlets carry no Yes/No labels", "validation: ambiguous branches"),
]


# ----------------------------------------------------------------- export
def export(name, chart, folders):
    svg = render_svg(chart)
    png_bytes = cairosvg.svg2png(bytestring=svg.encode(), scale=2)
    d = lambda k: os.path.join(OUT, folders, k) if folders else os.path.join(OUT, k)
    for k in ("svg", "png", "jpg", "jpeg", "pdf"):
        os.makedirs(d(k), exist_ok=True)
    open(f"{d('svg')}/{name}.svg", "w").write(svg)
    open(f"{d('png')}/{name}.png", "wb").write(png_bytes)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=f"{d('pdf')}/{name}.pdf")
    im = Image.open(f"{d('png')}/{name}.png").convert("RGB")
    im.save(f"{d('jpg')}/{name}.jpg", quality=95)
    im.save(f"{d('jpeg')}/{name}.jpeg", quality=95)
    return im


def stress(name, im):
    sd = os.path.join(OUT, "stress")
    os.makedirs(sd, exist_ok=True)
    w, h = im.size
    im.resize((w // 3, h // 3), Image.LANCZOS).save(f"{sd}/{name}_lowres.png")
    bg = Image.new("RGB", im.size, (225, 225, 225))
    rot = im.rotate(2.0, resample=Image.BICUBIC, expand=True, fillcolor=(225, 225, 225))
    rot.save(f"{sd}/{name}_rotated_2deg.jpg", quality=70)
    arr = np.asarray(im.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)
    rng = np.random.default_rng(4366)
    arr = arr * 0.92 + rng.normal(0, 12, arr.shape) - 6
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(f"{sd}/{name}_noisy_blur.jpg", quality=35)


if __name__ == "__main__":
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    rows = []
    for name, fn, desc, tests in VALID:
        im = export(name, fn(), "")
        rows.append([name, "valid", desc, tests])
        if name in ("05_while_loop_count", "08_nested_loops_table", "15_login_retry"):
            stress(name, im)
    for name, fn, desc, tests in INVALID:
        export(name, fn(), "invalid")
        rows.append([name, "invalid", desc, tests])
    with open(f"{OUT}/manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_stem", "type", "description", "constructs_tested"])
        w.writerows(rows)
    shutil.copy(__file__, f"{OUT}/gen_flowcharts.py")
    zp = "/mnt/user-data/outputs/modulo_test_flowcharts.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for fn_ in files:
                p = os.path.join(root, fn_)
                z.write(p, os.path.relpath(p, os.path.dirname(OUT)))
    print("done", len(rows))
