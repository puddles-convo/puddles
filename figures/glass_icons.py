"""Animated line icons for glass-diagrams recipes.

Each icon is a function frame(u) -> list of polylines in local units (about 64 x 64, centred on 0,0),
for u in [0, 1). `icon_animations` turns one into polyline overlays plus `points` tracks that loop
seamlessly. Point counts stay fixed per polyline, as the engine's `points` tracks require.
"""
import math

TAU = 2 * math.pi


# ---------------------------------------------------------------- geometry helpers
def arc(cx, cy, r, a0, a1, n):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]


def xform(line, s=1.0, rot=0.0, dx=0.0, dy=0.0, about=(0.0, 0.0)):
    ax, ay = about
    c, si = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    out = []
    for x, y in line:
        x, y = (x - ax) * s, (y - ay) * s
        x, y = x * c - y * si, x * si + y * c
        out.append((x + ax + dx, y + ay + dy))
    return out


def bump(u, centre, width):
    d = min(abs(u - centre), 1 - abs(u - centre))
    return math.exp(-(d / width) ** 2)


# ---------------------------------------------------------------- icons
def person(u):
    """Head and shoulders, breathing gently."""
    b = math.sin(TAU * u)
    head = xform(arc(0, -14, 11, 0, 360, 36), dy=-1.8 * b)
    shoulders = xform(arc(0, 30, 24, 180, 360, 24), s=1 + 0.05 * b, about=(0, 30))
    return [head, shoulders]


def _bubble_outline():
    # rounded rectangle (-28,-26)..(28,12), r=10, tail at the bottom left, drawn in one pass
    r = 10
    pts = []
    pts += arc(18, -16, r, -90, 0, 6)          # top right corner
    pts += arc(18, 2, r, 0, 90, 6)             # bottom right corner
    pts += [(-6, 12), (-16, 26), (-12, 12)]    # tail
    pts += arc(-18, 2, r, 90, 180, 6)          # bottom left corner
    pts += arc(-18, -16, r, 180, 270, 6)       # top left corner
    pts.append(pts[0])
    return pts


def _heart(n=32):
    out = []
    for k in range(n + 1):
        t = TAU * k / n
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        out.append((x * 0.62, y * 0.62 - 7))
    return out


def helper(u):
    """Speech bubble with a heart that beats twice, then rests."""
    beat = 1 + 0.22 * bump(u, 0.12, 0.05) + 0.14 * bump(u, 0.3, 0.05)
    return [_bubble_outline(), xform(_heart(), s=beat, about=(0, -8))]


def scales(u):
    """Scales of justice, the beam tipping from side to side."""
    th = 10 * math.sin(TAU * u)
    pivot = (0, -18)
    lend = xform([(-26, -18)], rot=th, about=pivot)[0]
    rend = xform([(26, -18)], rot=th, about=pivot)[0]
    post = [(0, -26), (0, 24), (-14, 24), (14, 24)]
    strings = [(lend[0] - 8, lend[1] + 18), lend, (lend[0] + 8, lend[1] + 18), lend, rend,
               (rend[0] - 8, rend[1] + 18), rend, (rend[0] + 8, rend[1] + 18)]
    lpan = arc(lend[0], lend[1] + 18, 8, 0, 180, 12)
    rpan = arc(rend[0], rend[1] + 18, 8, 0, 180, 12)
    return [post, strings, lpan, rpan]


def critic(u):
    """Magnifier with an alert mark, sweeping in a small circle."""
    dx, dy = 5 * math.cos(TAU * u), 4 * math.sin(TAU * u)
    lens = xform(arc(-6, -6, 18, 0, 360, 36), dx=dx, dy=dy)
    handle = xform([(7, 7), (26, 26)], dx=dx, dy=dy)
    mark = xform([(-6, -16), (-6, -3)], dx=dx, dy=dy)
    dot = xform([(-6, 3), (-6, 5)], dx=dx, dy=dy)
    return [lens, handle, mark, dot]


def researcher(u):
    """Globe turning: two meridians sweep across."""
    R = 24
    outline = arc(0, 0, R, 0, 360, 36)
    lines = [(-R, 0), (R, 0), (R * 0.87, -12), (-R * 0.87, -12), (-R * 0.87, 12), (R * 0.87, 12)]
    mer = []
    for off in (0.0, 0.25):
        phi = TAU * (u + off) / 2  # a meridian crosses the face in half a cycle
        w = R * math.sin(phi)
        mer.append([(w * math.cos(math.radians(a)), R * math.sin(math.radians(a))) for a in range(-90, 91, 10)])
    return [outline, lines] + mer


def proposer(u):
    """Pencil writing a line from left to right, then lifting back."""
    if u < 0.75:
        p = u / 0.75
        dx = -14 + 30 * p
        dy = 0.0
    else:
        p = (u - 0.75) / 0.25
        dx = 16 - 30 * p
        dy = -6 * math.sin(math.pi * p)
    body = xform([(-20, 18), (14, -16), (22, -8), (-12, 26), (-22, 28), (-20, 18)], dx=dx, dy=dy)
    band = xform([(8, -10), (16, -2)], dx=dx, dy=dy)
    tip = -22 + dx
    line_end = max(-25.5, tip) if u < 0.75 else -25.5  # the line follows the tip, cleared while lifting
    return [body, band, [(-26, 31), (line_end, 31)]]


def gatekeeper(u):
    """Shield whose check mark draws itself, holds, then resets."""
    A, B, C = (-11, 0), (-3, 9), (12, -9)
    u = (u + 0.5) % 1  # start mid-hold, so the static frame shows the full check
    p = min(1.0, u / 0.35)  # drawn over the first 35 % of the cycle, then held
    if p < 0.35:
        q = p / 0.35
        mid = (A[0] + (B[0] - A[0]) * q, A[1] + (B[1] - A[1]) * q)
        check = [A, mid, mid]
    else:
        q = (p - 0.35) / 0.65
        check = [A, B, (B[0] + (C[0] - B[0]) * q, B[1] + (C[1] - B[1]) * q)]
    glow = 1 + 0.05 * bump(u, 0.4, 0.08)
    shield = xform([(0, -28), (24, -18), (22, 6), (12, 20), (0, 28), (-12, 20), (-22, 6), (-24, -18), (0, -28)],
                   s=glow)
    return [shield, check]


ICONS = {"person": (person, 4.0), "helper": (helper, 1.8), "scales": (scales, 3.2),
         "critic": (critic, 3.0), "researcher": (researcher, 4.0), "proposer": (proposer, 3.0),
         "gatekeeper": (gatekeeper, 3.2)}


# ---------------------------------------------------------------- recipe output
def _fmt(line, cx, cy, scale):
    return " ".join(f"{cx + x * scale:.2f},{cy + y * scale:.2f}" for x, y in line)


def icon_animations(block_id, name, color, cx, cy, scale=0.85, stroke=3.0, samples=48):
    """Overlays + points tracks for one icon centred at (cx, cy) relative to the block centre."""
    fn, duration = ICONS[name]
    frames = [fn(k / samples) for k in range(samples)]
    out = []
    for i, line0 in enumerate(frames[0]):
        oid = f"ic_{block_id}_{i}"
        out.append({"id": oid, "type": "overlay", "shape": "polyline", "visible": True,
                    "anchor": {"component": block_id, "slot": "centre"},
                    "attrs": {"points": [[round(cx + x * scale, 2), round(cy + y * scale, 2)] for x, y in line0],
                              "stroke": color, "stroke_width": stroke, "opacity": 0.95}})
        keys = [{"t": round(k / samples, 6), "value": _fmt(frames[k][i], cx, cy, scale)} for k in range(samples)]
        keys.append({"t": 1, "value": _fmt(frames[0][i], cx, cy, scale)})
        out.append({"id": f"{oid}_move", "type": "track", "target": oid, "attribute": "points",
                    "keyframes": keys, "duration": duration, "easing": "linear"})
    return out
