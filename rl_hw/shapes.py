"""
Shapes and letters for the ball to draw (2026-09-27). Each function returns one continuous
open path (N, 2) in plate metres, drawn once from its first point to its last: the ball
cannot lift a pen, so multi-stroke letters are joined in writing order (the joins are part
of the drawing). Paths are resampled to ~5 mm spacing like the red-line follower's.
"""
import numpy as np

SPACING = 0.005

# single-stroke block letters on a 0..1 box (x right, y up); each letter = list of strokes
_L = {
    "A": [[(0, 0), (0.5, 1), (1, 0)], [(0.25, 0.5), (0.75, 0.5)]],
    "B": [[(0, 0), (0, 1), (0.7, 1), (0.9, 0.85), (0.7, 0.55), (0, 0.55), (0.75, 0.55), (1, 0.3), (0.8, 0), (0, 0)]],
    "C": [[(1, 0.9), (0.7, 1), (0.2, 1), (0, 0.7), (0, 0.3), (0.2, 0), (0.7, 0), (1, 0.1)]],
    "D": [[(0, 0), (0, 1), (0.6, 1), (1, 0.7), (1, 0.3), (0.6, 0), (0, 0)]],
    "E": [[(1, 1), (0, 1), (0, 0.5), (0.7, 0.5), (0, 0.5), (0, 0), (1, 0)]],
    "F": [[(1, 1), (0, 1), (0, 0.5), (0.7, 0.5), (0, 0.5), (0, 0)]],
    "G": [[(1, 0.9), (0.7, 1), (0.2, 1), (0, 0.7), (0, 0.3), (0.2, 0), (0.8, 0), (1, 0.3), (1, 0.5), (0.55, 0.5)]],
    "H": [[(0, 1), (0, 0), (0, 0.5), (1, 0.5), (1, 1), (1, 0)]],
    "I": [[(0.2, 1), (0.8, 1), (0.5, 1), (0.5, 0), (0.2, 0), (0.8, 0)]],
    "J": [[(0.3, 1), (1, 1), (0.8, 1), (0.8, 0.2), (0.6, 0), (0.2, 0), (0, 0.2)]],
    "K": [[(0, 1), (0, 0), (0, 0.45), (1, 1), (0.3, 0.6), (1, 0)]],
    "L": [[(0, 1), (0, 0), (1, 0)]],
    "M": [[(0, 0), (0, 1), (0.5, 0.45), (1, 1), (1, 0)]],
    "N": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "O": [[(0.5, 1), (0.1, 0.85), (0, 0.5), (0.1, 0.15), (0.5, 0), (0.9, 0.15), (1, 0.5), (0.9, 0.85), (0.5, 1)]],
    "P": [[(0, 0), (0, 1), (0.75, 1), (1, 0.8), (0.75, 0.55), (0, 0.55)]],
    "Q": [[(0.5, 1), (0.1, 0.85), (0, 0.5), (0.1, 0.15), (0.5, 0), (0.9, 0.15), (1, 0.5), (0.9, 0.85), (0.5, 1)],
          [(0.6, 0.3), (1, 0)]],
    "R": [[(0, 0), (0, 1), (0.75, 1), (1, 0.8), (0.75, 0.55), (0, 0.55), (0.4, 0.55), (1, 0)]],
    "S": [[(1, 0.9), (0.7, 1), (0.2, 1), (0, 0.8), (0.2, 0.55), (0.8, 0.45), (1, 0.2), (0.8, 0), (0.3, 0), (0, 0.1)]],
    "T": [[(0, 1), (1, 1), (0.5, 1), (0.5, 0)]],
    "U": [[(0, 1), (0, 0.25), (0.25, 0), (0.75, 0), (1, 0.25), (1, 1)]],
    "V": [[(0, 1), (0.5, 0), (1, 1)]],
    "W": [[(0, 1), (0.25, 0), (0.5, 0.6), (0.75, 0), (1, 1)]],
    "X": [[(0, 1), (1, 0)], [(1, 1), (0, 0)]],
    "Y": [[(0, 1), (0.5, 0.5), (1, 1), (0.5, 0.5), (0.5, 0)]],
    "Z": [[(0, 1), (1, 1), (0, 0), (1, 0)]],
}


def resample(path, spacing=SPACING):
    path = np.asarray(path, dtype=float)
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / spacing) + 1)
    si = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(si, s, path[:, 0]), np.interp(si, s, path[:, 1])])


def shape(name, center=(0.0, 0.0), size=0.04):
    """name: circle, figure8, heart, star, square, spiral -> path around `center`,
    about +-size in each direction"""
    t = np.linspace(0, 2 * np.pi, 240)
    if name == "circle":
        p = np.column_stack([np.cos(t), np.sin(t)])
    elif name == "figure8":
        p = np.column_stack([np.sin(t), np.sin(t) * np.cos(t) * 1.2])
    elif name == "heart":
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        p = np.column_stack([x, y + 2]) / 17.0
    elif name == "star":
        k = np.arange(11)
        r = np.where(k % 2 == 0, 1.0, 0.42)
        a = np.pi / 2 + k * np.pi / 5
        p = np.column_stack([r * np.cos(a), r * np.sin(a)])
    elif name == "square":
        p = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)], dtype=float) * 0.9
    elif name == "spiral":
        tt = np.linspace(0, 3 * 2 * np.pi, 400)
        r = 0.15 + 0.85 * tt / tt[-1]
        p = np.column_stack([r * np.cos(tt), r * np.sin(tt)])
    else:
        raise ValueError(f"unknown shape {name}")
    return resample(np.asarray(center) + p * size)


def text(s, center=(0.0, 0.0), height=0.05, gap=0.25):
    """block letters, joined into one continuous path, centred on `center`"""
    s = "".join(ch for ch in s.upper() if ch in _L)[:4]
    if not s:
        raise ValueError("no drawable letters (A-Z)")
    w = height * 0.7
    total = len(s) * w + (len(s) - 1) * w * gap
    x0 = center[0] - total / 2
    pts = []
    for i, ch in enumerate(s):
        ox = x0 + i * w * (1 + gap)
        for stroke in _L[ch]:
            pts += [(ox + x * w, center[1] - height / 2 + y * height) for x, y in stroke]
    return resample(pts)
