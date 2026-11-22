"""Line / circle pieces of a PathG2 built with q = 0, and which piece a point belongs to (closed form per piece:
a line covers 0 <= (p - a).u <= len, a circle arc covers 0 <= angle <= th; windows meet at the joint planes)."""
import numpy as np


def pieces_of(path):
    out = []
    for pc in path.pieces:
        if pc[0] == "line":
            _, s0, ln, a, u = pc
            out.append(dict(kind="line", s0=s0, len=ln, a=a, u=u))
        else:
            _, s0, ln, c = pc
            assert c.q == 0
            ctr = c.P0 + c.r * c.e2
            out.append(dict(kind="arc", s0=s0, len=ln, c=ctr, e1=c.e1, e2=c.e2, n=np.cross(c.e1, c.e2), R=c.r, th=c.th))
    return out


def piece_index(pcs, p):
    """Piece whose window contains p; where windows overlap (farther from the path than the arc radius) the
    one with the smaller offset; outside every window, the piece with the closest start."""
    best, bd = None, np.inf
    for i, P in enumerate(pcs):
        if P["kind"] == "line":
            lam = (p - P["a"]) @ P["u"]
            if not (0 <= lam <= P["len"]):
                continue
            dd = np.linalg.norm(p - P["a"] - lam * P["u"])
        else:
            q = p - P["c"]
            x, y, h = q @ P["e1"], q @ P["e2"], q @ P["n"]
            al = np.arctan2(x, -y)
            if not (0 <= al <= P["th"]):
                continue
            dd = np.hypot(np.hypot(x, y) - P["R"], h)
        if dd < bd:
            best, bd = i, dd
    if best is None:
        ends = [np.linalg.norm(p - (P["a"] if P["kind"] == "line" else P["c"] - P["R"] * P["e2"])) for P in pcs]
        best = int(np.argmin(ends))
    return best
