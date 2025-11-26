"""
"""

import os, math, argparse
import numpy as np
from dataclasses import dataclass
from numpy.random import default_rng
from PIL import Image, ImageDraw

# ============================================================
# Cassini-style polygon generator
# ============================================================

def cassini_polygon(cx, cy, shape, nvertices, rng, startradius=100):
    """Generate a Cassini-like irregular polygon"""
    step = 360.0 / (nvertices - 1)
    angle = 0.0
    coords = []
    b_vals, a_vals = [], []

    if shape == "cassini90":
        a_ratio = 0.9
    elif shape == "cassini70":
        a_ratio = 0.7
    elif shape == "cassini0":
        a_ratio = 0.0
    else:
        raise ValueError("shape must be one of: cassini0, cassini70, cassini90")

    while angle < 360.0:
        angle += step + (rng.random() * 4.0 - 2.0)
        ang = np.deg2rad(angle)
        b_v = startradius + rng.integers(-25, 26)
        a_v = a_ratio * b_v
        c = math.sqrt(a_v*a_v * math.cos(2*ang) + math.sqrt(b_v**4 - a_v*a_v * (math.sin(2*ang)**2)))
        y = math.cos(ang) * c
        x = math.sin(ang) * c
        coords.append([cx + x, cy + y])
        b_vals.append(b_v)
        a_vals.append(a_v)

    coords.reverse()
    V = np.asarray(coords, dtype=float)
    return V, float(np.mean(a_vals)), float(np.mean(b_vals))

# ============================================================
# Concavity % control
# ============================================================

def _ensure_ccw(xy):
    area = 0.5 * np.sum(xy[:,0]*np.roll(xy[:,1],-1) - xy[:,1]*np.roll(xy[:,0],-1))
    return xy if area > 0 else xy[::-1]

def _concave_mask_ccw(xy):
    # assumes CCW orientation
    v_prev = xy - np.roll(xy, 1, axis=0)
    v_next = np.roll(xy, -1, axis=0) - xy
    cross = v_prev[:,0]*v_next[:,1] - v_prev[:,1]*v_next[:,0]
    return cross < 0

def concavity_targets_for_vertices(N, percents=(30,40,50)):
    """Return integer target concavity counts for this N (percent of vertices)."""
    return [int(round(N * p / 100.0)) for p in percents]

def enforce_concavity_count(V, target_count, rng,
                            max_iter=5000,
                            inward_step=0.10,     # was 0.12
                            outward_step=0.75,     # was 1.2
                            rad_band=(0.65, 1.35),# clamp radii relative to median
                            reg_every=60,         # Laplacian regularization cadence. 0=no regularization. try 120
                            reg_lam=0.08):        # 0=no regularization, 0.05–0.12 is mild
    """
    Nudge vertices to reach target concavity count, while avoiding long spikes:
    - smaller steps
    - clamp radius to [rad_band[0]*r_med, rad_band[1]*r_med]
    - periodic light Laplacian smoothing
    """
    V = _ensure_ccw(V.copy())

    def _laplacian(X, lam):
        return (1.0 - lam) * X + lam * (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0)) / 2.0

    for it in range(max_iter):
        mask = _concave_mask_ccw(V)
        ccount = int(mask.sum())
        if ccount == target_count:
            break

        # centroid & per-vertex radii
        C = V.mean(axis=0)
        R = np.linalg.norm(V - C, axis=1)
        r_med = np.median(R)
        r_min = rad_band[0] * r_med
        r_max = rad_band[1] * r_med

        # choose a vertex to adjust (bias a bit toward extremes to kill spikes)
        if ccount > target_count:
            # too many concave → we will push *outward* a bit; avoid the very largest radius
            i = int(rng.integers(0, len(V)))
        else:
            # need more concave → pull inward; bias away from very small radii
            i = int(rng.integers(0, len(V)))

        v = V[i]
        dirv = v - C
        n = np.linalg.norm(dirv) + 1e-9
        dirv = dirv / n

        if ccount < target_count:
            v_new = v - inward_step * (v - C)         # inward
        else:
            v_new = v + outward_step * dirv           # outward (softer)

        # clamp radius to avoid needles
        r_new = np.linalg.norm(v_new - C)
        if r_new < r_min:
            v_new = C + (v_new - C) * (r_min / (r_new + 1e-9))
        elif r_new > r_max:
            v_new = C + (v_new - C) * (r_max / (r_new + 1e-9))

        V[i] = v_new

        # light regularization occasionally (pulls extreme single-vertex tips toward neighbors)
        if reg_every and (it % reg_every == 0):
            V = _laplacian(V, reg_lam)

    return V

# Resampling & smoothing)
# ============================================================

def ensure_closed_unique(P):
    return P[:-1] if np.allclose(P[0], P[-1]) else P

def close_loop(P):
    return P if np.allclose(P[0], P[-1]) else np.vstack([P, P[0]])

def densify_polyline(P, n=3000):
    P = close_loop(P)
    seg = np.sqrt(((np.roll(P, -1, axis=0) - P) ** 2).sum(1))
    t = np.r_[0.0, np.cumsum(seg[:-1])]
    if t[-1] == 0:
        return ensure_closed_unique(P.copy())
    t_new = np.linspace(0, t[-1], n)
    x = np.interp(t_new, t, P[:,0])
    y = np.interp(t_new, t, P[:,1])
    return np.c_[x, y]

def moving_average_closed(P, win=50, passes=2):
    """Circular moving-average on a closed path."""
    Q = ensure_closed_unique(P)
    if win % 2 == 0:
        win += 1
    h = win // 2
    k = np.ones(win, dtype=float) / win
    for _ in range(passes):
        xpad = np.r_[Q[-h:,0], Q[:,0], Q[:h,0]]
        ypad = np.r_[Q[-h:,1], Q[:,1], Q[:h,1]]
        x = np.convolve(xpad, k, mode="valid")
        y = np.convolve(ypad, k, mode="valid")
        Q = np.c_[x, y]
    return Q

# ============================================================
# Transforms & drawing
# ============================================================

def rotate_points(P, degrees):
    if degrees == 0:
        return P
    th = np.deg2rad(degrees)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=float)
    return P @ R.T

def center_and_scale_to_canvas(P, canvas_size=1200, margin=60):
    P = P - P.mean(axis=0, keepdims=True)
    max_extent = np.max(np.abs(P))
    if max_extent > 0:
        scale = (canvas_size/2 - margin) / max_extent
        P = P * scale
    P = P + np.array([canvas_size/2, canvas_size/2])
    return P

def save_path_as_png(P, out_path, canvas_size=1200, line_width=8, line_color=(0,0,0,255)):
    """
    Draws an outline-only stimulus with transparent background.
    """
    from PIL import Image, ImageDraw

    # transparent RGBA background (alpha=0)
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ensure explicitly closed path
    Q = P if np.allclose(P[0], P[-1]) else np.vstack([P, P[0]])
    pts = [tuple(map(float, xy)) for xy in Q]

    # black outline, fully opaque
    draw.line(pts, fill=line_color, width=line_width, joint="curve")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "PNG")

# ============================================================
# Stimulus generation
# ============================================================

def make_stimuli(outdir,
                 n_per_cell=1,
                 seed=1,
                 vertices=(22, 26),
                 families=("cassini0", "cassini70", "cassini90"),
                 orientations=(0, 45, -45),
                 canvas_px=1200,
                 line_px=8,
                 mov1_win=50,  mov1_passes=2,   # curv1 (light)
                 mov2_win=120, mov2_passes=4):  # curv2 (strong smoothing)

    rng = default_rng(seed)

    MOVAVG = {
        0: dict(win=None,       passes=0),
        1: dict(win=mov1_win,   passes=mov1_passes),
        2: dict(win=mov2_win,   passes=mov2_passes),
    }

    for d in ("curv0", "curv1", "curv2"):
        os.makedirs(os.path.join(outdir, d), exist_ok=True)

    for N in vertices:
        conc_counts = concavity_targets_for_vertices(N, percents=(30,40,50))
        for fam_name in families:
            for ori in orientations:
                for cc in conc_counts:
                    for rep in range(n_per_cell):
                        V0, _, _ = cassini_polygon(0, 0, fam_name, N, rng=rng)
                        V0 = enforce_concavity_count(V0, cc, rng)
                        base = densify_polyline(V0, n=3000)

                        curves = {}
                        for curv in (0, 1, 2):
                            if curv == 0:
                                P = base
                            else:
                                params = MOVAVG[curv]
                                P = moving_average_closed(base, win=params["win"], passes=params["passes"])
                            P = rotate_points(P, ori)
                            P = center_and_scale_to_canvas(P, canvas_size=canvas_px, margin=60)
                            curves[curv] = P

                        label_index = conc_counts.index(cc)
                        nominal_label = [30, 40, 50][label_index]
                        fname = f"stim_conc{nominal_label}_V{N}_fam-{fam_name}_rot{ori:+d}_rep{rep:02d}.png"

                        save_path_as_png(curves[0], os.path.join(outdir, "curv0", fname),
                                         canvas_size=canvas_px, line_width=line_px)
                        save_path_as_png(curves[1], os.path.join(outdir, "curv1", fname),
                                         canvas_size=canvas_px, line_width=line_px)
                        save_path_as_png(curves[2], os.path.join(outdir, "curv2", fname),
                                         canvas_size=canvas_px, line_width=line_px)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Generate Bertamini-style stimuli with concavity % and MOVAVG dials.")
    p.add_argument("--outdir", type=str, default=os.path.expanduser("~/Documents/Curvature_Stimuli/concavity_perc_fixed2"))
    p.add_argument("--n-per-cell", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--canvas", type=int, default=1200)
    p.add_argument("--linewidth", type=int, default=8)
    p.add_argument("--vertices", type=int, nargs="+", default=[22, 26])
    p.add_argument("--families", type=str, nargs="+", default=["cassini0","cassini70","cassini90"])
    p.add_argument("--orientations", type=int, nargs="+", default=[0, 45, -45])
    p.add_argument("--mov1-win", type=int, default=50)
    p.add_argument("--mov1-passes", type=int, default=2)
    p.add_argument("--mov2-win", type=int, default=120)
    p.add_argument("--mov2-passes", type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    make_stimuli(
        outdir=args.outdir,
        n_per_cell=args.n_per_cell,
        seed=args.seed,
        vertices=tuple(args.vertices),
        families=tuple(args.families),
        orientations=tuple(args.orientations),
        canvas_px=args.canvas,
        line_px=args.linewidth,
        mov1_win=args.mov1_win, mov1_passes=args.mov1_passes,
        mov2_win=args.mov2_win, mov2_passes=args.mov2_passes,
    )

if __name__ == "__main__":
    main()
