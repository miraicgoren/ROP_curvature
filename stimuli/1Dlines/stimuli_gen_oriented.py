#!/usr/bin/env python3
"""
Stimuli Generator (FREE-END tangents) — angle (3) × vertices (3) × curvature (3) × orientation (8)
=================================================================================================
- Base set: 27 condition combos (no concavity)
- For each base path, render 8 orientations: 0, +45, +90, +135, 180, -45, -90, -135
- Transparent background, anti-aliased
- Folder structure unchanged: out/images/curv{0,1,2}/
- Filenames now include _rot{+/-deg}: stim_nv{n}_ang{a}_curv{k}_rot+045.png
"""

import os, math, argparse
import numpy as np
from PIL import Image, ImageDraw

# ------------------------- Utilities -------------------------

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def hermite_sample(p0, m0, p1, m1, n=80):
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = (2*t**3 - 3*t**2 + 1)
    h10 = (t**3 - 2*t**2 + t)
    h01 = (-2*t**3 + 3*t**2)
    h11 = (t**3 - t**2)
    return h00*p0 + h10*m0 + h01*p1 + h11*m1

def format_rot_label(deg: int) -> str:
    """Return string like '+045', '-090', '+000', '+135', '+180'."""
    sign = '+' if deg >= 0 else '-'
    return f"{sign}{abs(int(deg)):03d}"

# ------------------------- Base geometry -------------------------

def build_polyline_by_turns(n_vertices: int, angle_deg: float, seg_len: float = 1.0) -> np.ndarray:
    assert n_vertices >= 1
    n_segments = 2 * n_vertices
    pts = [np.array([0.0, 0.0], float)]
    heading = 0.0
    turn = math.radians(angle_deg)
    sign = +1  # start by turning up
    for s in range(n_segments):
        dx = seg_len * math.cos(heading)
        dy = seg_len * math.sin(heading)
        pts.append(pts[-1] + np.array([dx, dy], float))
        if s < n_segments - 1:
            heading += sign * turn
            sign *= -1
    return np.vstack(pts)  # y-up

def fit_to_frame_general(poly_xy_up: np.ndarray, W: int, H: int, margin: int, stroke: int,
                         mode: str = "width", edge_to_edge: bool = True) -> np.ndarray:
    """
    Map y-up polyline to image coords, rotate chord horizontal, then scale.
    mode:
      - "width":  scale so horizontal span matches (W - 2*margin - stroke)
      - "min":    scale so BOTH width and height fit
    edge_to_edge:
      - If True, normalize X to touch left/right borders.
    """
    p0, pN = poly_xy_up[0], poly_xy_up[-1]
    P = poly_xy_up - (p0 + pN) / 2.0

    # rotate chord to horizontal in y-up
    v = pN - p0
    ang = math.atan2(v[1], v[0])
    R = np.array([[math.cos(-ang), -math.sin(-ang)],
                  [math.sin(-ang),  math.cos(-ang)]], float)
    P = (R @ P.T).T

    # spans (before y-flip)
    span_x = max(1e-6, P[:,0].max() - P[:,0].min())
    span_y = max(1e-6, P[:,1].max() - P[:,1].min())

    width_lim  = max(1.0, (W - 2*margin - stroke))
    height_lim = max(1.0, (H - 2*margin - stroke))

    if mode == "min":
        s = min(width_lim / span_x, height_lim / span_y)
    else:  # "width"
        s = width_lim / span_x

    P *= s

    # y-up -> image coords
    P[:,1] = -P[:,1]
    P[:,0] += W/2.0

    if edge_to_edge:
        # normalize X to exactly touch borders
        x_min, x_max = P[:,0].min(), P[:,0].max()
        x_rng = max(1e-6, x_max - x_min)
        P[:,0] = (P[:,0] - x_min) / x_rng * (W - stroke) + stroke/2.0

    return P

# ------------------------- Smoothing (FREE-END tangents) -------------------------

def build_smooth_path_free_ends(ctrl: np.ndarray, smooth_strength: float,
                                sample_density: float = 0.30) -> np.ndarray:
    P = np.asarray(ctrl, float)

    M = np.zeros_like(P)
    if len(P) == 2:
        v = P[1] - P[0]
        M[0]  = smooth_strength * v
        M[-1] = smooth_strength * v
    else:
        M[0]  = smooth_strength * (P[1] - P[0])
        M[-1] = smooth_strength * (P[-1] - P[-2])
        for i in range(1, len(P)-1):
            M[i] = smooth_strength * 0.5 * (P[i+1] - P[i-1])

    out = [P[0]]
    for i in range(len(P)-1):
        seg_len = float(np.linalg.norm(P[i+1] - P[i]))
        samples = int(max(40, min(140, seg_len * sample_density)))  # cap 40–140
        seg = hermite_sample(P[i], M[i], P[i+1], M[i+1], n=samples)
        out.append(seg[1:])
    return np.vstack(out)

# ------------------------- Centering -------------------------

def center_by_internal_vertices(ctrl_img: np.ndarray, path: np.ndarray, H: int) -> np.ndarray:
    if len(ctrl_img) <= 2:
        y_center = float((path[:,1].min() + path[:,1].max()) / 2.0)
    else:
        ys = ctrl_img[1:-1, 1]
        y_center = float((ys.min() + ys.max()) / 2.0)
    dy = (H / 2.0) - y_center
    P = np.asarray(path, float).copy()
    P[:, 1] += dy
    return P

def center_by_length_weighted_mean(path: np.ndarray, H: int, clip: float) -> np.ndarray:
    P = np.asarray(path, float).copy()
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    if L <= 1e-9:
        y_ctr = float((P[:,1].min() + P[:,1].max()) / 2.0)
        P[:,1] += (H/2.0 - y_ctr)
        return P

    u = s / L
    u_mid = 0.5*(u[:-1] + u[1:])
    seg_mask = (u_mid >= clip) & (u_mid <= 1.0 - clip)
    if not np.any(seg_mask):
        seg_mask = slice(None)

    y_seg = 0.5*(P[:-1,1] + P[1:,1])
    y_mean = float(np.average(y_seg[seg_mask], weights=d[seg_mask]))
    P[:, 1] += (H / 2.0 - y_mean)
    return P

# ------------------------- Orientation helpers -------------------------

def rotate_path_about_center(path_img: np.ndarray, W: int, H: int, deg: float) -> np.ndarray:
    """Rotate image-coordinate path around the image center by deg degrees."""
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], float)
    P = np.asarray(path_img, float).copy()
    P -= np.array([W/2.0, H/2.0])
    P = (R @ P.T).T
    P += np.array([W/2.0, H/2.0])
    return P

def fit_inside_canvas_imgcoords(path_img: np.ndarray, W: int, H: int, margin: int, stroke: int) -> np.ndarray:
    """
    Scale and center an already-in-image-coords path so it fits within margins.
    Does not force edge-to-edge; preserves relative geometry after rotation.
    """
    P = np.asarray(path_img, float).copy()
    # Shift to center for uniform scaling
    cx, cy = W/2.0, H/2.0
    P -= np.array([cx, cy])

    # Current spans
    span_x = max(1e-9, P[:,0].max() - P[:,0].min())
    span_y = max(1e-9, P[:,1].max() - P[:,1].min())

    width_lim  = max(1.0, (W - 2*margin - stroke))
    height_lim = max(1.0, (H - 2*margin - stroke))

    s = min(width_lim / span_x, height_lim / span_y)
    P *= s

    # Back to center
    P += np.array([cx, cy])
    return P

# ------------------------- Rendering -------------------------

def render_aa(path: np.ndarray, W: int, H: int, stroke: int,
              bg=(0, 0, 0, 0), fg=(0, 0, 0, 255), aa: int = 2) -> Image.Image:
    W2, H2 = W * aa, H * aa
    pts2 = np.asarray(path, float) * aa

    img = Image.new("RGBA", (W2, H2), bg)  # transparent
    drw = ImageDraw.Draw(img)
    drw.line([tuple(p) for p in pts2], fill=fg, width=stroke * aa, joint="curve")

    if aa > 1:
        img = img.resize((W, H), Image.LANCZOS)
    return img

# ------------------------- Generation -------------------------

import csv
from datetime import datetime

def generate_set_free_oriented(out_root: str,
                               img_size: int = 512,
                               stroke: int = 9,
                               margin: int = 40,
                               vertices=(1,2,3),
                               angles=(45,90,135),
                               curvature_levels=(0,1,2),
                               orientations=(0, 45, 90, 135, 180, -45, -90, -135),
                               s1: float = 0.6,
                               s2: float = 1.1,
                               aa: int = 4,
                               seg_len: float = 1.0):
    """
    Build and save stimuli with FREE-END tangents, then render each at multiple orientations.
    Folder structure unchanged: images/curv{k}/
    A CSV log (stimuli_log.csv) is saved under the output root directory.
    """
    ensure_dir(out_root)
    for k in curvature_levels:
        ensure_dir(os.path.join(out_root, f"images/curv{k}"))

    # --- Initialize CSV log ---
    log_path = os.path.join(out_root, "stimuli_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "filename", "out_path",
            "vertices", "angle_deg", "curvature_level", "orientation_deg",
            "smooth_strength", "img_size", "stroke", "margin"
        ])

        W = H = img_size

        for n in vertices:
            for a in angles:
                if a == 45:
                    clip_for_n1 = 0.30
                elif a == 90:
                    clip_for_n1 = 0.25
                else:
                    clip_for_n1 = 0.20

                ctrl_up = build_polyline_by_turns(n_vertices=n, angle_deg=a, seg_len=seg_len)
                ctrl_oriented = ctrl_up.copy()
                ctrl_oriented[:, 1] *= -1.0

                is_narrow_v = (n == 1 and a == 135)
                fit_mode = "min" if is_narrow_v else "width"
                edge2edge = False if is_narrow_v else True
                ctrl_img = fit_to_frame_general(ctrl_oriented, W=W, H=H, margin=margin,
                                                stroke=stroke, mode=fit_mode, edge_to_edge=edge2edge)

                for k in curvature_levels:
                    if k == 0:
                        base_path = ctrl_img
                        s_used = 0.0
                    else:
                        s = s1 if k == 1 else s2
                        s_used = float(s)
                        base_path = build_smooth_path_free_ends(ctrl_img, smooth_strength=s_used, sample_density=0.30)

                    if n == 1:
                        base_path = center_by_length_weighted_mean(base_path, H=H, clip=clip_for_n1)
                    else:
                        base_path = center_by_internal_vertices(ctrl_img, base_path, H=H)

                    for ori in orientations:
                        path_rot = rotate_path_about_center(base_path, W=W, H=H, deg=float(ori))
                        path_fit = fit_inside_canvas_imgcoords(path_rot, W=W, H=H, margin=margin, stroke=stroke)

                        img = render_aa(path_fit, W=W, H=H, stroke=stroke, aa=aa)
                        rot_lbl = format_rot_label(int(ori))
                        fname = f"stim_nv{n}_ang{a}_curv{k}_rot{rot_lbl}.png"
                        out_path = os.path.join(out_root, f"images/curv{k}", fname)
                        img.save(out_path)

                        # --- Log entry ---
                        writer.writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            fname, out_path,
                            n, a, k, ori,
                            s_used, img_size, stroke, margin
                        ])

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate FREE-END line contour stimuli with multiple orientations (angle × vertices × curvature × orientation)")
    p.add_argument("--out", type=str,
        default=os.path.expanduser("~/Documents/Curvature_Stimuli/stimuli27_oriented"))
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--stroke", type=int, default=9)
    p.add_argument("--margin", type=int, default=40)
    p.add_argument("--aa", type=int, default=4, help="Anti-alias factor")
    p.add_argument("--seg-len", type=float, default=1.0)
    # Optional: override smooth strengths
    p.add_argument("--s1", type=float, default=0.6, help="smooth strength for curvature_level=1")
    p.add_argument("--s2", type=float, default=1.1, help="smooth strength for curvature_level=2")
    # Optional: custom orientations as comma-separated list, e.g. "0,45,90,135,180,-45,-90,-135"
    p.add_argument("--orientations", type=str, default="0,45,90,135,180,-45,-90,-135")
    return p.parse_args()

def parse_orientations(s: str):
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return tuple(vals)

if __name__ == "__main__":
    args = parse_args()
    orientations = parse_orientations(args.orientations)
    generate_set_free_oriented(out_root=args.out,
                               img_size=args.img_size,
                               stroke=args.stroke,
                               margin=args.margin,
                               s1=args.s1,
                               s2=args.s2,
                               aa=args.aa,
                               seg_len=args.seg_len,
                               orientations=orientations)

