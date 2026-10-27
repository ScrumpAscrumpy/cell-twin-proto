"""Initial video -> tracks -> population fit -> simulation verification.

This script is deliberately modest:

- input: a 2D time-lapse (GIF natively; MP4/AVI via ffmpeg)
- segmentation: Otsu threshold + connected components (bright nuclei/cells)
- tracking: greedy nearest-neighbour linking (for QC / visualisation only)
- statistics: *tracking-free* frame-level statistics, so identity switches do
  not poison the fit
- model: the same M1/M3 mechanisms as ``model.py``, initialised from the
  observed first frame
- output: figures + ``results.json``

It is a feasibility check, not a production cell-tracking pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageSequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import DENSITY_EDGES, Params, R_EDGES, simulate  # noqa: E402
from infer import nelder_mead, vec_to_params  # noqa: E402
from plotting import Plot, viridis  # noqa: E402

ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "outputs_video")
FIGDIR = os.path.join(OUTDIR, "figures")

L_MODEL = 42.0
R_MODEL = 3.0
DT_FRAME = 1.0

VIDEO_STAT_NAMES = (
    [f"speed@rho_bin{b}" for b in range(len(DENSITY_EDGES) - 1)]
    + ["nn_mean", "nn_std", "speed_mean", "speed_std"]
    + [f"g(r)_{R_EDGES[b]:.0f}-{R_EDGES[b+1]:.0f}" for b in range(len(R_EDGES) - 1)]
    + ["local_alignment", "density_contrast"]
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def _load_with_pillow(path: str, max_frames: int, step: int, size: Optional[int]):
    im = Image.open(path)
    frames = []
    for i, fr in enumerate(ImageSequence.Iterator(im)):
        if i % step:
            continue
        g = fr.convert("L")
        if size:
            g = g.resize((size, size), Image.BILINEAR)
        frames.append(np.asarray(g, dtype=np.float32))
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def _load_with_ffmpeg(path: str, max_frames: int, step: int, size: Optional[int]):
    tmp = tempfile.mkdtemp(prefix="cellvideo_")
    try:
        vf = []
        if size:
            vf.append(f"scale={size}:{size}")
        cmd = ["ffmpeg", "-v", "error", "-i", path]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-vsync", "0", os.path.join(tmp, "%05d.png")]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        files = sorted(os.listdir(tmp))
        frames = []
        for i, f in enumerate(files):
            if i % step:
                continue
            g = Image.open(os.path.join(tmp, f)).convert("L")
            frames.append(np.asarray(g, dtype=np.float32))
            if max_frames and len(frames) >= max_frames:
                break
        return frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_video(path: str, max_frames: int = 36, step: int = 2, size: Optional[int] = 512):
    if path.lower().endswith((".gif", ".tif", ".tiff", ".png")):
        try:
            return _load_with_pillow(path, max_frames, step, size)
        except Exception:
            if path.lower().endswith((".gif", ".png")):
                raise
    return _load_with_ffmpeg(path, max_frames, step, size)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, img)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    return out


def otsu_threshold(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    hist = hist.astype(float)
    total = hist.sum()
    sum_all = float((np.arange(256) * hist).sum())
    w_b = 0.0
    sum_b = 0.0
    best = -1.0
    thr = 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best:
            best = var
            thr = t
    return thr


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    comps: List[List[Tuple[int, int]]] = []
    for y, x in np.argwhere(mask):
        if lab[y, x] >= 0:
            continue
        idx = len(comps)
        q = deque([(int(y), int(x))])
        lab[y, x] = idx
        pix = []
        while q:
            cy, cx = q.popleft()
            pix.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] < 0:
                        lab[ny, nx] = idx
                        q.append((ny, nx))
        comps.append(pix)
    return comps


def segment(
    frame: np.ndarray,
    *,
    thr: Optional[float] = None,
    amin: int = 30,
    amax: int = 2000,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, float]:
    b = _blur(frame, sigma)
    if thr is None:
        thr = otsu_threshold(b)
    mask = b > thr
    comps = _components(mask)
    out = []
    for pix in comps:
        if not (amin <= len(pix) <= amax):
            continue
        ys = np.array([p[0] for p in pix], dtype=float)
        xs = np.array([p[1] for p in pix], dtype=float)
        out.append((float(xs.mean()), float(ys.mean()), float(len(pix)), float(b[ys.astype(int), xs.astype(int)].mean())))
    return (np.asarray(out, dtype=float) if out else np.zeros((0, 4))), float(thr)


# ---------------------------------------------------------------------------
# Tracking (QC / visualisation)
# ---------------------------------------------------------------------------

def track_detections(dets: Sequence[np.ndarray], max_dist: float = 22.0) -> List[List[Tuple[int, float, float]]]:
    tracks: List[List[Tuple[int, float, float]]] = []
    active: Dict[int, int] = {}
    nid = 0
    for t, d in enumerate(dets):
        if not active:
            for k in range(len(d)):
                active[nid] = k
                tracks.append([(t, float(d[k, 0]), float(d[k, 1]))])
                nid += 1
            continue
        ids = list(active.keys())
        prev = np.asarray([[tracks[i][-1][1], tracks[i][-1][2]] for i in ids], dtype=float)
        cur = d[:, :2]
        pairs = []
        for pi in range(len(prev)):
            for ci in range(len(cur)):
                dist = float(np.hypot(*(prev[pi] - cur[ci])))
                if dist <= max_dist:
                    pairs.append((dist, pi, ci))
        pairs.sort()
        used_p, used_c = set(), set()
        for _, pi, ci in pairs:
            if pi in used_p or ci in used_c:
                continue
            used_p.add(pi)
            used_c.add(ci)
            tracks[ids[pi]].append((t, float(cur[ci, 0]), float(cur[ci, 1])))
        for ci in range(len(cur)):
            if ci not in used_c:
                active[nid] = ci
                tracks.append([(t, float(cur[ci, 0]), float(cur[ci, 1]))])
                nid += 1
        for pi in range(len(prev)):
            if pi not in used_p:
                active.pop(ids[pi], None)
    return tracks


# ---------------------------------------------------------------------------
# Tracking-free frame statistics
# ---------------------------------------------------------------------------

def _pairwise(pos: np.ndarray, L: float, periodic: bool) -> np.ndarray:
    d = pos[None, :, :] - pos[:, None, :]
    if periodic:
        d = (d + 0.5 * L) % L - 0.5 * L
    return np.sqrt(np.sum(d * d, axis=-1))


def _link_frames(pos: np.ndarray, nxt: np.ndarray, max_dist: float, L: float, periodic: bool):
    if len(pos) == 0 or len(nxt) == 0:
        return [], []
    pairs = []
    for i in range(len(pos)):
        for j in range(len(nxt)):
            d = nxt[j] - pos[i]
            if periodic:
                d = (d + 0.5 * L) % L - 0.5 * L
            dist = float(np.hypot(*d))
            if dist <= max_dist:
                pairs.append((dist, i, j))
    pairs.sort()
    ui, uj = set(), set()
    ii, jj = [], []
    for _, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        ii.append(i)
        jj.append(j)
    return ii, jj


def frame_stats(
    frames,
    *,
    L: float = L_MODEL,
    R: float = R_MODEL,
    dt_frame: float = DT_FRAME,
    periodic: bool = True,
    burn: float = 0.12,
    max_link: float = 3.0,
) -> np.ndarray:
    """Statistics that do not require persistent track identities.

    ``frames`` may be a list of (N_t, 2) arrays (video) or an array
    (F, N, 2) (model).  Cells are linked frame-to-frame by nearest
    neighbour only to estimate local velocities.
    """
    if isinstance(frames, np.ndarray):
        fl = [frames[t] for t in range(frames.shape[0])]
    else:
        fl = [np.asarray(f, dtype=float) for f in frames]
    F = len(fl)
    f0 = int(burn * F)
    nn_all, speed_all, rho_all = [], [], []
    g_frames = []
    align_frames = []
    contrast_frames = []
    prof_speed = [[] for _ in range(len(DENSITY_EDGES) - 1)]

    for t in range(f0, F - 1):
        pos = np.asarray(fl[t], dtype=float)
        nxt = np.asarray(fl[t + 1], dtype=float)
        n = len(pos)
        if n < 2 or len(nxt) < 1:
            continue
        dist = _pairwise(pos, L, periodic)
        np.fill_diagonal(dist, np.inf)
        nn = dist.min(axis=1)
        nn_all.append(nn)
        density = (dist < R).sum(axis=1) / (np.pi * R * R)
        rho_all.append(density)
        contrast_frames.append(float(density.std() / max(density.mean(), 1e-9)))

        iu = np.triu_indices(n, k=1)
        r = dist[iu]
        counts, _ = np.histogram(r, bins=R_EDGES)
        shell = np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2)
        expected = 0.5 * n * (n - 1) * shell / (L * L)
        g_frames.append(counts / np.maximum(expected, 1e-9))

        ii, jj = _link_frames(pos, nxt, max_link, L, periodic)
        if not ii:
            continue
        ii = np.asarray(ii)
        jj = np.asarray(jj)
        d = nxt[jj] - pos[ii]
        if periodic:
            d = (d + 0.5 * L) % L - 0.5 * L
        v = d / dt_frame
        speed = np.linalg.norm(v, axis=1)
        speed_all.append(speed)
        for b in range(len(DENSITY_EDGES) - 1):
            m = (density[ii] >= DENSITY_EDGES[b]) & (density[ii] < DENSITY_EDGES[b + 1])
            if np.any(m):
                prof_speed[b].append(speed[m])

        # local velocity alignment
        e = v / np.maximum(speed[:, None], 1e-9)
        vel = np.full((n, 2), np.nan)
        vel[ii] = e
        vals = []
        for i in range(n):
            if not np.isfinite(vel[i]).all():
                continue
            nb = (dist[i] < R) & np.isfinite(vel).all(axis=1)
            if not np.any(nb):
                continue
            mean_e = vel[nb].mean(axis=0)
            norm = np.linalg.norm(mean_e)
            if norm > 1e-9:
                vals.append(float(np.dot(vel[i], mean_e / norm)))
        if vals:
            align_frames.append(float(np.mean(vals)))

    prof = np.full(len(DENSITY_EDGES) - 1, np.nan)
    for b in range(prof.size):
        if prof_speed[b]:
            prof[b] = float(np.mean(np.concatenate(prof_speed[b])))
    good = np.isfinite(prof)
    if good.any():
        prof = np.interp(np.arange(prof.size), np.flatnonzero(good), prof[good])
    else:
        prof[:] = 0.0

    nn_all = np.concatenate(nn_all) if nn_all else np.array([0.0])
    speed_all = np.concatenate(speed_all) if speed_all else np.array([0.0])
    g_mean = np.mean(g_frames, axis=0) if g_frames else np.zeros(len(R_EDGES) - 1)
    align = float(np.mean(align_frames)) if align_frames else 0.0
    contrast = float(np.mean(contrast_frames)) if contrast_frames else 0.0

    return np.concatenate([
        prof,
        [float(nn_all.mean()), float(nn_all.std()),
         float(speed_all.mean()), float(speed_all.std())],
        g_mean,
        [align, contrast],
    ])


# ---------------------------------------------------------------------------
# Model fit
# ---------------------------------------------------------------------------

def rescale(dets: np.ndarray, size: int, L: float = L_MODEL) -> np.ndarray:
    p = dets[:, :2].astype(float).copy()
    p[:, 0] = p[:, 0] / float(size) * L
    p[:, 1] = p[:, 1] / float(size) * L
    return p


def initial_theta(pos: np.ndarray, nxt: np.ndarray, L: float, periodic: bool = True,
                  max_link: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.random(len(pos)) * 2.0 * np.pi
    ii, jj = _link_frames(pos, nxt, max_link, L, periodic)
    for i, j in zip(ii, jj):
        d = nxt[j] - pos[i]
        if periodic:
            d = (d + 0.5 * L) % L - 0.5 * L
        theta[i] = float(np.arctan2(d[1], d[0]))
    return theta


def model_stats(
    params: Params,
    init_pos: np.ndarray,
    init_theta: np.ndarray,
    T: float,
    seed: int = 0,
) -> Tuple[np.ndarray, "object"]:
    # Nuclei in this assay show almost no directional alignment (obs ~0.04),
    # so we switch off the alignment term for the video fit.  It is a
    # diagnostic, not a fitted parameter.
    # Nuclei are ~15 px across; in the 42-unit model box that is ~1.2
    # units, so the repulsion radius must be larger than the default 0.85.
    p = params.with_updates(beta=0.0, Dr=0.5, d0=1.6, k_rep=5.0)
    res = simulate(
        p, N=len(init_pos), L=L_MODEL, T=float(T), dt=0.05, save_every=20,
        seed=seed, init="uniform", init_positions=init_pos, init_theta=init_theta,
    )
    return frame_stats(res.frames, L=L_MODEL, R=R_MODEL, dt_frame=DT_FRAME, periodic=True), res


def fit_video_model(
    model: str,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    init_pos: np.ndarray,
    init_theta: np.ndarray,
    T: float,
    *,
    n_random: int = 10,
    n_nm: int = 22,
    n_restarts: int = 1,
    rng_seed: int = 0,
) -> Dict:
    if model == "M1_adaptive":
        bounds = np.array([[0.05, 1.5], [0.5, 20.0], [0.0, 2.5]], dtype=float)
    elif model == "M1_signed":
        # Diagnostic: allow a crowding-induced speed-up (alpha < 0) to test
        # the sign of the observed speed-density relation.
        bounds = np.array([[0.05, 1.5], [0.5, 20.0], [-2.5, 2.5]], dtype=float)
    elif model == "M3_instant":
        bounds = np.array([[0.05, 1.5], [0.0, 2.5]], dtype=float)
    else:
        raise ValueError(model)
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)
    rng = np.random.default_rng(rng_seed)

    # speed_std (10) is inflated by frame-linking errors, and
    # local_alignment (17) is a model assumption we do not fit here.
    mask = np.ones_like(obs_mean, dtype=bool)
    mask[[10, 17]] = False

    base_model = "M1_adaptive" if model.startswith("M1") else model

    def loss(vec):
        params = vec_to_params(base_model, vec)
        s, _ = model_stats(params, init_pos, init_theta, T, seed=0)
        r = (s[mask] - obs_mean[mask]) / obs_std[mask]
        return float(np.mean(r * r))

    X = rng.uniform(lo, hi, size=(n_random, dim))
    fvals = np.asarray([loss(x) for x in X])
    order = np.argsort(fvals)
    best_x, best_f = X[order[0]], float(fvals[order[0]])
    for k in range(min(n_restarts, len(order))):
        x, f = nelder_mead(
            loss, X[order[k]], step=(hi - lo) * 0.15, lo=lo, hi=hi,
            max_iter=n_nm, xatol=1e-3, fatol=1e-4,
        )
        if f < best_f:
            best_x, best_f = x, f
    names = ["s0", "tau_s", "alpha"] if model.startswith("M1") else ["s0", "alpha"]
    return {
        "model": model,
        "names": names,
        "vec": [float(v) for v in best_x],
        "loss": float(best_f),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def density_midpoints():
    mids = []
    for a, b in zip(DENSITY_EDGES[:-1], DENSITY_EDGES[1:]):
        mids.append(0.5 * (a + b) if np.isfinite(b) else a + 0.12)
    return np.asarray(mids)


def shell_midpoints():
    return 0.5 * (R_EDGES[1:] + R_EDGES[:-1])


def combine(paths, out, cols=1):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (w * cols, h * rows), "white")
    for k, img in enumerate(imgs):
        r, c = divmod(k, cols)
        canvas.paste(img, (c * w, r * h))
    canvas.save(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "data", "deepcell_raw_docs.gif"))
    ap.add_argument("--max-frames", type=int, default=36)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    log(f"loading {args.video}")
    frames = load_video(args.video, max_frames=args.max_frames, step=args.step, size=args.size)
    if len(frames) < 8:
        raise SystemExit(f"need >=8 frames, got {len(frames)}")
    log(f"loaded {len(frames)} frames at {frames[0].shape}")

    log("segmenting frames")
    dets = []
    thrs = []
    for i, f in enumerate(frames):
        d, thr = segment(f)
        dets.append(d)
        thrs.append(thr)
        if (i + 1) % 5 == 0:
            log(f"  {i+1}/{len(frames)}: {len(d)} objects")
    counts = np.array([len(d) for d in dets])
    log(f"detections per frame: median={np.median(counts):.0f}, min={counts.min()}, max={counts.max()}")

    # Use the whole loaded window; shifting to a "median count" frame would
    # throw away most of the movie when max_frames is already the full length.
    t0 = 0
    frames = frames[t0:]
    dets = dets[t0:]
    thrs = thrs[t0:]
    F = len(frames)
    log(f"using frames {t0}..{t0+F-1} (F={F}), start count={len(dets[0])}")

    log("tracking (for QC / visualisation)")
    tracks = track_detections(dets)
    lengths = np.array([len(t) for t in tracks])
    long_tracks = int((lengths >= 0.6 * F).sum())
    log(f"tracks={len(tracks)}, median length={np.median(lengths):.0f}, >=60% length={long_tracks}")

    # Observation statistics, in model coordinates.
    obs_frames = [rescale(d, args.size) for d in dets]
    obs_stats = frame_stats(obs_frames, L=L_MODEL, R=R_MODEL, dt_frame=DT_FRAME,
                            periodic=True, burn=0.12)
    obs_std = np.maximum(0.12 * np.abs(obs_stats), 0.02)
    obs_std[11:17] = np.maximum(obs_std[11:17], 0.05)  # g(r) is noisier

    init_pos = obs_frames[0]
    init_th = initial_theta(init_pos, obs_frames[1], L_MODEL, periodic=True,
                            max_link=3.0, seed=0)
    T = float(F - 1)
    log(f"initial cells={len(init_pos)}, T={T}")

    log("fitting M1_adaptive")
    t = time.time()
    fit1 = fit_video_model("M1_adaptive", obs_stats, obs_std, init_pos, init_th, T)
    log(f"  loss={fit1['loss']:.3f}  {dict(zip(fit1['names'], np.round(fit1['vec'], 3)))}  ({time.time()-t:.1f}s)")

    log("fitting M3_instant")
    t = time.time()
    fit3 = fit_video_model("M3_instant", obs_stats, obs_std, init_pos, init_th, T)
    log(f"  loss={fit3['loss']:.3f}  {dict(zip(fit3['names'], np.round(fit3['vec'], 3)))}  ({time.time()-t:.1f}s)")

    log("fitting M1_signed (diagnostic: alpha may be negative)")
    t = time.time()
    fit_s = fit_video_model("M1_signed", obs_stats, obs_std, init_pos, init_th, T)
    log(f"  loss={fit_s['loss']:.3f}  {dict(zip(fit_s['names'], np.round(fit_s['vec'], 3)))}  ({time.time()-t:.1f}s)")

    log("simulating best M1")
    p1 = vec_to_params("M1_adaptive", fit1["vec"])
    s1, res1 = model_stats(p1, init_pos, init_th, T, seed=1)
    p3 = vec_to_params("M3_instant", fit3["vec"])
    s3, _ = model_stats(p3, init_pos, init_th, T, seed=1)
    ps = vec_to_params("M1_adaptive", fit_s["vec"])
    ss, _ = model_stats(ps, init_pos, init_th, T, seed=1)

    # --- Figure 1: observed frame, detections, simulation -------------------
    panels = []
    for k, (idx, title) in enumerate([(0, f"observed t={0}"), (F - 1, f"observed t={F-1}")]):
        p = Plot(560, 560, margin=(45, 75, 55, 65))
        p.axes((0, L_MODEL), (0, L_MODEL), xlabel="x", ylabel="y", title=title,
               n_xticks=4, n_yticks=4, xtick_fmt="{:.0f}", ytick_fmt="{:.0f}")
        p.image(frames[idx], vmin=0, vmax=255)
        d = obs_frames[idx]
        p.scatter(d[:, 0], d[:, 1], ["#ff3030"] * len(d), r=3.5)
        path = os.path.join(FIGDIR, f"_obs{k}.png")
        p.save(path)
        panels.append(path)

    p = Plot(560, 560, margin=(45, 75, 55, 65))
    p.axes((0, L_MODEL), (0, L_MODEL), xlabel="x", ylabel="y",
           title=f"M1 simulation t={F-1}", n_xticks=4, n_yticks=4,
           xtick_fmt="{:.0f}", ytick_fmt="{:.0f}")
    p.scatter(res1.frames[-1, :, 0], res1.frames[-1, :, 1],
              ["#1f77b4"] * len(init_pos), r=4)
    path = os.path.join(FIGDIR, "_sim.png")
    p.save(path)
    panels.append(path)

    p = Plot(560, 560, margin=(45, 75, 55, 65))
    p.axes((0, L_MODEL), (0, L_MODEL), xlabel="x", ylabel="y",
           title="observed trajectories (long)", n_xticks=4, n_yticks=4,
           xtick_fmt="{:.0f}", ytick_fmt="{:.0f}")
    for tr in tracks:
        if len(tr) < 0.6 * F:
            continue
        a = np.asarray(tr, dtype=float)
        xy = rescale(np.column_stack([a[:, 1], a[:, 2]]), args.size)
        p.line(xy[:, 0], xy[:, 1], color="#1f77b4", width=1)
    path = os.path.join(FIGDIR, "_tracks.png")
    p.save(path)
    panels.append(path)
    combine(panels, os.path.join(FIGDIR, "fig_video_overview.png"), cols=2)

    # --- Figure 2: observation vs simulation statistics ---------------------
    mids = density_midpoints()
    p = Plot(700, 470)
    p.axes((0, 0.7), (0, 1.1), xlabel="local density", ylabel="mean speed",
           title="Speed-density: observation vs fit", n_xticks=5, n_yticks=5)
    p.line(mids, obs_stats[:7], color="#111111", width=3, label="observed")
    p.scatter(mids, obs_stats[:7], ["#111111"] * 7, r=3)
    p.line(mids, s1[:7], color="#1f77b4", width=2, label="M1")
    p.scatter(mids, s1[:7], ["#1f77b4"] * 7, r=3)
    p.line(mids, s3[:7], color="#ff7f0e", width=2, label="M3")
    p.scatter(mids, s3[:7], ["#ff7f0e"] * 7, r=3)
    p.line(mids, ss[:7], color="#2ca02c", width=2, label="M1 signed")
    p.scatter(mids, ss[:7], ["#2ca02c"] * 7, r=3)
    p.legend()
    p.save(os.path.join(FIGDIR, "_fit_speed.png"))

    p = Plot(700, 470)
    p.axes((0, 12.5), (0, 4.0), xlabel="r (model units)", ylabel="g(r)",
           title="Pair correlation: observation vs fit", n_xticks=5, n_yticks=5)
    sm = shell_midpoints()
    p.line(sm, obs_stats[11:17], color="#111111", width=3, label="observed")
    p.scatter(sm, obs_stats[11:17], ["#111111"] * 6, r=3)
    p.line(sm, s1[11:17], color="#1f77b4", width=2, label="M1")
    p.line(sm, s3[11:17], color="#ff7f0e", width=2, label="M3")
    p.line(sm, ss[11:17], color="#2ca02c", width=2, label="M1 signed")
    p.legend()
    p.save(os.path.join(FIGDIR, "_fit_g.png"))
    combine([os.path.join(FIGDIR, "_fit_speed.png"), os.path.join(FIGDIR, "_fit_g.png")],
            os.path.join(FIGDIR, "fig_video_fit.png"), cols=2)

    for f in os.listdir(FIGDIR):
        if f.startswith("_"):
            os.remove(os.path.join(FIGDIR, f))

    results = {
        "video": args.video,
        "frames_used": F,
        "frame_size": args.size,
        "segmentation": {
            "thresholds": [float(v) for v in thrs],
            "counts": [int(v) for v in counts],
            "median_count": float(np.median(counts)),
        },
        "tracking": {
            "n_tracks": len(tracks),
            "median_length": float(np.median(lengths)),
            "long_tracks": long_tracks,
        },
        "observation": {
            "n_cells": int(len(init_pos)),
            "stats": [float(v) for v in obs_stats],
            "names": VIDEO_STAT_NAMES,
        },
        "fits": {"M1_adaptive": fit1, "M1_signed": fit_s, "M3_instant": fit3},
        "simulated_stats": {
            "M1_adaptive": [float(v) for v in s1],
            "M1_signed": [float(v) for v in ss],
            "M3_instant": [float(v) for v in s3],
        },
    }
    with open(os.path.join(OUTDIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("done")
    log(f"figures: {FIGDIR}")
    log(f"results: {os.path.join(OUTDIR, 'results.json')}")


if __name__ == "__main__":
    main()
