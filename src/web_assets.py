"""Build the static GitHub Pages assets from the prototype outputs."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
ASSETS = os.path.join(DOCS, "assets")
FRAMES = os.path.join(ASSETS, "frames")
FIGS = os.path.join(ASSETS, "figs")

import sys
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from infer import run_stats, vec_to_params  # noqa: E402
from model import simulate  # noqa: E402
from video_pipeline import (  # noqa: E402
    L_MODEL,
    initial_theta,
    load_video,
    rescale,
    segment,
)

SYNTHETIC_SIM = dict(N=90, L=42.0, T=16.0)
VIDEO_PATH = os.path.join(ROOT, "data", "deepcell_raw_docs.gif")


def density_midpoints():
    from model import DENSITY_EDGES

    mids = []
    for a, b in zip(DENSITY_EDGES[:-1], DENSITY_EDGES[1:]):
        mids.append(0.5 * (a + b) if np.isfinite(b) else a + 0.12)
    return [float(v) for v in mids]


def copy_figures():
    os.makedirs(FIGS, exist_ok=True)
    pairs = [
        (os.path.join(ROOT, "outputs", "figures", "fig1_snapshots.png"), "synthetic_snapshots.png"),
        (os.path.join(ROOT, "outputs", "figures", "fig3_model_comparison.png"), "model_comparison.png"),
        (os.path.join(ROOT, "outputs", "figures", "fig4_identifiability.png"), "identifiability.png"),
        (os.path.join(ROOT, "outputs_video", "figures", "fig_video_overview.png"), "video_overview.png"),
        (os.path.join(ROOT, "outputs_video", "figures", "fig_video_fit.png"), "video_fit.png"),
    ]
    for src, dst in pairs:
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(FIGS, dst))


def build_synthetic(syn):
    curves = {"observed": [float(v) for v in syn["observation"]["mean"][:7]]}
    for model in ["M0_constant", "M1_adaptive", "M2_turning", "M3_instant"]:
        stats = run_stats(
            vec_to_params(model, syn["fits"][model]["vec"]),
            [901],
            sim_kw=SYNTHETIC_SIM,
        )
        curves[model] = [float(v) for v in stats[:7]]

    idn = syn["identifiability"]
    return {
        "models": ["M0", "M1", "M2", "M3"],
        "model_keys": ["M0_constant", "M1_adaptive", "M2_turning", "M3_instant"],
        "train": [float(syn["fits"][m]["loss"]) for m in
                  ["M0_constant", "M1_adaptive", "M2_turning", "M3_instant"]],
        "heldout": [float(syn["heldout_loss"][m]) for m in
                    ["M0_constant", "M1_adaptive", "M2_turning", "M3_instant"]],
        "speed_density": curves,
        "mids": density_midpoints(),
        "identifiability": {
            "alphas": [float(v) for v in idn["alphas"]],
            "taus": [float(v) for v in idn["taus"]],
            "true": [0.8, 2.0],
            "panels": {
                "single": idn["single_cell_grid"],
                "population": idn["population_grid"],
                "two": idn["two_density_grid"],
                "pulse": idn["pulse_grid"],
            },
        },
    }


def draw_dots(frame, dets, color=(255, 48, 48), radius=7, size=512):
    img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in dets:
        x, y = float(d[0]), float(d[1])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    return img.resize((size, size), Image.LANCZOS)


def draw_sim(pos, color=(31, 119, 180), radius=6, size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x, y in pos:
        px = float(x) / L_MODEL * size
        py = float(y) / L_MODEL * size
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color)
    return img


def build_video(vid):
    os.makedirs(FRAMES, exist_ok=True)
    frames = load_video(VIDEO_PATH, max_frames=36, step=2, size=512)
    dets = [segment(f)[0] for f in frames]
    obs_frames = [rescale(d, 512) for d in dets]

    init_pos = obs_frames[0]
    init_th = initial_theta(init_pos, obs_frames[1], L_MODEL, periodic=True,
                            max_link=3.0, seed=0)
    p1 = vec_to_params("M1_adaptive", vid["fits"]["M1_adaptive"]["vec"]).with_updates(
        beta=0.0, Dr=0.5, d0=1.6, k_rep=5.0
    )
    res = simulate(
        p1, N=len(init_pos), L=L_MODEL, T=float(len(frames) - 1),
        dt=0.05, save_every=20, seed=1, init="uniform",
        init_positions=init_pos, init_theta=init_th,
    )

    idx = np.linspace(0, len(frames) - 1, 12).astype(int)
    meta = []
    for k, i in enumerate(idx):
        obs = draw_dots(frames[int(i)], dets[int(i)], size=384)
        sim = draw_sim(res.frames[int(i)], size=384)
        obs_name = f"obs_{k:02d}.jpg"
        sim_name = f"sim_{k:02d}.jpg"
        obs.save(os.path.join(FRAMES, obs_name), quality=88, optimize=True)
        sim.save(os.path.join(FRAMES, sim_name), quality=88, optimize=True)
        meta.append({"t": int(i), "obs": f"assets/frames/{obs_name}",
                     "sim": f"assets/frames/{sim_name}"})

    stats = [float(v) for v in vid["observation"]["stats"]]
    sim_stats = {k: [float(v) for v in v] for k, v in vid["simulated_stats"].items()}
    return {
        "source": "DeepCell docs raw.gif (512x512, 71 frames; 36 used)",
        "n_frames": len(frames),
        "n_cells": int(len(init_pos)),
        "counts": [int(v) for v in vid["segmentation"]["counts"]],
        "counts_median": float(vid["segmentation"]["median_count"]),
        "tracking": vid["tracking"],
        "fits": vid["fits"],
        "observed": {
            "speed": stats[:7],
            "g": stats[11:17],
            "nn_mean": stats[7],
            "nn_std": stats[8],
            "speed_mean": stats[9],
            "speed_std": stats[10],
            "alignment": stats[17],
            "contrast": stats[18],
        },
        "simulated": {
            "M1_adaptive": {
                "speed": sim_stats["M1_adaptive"][:7],
                "g": sim_stats["M1_adaptive"][11:17],
            },
            "M1_signed": {
                "speed": sim_stats["M1_signed"][:7],
                "g": sim_stats["M1_signed"][11:17],
            },
            "M3_instant": {
                "speed": sim_stats["M3_instant"][:7],
                "g": sim_stats["M3_instant"][11:17],
            },
        },
        "frames": meta,
    }


def main():
    os.makedirs(ASSETS, exist_ok=True)
    syn = json.load(open(os.path.join(ROOT, "outputs", "results.json")))
    vid = json.load(open(os.path.join(ROOT, "outputs_video", "results.json")))

    print("copying figures")
    copy_figures()

    print("building synthetic data")
    synthetic = build_synthetic(syn)

    print("building video frames (segment + simulate)")
    video = build_video(vid)

    data = {"synthetic": synthetic, "video": video}
    with open(os.path.join(ASSETS, "data.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote {os.path.join(ASSETS, 'data.json')}")


if __name__ == "__main__":
    main()
