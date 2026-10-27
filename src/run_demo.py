"""End-to-end minimal prototype.

Run:
    python src/run_demo.py            # full run (~60-90 s on a laptop)
    python src/run_demo.py --quick    # fast smoke test (~15 s)

Outputs:
    outputs/figures/*.png
    outputs/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
from PIL import Image

# Make ``src`` importable when the script is run from the project root.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import (  # noqa: E402
    DENSITY_EDGES,
    Params,
    pulse_protocol,
    pulse_speed_trace,
    simulate,
)
from infer import (  # noqa: E402
    fit_all,
    fit_s0_single,
    loss_for_vec,
    loss_grid,
    make_observation,
    pulse_loss_grid,
    run_stats,
    single_cell_loss_grid,
    vec_to_params,
)
from plotting import Plot, viridis  # noqa: E402

ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "outputs", "figures")
OUTDIR = os.path.join(ROOT, "outputs")

COLORS = {
    "observed": "#111111",
    "M0_constant": "#8c8c8c",
    "M1_adaptive": "#1f77b4",
    "M2_turning": "#2ca02c",
    "M3_instant": "#ff7f0e",
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def density_midpoints():
    mids = []
    for a, b in zip(DENSITY_EDGES[:-1], DENSITY_EDGES[1:]):
        if np.isfinite(b):
            mids.append(0.5 * (a + b))
        else:
            mids.append(a + 0.12)
    return np.asarray(mids)


def best_point(grid, alphas, taus):
    # grid shape is (n_taus, n_alphas)
    j, i = np.unravel_index(np.argmin(grid), grid.shape)
    return float(alphas[i]), float(taus[j]), float(grid[j, i])


def main(quick: bool = False):
    global OUTDIR, FIGDIR
    if quick:
        OUTDIR = os.path.join(ROOT, "outputs_quick")
        FIGDIR = os.path.join(OUTDIR, "figures")
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    true = Params()

    obs_seeds = [101, 102, 103, 104] if quick else [101, 102, 103, 104, 105, 106]
    fit_seeds = [201] if quick else [201, 202]
    n_random = 8 if quick else 32
    n_nm = 15 if quick else 45
    n_restarts = 1 if quick else 3
    n_workers = 1 if quick else 4
    grid_n = 5 if quick else 9

    log("1/7  generating synthetic population observation")
    obs_mean, obs_std, _ = make_observation(true, obs_seeds)
    log(f"     {len(obs_mean)} summary statistics from {len(obs_seeds)} replicates")

    log("2/7  fitting four candidate models (parallel)")
    t0 = time.time()
    fits = fit_all(
        obs_mean, obs_std, fit_seeds,
        n_workers=n_workers, n_random=n_random, n_nm=n_nm,
        n_restarts=n_restarts, rng_seed=3,
    )
    log(f"     done in {time.time() - t0:.1f}s")
    for name, f in fits.items():
        vals = ", ".join(f"{k}={v:.3f}" for k, v in zip(f["names"], f["vec"]))
        log(f"     {name:12s} train={f['loss']:.3f}  {vals}")

    log("3/7  held-out evaluation (different density / time / seeds)")
    held_kw = dict(N=110 if not quick else 100, L=42.0, T=18.0)
    hmean, hsd, _ = make_observation(true, [301, 302, 303], sim_kw=held_kw)
    held = {}
    for name, f in fits.items():
        held[name] = loss_for_vec(
            name, f["vec"], hmean, hsd, [401, 402], sim_kw=held_kw
        )
    for name, v in held.items():
        log(f"     {name:12s} held-out={v:.3f}")

    log("4/7  identifiability: single cell free vs population vs two densities")
    sc_kw = dict(N=1, L=42.0, T=16.0, init="uniform")
    sc_mean, sc_sd, _ = make_observation(true, [801, 802, 803, 804], sim_kw=sc_kw, single=True)
    s0_sc, _ = fit_s0_single(sc_mean, sc_sd, [811, 812], sim_kw=sc_kw)

    alphas = np.linspace(0.05, 1.8, grid_n)
    taus = np.linspace(0.3, 5.0, grid_n)
    s0_fit = float(fits["M1_adaptive"]["vec"][0])

    single_grid = single_cell_loss_grid(
        s0_sc, alphas, taus, sc_mean, sc_sd, [811, 812],
        sim_kw=sc_kw, n_workers=1 if quick else 8,
    )

    pop_kw = dict(N=90, L=42.0, T=16.0)
    pop_mean, pop_sd, _ = make_observation(true, [501, 502, 503, 504], sim_kw=pop_kw)
    pop_grid = loss_grid(
        s0_fit, alphas, taus,
        [(pop_mean, pop_sd, pop_kw)], [511, 512],
        n_workers=1 if quick else 8,
    )

    d1_kw = dict(N=70, L=42.0, T=16.0)
    d2_kw = dict(N=120, L=42.0, T=16.0)
    d1_mean, d1_sd, _ = make_observation(true, [601, 602, 603, 604], sim_kw=d1_kw)
    d2_mean, d2_sd, _ = make_observation(true, [605, 606, 607, 608], sim_kw=d2_kw)
    two_grid = loss_grid(
        s0_fit, alphas, taus,
        [(d1_mean, d1_sd, d1_kw), (d2_mean, d2_sd, d2_kw)], [611, 612],
        n_workers=1 if quick else 8,
    )

    log("5/7  identifiability: controlled crowding pulse (single cell)")
    T_p, dt_p, npt = 22.0, 0.05, 18
    protocol = pulse_protocol(T_p, dt_p, start=6.0, end=13.0, amplitude=0.4)
    ptraces = []
    for sd in ([701, 702, 703] if quick else [701, 702, 703, 704, 705]):
        res = simulate(
            true, N=1, L=42.0, T=T_p, dt=dt_p, save_every=10,
            seed=sd, init="uniform", external_density=protocol,
        )
        ptraces.append(pulse_speed_trace(res, npt))
    ptraces = np.asarray(ptraces)
    pmean = ptraces.mean(axis=0)
    psd = np.maximum(ptraces.std(axis=0, ddof=1), 0.02)
    s0_pulse = float(pmean[:4].mean())
    pulse_grid = pulse_loss_grid(
        s0_pulse, alphas, taus, pmean, psd, protocol, [711, 712],
        T=T_p, dt=dt_p, n_points=npt, n_workers=1 if quick else 8,
    )

    log("6/7  drawing figures")
    # --- Figure 1: snapshots ------------------------------------------------
    res_true = simulate(true, N=90, L=42.0, T=16.0, seed=101, init="mixed")
    res_fit = simulate(
        vec_to_params("M1_adaptive", fits["M1_adaptive"]["vec"]),
        N=90, L=42.0, T=16.0, seed=101, init="mixed",
    )
    panels = []
    for k, (res, title) in enumerate([
        (res_true, "observed  t = 0"),
        (res_true, "observed  t = end"),
        (res_fit, "M1 fit  t = end"),
    ]):
        t = 0 if k == 0 else -1
        p = Plot(520, 520, margin=(40, 70, 55, 60))
        p.axes((0, res.L), (0, res.L), xlabel="x", ylabel="y", title=title,
               n_xticks=3, n_yticks=3)
        sp = res.speed[t]
        cols = [viridis(float(np.clip(v / 1.0, 0, 1))) for v in sp]
        p.scatter(res.frames[t, :, 0], res.frames[t, :, 1], cols, r=4.2)
        p.gradient_bar(0.0, 1.0, "speed")
        path = os.path.join(FIGDIR, f"_snap{k}.png")
        p.save(path)
        panels.append(path)
    combine(panels, os.path.join(FIGDIR, "fig1_snapshots.png"), cols=3)

    # --- Figure 2: speed-density profile ------------------------------------
    mids = density_midpoints()
    p = Plot(780, 500)
    p.axes((0, 0.68), (0, 1.05), xlabel="local density (cells / area)",
           ylabel="mean speed", title="Speed-density relation")
    p.line(mids, obs_mean[:7], color=COLORS["observed"], width=3, label="observed")
    p.scatter(mids, obs_mean[:7], [COLORS["observed"]] * 7, r=3.5)
    for name in ["M0_constant", "M2_turning", "M3_instant", "M1_adaptive"]:
        s = run_stats(vec_to_params(name, fits[name]["vec"]), [901], sim_kw=pop_kw)
        p.line(mids, s[:7], color=COLORS[name], width=2, label=name)
        p.scatter(mids, s[:7], [COLORS[name]] * 7, r=3)
    p.legend()
    p.save(os.path.join(FIGDIR, "fig2_speed_density.png"))

    # --- Figure 3: model comparison -----------------------------------------
    names = ["M0_constant", "M1_adaptive", "M2_turning", "M3_instant"]
    labels = ["M0", "M1", "M2", "M3"]
    p = Plot(760, 480)
    p.bar_group(
        labels,
        [[fits[n]["loss"] for n in names], [held[n] for n in names]],
        ["#9ecae1", "#08519c"],
        ["train", "held-out"],
    )
    p.legend()
    p.save(os.path.join(FIGDIR, "fig3_model_comparison.png"))

    # --- Figure 4: identifiability heatmaps ---------------------------------
    panels = []
    specs = [
        (single_grid, "single cell, free\nalpha unidentifiable", None),
        (pop_grid, "population, 1 density", 6.0),
        (two_grid, "population, 2 densities", 6.0),
        (pulse_grid, "single cell + crowding pulse", 6.0),
    ]
    for k, (grid, title, vmax) in enumerate(specs):
        p = Plot(520, 470, margin=(80, 62, 55, 80))
        p.axes((float(alphas[0]), float(alphas[-1])), (float(taus[0]), float(taus[-1])),
               xlabel="alpha (crowding gain)", ylabel="tau_s (memory)",
               title=title, n_xticks=4, n_yticks=4)
        p.heatmap(grid, (float(alphas[0]), float(alphas[-1])),
                  (float(taus[0]), float(taus[-1])), vmin=0.0, vmax=vmax)
        p.marker(0.8, 2.0, "white", r=5)
        path = os.path.join(FIGDIR, f"_id{k}.png")
        p.save(path)
        panels.append(path)
    combine(panels, os.path.join(FIGDIR, "fig4_identifiability.png"), cols=2)

    # --- Figure 5: pulse traces ---------------------------------------------
    a_best, t_best, _ = best_point(pulse_grid, alphas, taus)
    res_m1 = simulate(
        true.with_updates(s0=s0_pulse, alpha=a_best, tau_s=t_best, adaptive=True),
        N=1, L=42.0, T=T_p, dt=dt_p, save_every=10, seed=711,
        init="uniform", external_density=protocol,
    )
    res_m3 = simulate(
        vec_to_params("M3_instant", fits["M3_instant"]["vec"]),
        N=1, L=42.0, T=T_p, dt=dt_p, save_every=10, seed=711,
        init="uniform", external_density=protocol,
    )
    x = np.linspace(0, T_p, npt)
    p = Plot(780, 480)
    p.axes((0, T_p), (0, 1.1), xlabel="time", ylabel="motility drive / speed",
           title="Crowding pulse: observed vs candidate mechanisms")
    p.line(x, pmean, color=COLORS["observed"], width=3, label="observed")
    p.line(x, pulse_speed_trace(res_m1, npt), color=COLORS["M1_adaptive"], width=2,
           label=f"M1 (alpha={a_best:.2f}, tau={t_best:.2f})")
    p.line(x, pulse_speed_trace(res_m3, npt), color=COLORS["M3_instant"], width=2,
           label="M3 (memoryless)")
    p.legend()
    p.save(os.path.join(FIGDIR, "fig5_pulse.png"))

    # Remove the intermediate panel images used to build the composites.
    for f in os.listdir(FIGDIR):
        if f.startswith("_"):
            os.remove(os.path.join(FIGDIR, f))

    log("7/7  writing results.json")
    def _grid_out(g):
        return [[float(v) for v in row] for row in g]

    results = {
        "true_params": asdict(true),
        "observation": {
            "mean": [float(v) for v in obs_mean],
            "sd": [float(v) for v in obs_std],
            "seeds": obs_seeds,
        },
        "fits": fits,
        "heldout_loss": {k: float(v) for k, v in held.items()},
        "identifiability": {
            "alphas": [float(v) for v in alphas],
            "taus": [float(v) for v in taus],
            "single_cell_s0": float(s0_sc),
            "single_cell_best": best_point(single_grid, alphas, taus),
            "population_best": best_point(pop_grid, alphas, taus),
            "two_density_best": best_point(two_grid, alphas, taus),
            "pulse_s0": float(s0_pulse),
            "pulse_best": best_point(pulse_grid, alphas, taus),
            "single_cell_grid": _grid_out(single_grid),
            "population_grid": _grid_out(pop_grid),
            "two_density_grid": _grid_out(two_grid),
            "pulse_grid": _grid_out(pulse_grid),
        },
    }
    with open(os.path.join(OUTDIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("done")
    log(f"figure: {os.path.join(FIGDIR, 'fig1_snapshots.png')} ... fig5_pulse.png")
    log(f"results: {os.path.join(OUTDIR, 'results.json')}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small budget smoke test")
    args = ap.parse_args()
    main(quick=args.quick)
