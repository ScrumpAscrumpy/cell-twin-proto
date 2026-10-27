"""Parameter recovery, model comparison and identifiability analysis.

The workflow is deliberately small:

1. Simulate a synthetic "ground truth" population in which each cell
   carries a hidden intracellular motility drive ``s`` that adapts to
   local crowding.
2. Fit four candidate models to population-level summary statistics.
   Only one of them contains the hidden adaptive state.
3. Show that isolated-cell statistics are blind to the mechanism, while
   population statistics can identify it (at least partially).
4. Compare held-out predictive loss to select the correct mechanism.

All optimisation is derivative-free (random search + Nelder-Mead) so the
prototype depends only on NumPy.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from model import (
    Params,
    STAT_NAMES,
    pulse_protocol,
    pulse_speed_trace,
    simulate,
    single_cell_summarize,
    summarize,
)

# Default simulation configuration used for both observation and fitting.
SIM: Dict = dict(N=90, L=42.0, T=16.0, dt=0.05, save_every=10, init="mixed")
BURN = 0.2


# ---------------------------------------------------------------------------
# Running simulations and building observations
# ---------------------------------------------------------------------------

def _sim_kw(overrides: Optional[Dict] = None) -> Dict:
    kw = dict(SIM)
    if overrides:
        kw.update(overrides)
    return kw


def run_stats(
    params: Params,
    seeds: Sequence[int],
    *,
    sim_kw: Optional[Dict] = None,
    single: bool = False,
) -> np.ndarray:
    """Average summary statistics over stochastic replicates."""
    kw = _sim_kw(sim_kw)
    rows = []
    for sd in seeds:
        res = simulate(params, seed=int(sd), **kw)
        rows.append(single_cell_summarize(res) if single else summarize(res, burn=BURN))
    return np.mean(np.asarray(rows, dtype=float), axis=0)


def make_observation(
    true_params: Params,
    seeds: Sequence[int],
    *,
    sim_kw: Optional[Dict] = None,
    single: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, sd, raw) summary statistics for a synthetic dataset."""
    kw = _sim_kw(sim_kw)
    rows = []
    for sd in seeds:
        res = simulate(true_params, seed=int(sd), **kw)
        rows.append(single_cell_summarize(res) if single else summarize(res, burn=BURN))
    raw = np.asarray(rows, dtype=float)
    mean = raw.mean(axis=0)
    sd = raw.std(axis=0, ddof=1) if len(rows) > 1 else np.ones_like(mean)
    # Floor the observation noise: very stable statistics should not get
    # infinite weight, and we do not want to fit stochastic noise.
    sd = np.maximum(sd, np.maximum(0.05 * np.abs(mean), 1e-4))
    return mean, sd, raw


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

MODEL_SPECS: Dict[str, Dict] = {
    # no intracellular state: constant speed, no crowding response
    "M0_constant": dict(names=["s0"], bounds=[(0.3, 1.8)]),
    # the true mechanism: adaptive speed with memory
    "M1_adaptive": dict(
        names=["s0", "tau_s", "alpha"],
        bounds=[(0.3, 1.8), (0.2, 10.0), (0.0, 2.5)],
    ),
    # alternative mechanism: crowding increases turning, speed stays constant
    "M2_turning": dict(names=["s0", "gamma_turn"], bounds=[(0.3, 1.8), (0.0, 6.0)]),
    # memoryless speed reduction: same steady state, no adaptation time
    "M3_instant": dict(names=["s0", "alpha"], bounds=[(0.3, 1.8), (0.0, 2.5)]),
}


def vec_to_params(model: str, vec: Sequence[float]) -> Params:
    v = np.asarray(vec, dtype=float)
    base = Params()
    if model == "M0_constant":
        return base.with_updates(s0=float(v[0]), adaptive=False, alpha=0.0)
    if model == "M1_adaptive":
        return base.with_updates(
            s0=float(v[0]), tau_s=float(v[1]), alpha=float(v[2]), adaptive=True
        )
    if model == "M2_turning":
        return base.with_updates(
            s0=float(v[0]), gamma_turn=float(v[1]), adaptive=False, alpha=0.0
        )
    if model == "M3_instant":
        return base.with_updates(
            s0=float(v[0]), alpha=float(v[1]), adaptive=False, instantaneous=True
        )
    raise ValueError(f"unknown model: {model}")


def loss_for_vec(
    model: str,
    vec: Sequence[float],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seeds: Sequence[int],
    *,
    sim_kw: Optional[Dict] = None,
    single: bool = False,
) -> float:
    params = vec_to_params(model, vec)
    s = run_stats(params, seeds, sim_kw=sim_kw, single=single)
    r = (s - obs_mean) / obs_std
    return float(np.mean(r * r))


# ---------------------------------------------------------------------------
# Derivative-free optimisation
# ---------------------------------------------------------------------------

def nelder_mead(
    f,
    x0: np.ndarray,
    *,
    step: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    max_iter: int = 60,
    xatol: float = 1e-3,
    fatol: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """Bounded Nelder-Mead simplex (small dimension, noisy objective)."""
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    n = x0.size
    simplex = [x0.copy()]
    for i in range(n):
        x = x0.copy()
        x[i] = np.clip(x[i] + step[i], lo[i], hi[i])
        if np.allclose(x, x0):
            x[i] = np.clip(x[i] - step[i], lo[i], hi[i])
        simplex.append(x)
    simplex = np.asarray(simplex)
    fvals = np.asarray([f(x) for x in simplex], dtype=float)

    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        if np.max(np.abs(simplex[1:] - simplex[0])) < xatol and np.max(
            np.abs(fvals[1:] - fvals[0])
        ) < fatol:
            break

        centroid = simplex[:-1].mean(axis=0)
        xr = np.clip(centroid + (centroid - simplex[-1]), lo, hi)
        fr = f(xr)
        if fr < fvals[0]:
            xe = np.clip(centroid + 2.0 * (centroid - simplex[-1]), lo, hi)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), lo, hi)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                fvals[1:] = np.asarray([f(x) for x in simplex[1:]])

    best = int(np.argmin(fvals))
    return simplex[best], float(fvals[best])


def fit_model(
    model: str,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seeds: Sequence[int],
    *,
    n_random: int = 28,
    n_nm: int = 55,
    n_restarts: int = 1,
    rng_seed: int = 0,
    sim_kw: Optional[Dict] = None,
) -> Dict:
    spec = MODEL_SPECS[model]
    bounds = np.asarray(spec["bounds"], dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)
    rng = np.random.default_rng(rng_seed)

    X = rng.uniform(lo, hi, size=(n_random, dim))
    fvals = np.asarray(
        [loss_for_vec(model, x, obs_mean, obs_std, seeds, sim_kw=sim_kw) for x in X]
    )
    order = np.argsort(fvals)

    best_x = X[order[0]]
    best_f = float(fvals[order[0]])
    for k in range(min(n_restarts, len(order))):
        x, f = nelder_mead(
            lambda v: loss_for_vec(model, v, obs_mean, obs_std, seeds, sim_kw=sim_kw),
            X[order[k]],
            step=(hi - lo) * 0.15,
            lo=lo,
            hi=hi,
            max_iter=n_nm,
        )
        if f < best_f:
            best_x, best_f = x, f

    return dict(
        model=model,
        names=spec["names"],
        vec=[float(v) for v in best_x],
        loss=best_f,
        n_params=dim,
    )


def _fit_worker(args) -> Dict:
    (model, obs_mean, obs_std, seeds, n_random, n_nm,
     n_restarts, rng_seed, sim_kw) = args
    return fit_model(
        model, obs_mean, obs_std, seeds,
        n_random=n_random, n_nm=n_nm, n_restarts=n_restarts,
        rng_seed=rng_seed, sim_kw=sim_kw,
    )


def fit_all(
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seeds: Sequence[int],
    *,
    models: Optional[Sequence[str]] = None,
    n_workers: int = 4,
    **fit_kw,
) -> Dict[str, Dict]:
    models = list(models or MODEL_SPECS.keys())
    # fit_kw is identical for all models; pass as a tuple for pickling.
    jobs = [
        (m, obs_mean, obs_std, seeds, fit_kw.get("n_random", 28),
         fit_kw.get("n_nm", 55), fit_kw.get("n_restarts", 1),
         fit_kw.get("rng_seed", 0), fit_kw.get("sim_kw", None))
        for m in models
    ]
    if n_workers > 1 and len(models) > 1:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(models))) as ex:
            results = list(ex.map(_fit_worker, jobs))
    else:
        results = [_fit_worker(j) for j in jobs]
    return {r["model"]: r for r in results}


# ---------------------------------------------------------------------------
# Identifiability surfaces
# ---------------------------------------------------------------------------

def _grid_worker(args) -> float:
    alpha, tau_s, s0, obs_list, seeds = args
    params = Params().with_updates(s0=float(s0), tau_s=float(tau_s), alpha=float(alpha), adaptive=True)
    total = 0.0
    for obs_mean, obs_std, sim_kw in obs_list:
        s = run_stats(params, seeds, sim_kw=sim_kw)
        r = (s - obs_mean) / obs_std
        total += float(np.mean(r * r))
    return total / len(obs_list)


def _single_grid_worker(args) -> float:
    alpha, tau_s, s0, obs_mean, obs_std, seeds, sim_kw = args
    params = Params().with_updates(s0=float(s0), tau_s=float(tau_s), alpha=float(alpha), adaptive=True)
    s = run_stats(params, seeds, sim_kw=sim_kw, single=True)
    r = (s - obs_mean) / obs_std
    return float(np.mean(r * r))


def loss_grid(
    s0: float,
    alphas: Sequence[float],
    taus: Sequence[float],
    obs_list: Sequence[Tuple[np.ndarray, np.ndarray, Dict]],
    seeds: Sequence[int],
    *,
    n_workers: int = 8,
) -> np.ndarray:
    args = [
        (float(a), float(t), float(s0), list(obs_list), seeds)
        for a in alphas
        for t in taus
    ]
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(args))) as ex:
            vals = list(ex.map(_grid_worker, args))
    else:
        vals = [_grid_worker(a) for a in args]
    return np.asarray(vals).reshape(len(alphas), len(taus)).T


def single_cell_loss_grid(
    s0: float,
    alphas: Sequence[float],
    taus: Sequence[float],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seeds: Sequence[int],
    *,
    sim_kw: Optional[Dict] = None,
    n_workers: int = 8,
) -> np.ndarray:
    args = [
        (float(a), float(t), float(s0), obs_mean, obs_std, seeds, sim_kw)
        for a in alphas
        for t in taus
    ]
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(args))) as ex:
            vals = list(ex.map(_single_grid_worker, args))
    else:
        vals = [_single_grid_worker(a) for a in args]
    return np.asarray(vals).reshape(len(alphas), len(taus)).T


def fit_s0_single(
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seeds: Sequence[int],
    *,
    sim_kw: Optional[Dict] = None,
) -> Tuple[float, float]:
    """1-D fit of s0 to isolated-cell statistics."""
    def f(v):
        return loss_for_vec("M1_adaptive", [v[0], 2.0, 0.8], obs_mean, obs_std, seeds,
                            sim_kw=sim_kw, single=True)

    x, fv = nelder_mead(
        f,
        np.array([1.0]),
        step=np.array([0.2]),
        lo=np.array([0.3]),
        hi=np.array([1.8]),
        max_iter=30,
    )
    return float(x[0]), float(fv)


# ---------------------------------------------------------------------------
# Controlled crowding pulse: a single-cell perturbation experiment
# ---------------------------------------------------------------------------

def _pulse_grid_worker(args) -> float:
    alpha, tau_s, s0, obs_trace, obs_std, protocol, seeds, T, dt, n_points = args
    params = Params().with_updates(
        s0=float(s0), tau_s=float(tau_s), alpha=float(alpha), adaptive=True
    )
    sims = []
    for sd in seeds:
        res = simulate(
            params, N=1, L=42.0, T=float(T), dt=float(dt), save_every=10,
            seed=int(sd), init="uniform", external_density=protocol,
        )
        sims.append(pulse_speed_trace(res, int(n_points)))
    sim = np.mean(sims, axis=0)
    r = (sim - obs_trace) / obs_std
    return float(np.mean(r * r))


def pulse_loss_grid(
    s0: float,
    alphas: Sequence[float],
    taus: Sequence[float],
    obs_trace: np.ndarray,
    obs_std: np.ndarray,
    protocol: np.ndarray,
    seeds: Sequence[int],
    *,
    T: float,
    dt: float,
    n_points: int,
    n_workers: int = 8,
) -> np.ndarray:
    args = [
        (float(a), float(t), float(s0), obs_trace, obs_std, protocol, seeds,
         float(T), float(dt), int(n_points))
        for a in alphas
        for t in taus
    ]
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(args))) as ex:
            vals = list(ex.map(_pulse_grid_worker, args))
    else:
        vals = [_pulse_grid_worker(a) for a in args]
    return np.asarray(vals).reshape(len(alphas), len(taus)).T
