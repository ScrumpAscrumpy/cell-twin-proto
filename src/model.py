"""Minimal 2D cell-population simulator.

Each cell carries one *intracellular* state, its motility drive ``s``.
The drive relaxes towards ``s0`` and is reduced by local crowding
(contact inhibition of speed).  This is the "hidden intracellular
mechanism" that population-level data are supposed to identify.

The model is deliberately small and NumPy-only so the whole prototype
runs anywhere.  It is a toy, not a biological model of a particular cell.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Params:
    # single-cell / intracellular
    s0: float = 1.0          # baseline motility drive (speed units)
    tau_s: float = 2.0       # relaxation time of the drive
    alpha: float = 0.8       # crowding-induced speed reduction
    sigma_s: float = 0.03    # drive noise
    # polarity
    beta: float = 1.6        # alignment strength
    Dr: float = 0.25         # rotational diffusion
    gamma_turn: float = 0.0  # crowding-induced turning (alternative mechanism)
    # interactions / mechanics
    R: float = 3.0           # interaction radius
    d0: float = 0.85         # repulsion radius
    k_rep: float = 2.0       # repulsion strength
    # mechanism switch
    adaptive: bool = True    # if False: s == s0 (no intracellular state dynamics)
    instantaneous: bool = False  # if True: s = s0 - alpha*rho (no memory)

    def with_updates(self, **kw) -> "Params":
        return replace(self, **kw)


@dataclass
class SimResult:
    frames: np.ndarray       # (F, N, 2) positions
    theta: np.ndarray        # (F, N) orientations
    speed: np.ndarray        # (F, N) intracellular drive
    density: np.ndarray      # (F, N) local density
    dt_frame: float
    L: float
    N: int

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]


def _wrap(d: np.ndarray, L: float) -> np.ndarray:
    return (d + 0.5 * L) % L - 0.5 * L


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _pairwise(frames_t: np.ndarray, L: float) -> np.ndarray:
    """Periodic pairwise distances for one frame, shape (N, N)."""
    d = _wrap(frames_t[None, :, :] - frames_t[:, None, :], L)
    return np.sqrt(np.sum(d * d, axis=-1))


def simulate(
    params: Params,
    *,
    N: int = 150,
    L: float = 42.0,
    T: float = 24.0,
    dt: float = 0.05,
    save_every: int = 10,
    seed: int = 0,
    init: str = "blob",
    blob_radius: float = 7.0,
    center: Optional[Tuple[float, float]] = None,
    external_density: Optional[np.ndarray] = None,
    init_positions: Optional[np.ndarray] = None,
    init_theta: Optional[np.ndarray] = None,
    init_speed: Optional[np.ndarray] = None,
) -> SimResult:
    """Run one simulation and return saved frames.

    ``init="blob"`` starts from a dense disc and lets it expand, which
    produces a wide range of local densities in a single run.  ``init=
    "uniform"`` starts from a uniform random configuration.
    """
    rng = np.random.default_rng(seed)
    steps = int(round(T / dt))
    n_save = steps // save_every + 1
    if external_density is not None:
        external_density = np.asarray(external_density, dtype=float)
        if external_density.size != steps + 1:
            raise ValueError("external_density must have length steps+1")
    cx, cy = center if center is not None else (0.5 * L, 0.5 * L)

    if init_positions is not None:
        x = np.asarray(init_positions, dtype=float).copy()
        if x.shape != (N, 2):
            raise ValueError(f"init_positions must have shape ({N}, 2), got {x.shape}")
        x = x % L
    elif init == "blob":
        r = blob_radius * np.sqrt(rng.random(N))
        a = 2.0 * np.pi * rng.random(N)
        x = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    elif init == "uniform":
        x = rng.random((N, 2)) * L
    elif init == "mixed":
        # A dense core plus a dilute background: spans the density range
        # in a single run, which is what makes the speed-density curve
        # identifiable from population data.
        n_blob = int(round(0.7 * N))
        n_bg = N - n_blob
        r = blob_radius * np.sqrt(rng.random(n_blob))
        a = 2.0 * np.pi * rng.random(n_blob)
        xb = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
        xg = rng.random((n_bg, 2)) * L
        x = np.concatenate([xb, xg], axis=0)
    else:
        raise ValueError(f"unknown init: {init}")

    if init_theta is None:
        theta = rng.random(N) * 2.0 * np.pi
    else:
        theta = np.asarray(init_theta, dtype=float).copy()
        if theta.shape != (N,):
            raise ValueError(f"init_theta must have shape ({N},), got {theta.shape}")
    if init_speed is None:
        s = np.full(N, params.s0, dtype=float)
    else:
        s = np.asarray(init_speed, dtype=float).copy()
        if s.shape != (N,):
            raise ValueError(f"init_speed must have shape ({N},), got {s.shape}")

    frames = np.empty((n_save, N, 2), dtype=float)
    theta_hist = np.empty((n_save, N), dtype=float)
    speed_hist = np.empty((n_save, N), dtype=float)
    density_hist = np.empty((n_save, N), dtype=float)

    idx = 0
    for step in range(steps + 1):
        # --- interactions -------------------------------------------------
        d = _wrap(x[None, :, :] - x[:, None, :], L)       # x_j - x_i
        dist = np.sqrt(np.sum(d * d, axis=-1))
        np.fill_diagonal(dist, np.inf)
        mask = dist < params.R
        density = mask.sum(axis=1) / (np.pi * params.R * params.R)

        if step % save_every == 0:
            frames[idx] = x
            theta_hist[idx] = theta
            speed_hist[idx] = s
            density_hist[idx] = density
            idx += 1

        if step == steps:
            break

        # --- polarity alignment ------------------------------------------
        e = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        sum_e = mask.astype(float) @ e
        norm = np.linalg.norm(sum_e, axis=1)
        has_nb = norm > 1e-9
        phi = np.arctan2(sum_e[:, 1], sum_e[:, 0])
        dtheta = _wrap_angle(phi - theta)
        theta = theta + params.beta * dt * dtheta * has_nb

        # --- intracellular motility drive --------------------------------
        rho_eff = density if external_density is None else density + external_density[step]
        if params.instantaneous:
            s = np.clip(params.s0 - params.alpha * rho_eff, 0.0, 2.0 * params.s0)
        elif params.adaptive:
            ds = (params.s0 - s) / params.tau_s - params.alpha * rho_eff
            s = s + dt * ds + params.sigma_s * np.sqrt(dt) * rng.standard_normal(N)
            s = np.clip(s, 0.0, 2.0 * params.s0)
        else:
            s = np.full(N, params.s0, dtype=float)

        # --- rotational noise --------------------------------------------
        Dr_eff = params.Dr * (1.0 + params.gamma_turn * rho_eff)
        theta = theta + np.sqrt(2.0 * Dr_eff * dt) * rng.standard_normal(N)
        theta = _wrap_angle(theta)

        # --- repulsion ----------------------------------------------------
        rep = (dist < params.d0) & np.isfinite(dist)
        if np.any(rep):
            force_mag = np.where(rep, params.k_rep * (params.d0 - dist) / params.d0, 0.0)
            unit = np.divide(d, dist[..., None], out=np.zeros_like(d), where=np.isfinite(dist)[..., None])
            force = np.sum(force_mag[..., None] * unit, axis=1)
            x = x + dt * force

        # --- move ---------------------------------------------------------
        e = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        x = x + dt * s[:, None] * e
        x = x % L

    return SimResult(
        frames=frames,
        theta=theta_hist,
        speed=speed_hist,
        density=density_hist,
        dt_frame=dt * save_every,
        L=L,
        N=N,
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

# Fixed density bins (cells / area) so statistics are comparable across runs.
DENSITY_EDGES = np.array([0.0, 0.03, 0.06, 0.10, 0.16, 0.26, 0.45, np.inf])

# Pair-correlation shells (edge radii).
R_EDGES = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])

MSD_LAGS = (1, 2, 4, 8)  # in saved frames


def _unwrap(frames: np.ndarray, L: float) -> np.ndarray:
    out = np.empty_like(frames)
    out[0] = frames[0]
    for t in range(1, frames.shape[0]):
        out[t] = out[t - 1] + _wrap(frames[t] - frames[t - 1], L)
    return out


def _speed_from_positions(frames: np.ndarray, L: float, dt_frame: float, lag: int = 2) -> np.ndarray:
    """Measured speed over ``lag`` saved frames; shape (F-lag, N)."""
    d = _wrap(frames[lag:] - frames[:-lag], L)
    return np.linalg.norm(d, axis=-1) / (lag * dt_frame)


def _speed_density_profile(frames, density, L, dt_frame, lag=2):
    v = _speed_from_positions(frames, L, dt_frame, lag=lag)
    rho = density[: v.shape[0]]
    out = np.full(len(DENSITY_EDGES) - 1, np.nan)
    for b in range(out.size):
        m = (rho >= DENSITY_EDGES[b]) & (rho < DENSITY_EDGES[b + 1])
        if np.any(m):
            out[b] = float(v[m].mean())
    # Empty density bins are common at low/high N.  Fill them by linear
    # interpolation between populated bins (nearest value at the edges)
    # so the statistic vector stays finite and comparable across runs.
    good = np.isfinite(out)
    if good.any():
        out = np.interp(np.arange(out.size), np.flatnonzero(good), out[good])
    else:
        out[:] = float(v.mean())
    return out


def _order_parameter(theta, burn=0.2):
    th = theta[int(burn * theta.shape[0]):]
    return float(np.mean(np.abs(np.mean(np.exp(1j * th), axis=1))))


def _local_alignment(frames, theta, L, R, n_frames=10):
    """Mean length of the neighbour-averaged polarity vector.

    Unlike the global order parameter this is insensitive to the
    overall direction of motion, so a randomly oriented background does
    not dilute it.
    """
    F = frames.shape[0]
    step = max(1, F // n_frames)
    vals = []
    for t in range(0, F, step):
        d = _wrap(frames[t][None, :, :] - frames[t][:, None, :], L)
        dist = np.sqrt(np.sum(d * d, axis=-1))
        np.fill_diagonal(dist, np.inf)
        mask = dist < R
        n = mask.sum(axis=1)
        e = np.stack([np.cos(theta[t]), np.sin(theta[t])], axis=1)
        sum_e = mask.astype(float) @ e
        norm = np.linalg.norm(sum_e, axis=1) / np.maximum(n, 1)
        ok = n > 0
        if np.any(ok):
            vals.append(float(norm[ok].mean()))
    return float(np.mean(vals)) if vals else 0.0


def _pair_correlation(frames, L, n_frames=12):
    N = frames.shape[1]
    rho_global = N / (L * L)
    counts = np.zeros(len(R_EDGES) - 1)
    step = max(1, frames.shape[0] // n_frames)
    used = 0
    for t in range(0, frames.shape[0], step):
        d = _pairwise(frames[t], L)
        iu = np.triu_indices(N, k=1)
        r = d[iu]
        h, _ = np.histogram(r, bins=R_EDGES)
        counts += h
        used += 1
    shell = np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2)
    expected = 0.5 * N * (N - 1) * shell / (L * L)
    return counts / (expected * used)


def _msd(frames, L, lags=MSD_LAGS):
    u = _unwrap(frames, L)
    out = []
    for k in lags:
        d = u[k:] - u[:-k]
        out.append(float(np.mean(np.sum(d * d, axis=-1))))
    return np.array(out)


def _speed_autocorr(frames, L, dt_frame, lag=1):
    v = _speed_from_positions(frames, L, dt_frame, lag=1)  # (F-1, N)
    a = v[:-lag].ravel()
    b = v[lag:].ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _speed_density_xcorr(frames, density, L, dt_frame, lags=(0, 1, 2)):
    """Lagged correlation between measured speed and local density.

    A memoryless response peaks at lag 0; an adaptive response with
    relaxation time tau_s peaks at a positive lag.  This is the statistic
    that distinguishes M1 from M3.
    """
    v = _speed_from_positions(frames, L, dt_frame, lag=2)
    rho = density[: v.shape[0]]
    out = []
    for k in lags:
        if k == 0:
            a, b = v.ravel(), rho.ravel()
        else:
            a, b = v[k:].ravel(), rho[:-k].ravel()
        if a.std() < 1e-9 or b.std() < 1e-9:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.asarray(out)


def _density_contrast(density):
    m = density.mean(axis=1)
    sd = density.std(axis=1)
    return float(np.mean(sd / np.maximum(m, 1e-9)))


def summarize(res: SimResult, burn: float = 0.15) -> np.ndarray:
    """Return a fixed-length vector of population summary statistics."""
    f0 = int(burn * res.n_frames)
    frames = res.frames[f0:]
    theta = res.theta[f0:]
    density = res.density[f0:]

    prof = _speed_density_profile(frames, density, res.L, res.dt_frame)
    local_align = _local_alignment(res.frames[f0:], res.theta[f0:], res.L, Params.R)
    g = _pair_correlation(frames, res.L)
    msd = _msd(frames, res.L)
    ac = _speed_autocorr(frames, res.L, res.dt_frame)
    xcorr = _speed_density_xcorr(frames, density, res.L, res.dt_frame)
    contrast = _density_contrast(density)

    return np.concatenate([prof, [local_align], g, msd, [ac], xcorr, [contrast]])


STAT_NAMES = (
    [f"speed@rho_bin{b}" for b in range(len(DENSITY_EDGES) - 1)]
    + ["local_alignment"]
    + [f"g(r)_{R_EDGES[b]:.0f}-{R_EDGES[b+1]:.0f}" for b in range(len(R_EDGES) - 1)]
    + [f"msd_lag{k}" for k in MSD_LAGS]
    + ["speed_autocorr"]
    + [f"xcorr_speed_density_lag{k}" for k in (0, 1, 2)]
    + ["density_contrast"]
)


def single_cell_summarize(res: SimResult) -> np.ndarray:
    """Statistics available from isolated cells (no density information)."""
    frames = res.frames
    v = _speed_from_positions(frames, res.L, res.dt_frame, lag=2)
    msd = _msd(frames, res.L)
    ac = _speed_autocorr(frames, res.L, res.dt_frame)
    return np.concatenate([[float(v.mean())], msd, [ac]])


SINGLE_STAT_NAMES = ["mean_speed"] + [f"msd_lag{k}" for k in MSD_LAGS] + ["speed_autocorr"]


# ---------------------------------------------------------------------------
# Controlled crowding pulse (for experiment-design / identifiability)
# ---------------------------------------------------------------------------

def pulse_protocol(T: float, dt: float, start: float = 6.0, end: float = 12.0,
                   amplitude: float = 0.9) -> np.ndarray:
    steps = int(round(T / dt))
    t = np.arange(steps + 1) * dt
    return np.where((t >= start) & (t < end), amplitude, 0.0)


def pulse_speed_trace(res: SimResult, n_points: int = 16) -> np.ndarray:
    """Interpolated speed of a single cell at ``n_points`` times."""
    s = res.speed[:, 0]
    t_src = np.arange(len(s)) * res.dt_frame
    t_dst = np.linspace(0.0, t_src[-1], n_points)
    return np.interp(t_dst, t_src, s)
