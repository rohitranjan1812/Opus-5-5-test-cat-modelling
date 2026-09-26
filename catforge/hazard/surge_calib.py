"""Low-fidelity → full-model surge calibration: a two-part (hurdle) model with crossed random effects.

For event e and coastal node n (a 2′ DEM cell: the stencil centre the engine rounds a building to),
the full model's peak water surface in the node's 3×3 stencil is either *dry* (the water never reached
the stencil) or a level y. Two physically distinct processes decide the outcome:

- **connectivity** — whether the water body reaches the node at all (barriers, sheltered basins,
  channels a 4′ grid cannot see). This is a logistic model:
  ``P(wet) = σ(θ · [1, x, z, shore, x·z, x − z, (x − z)₊, Δ, Δ·shore])``.
- **level given wet** — a linear mixed model with crossed random effects:
  ``y = s(x) + γ · [shore, z, x·shore, x·z, Δ, Δ·shore, min(Δ, 3)·x] + u_e + δ_n + ε``,
  where s is a linear spline, u_e ~ N(0, σ_e²) an event term, δ_n ~ N(0, τ²) a persistent **site
  response** (the harbour/bay term, the surge analogue of a Vs30 site term), and ε ~ N(0, σ²).

Here x is the low-fidelity site level, z the node's 2′ ground (clipped to 0…10 m), and shore = 1 when
the stencil touches the sea. Δ = x(α = 0) − x is the friction loss the site rule applied. A bay
carries distant water over water, where it amplifies instead of attenuating, and Δ lets the model
learn that. It cuts held-out depth RMSE in water over 1 m deep by 8 %. A per-node random *slope* on x
was tried instead: with two or three big storms per bay it overfits, and held-out error rises. The expected flood depth over building ground g is then:

    E[max(ζ − g, 0)] = P(wet) · E_N[max(y − g, 0)]

**Why not a single censored (Tobit) model?** A Tobit assumes one latent Gaussian decides both
connectivity and level. The calibration data reject that assumption. Where x ≥ 2 m, the held-out
residuals have a positive median, skew −0.8 to −2.1 and excess kurtosis 3–7, because connected nodes
sit above the fit and sheltered ones far below it. On the design set (224 storms, 341k node-events
with low-fidelity water), a pooled Gaussian understates held-out flood depth by 23–37 % where
x ≥ 2 m, and by 35–92 % on ground above 2 m. The hurdle model stays within 0.93–1.01 in every bin
with x ≥ 1 m.

A naive OLS on wet-in-both pairs is not a fix either: it is a truncated sample, which biases the
intercept up and flattens the slope. ``calibrate`` reports all three models side by side.

**Event error and spatially correlated residual.** The event term has a random slope on x, because
an event's error grows with its surge. Within an event the wet-level residual is far from
independent across nodes: its correlation is about 0.8 at 3 km, 0.44 at 9 km and 0.2 at 17 km,
because whole bays are off together. ``residual_correlation`` fits σ²[s₁·exp(−d²/4h₁²) + s₂·exp(−d²/4h₂²)], plus a nugget,
to the binned within-event covariance. The kernel samples it as a two-lattice process convolution, so
the event-level loss uncertainty — the input to fidelity allocation — is not averaged away.

The mixed model is fitted by EM. With censoring (the Tobit comparison), the E-step uses truncated-normal
moments. The M-step updates, in turn, the fixed effects by least squares, the random effects as BLUPs
(δ_n = Σ r / (m_n + σ²/τ²) shrinks nodes seen in few events toward 0), and the variance components
from completed-data second moments (a mean-field treatment of the crossed effects).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, log_ndtr, ndtr

from .surge import KNOTS, SurgeCells, level_design, site_covariates, site_wse, wet_features

GROUND_MIN = 1.0  # the engine's default building ground (2′ DEM floored at 1 m)


# ------------------------------------------------------------------------------------------ estimators
def logit_fit(F: np.ndarray, w: np.ndarray, l2: float = 1e-3, n_iter: int = 60) -> np.ndarray:
    """Ridge-stabilised logistic regression by IRLS (Newton)."""
    b = np.zeros(F.shape[1])
    for _ in range(n_iter):
        p = expit(F @ b)
        H = (F * (p * (1 - p))[:, None]).T @ F + l2 * np.eye(F.shape[1])
        step = np.linalg.solve(H, F.T @ (w - p) - l2 * b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def _trunc_moments(mu, sig, c):
    """Mean and variance of N(mu, sig²) truncated to (−∞, c]."""
    al = (c - mu) / sig
    lam = np.exp(-0.5 * al * al - 0.5 * math.log(2 * math.pi) - log_ndtr(al))
    return mu - sig * lam, sig * sig * np.maximum(1.0 - al * lam - lam * lam, 1e-9)


def fit_mixed(A, y, ev, node, c=None, site_effects: bool = True, x_event=None, x0: float = 2.0,
              n_iter: int = 300, tol: float = 1e-7) -> dict:
    """Crossed random-effects model y = Aβ + u_e0 [+ u_e1·(x − x0)] + δ_n + ε, fitted by EM.

    With ``x_event`` the event term has a random **slope** on the low-fidelity level:
    (u_e0, u_e1) ~ N(0, Ψ). This is an event-wide error that grows with the surge. It is well
    identified, because each event carries hundreds of nodes; a per-*node* slope, which sees two or
    three big storms per bay, is not. The event BLUPs are 2×2 solves, and Ψ is updated from
    u uᵀ + V_u.

    If ``c`` is given, NaN entries of ``y`` are censored at ``c`` (y ≤ c), the Tobit case.
    ``ev`` and ``node`` are dense 0-based group ids.
    """
    obs = np.isfinite(y)
    y = np.where(obs, y, 0.0)
    censored = c is not None and not obs.all()
    slope = x_event is not None
    t = (np.asarray(x_event, float) - x0) if slope else np.zeros(y.size)
    ne, nn = int(ev.max()) + 1, int(node.max()) + 1
    S0 = np.bincount(ev, minlength=ne).astype(float)
    S1 = np.bincount(ev, t, minlength=ne)
    S2 = np.bincount(ev, t * t, minlength=ne)
    m_n = np.bincount(node, minlength=nn).astype(float)
    beta = np.linalg.lstsq(A[obs], y[obs], rcond=None)[0]
    s2 = float(np.var(y[obs] - A[obs] @ beta))
    Psi = np.diag([0.1 * s2, 0.01 * s2 if slope else 0.0])
    t2 = 0.1 * s2 if site_effects else 0.0
    u = np.zeros((ne, 2))
    Vu = np.zeros((ne, 2, 2))
    d, pv_n = np.zeros(nn), np.zeros(nn)
    ll_old, iters = -np.inf, 0
    for iters in range(1, n_iter + 1):  # noqa: B007  (reported after the loop)
        sig = math.sqrt(s2)
        ue = u[ev, 0] + u[ev, 1] * t
        if censored:
            ey, vy = _trunc_moments(A @ beta + ue + d[node], sig, c)
            yt, wv = np.where(obs, y, ey), np.where(obs, 0.0, vy)
        else:
            yt, wv = y, 0.0
        beta = np.linalg.lstsq(A, yt - ue - d[node], rcond=None)[0]
        f = A @ beta
        r = yt - f - d[node]
        R0, R1 = np.bincount(ev, r, minlength=ne), np.bincount(ev, t * r, minlength=ne)
        if slope:
            Pi = np.linalg.inv(Psi + 1e-12 * np.eye(2))
            P = np.empty((ne, 2, 2))
            P[:, 0, 0], P[:, 0, 1], P[:, 1, 1] = S0 / s2 + Pi[0, 0], S1 / s2 + Pi[0, 1], S2 / s2 + Pi[1, 1]
            P[:, 1, 0] = P[:, 0, 1]
            Vu = np.linalg.inv(P)
            u = np.einsum("nij,nj->ni", Vu, np.stack([R0, R1], 1) / s2)
            Psi = (np.einsum("ni,nj->ij", u, u) + Vu.sum(0)) / ne
        else:
            k_e = s2 / max(Psi[0, 0], 1e-12)
            u[:, 0] = R0 / (S0 + k_e)
            Vu[:, 0, 0] = s2 / (S0 + k_e)
            Psi[0, 0] = float(np.mean(u[:, 0] ** 2 + Vu[:, 0, 0]))
        ue = u[ev, 0] + u[ev, 1] * t
        pv_e = Vu[ev, 0, 0] + 2 * t * Vu[ev, 0, 1] + t * t * Vu[ev, 1, 1]
        if site_effects:
            k_n = s2 / max(t2, 1e-12)
            d = np.bincount(node, yt - f - ue, minlength=nn) / (m_n + k_n)
            pv_n = s2 / (m_n + k_n)
        rr = yt - (f + ue + d[node])
        s2 = float(np.mean(rr * rr + wv + pv_e + pv_n[node]))
        if site_effects:
            t2 = float(np.mean(d * d + pv_n))
        # observed-data log-likelihood with the random effects at their BLUPs (convergence monitor)
        sig = math.sqrt(s2)
        mu = f + ue + d[node]
        ll = float(np.sum(np.where(obs, -0.5 * ((y - mu) / sig) ** 2 - math.log(sig),
                                   log_ndtr((c - mu) / sig) if censored else 0.0)))
        if abs(ll - ll_old) < tol * abs(ll):
            break
        ll_old = ll
    return {"beta": beta, "sigma": math.sqrt(s2), "sigma_event": math.sqrt(Psi[0, 0]), "Psi_event": Psi, "x0": x0,
            "tau": math.sqrt(t2), "u": u[:, 0], "u_event": u, "delta": d, "pv_delta": pv_n, "m_node": m_n,
            "iterations": iters}


def event_var(m: dict, x) -> np.ndarray:
    """Variance of the event error u_e0 + u_e1·(x − x0) at low-fidelity level x."""
    P = np.asarray(m.get("Psi_event", [[m["sigma_event"] ** 2, 0.0], [0.0, 0.0]]))
    t = np.asarray(x, float) - float(m.get("x0", 2.0))
    return P[0, 0] + 2 * t * P[0, 1] + t * t * P[1, 1]


def naive_fit(x, y) -> dict:
    """The truncated estimator: OLS on pairs where both models are wet (kept for comparison)."""
    both = np.isfinite(x) & np.isfinite(y)
    a, b = np.linalg.lstsq(np.stack([np.ones(both.sum()), x[both]], 1), y[both], rcond=None)[0]
    return {"a": float(a), "b": float(b), "sigma": float(np.std(y[both] - a - b * x[both]))}


def expected_depth(mu, s, g):
    """E[max(ζ − g, 0)] for ζ ~ N(mu, s²)."""
    k = (mu - g) / s
    return s * (np.exp(-0.5 * k * k) / math.sqrt(2 * math.pi) + k * ndtr(k))


def _dense(v):
    return np.unique(v, return_inverse=True)


def _lookup(keys_fit, values, keys, default):
    pos = np.minimum(np.searchsorted(keys_fit, keys), keys_fit.size - 1)
    return np.where(keys_fit[pos] == keys, values[pos], default)


# ------------------------------------------------------------------------------------------ models
def fit_hurdle(x, dlt, y, zc, shore, ev, keys, knots=KNOTS) -> dict:
    """The engine's model: logistic connectivity × wet-level crossed mixed model."""
    wet = np.isfinite(y)
    theta = logit_fit(wet_features(x, dlt, zc, shore), wet.astype(float))
    nk, ninv = _dense(keys[wet])
    _, einv = _dense(ev[wet])
    knots = tuple(k for k in knots if np.sum(x[wet] > k) >= 50)  # a knot needs data beyond it
    p = fit_mixed(level_design(x[wet], dlt[wet], zc[wet], shore[wet], knots), y[wet], einv, ninv, x_event=x[wet])
    return {"theta": theta, "beta": p["beta"], "knots": knots, "sigma": p["sigma"], "sigma_event": p["sigma_event"],
            "Psi_event": p["Psi_event"], "x0": p["x0"], "u_event": p["u_event"], "event_ids": np.unique(ev[wet]),
            "tau": p["tau"], "site_key": nk, "delta": p["delta"], "pv_delta": p["pv_delta"], "m_node": p["m_node"],
            "iterations": p["iterations"]}


def predict_hurdle(m: dict, x, dlt, zc, shore, keys, g) -> np.ndarray:
    pw = expit(wet_features(x, dlt, zc, shore) @ m["theta"])
    dl = _lookup(m["site_key"], m["delta"], keys, 0.0)
    pv = _lookup(m["site_key"], m["pv_delta"], keys, m["tau"] ** 2)
    mu = level_design(x, dlt, zc, shore, m["knots"]) @ m["beta"] + dl
    return pw * expected_depth(mu, np.sqrt(m["sigma"] ** 2 + event_var(m, x) + pv), g)


def _fit_predict(kind, tr, te, x, dlt, y, c, g, zc, shore, ev, keys):
    if kind == "hurdle":
        return predict_hurdle(fit_hurdle(x[tr], dlt[tr], y[tr], zc[tr], shore[tr], ev[tr], keys[tr]),
                              x[te], dlt[te], zc[te], shore[te], keys[te], g[te])
    if kind == "naive":
        p = naive_fit(x[tr], y[tr])
        return expected_depth(p["a"] + p["b"] * x[te], p["sigma"], g[te])
    # Tobit: censored linear model with the site term
    nk, ninv = _dense(keys[tr])
    _, einv = _dense(ev[tr])
    A = lambda i: np.stack([np.ones(i.sum()), x[i]], 1)  # noqa: E731
    p = fit_mixed(A(tr), y[tr], einv, ninv, c=c[tr])
    mu = A(te) @ p["beta"] + _lookup(nk, p["delta"], keys[te], 0.0)
    pv = _lookup(nk, p["pv_delta"], keys[te], p["tau"] ** 2)
    return expected_depth(mu, np.sqrt(p["sigma"] ** 2 + p["sigma_event"] ** 2 + pv), g[te])


def cross_validate(kind, x, dlt, y, c, g, zc, shore, ev, keys, folds: int = 5, seed: int = 0) -> dict:
    """Event-grouped K-fold: fit on the other events, predict E[depth] on the held-out ones."""
    ue = np.unique(ev)
    f_of = dict(zip(ue, np.random.default_rng(seed).permutation(np.arange(ue.size) % folds)))
    fold = np.array([f_of[e] for e in ev])
    pred = np.zeros(x.size)
    for f in range(folds):
        pred[fold == f] = _fit_predict(kind, fold != f, fold == f, x, dlt, y, c, g, zc, shore, ev, keys)
    true = np.where(np.isfinite(y), np.maximum(y - g, 0.0), 0.0)
    m = (true > 0.05) | (pred > 0.05)
    deep = true > 1.0
    return {"depth_rmse_m": math.sqrt(float(np.mean((pred[m] - true[m]) ** 2))) if m.any() else 0.0,
            "deep_depth_rmse_m": math.sqrt(float(np.mean((pred[deep] - true[deep]) ** 2))) if deep.any() else 0.0,
            "depth_bias": float(pred.sum() / max(true.sum(), 1e-9) - 1.0), "n": int(m.sum()), "pred": pred, "true": true}


X_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, np.inf)
Z_BINS = (0.0, 1.0, 2.0, 3.0, 5.0, np.inf)


def conditional_table(v, pred, true, bins=X_BINS, sel=None) -> list[dict]:
    """Calibration conditional on a predictor: mean full-model vs predicted depth by bin of ``v``."""
    rows = []
    sel = np.ones(v.size, bool) if sel is None else sel
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = sel & (v >= lo) & (v < hi)
        if m.any():
            rows.append({"lo": lo, "hi": None if not np.isfinite(hi) else hi, "n": int(m.sum()),
                         "true_depth_m": round(float(true[m].mean()), 4), "pred_depth_m": round(float(pred[m].mean()), 4),
                         "ratio": round(float(pred[m].sum() / max(true[m].sum(), 1e-9)), 4)})
    return rows


# ------------------------------------------------------------------------------------------ spatial correlation
CORR_BINS = np.array([0.0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 60, 80, 120])


def gauss2(d, s1, h1, s2, h2):
    """Two-scale Gaussian correlation s₁·exp(−d²/4h₁²) + s₂·exp(−d²/4h₂²), the covariance form a
    process convolution of white noise with Gaussian kernels of width h₁, h₂ produces."""
    d = np.asarray(d, float)
    return s1 * np.exp(-d * d / (4 * h1 * h1)) + s2 * np.exp(-d * d / (4 * h2 * h2))


def residual_correlation(m: dict, X, D, Y, ZC, SH, E, K, LA, LO, max_nodes: int = 400, seed: int = 0) -> dict:
    """Within-event spatial correlation of the wet-level residual (event term removed), binned by node
    distance and fitted as a two-scale Gaussian by weighted least squares. The rest is nugget."""
    from scipy.optimize import least_squares

    wet = np.isfinite(Y)
    fixed = level_design(X, D, ZC, SH, m["knots"]) @ m["beta"] + _lookup(m["site_key"], m["delta"], K, 0.0)
    ue, einv = np.unique(E, return_inverse=True)
    uev = np.zeros((ue.size, 2))  # the event term (intercept + slope) as fitted
    uev[np.searchsorted(ue, m["event_ids"])] = m["u_event"]
    rr = np.where(wet, Y - fixed - uev[einv, 0] - uev[einv, 1] * (X - m["x0"]), 0.0)
    var = float(np.mean(rr[wet] ** 2))
    nb_ = CORR_BINS.size - 1
    sxy, cnt = np.zeros(nb_), np.zeros(nb_)
    rng = np.random.default_rng(seed)
    for g in range(ue.size):
        i = np.nonzero((einv == g) & wet)[0]
        if i.size < 10:
            continue
        if i.size > max_nodes:
            i = rng.choice(i, max_nodes, replace=False)
        la, lo, v = LA[i], LO[i], rr[i]
        dd = np.hypot((lo[:, None] - lo[None, :]) * 111.32 * math.cos(math.radians(float(la.mean()))),
                      (la[:, None] - la[None, :]) * 110.574)
        iu = np.triu_indices(i.size, 1)
        b = np.digitize(dd[iu], CORR_BINS) - 1
        ok = (b >= 0) & (b < nb_)
        sxy += np.bincount(b[ok], (v[:, None] * v[None, :])[iu][ok], minlength=nb_)
        cnt += np.bincount(b[ok], minlength=nb_)
    corr = sxy / np.maximum(cnt, 1) / var
    mid = 0.5 * (CORR_BINS[1:] + CORR_BINS[:-1])
    use = cnt > 0
    wts = np.sqrt(cnt[use]) / np.sqrt(cnt[use]).max()
    fit_ = least_squares(lambda q: (gauss2(mid[use], *q) - corr[use]) * wts, [0.5, 4.0, 0.3, 14.0],
                         bounds=([0, 0.5, 0, 2.0], [1, 30, 1, 80]))
    s1, h1, s2, h2 = (float(v) for v in fit_.x)
    if h1 > h2:
        s1, h1, s2, h2 = s2, h2, s1, h1
    tot = s1 + s2
    if tot > 0.99:  # keep a nugget: residuals at one node are not perfectly predictable from neighbours
        s1, s2 = s1 * 0.99 / tot, s2 * 0.99 / tot
    return {"scales": [{"frac": round(s1, 4), "h_km": round(h1, 3)}, {"frac": round(s2, 4), "h_km": round(h2, 3)}],
            "nugget_frac": round(1.0 - s1 - s2, 4),
            "empirical": [{"lo_km": float(a), "hi_km": float(b), "corr": round(float(c), 4), "pairs": int(n)}
                          for a, b, c, n in zip(CORR_BINS[:-1], CORR_BINS[1:], corr, cnt) if n > 0]}


# ------------------------------------------------------------------------------------------ driver
def _lf_levels(d, alpha, rmax, r0):
    cells = SurgeCells(ptr=d["cptr"], lat=d["cell_lat"], lon=d["cell_lon"], eta=d["cell_eta"], lf={})
    n = d["lat"].size
    return site_wse(cells, d["nptr"], np.arange(n), d["lat"], d["lon"], np.ones(n, bool), alpha, r0, rmax).astype(float)


def calibrate(d: dict, alphas, rmaxs, r0: float = 3.5, progress=print, tie: float = 0.01) -> dict:
    """Grid-search (α, R) by cross-validated depth error, then fit the engine's model on all events.

    The collected data (``scripts/calibrate_surge.py collect``) holds, per node-event: the node's
    position, its ground z, the full model's stencil water level ``hf`` (NaN = dry), the censoring
    level ``cz`` (lowest stencil ground + H_dry), and the low-fidelity cells per event.

    Among grid points within ``tie`` (relative) of the best RMSE, the largest R wins. Nodes that no
    low-fidelity cell reaches are one-sided misses, and the CV surface is flat there.
    """
    n_ev = d["nptr"].size - 1
    ev = np.repeat(np.arange(n_ev), np.diff(d["nptr"]))
    y = d["hf"].astype(float)
    c = d["cz"].astype(float)
    z = d["z"].astype(float)
    g = np.maximum(z, GROUND_MIN)
    keys, zc, shore = site_covariates(d["lat"], d["lon"])  # exactly the features the engine computes
    grid, xs, free = [], {}, {}
    for alpha in alphas:
        for rmax in rmaxs:
            x = _lf_levels(d, alpha, rmax, r0)
            if rmax not in free:
                free[rmax] = _lf_levels(d, 0.0, rmax, r0)
            f = np.isfinite(x)
            dlt = np.where(f, free[rmax] - x, 0.0)
            cv = cross_validate("hurdle", x[f], dlt[f], y[f], c[f], g[f], zc[f], shore[f], ev[f], keys[f])
            # misses: the full model floods but no low-fidelity cell reaches the node
            miss = ~f & np.isfinite(y) & (y - g > 0.05)
            dmiss = y[miss] - g[miss]
            rmse = math.sqrt((cv["depth_rmse_m"] ** 2 * cv["n"] + float(np.sum(dmiss ** 2))) / max(cv["n"] + miss.sum(), 1))
            bias = float(cv["pred"].sum()) / max(float(cv["true"].sum() + dmiss.sum()), 1e-9) - 1.0
            grid.append({"alpha": alpha, "rmax": rmax, "cv_depth_rmse_m": round(rmse, 4), "cv_depth_bias": round(bias, 4),
                         "misses": int(miss.sum())})
            xs[(alpha, rmax)] = x
            progress(f"α={alpha} R={rmax}: CV depth RMSE {rmse:.3f} m, bias {bias:+.3f}, misses {miss.sum()}")
    best = min(gr["cv_depth_rmse_m"] for gr in grid)
    pick = max((gr for gr in grid if gr["cv_depth_rmse_m"] <= best * (1 + tie)),
               key=lambda gr: (gr["rmax"], -gr["cv_depth_rmse_m"]))
    alpha, rmax = pick["alpha"], pick["rmax"]
    x = xs[(alpha, rmax)]
    f = np.isfinite(x)
    dlt = np.where(f, free[rmax] - x, 0.0)
    X, D, Y, Cc, G, ZC, SH, E, K, Zr = x[f], dlt[f], y[f], c[f], g[f], zc[f], shore[f], ev[f], keys[f], z[f]
    m = fit_hurdle(X, D, Y, ZC, SH, E, K)
    field = residual_correlation(m, X, D, Y, ZC, SH, E, K, d["lat"][f].astype(float), d["lon"][f].astype(float))
    progress(f"residual field: {field['scales']}, nugget {field['nugget_frac']}")
    variants = {k: cross_validate(k, X, D, Y, Cc, G, ZC, SH, E, K) for k in ("hurdle", "tobit", "naive")}
    for name, cv_ in variants.items():
        progress(f"{name}: CV depth RMSE {cv_['depth_rmse_m']:.3f} m, bias {cv_['depth_bias']:+.3f}")
    sel = variants["hurdle"]
    wet_t, wet_p = sel["true"] > 0.3, sel["pred"] > 0.3
    r = lambda v: round(float(v), 4)  # noqa: E731
    cal = {
        "model": "hurdle", "alpha_m_per_km": alpha, "r0_km": r0, "rmax_km": rmax,
        "wet_logit": [r(v) for v in m["theta"]], "level_beta": [r(v) for v in m["beta"]], "knots": list(m["knots"]),
        "sigma_event_m": r(m["sigma_event"]), "sigma_site_m": r(m["sigma"]), "tau_site_response_m": r(m["tau"]),
        # event-wide error u0 + u1·(x − x0), (u0, u1) ~ N(0, cov): grows with the surge level
        "event_error": {"x0": m["x0"], "cov": [[r(v) for v in row] for row in m["Psi_event"]]},
        "field": field,  # within-event spatial correlation of the site residual (kernel: process convolution)
        "cv_depth_rmse_m": pick["cv_depth_rmse_m"], "cv_depth_bias": pick["cv_depth_bias"],  # held-out, misses included
        "cv_hit_rate": r((wet_t & wet_p).sum() / max(wet_t.sum(), 1)),
        "cv_false_alarm_ratio": r((wet_p & ~wet_t).sum() / max(wet_p.sum(), 1)),
        "cv": {name: {"depth_rmse_m": r(cv_["depth_rmse_m"]), "deep_depth_rmse_m": r(cv_["deep_depth_rmse_m"]),
                      "depth_bias": r(cv_["depth_bias"]), "n": cv_["n"],
                      "by_level": conditional_table(X, cv_["pred"], cv_["true"]),
                      "by_ground_where_level_ge_2m": conditional_table(Zr, cv_["pred"], cv_["true"], Z_BINS, sel=X >= 2)}
               for name, cv_ in variants.items()},
        "grid": grid,
        "n_events": int(n_ev), "n_node_events": int(y.size), "n_used": int(f.sum()),
        "n_dry": int((~np.isfinite(Y)).sum()), "n_nodes": int(m["site_key"].size), "em_iterations": m["iterations"],
    }
    site = {"key": m["site_key"].astype(np.int64), "delta": m["delta"].astype(np.float32),
            "pv": m["pv_delta"].astype(np.float32), "m": m["m_node"].astype(np.int16)}
    return {"calibration": cal, "site_response": site}
