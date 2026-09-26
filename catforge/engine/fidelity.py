"""Uncertainty-driven fidelity allocation for storm surge.

The engine prices every hurricane with the low-fidelity surge and its calibrated error model. The
full 2′ model is exact relative to that reference but costs about 6 s per event. This module spends a
budget of full-model runs where they reduce the error of a *tail* metric the most.

**Tail sensitivity.** For AEP TVaR at return period T, shifting event e's loss by ε moves the reported
metric by ε · a_e. Here a_e is the *realised* Euler gradient: the number of e's occurrences in this
simulation's k = n/T worst years, divided by k. The weight tapers linearly to 0 at rank 2k, to hedge
against years crossing the tail boundary once events are upgraded. The error being corrected is in
the reported number, and an event that never lands in a simulated tail year contributes nothing to
it. The *expected* participation λ_e·T·E_s[P(A ≥ VaR − L_e,s)] (``tail_sensitivity``) was tried first.
On the demo book it ranked no better than random: the TVaR₂₅₀ error stayed at −1.2 % after 48
upgrades. The realised gradient cut it to −0.4 % at 24 and −0.1 % at 96.

**Resolvable uncertainty.** V_e is the variance of event e's surge-attributed loss over its ELT
samples. The track is fixed per catalog event, so the surge draws — the event term, the correlated
residual field, the nugget and connectivity — are all epistemic with respect to the full model: one
full-model run removes them.

**Allocation.** With a common correlation ρ̄ between event errors, the TVaR error has variance
U² = (1 − ρ̄)·Σ_e a_e² V_e + ρ̄·(Σ_e a_e sd_e)². Events are upgraded greedily by a_e·sd_e, which is
the optimal order at both ends of ρ̄, until the remaining U ≤ tol · TVaR_T or the budget is spent. The chosen events take the full model's
water levels, with the residual and connectivity draws switched off. Exactly their occurrences are
then re-simulated: every other draw is unchanged (common random numbers).

Errors correlated across events (a regional bias shared by all storms in one bay) make U optimistic.
The realised change of the metric after upgrading is reported alongside, as a check.
"""

from __future__ import annotations

import time

import numpy as np

DEFAULTS = {"budget": 0, "tol": 0.01, "rp": 250.0, "rho": 0.04, "taper": 2.0}  # ρ̄, taper: see docs §1.5


def tail_stats(annual: np.ndarray, rp: float) -> tuple[float, float]:
    """(VaR, TVaR) of the annual loss at return period ``rp`` (order statistics)."""
    a = np.sort(annual)[::-1]
    k = max(int(np.floor(a.size / rp)), 1)
    return float(a[k - 1]), float(a[:k].mean())


def tail_sensitivity(rate: np.ndarray, loss_s: np.ndarray, annual: np.ndarray, rp: float) -> np.ndarray:
    """a_e = λ_e · T · mean_s P(A ≥ VaR − L_e,s), with P from the simulated annual distribution."""
    var, _ = tail_stats(annual, rp)
    srt = np.sort(annual)
    surv = (srt.size - np.searchsorted(srt, var - loss_s, side="left")) / srt.size
    return rate * rp * surv.mean(1)


def tail_participation(year: np.ndarray, event: np.ndarray, annual: np.ndarray, rp: float, n_events: int,
                       taper: float = 2.0) -> np.ndarray:
    """Realised Euler gradient of the simulated AEP TVaR: a_e = Σ over e's occurrences of w(rank of year) / k.

    w = 1 for the k = n/T worst years, and w tapers linearly to 0 at rank ``taper``·k. The taper hedges
    against years crossing the tail boundary once events are upgraded.
    """
    k = max(int(annual.size / rp), 1)
    rank = np.empty(annual.size, np.int64)
    rank[np.argsort(-annual, kind="stable")] = np.arange(annual.size)
    w = (rank < k).astype(float)
    if taper > 1.0:
        w = np.maximum(w, np.clip((taper * k - rank) / ((taper - 1.0) * k), 0.0, 1.0))
    return np.bincount(event, weights=w[year], minlength=n_events) / k


def plan(rate, loss_s, surge_s, annual, eligible, budget: int, tol: float, rp: float, rho: float = 0.0,
         a: np.ndarray | None = None) -> dict:
    """Rank events by a_e·sd_e and choose the smallest set meeting the tolerance, within the budget.

    The tail error has variance U² = (1 − ρ̄)·Σ a²V + ρ̄·(Σ a·sd)², for a common correlation ρ̄
    between event errors. The ranking by a·sd is optimal both at ρ̄ = 0 (the greedy order of a²V is
    the same) and at ρ̄ = 1, so only the stopping point depends on ρ̄.
    """
    _, tvar = tail_stats(annual, rp)
    a = tail_sensitivity(rate, loss_s, annual, rp) if a is None else a
    v = surge_s.var(1, ddof=1) if surge_s.shape[1] > 1 else np.zeros(surge_s.shape[0])
    lin = np.where(eligible, a * np.sqrt(v), 0.0)
    order = np.argsort(-lin)
    order = order[lin[order] > 0]
    rem_ind = float(np.sum(lin ** 2)) - np.concatenate([[0.0], np.cumsum(lin[order] ** 2)])
    rem_lin = float(np.sum(lin)) - np.concatenate([[0.0], np.cumsum(lin[order])])
    u = np.sqrt(np.maximum((1.0 - rho) * rem_ind + rho * rem_lin ** 2, 0.0))
    meets = np.nonzero(u <= tol * tvar)[0]
    k = min(int(meets[0]) if meets.size else order.size, int(budget))
    return {"rows": order[:k], "a": a, "v": v, "lin": lin, "tvar": tvar, "u_before": float(u[0]),
            "u_after": float(u[k]), "tol_met": bool(u[k] <= tol * tvar), "n_candidates": int(order.size), "rho": rho,
            "u_independent": float(np.sqrt(rem_ind[0])), "u_coherent": float(rem_lin[0])}


def upgrade(model, ctx, events: np.ndarray, progress=None) -> dict:
    """Give ``events`` (combined indices) the full 2′ model's water levels, in place.

    Residual and connectivity draws are switched off for them (``ev_hf``).
    """
    from ..hazard.surge import coastal_mask, hf_event_fields, hf_site_wse

    L = ctx.portfolio.locations
    lat, lon = L["lat"].to_numpy(float), L["lon"].to_numpy(float)
    co = coastal_mask(lat, lon)
    cat = model.catalogs["TC"]
    names = ctx.events["name"].to_numpy()
    cat_index = ctx.events["cat_index"].to_numpy()
    t0 = time.time()
    events = np.asarray(events, np.int64)
    fields = hf_event_fields(cat, cat_index[events], progress=None if progress is None else (
        lambda f, i: progress(f, f"Full-fidelity surge: {names[events[np.nonzero(cat_index[events] == i)[0][0]]]}")))
    for e in events:
        a, b = int(ctx.ev_ptr[e]), int(ctx.ev_ptr[e + 1])
        loc = ctx.pair_loc[a:b]
        w = np.where(co[loc], hf_site_wse(fields[int(cat_index[e])], lat[loc], lon[loc]), np.nan).astype(np.float32)
        ctx.pair_wse[a:b] = w
        ctx.pair_pwet[a:b] = np.isfinite(w).astype(np.float32)
        ctx.ev_hf[e] = 1
    return {"hf_seconds": round(time.time() - t0, 2)}
