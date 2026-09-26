"""Tail validation of the multi-fidelity surge and of uncertainty-driven fidelity allocation.

    python scripts/validate_surge_tail.py [--locations 5000] [--years 10000] [--min-rp 50] [--budgets 12 24 48]

1. **Baseline.** The demo book's hurricane analysis, with the low-fidelity surge and its calibrated error
   model and no allocation.
2. **Allocation.** The same analysis with ``surge_fidelity = {budget: K, tol: 0}`` for each K. The
   engine chooses the events by a_e·sd_e, without seeing any full-model result.
3. **Reference.** Every event that matters takes the full 2′ model's water levels, through the
   engine's own path (``fidelity.upgrade``), and exactly its occurrences are re-simulated. That is
   the baseline tail — the largest occurrence of every year at or beyond ``--min-rp``, plus
   occurrences in those years with over 10 % surge share — united with every event of the 3k worst
   years (k = n/250) and every event any budget picked. The occurrence keys are unchanged, so wind uses common random numbers.
4. **Honesty of the surrogate's uncertainty.** For each tail event, the full model's expected surge
   loss is compared with the surrogate's distribution under identical wind draws (64 ELT samples).
   z = (full − mean) / epistemic sd should be about N(0, 1).
5. **Error vs budget,** and the ρ̄ implied by the realised baseline error: it sits between the
   independent (ρ̄ = 0) and coherent (ρ̄ = 1) bounds.

Writes ``docs/validation/surge_tail.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from catforge.analytics.ep import quantile_at_rp
from catforge.config import PERIL_INDEX
from catforge.engine import fidelity as F
from catforge.engine.model import AnalysisConfig, CatModel, _elt_occurrences
from catforge.exposure.synthetic import generate_portfolio
from catforge.hazard.surge import cache_dir

OUT = Path(__file__).resolve().parent.parent / "docs" / "validation" / "surge_tail.json"
RPS = (100, 250, 500)


def annual(ylt, v):
    return np.bincount(ylt.year, weights=v, minlength=ylt.n_years)


def aep(ylt, v):
    a = annual(ylt, v)
    s = np.sort(a)
    return {**{f"aep{rp}": quantile_at_rp(s, rp) for rp in RPS}, **{f"tvar{rp}": F.tail_stats(a, rp)[1] for rp in RPS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", type=int, default=5000)
    ap.add_argument("--years", type=int, default=10000)
    ap.add_argument("--min-rp", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--budgets", type=int, nargs="+", default=[6, 12, 24, 48, 96])
    args = ap.parse_args()
    model = CatModel.default()
    pf = generate_portfolio(args.locations, seed=11)
    cfg = AnalysisConfig(perils=["TC"], n_years=args.years, seed=args.seed)
    t0 = time.time()
    res = model.run(pf, cfg)
    t_run = time.time() - t0
    ctx, ylt = res.ctx, res.ylt
    assert np.all(ylt.peril == PERIL_INDEX["TC"])
    occ_s = ctx.kernel(ylt.event, ylt.key, split=True)[8]
    ann = annual(ylt, res.occ_gu)
    tail_years = np.argsort(-ann)[: int(ylt.n_years / args.min_rp)]
    in_tail = np.isin(ylt.year, tail_years)
    big = np.zeros(ylt.event.size, bool)
    for y in tail_years:
        k = np.nonzero(ylt.year == y)[0]
        big[k[np.argmax(res.occ_gu[k])]] = True
    tail_ev = np.unique(ylt.event[big | (in_tail & (occ_s / np.maximum(ann[ylt.year], 1.0) > 0.10))])
    print(f"baseline {t_run:.0f}s; {tail_years.size} tail years → {tail_ev.size} tail events", flush=True)

    # ---- allocation runs: the engine picks events by a_e·sd_e, without seeing any full-model result --
    runs, picked = [], set()
    for K in args.budgets:
        t2 = time.time()
        r = model.run(pf, AnalysisConfig(perils=["TC"], n_years=args.years, seed=args.seed,
                                         surge_fidelity={"budget": K, "tol": 0.0}))
        fid = r.extras["surge"]["fidelity"]
        up = [u["event"] for u in fid["upgraded"]]
        picked.update(up)
        runs.append({"budget": K, "seconds": round(time.time() - t2, 1), "hf_seconds": fid.get("hf_seconds"),
                     "u_before_rel": fid["u_before_rel"], "u_after_rel": fid["u_after_rel"],
                     "u_independent_rel": fid["u_independent"] / fid["tvar_before"],
                     "u_coherent_rel": fid["u_coherent"] / fid["tvar_before"],
                     "upgraded_in_baseline_tail": int(np.isin(up, tail_ev).sum()), "occ_gu": r.occ_gu})
        print(f"K={K}: {runs[-1]['seconds']}s, {runs[-1]['upgraded_in_baseline_tail']}/{K} in the baseline tail", flush=True)

    # ---- reference: the full model on every event that matters (baseline tail ∪ all picks) ---------
    rank = np.empty(ylt.n_years, int)
    rank[np.argsort(-ann)] = np.arange(ylt.n_years)
    near_tail = np.unique(ylt.event[rank[ylt.year] < 3 * int(ylt.n_years / 250)])  # every event of the 3k worst years
    ref_ev = np.union1d(np.union1d(tail_ev, near_tail), np.array(sorted(picked), int))
    ref = dataclasses.replace(ctx, pair_wse=ctx.pair_wse.copy(), pair_pwet=ctx.pair_pwet.copy(), ev_hf=ctx.ev_hf.copy())
    t1 = time.time()
    F.upgrade(model, ref, ref_ev)
    t_hf = time.time() - t1
    idx = np.isin(ylt.event, ref_ev)
    out = ref.kernel(ylt.event[idx], ylt.key[idx], split=True)
    gu_ref = res.occ_gu.copy()
    gu_ref[idx] = out[0]
    s_ref = occ_s.copy()
    s_ref[idx] = out[8]
    print(f"reference: {ref_ev.size} full-model events (tail ∪ 3k worst years ∪ picks) in {t_hf:.0f}s", flush=True)

    # ---- honesty of the surrogate's uncertainty (identical wind draws) ------------------------------
    S = 64
    oe, ok, _ = _elt_occurrences(ctx, tail_ev, S)
    sur = ctx.kernel(oe, ok, split=True)[8].reshape(tail_ev.size, S)
    full = ref.kernel(oe, ok, split=True)[8].reshape(tail_ev.size, S)
    ms, mf, vs, vf = sur.mean(1), full.mean(1), sur.var(1, ddof=1), full.var(1, ddof=1)
    epi = np.maximum(vs - vf, 0.0)
    z = (mf - ms) / np.sqrt(epi + (vs + vf) / S + 1.0)
    sig = mf > 5e6
    honesty = {"n_events": int(sig.sum()), "z_mean": float(z[sig].mean()), "z_sd": float(z[sig].std()),
               "share_abs_z_gt_2": float(np.mean(np.abs(z[sig]) > 2)),
               "surge_ratio_surrogate_over_full": float(ms.sum() / mf.sum()),
               "median_rel_epistemic_sd": float(np.median(np.sqrt(epi[sig]) / np.maximum(ms[sig], 1.0)))}
    print("honesty", honesty, flush=True)

    A_base, A_ref = aep(ylt, res.occ_gu), aep(ylt, gu_ref)
    for run in runs:
        A = aep(ylt, run.pop("occ_gu"))
        run["aep"], run["vs_reference"] = A, {k: A[k] / A_ref[k] - 1.0 for k in A}
        print(f"K={run['budget']}: " + ", ".join(f"{k} {A[k] / A_ref[k] - 1:+.2%}" for k in A), flush=True)
    base_err = A_base["tvar250"] / A_ref["tvar250"] - 1.0
    ui, uc = runs[0]["u_independent_rel"], runs[0]["u_coherent_rel"]
    rho_hat = float(np.clip((base_err ** 2 - ui ** 2) / max(uc ** 2 - ui ** 2, 1e-12), 0.0, 1.0))
    print(f"baseline TVaR250 error {base_err:+.2%}; U(ρ=0) {ui:.2%}, U(ρ=1) {uc:.2%} → ρ̂ {rho_hat:.3f}", flush=True)
    summary = {
        "n_years": ylt.n_years, "n_locations": pf.n, "tail_years": int(tail_years.size), "tail_events": int(tail_ev.size),
        "baseline": {"aep": A_base, "vs_reference": {k: A_base[k] / A_ref[k] - 1.0 for k in A_base}},
        "reference": {"aep": A_ref},
        "wind_only": {"aep": aep(ylt, res.occ_gu - occ_s)},
        "tail_event_surge_ratio_engine_over_full": float(occ_s[idx].sum() / max(s_ref[idx].sum(), 1.0)),
        "honesty": honesty, "allocation": runs, "rho_hat": rho_hat, "reference_events": int(ref_ev.size),
        "seconds": {"baseline_analysis": round(t_run, 1), "full_model_tail_events": round(t_hf, 1)},
        "engine_surge_summary": {k: v for k, v in res.extras.get("surge", {}).items() if k not in ("model", "fidelity")},
        "per_event": [{"event": int(e), "name": str(ctx.events["name"].iloc[e]), "surge_gu_surrogate": float(a),
                       "surge_gu_full": float(b), "surrogate_sd": float(np.sqrt(v)), "z": float(zz)}
                      for e, a, b, v, zz in zip(tail_ev, ms, mf, epi, z)],
    }
    if (cache_dir() / "calibration_pairs.npz").exists():
        with np.load(cache_dir() / "calibration_pairs.npz") as zf:
            design = {int(n.split(":")[1]) for n in zf["names"].astype(str) if n.startswith("cat:")}
        ci = ctx.events["cat_index"].to_numpy()[tail_ev]
        summary["tail_events_in_design"] = int(np.isin(ci, list(design)).sum())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_event"}, indent=1))


if __name__ == "__main__":
    main()
