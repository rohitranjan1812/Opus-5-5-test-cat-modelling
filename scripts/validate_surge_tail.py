"""Tail validation of the multi-fidelity surge: does the 1-in-250 hold up against the full model?

    python scripts/validate_surge_tail.py [--locations 5000] [--years 10000] [--min-rp 50]

1. Runs the demo book's hurricane analysis (low-fidelity surge + calibrated correction).
2. Picks the tail events: the largest occurrence of every year at or beyond ``--min-rp``, plus any
   occurrence in those years whose surge share of that year's loss is over 10 %.
3. Runs the full 2′ model for each tail event at the book's coastal sites (the peak water surface in
   each building's 3×3 stencil).
4. Re-simulates exactly those occurrences with the full model's water levels. The residual terms and
   the connectivity draw are switched off, because the full model is the reference. The occurrence keys are unchanged, so wind
   damage uses common random numbers and only the surge differs.
5. Compares the hurricane AEP (ground-up) at 100/250/500 years, and each tail event's mean surge loss.

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
from catforge.engine.model import AnalysisConfig, CatModel
from catforge.exposure.synthetic import generate_portfolio
from catforge.hazard.surge import _tracks, coastal_mask, lf_event_cells
from catforge.physics.surge2d import run_surge

OUT = Path(__file__).resolve().parent.parent / "docs" / "validation" / "surge_tail.json"


def annual(ylt, v, n):
    return np.bincount(ylt.year, weights=v, minlength=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", type=int, default=5000)
    ap.add_argument("--years", type=int, default=10000)
    ap.add_argument("--min-rp", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    model = CatModel.default()
    pf = generate_portfolio(args.locations, seed=11)
    cfg = AnalysisConfig(perils=["TC"], n_years=args.years, seed=args.seed)
    t0 = time.time()
    res = model.run(pf, cfg)
    t_run = time.time() - t0
    ctx, ylt = res.ctx, res.ylt
    n = ylt.n_years
    assert np.all(ylt.peril == PERIL_INDEX["TC"])
    occ_s = ctx.kernel(ylt.event, ylt.key, split=True)[8]
    ann = annual(ylt, res.occ_gu, n)
    order = np.argsort(-ann)
    tail_years = order[: int(n / args.min_rp)]
    in_tail = np.isin(ylt.year, tail_years)
    big = np.zeros(ylt.event.size, bool)
    for y in tail_years:  # the year's largest occurrence
        k = np.nonzero(ylt.year == y)[0]
        big[k[np.argmax(res.occ_gu[k])]] = True
    share = occ_s / np.maximum(ann[ylt.year], 1.0)
    tail_ev = np.unique(ylt.event[big | (in_tail & (share > 0.10))])
    print(f"analysis {t_run:.0f}s; {tail_years.size} tail years → {tail_ev.size} tail events", flush=True)

    L = pf.locations
    lat, lon = L["lat"].to_numpy(float), L["lon"].to_numpy(float)
    co = coastal_mask(lat, lon)
    cat = model.catalogs["TC"]
    wse_hf = ctx.pair_wse.copy()
    t_hf = t_lf = 0.0
    for k, e in enumerate(tail_ev):
        ci = int(ctx.events["cat_index"].iloc[e])
        a, b = int(ctx.ev_ptr[e]), int(ctx.ev_ptr[e + 1])
        loc = ctx.pair_loc[a:b]
        sel = co[loc]
        wse_hf[a:b] = np.nan
        if sel.any():
            tr = _tracks(cat, ci)
            t1 = time.time()
            w = run_surge(tr, -18.0, 12.0, keep_frames=False, max_cells=400_000,
                          sites_lat=lat[loc[sel]], sites_lon=lon[loc[sel]])["site_wse_max"]
            t_hf += time.time() - t1
            t1 = time.time()
            lf_event_cells(tr)
            t_lf += time.time() - t1
            wse_hf[a:b][sel] = w
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{tail_ev.size} full-model runs, {t_hf:.0f}s", flush=True)

    idx = np.isin(ylt.event, tail_ev)
    ref = dataclasses.replace(ctx, pair_wse=wse_hf, pair_pwet=np.isfinite(wse_hf).astype(np.float32),
                              loc_ssig=np.zeros(pf.n), surge_cal={**ctx.surge_cal, "sigma_event_m": 0.0})
    out = ref.kernel(ylt.event[idx], ylt.key[idx], split=True)
    gu_hf = res.occ_gu.copy()
    gu_hf[idx] = out[0]
    s_hf = occ_s.copy()
    s_hf[idx] = out[8]
    # the low-fidelity water levels without residual noise (to show what the error model contributes)
    flat = dataclasses.replace(ctx, loc_ssig=np.zeros(pf.n), surge_cal={**ctx.surge_cal, "sigma_event_m": 0.0})
    gu_flat = res.occ_gu.copy()
    gu_flat[idx] = flat.kernel(ylt.event[idx], ylt.key[idx])[0]

    A = {"engine (LF + calibration)": ann, "full model on tail events": annual(ylt, gu_hf, n),
         "LF, no residual": annual(ylt, gu_flat, n), "wind only": ann - annual(ylt, occ_s, n)}
    rows = []
    for rp in (100, 250, 500):
        row = {"rp": rp, **{k: quantile_at_rp(np.sort(v), rp) for k, v in A.items()}}
        row["engine_vs_full"] = row["engine (LF + calibration)"] / row["full model on tail events"] - 1.0
        rows.append(row)
        print(f"RP {rp}: " + ", ".join(f"{k} {v / 1e6:,.1f}m" for k, v in row.items() if isinstance(v, float) and k != "engine_vs_full")
              + f" → engine vs full {row['engine_vs_full']:+.1%}", flush=True)
    per_event = []
    for e in tail_ev:
        m = ylt.event == e
        per_event.append({"event": int(e), "name": str(ctx.events["name"].iloc[e]), "occurrences": int(m.sum()),
                          "surge_gu_engine": float(occ_s[m].mean()), "surge_gu_full": float(s_hf[m].mean())})
    se = np.array([p["surge_gu_engine"] for p in per_event])
    sf = np.array([p["surge_gu_full"] for p in per_event])
    summary = {
        "n_years": n, "n_locations": pf.n, "tail_years": int(tail_years.size), "tail_events": int(tail_ev.size),
        "aep": rows,
        "tail_event_surge_gu_ratio_engine_over_full": float(se.sum() / max(sf.sum(), 1.0)),
        "tail_event_surge_gu_log_ratio_sd": float(np.std(np.log((se + 1e5) / (sf + 1e5)))),
        "seconds_per_event": {"full_2min": t_hf / max(tail_ev.size, 1), "low_fidelity_serial": t_lf / max(tail_ev.size, 1)},
        "per_event": per_event,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_event"}, indent=1))


if __name__ == "__main__":
    main()
