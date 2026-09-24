"""Automated, quantified portfolio insights.

Each insight is a dict ``{id, category, severity, title, detail, metric, recommendation}`` derived
deterministically from the analysis result — no hand-written narrative, every number traceable.
Severity: ``critical`` > ``warning`` > ``info`` > ``positive``.
"""

from __future__ import annotations

import numpy as np

from ..config import CONSTRUCTION_CLASSES, PERIL_INDEX, PERIL_NAMES
from .allocation import location_frame, peril_tail_split
from .ep import quantile_at_rp


def _m(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}bn"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}m"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}k"
    return f"${x:,.0f}"


def _pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def generate_insights(res) -> list[dict]:
    out: list[dict] = []
    add = out.append
    try:
        df = location_frame(res)
    except Exception:  # pragma: no cover
        return out
    rp = res.config.allocation_rp
    tot_aal = float(df["aal"].sum()) or 1e-300
    tot_tiv = float(df["tiv"].sum()) or 1e-300
    tot_cot = float(df["cotvar"].sum()) or 1e-300
    g = res.ep["gross"]["all"]
    aep = {r["rp"]: r for r in g["aep"]}

    # 1 — geographic concentration --------------------------------------------------------------
    st = df.groupby("state").agg(aal=("aal", "sum"), tiv=("tiv", "sum"), cot=("cotvar", "sum"))
    st["aal_s"], st["tiv_s"], st["cot_s"] = st["aal"] / tot_aal, st["tiv"] / tot_tiv, st["cot"] / tot_cot
    top = st.sort_values("cot", ascending=False).head(1)
    for name, r in top.iterrows():
        sev = "critical" if r.cot_s > 0.5 else ("warning" if r.cot_s > 0.3 else "info")
        add({"id": "geo_concentration", "category": "Concentration", "severity": sev,
             "title": f"{name} drives {_pct(r.cot_s)} of 1-in-{rp:g} tail risk on {_pct(r.tiv_s)} of TIV",
             "detail": f"{name}: AAL share {_pct(r.aal_s)}, TVaR contribution {_pct(r.cot_s)}, TIV share {_pct(r.tiv_s)}. "
                       f"Risk-to-exposure leverage {r.cot_s / max(r.tiv_s, 1e-9):.1f}×.",
             "metric": {"state": name, "cotvar_share": r.cot_s, "aal_share": r.aal_s, "tiv_share": r.tiv_s},
             "recommendation": "Consider accumulation limits / reinsurance targeted at this zone, or rate adequacy "
                               "review where risk share materially exceeds premium share."})
    lev = st[(st["aal_s"] > 0.03) & (st["aal_s"] > 2.0 * st["tiv_s"])].sort_values("aal_s", ascending=False)
    if len(lev):
        names = ", ".join(f"{k} ({_pct(r.aal_s)} AAL / {_pct(r.tiv_s)} TIV)" for k, r in lev.head(4).iterrows())
        add({"id": "loss_cost_hotspots", "category": "Concentration", "severity": "warning",
             "title": f"{len(lev)} state(s) carry more than twice their exposure share of expected loss",
             "detail": names, "metric": {"states": lev.index.tolist()},
             "recommendation": "Verify pricing adequacy: technical premium should scale with loss cost, not TIV."})

    # 2 — peril contribution to the tail ----------------------------------------------------------
    split = peril_tail_split(res, rp)
    if sum(v["aal"] > 0 for v in split.values()) > 1:
        s_aal = sum(v["aal"] for v in split.values()) or 1e-300
        s_cot = sum(v["cotvar"] for v in split.values()) or 1e-300
        dom = max(split, key=lambda p: split[p]["cotvar"])
        add({"id": "peril_tail_mix", "category": "Peril mix", "severity": "info",
             "title": f"{PERIL_NAMES[dom]} is {_pct(split[dom]['cotvar'] / s_cot)} of the 1-in-{rp:g} TVaR "
                      f"vs {_pct(split[dom]['aal'] / s_aal)} of AAL",
             "detail": "; ".join(f"{p}: AAL {_m(v['aal'])} ({_pct(v['aal'] / s_aal)}), tail {_pct(v['cotvar'] / s_cot)}"
                                 for p, v in split.items()),
             "metric": split,
             "recommendation": "Peril whose tail share exceeds its AAL share is the capital driver; "
                               "prioritise it in reinsurance and aggregate management."})

    # 3 — location-level tail concentration -------------------------------------------------------
    cot = np.sort(df["cotvar"].to_numpy())[::-1]
    n1 = max(int(0.01 * cot.size), 1)
    share1 = cot[:n1].sum() / tot_cot
    acc = df.groupby("acc_id")["cotvar"].sum().sort_values(ascending=False)
    share_acc10 = acc.head(10).sum() / tot_cot
    add({"id": "tail_granularity", "category": "Concentration",
         "severity": "warning" if share1 > 0.2 or share_acc10 > 0.25 else "info",
         "title": f"Top 1% of locations ({n1}) contribute {_pct(share1)} of tail risk; top 10 accounts {_pct(share_acc10)}",
         "detail": "Largest account contributors: " + ", ".join(f"{k} {_m(v)}" for k, v in acc.head(5).items()),
         "metric": {"top1pct_location_share": share1, "top10_account_share": share_acc10},
         "recommendation": "Review line sizes / facultative cessions on the top contributors; marginal-impact "
                           "analysis (common random numbers) quantifies each account's capital cost."})

    # 4 — tail shape ------------------------------------------------------------------------------
    if 100 in aep and 250 in aep and aep[100]["loss"] > 0:
        ratio = aep[250]["loss"] / aep[100]["loss"]
        tv = aep[250]["tvar"] / max(aep[250]["loss"], 1e-300)
        cov = g["stats"]["cov"] or 0
        add({"id": "tail_shape", "category": "Tail risk", "severity": "warning" if ratio > 1.6 or tv > 1.5 else "info",
             "title": f"1-in-250 AEP is {ratio:.2f}× the 1-in-100; TVaR/VaR at 250 = {tv:.2f}",
             "detail": f"Annual loss CoV {cov:.1f}. AEP 100 {_m(aep[100]['loss'])}, 250 {_m(aep[250]['loss'])}, "
                       f"TVaR 250 {_m(aep[250]['tvar'])}.",
             "metric": {"pml_ratio_250_100": ratio, "tvar_var_250": tv, "cov": cov},
             "recommendation": "Heavy tails reward per-occurrence protection high up the curve and TVaR-based "
                               "capital metrics over VaR."})

    # 5 — convergence --------------------------------------------------------------------------------
    if 250 in aep and aep[250].get("loss_hi") is not None and aep[250]["loss"] > 0:
        w = (aep[250]["loss_hi"] - aep[250]["loss_lo"]) / (2 * aep[250]["loss"])
        se = g["stats"]["se_mean"] / max(g["stats"]["mean"], 1e-300)
        add({"id": "convergence", "category": "Model diagnostics", "severity": "warning" if w > 0.10 else "positive",
             "title": f"Sampling uncertainty: ±{_pct(w, 1)} at 1-in-250 AEP, ±{_pct(1.96 * se, 1)} on AAL (95%)",
             "detail": f"{res.n_years:,} simulated years; distribution-free order-statistic interval "
                       f"[{_m(aep[250]['loss_lo'])}, {_m(aep[250]['loss_hi'])}].",
             "metric": {"rel_halfwidth_250": w, "aal_rel_se": se},
             "recommendation": "Increase simulated years (∝ 1/width²) if decisions are sensitive at this precision."
             if w > 0.10 else "Simulation is converged for decision use at this return period."})

    # 6 — analytic vs simulation -----------------------------------------------------------------
    an = {r["rp"]: r for r in res.analytic.get("table", [])}
    diffs = [abs(an[r]["aep"] / aep[r]["loss"] - 1) for r in (50, 100, 250) if r in an and r in aep and aep[r]["loss"] > 0]
    if diffs:
        d = max(diffs)
        add({"id": "analytic_check", "category": "Model diagnostics", "severity": "positive" if d < 0.06 else "warning",
             "title": f"Analytic (FFT) and simulated AEP agree within {_pct(d, 1)} at 1-in-50…250",
             "detail": "Independent check: ELT → Beta event severities → mixed-Poisson PGF → tilted FFT aggregate.",
             "metric": {"max_rel_diff": d},
             "recommendation": "Large gaps indicate strong intra-event non-linearity (account terms) or "
                               "multi-modal event losses not captured by a Beta."})

    # 7 — deductible / terms effectiveness ---------------------------------------------------------
    gu = res.ep["gu"]["all"]["stats"]["mean"]
    gr = g["stats"]["mean"]
    if gu > 0:
        add({"id": "terms_effect", "category": "Financial terms", "severity": "info",
             "title": f"Policy terms absorb {_pct(1 - gr / gu)} of ground-up AAL ({_m(gu - gr)} p.a.)",
             "detail": f"Ground-up AAL {_m(gu)} → gross {_m(gr)}. Percentage hurricane/earthquake deductibles "
                       f"shift frequency losses to insureds while tail losses remain largely retained.",
             "metric": {"gu_aal": gu, "gross_aal": gr},
             "recommendation": "Deductible buy-backs or reductions should be priced off the GU-to-gross gap."})

    # 8 — vulnerability class ranking -----------------------------------------------------------
    cc = df.groupby("construction").agg(aal=("aal", "sum"), tiv=("tiv", "sum"))
    cc = cc[cc["tiv"] > 0.01 * tot_tiv]
    if len(cc) > 1:
        cc["lc"] = 1000 * cc["aal"] / cc["tiv"]
        base_lc = 1000 * tot_aal / tot_tiv
        worst = cc["lc"].idxmax()
        add({"id": "construction_loss_cost", "category": "Vulnerability", "severity": "info",
             "title": f"{CONSTRUCTION_CLASSES.get(worst, worst)} has the highest loss cost: "
                      f"{cc.loc[worst, 'lc']:.2f}‰ vs portfolio {base_lc:.2f}‰",
             "detail": ", ".join(f"{k}: {v:.2f}‰" for k, v in cc["lc"].sort_values(ascending=False).items()),
             "metric": cc["lc"].to_dict(),
             "recommendation": "Use construction-differentiated rating; target mitigation credits where loss "
                               "cost is highest."})

    # 9 — event concentration ------------------------------------------------------------------
    elt = res.elt
    if len(elt) > 20:
        k = max(int(0.01 * len(elt)), 1)
        share = float(elt["aal_contrib"].head(k).sum() / max(elt["aal_contrib"].sum(), 1e-300))
        add({"id": "event_concentration", "category": "Hazard", "severity": "info",
             "title": f"The top 1% of events ({k}) generate {_pct(share)} of AAL",
             "detail": "Largest AAL drivers: " + "; ".join(f"{r['name']} ({_m(r['aal_contrib'])}/yr)"
                                                          for _, r in elt.head(3).iterrows()),
             "metric": {"top1pct_event_share": share},
             "recommendation": "Deterministic scenario tests on these events validate model behaviour for "
                               "the losses that matter most."})

    # 10 — multi-event years / clustering ------------------------------------------------------
    occ = res.occ_gross
    if occ.size and 10 in {r["rp"] for r in g["oep"]}:
        thr = quantile_at_rp(np.sort(res.annual_max("gross")), 10)
        big = occ > thr
        cnt = np.bincount(res.ylt.year[big], minlength=res.n_years)
        p2 = float((cnt >= 2).mean())
        p1 = float((cnt >= 1).mean())
        add({"id": "clustering", "category": "Tail risk", "severity": "warning" if p1 > 0 and p2 / p1 > 0.15 else "info",
             "title": f"{_pct(p2, 2)} of years have ≥2 events above the 1-in-10 OEP loss ({_m(thr)})",
             "detail": f"P(≥1) = {_pct(p1, 1)}; conditional probability of a second large loss "
                       f"{_pct(p2 / p1 if p1 else 0, 1)} (frequency over-dispersion and climate regimes inflate this).",
             "metric": {"p_ge1": p1, "p_ge2": p2},
             "recommendation": "Ensure reinstatement provisions and aggregate covers reflect multi-event years."})

    # 11 — ENSO regime conditioning ----------------------------------------------------------------
    fm = res.ctx.freq.get("TC")
    if fm is not None and fm.regimes and "TC" in res.config.perils:
        reg = res.ylt.regime[PERIL_INDEX["TC"]]
        tc = res.annual("gross", "TC")
        vals = {g_.name: float(tc[reg == i].mean()) for i, g_ in enumerate(fm.regimes) if (reg == i).sum() > 50}
        if len(vals) >= 2:
            hi, lo = max(vals, key=vals.get), min(vals, key=vals.get)
            add({"id": "enso", "category": "Climate", "severity": "info",
                 "title": f"Hurricane AAL is {vals[hi] / max(vals[lo], 1e-300):.1f}× higher in {hi} than {lo} years",
                 "detail": ", ".join(f"{k}: {_m(v)}" for k, v in vals.items()),
                 "metric": vals,
                 "recommendation": "Seasonal forecasts can inform mid-year capacity purchases and pricing loads."})

    # 12 — reinsurance ---------------------------------------------------------------------------
    rein = res.reinsurance
    if rein:
        m = rein["metrics"]
        relief = m["capital_relief_var995"]
        cost = m["net_cost"]
        add({"id": "reinsurance", "category": "Reinsurance",
             "severity": "positive" if (m.get("relief_per_cost") or 0) > 5 else "info",
             "title": f"Programme cuts 1-in-200 AEP by {_m(relief)} for an expected net cost of {_m(cost)} p.a.",
             "detail": "; ".join(f"{c['name']}: P(attach) {_pct(c['prob_attach'], 2)}, EL {_m(c['expected_loss'])}, "
                                 f"ROL {_pct(c['rate_on_line'] or 0, 2)}" for c in m["contracts"]),
             "metric": {"capital_relief": relief, "net_cost": cost, "relief_per_cost": m.get("relief_per_cost")},
             "recommendation": "Compare against the efficient frontier (optimiser) to test attachment/limit choices."})
    elif 200 in aep:
        add({"id": "reinsurance_none", "category": "Reinsurance", "severity": "warning",
             "title": f"No reinsurance modelled: 1-in-200 AEP retained loss {_m(aep[200]['loss'])}",
             "detail": f"That is {_pct(aep[200]['loss'] / tot_tiv, 2)} of TIV and "
                       f"{aep[200]['loss'] / max(gr, 1e-300):.1f}× AAL.",
             "metric": {"aep_200": aep[200]["loss"]},
             "recommendation": "Run the reinsurance optimiser to find cost-efficient cat XL structures."})

    order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    out.sort(key=lambda d: order.get(d["severity"], 9))
    return out
