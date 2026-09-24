"""Sensitivity, climate conditioning, ENSO regimes and mitigation what-ifs.

Two exact variance-reduction devices are used:
* **Common random numbers** — kernel re-runs reuse the same YLT and occurrence keys, so parameter
  deltas are not swamped by Monte Carlo noise.
* **Importance re-weighting of years** — rate changes (frequency, intensity-distribution shifts)
  are evaluated without re-simulation through exact per-year likelihood ratios (see
  ``engine.ylt.rate_change_weights``); the effective sample size is reported.
"""

from __future__ import annotations

import numpy as np

from ..analytics.ep import quantile_at_rp, tvar_sorted
from ..config import PERIL_INDEX, PERILS
from ..engine.model import AnalysisResult, rerun_ylt, reweighted_metrics, run_elt
from ..exposure.portfolio import Portfolio
from ..hazard.tropical_cyclone import intensity_rate_multiplier
from ..vulnerability.damage import build_tables


def _metrics(ann: np.ndarray) -> dict:
    x = np.sort(ann)
    return {"aal": float(ann.mean()), "aep_100": quantile_at_rp(x, 100), "aep_250": quantile_at_rp(x, 250),
            "tvar_250": tvar_sorted(x, 250)}


def base_metrics(res: AnalysisResult) -> dict:
    return _metrics(res.annual("gross"))


def tornado(res: AnalysisResult, spread: float = 0.2) -> dict:
    """One-at-a-time sensitivities of AAL / 1-in-100 / 1-in-250 AEP to key model assumptions."""
    ctx = res.ctx
    base = base_metrics(res)
    rows = []

    def add(name, lo_label, hi_label, lo, hi):
        rows.append({"parameter": name, "low_label": lo_label, "high_label": hi_label, "low": lo, "high": hi})

    for p in PERILS:
        if p not in res.config.perils:
            continue
        i = PERIL_INDEX[p]
        out = []
        for f in (1 - spread, 1 + spread):
            d = ctx.p_dmg.copy()
            d[i] *= f
            out.append(_metrics(rerun_ylt(res, p_dmg=d)["annual"]))
        add(f"{p} vulnerability (damage ×{1 - spread:.1f} / ×{1 + spread:.1f})", f"×{1 - spread:.1f}",
            f"×{1 + spread:.1f}", *out)
    out = []
    for f in (0.5, 1.5):
        out.append(_metrics(rerun_ylt(res, p_rho_e=np.minimum(ctx.p_rho_e * f, 0.49),
                                      p_rho_c=np.minimum(ctx.p_rho_c * f, 0.49))["annual"]))
    add("Spatial correlation ρ (×0.5 / ×1.5)", "×0.5", "×1.5", *out)
    out = []
    for f in (0.5, 1.5):
        out.append(_metrics(rerun_ylt(res, p_sig_b=ctx.p_sig_b * f)["annual"]))
    add("Inter-event hazard σ (×0.5 / ×1.5)", "×0.5", "×1.5", *out)
    for p in PERILS:
        if p not in res.config.perils:
            continue
        out = []
        for f in (1 - spread, 1 + spread):
            nr = ctx.ev_rate * np.where(ctx.ev_peril == PERIL_INDEX[p], f, 1.0)
            r = reweighted_metrics(res, nr)
            aep = {x["rp"]: x for x in r["aep"]}
            out.append({"aal": r["aal"], "aep_100": aep.get(100, {}).get("loss"), "aep_250": aep.get(250, {}).get("loss"),
                        "tvar_250": aep.get(250, {}).get("tvar"), "ess": r["weights_ess"]})
        add(f"{p} frequency (×{1 - spread:.1f} / ×{1 + spread:.1f})", f"×{1 - spread:.1f}", f"×{1 + spread:.1f}", *out)
    if "TC" in res.config.perils:
        out = []
        for f in (0.95, 1.05):
            nr = ctx.ev_rate.copy()
            m = ctx.ev_peril == PERIL_INDEX["TC"]
            nr[m] = nr[m] * intensity_rate_multiplier(_tc_events(res), f)
            r = reweighted_metrics(res, nr)
            aep = {x["rp"]: x for x in r["aep"]}
            out.append({"aal": r["aal"], "aep_100": aep.get(100, {}).get("loss"), "aep_250": aep.get(250, {}).get("loss"),
                        "tvar_250": aep.get(250, {}).get("tvar"), "ess": r["weights_ess"]})
        add("TC landfall intensity scale (×0.95 / ×1.05)", "×0.95", "×1.05", *out)
    for r in rows:
        r["swing_aep_250"] = abs((r["high"].get("aep_250") or 0) - (r["low"].get("aep_250") or 0))
    rows.sort(key=lambda r: -r["swing_aep_250"])
    return {"base": base, "rows": rows}


def _tc_events(res: AnalysisResult):
    """TC catalog events, aligned with the TC rows of the global event table."""
    return res.ctx.catalog_events["TC"]


def climate_scenario(res: AnalysisResult, tc_frequency: float = 1.0, tc_intensity: float = 1.0,
                     eq_frequency: float = 1.0) -> dict:
    ctx = res.ctx
    nr = ctx.ev_rate.copy()
    if "TC" in res.config.perils:
        m = ctx.ev_peril == PERIL_INDEX["TC"]
        nr[m] *= tc_frequency * intensity_rate_multiplier(_tc_events(res), tc_intensity)
    if "EQ" in res.config.perils:
        nr[ctx.ev_peril == PERIL_INDEX["EQ"]] *= eq_frequency
    r = reweighted_metrics(res, nr)
    base = base_metrics(res)
    aep = {x["rp"]: x for x in r["aep"]}
    return {"params": {"tc_frequency": tc_frequency, "tc_intensity": tc_intensity, "eq_frequency": eq_frequency},
            "base": base, "scenario": {"aal": r["aal"], "aep_100": aep.get(100, {}).get("loss"),
                                       "aep_250": aep.get(250, {}).get("loss"), "tvar_250": aep.get(250, {}).get("tvar")},
            "aep": r["aep"], "oep": r["oep"], "effective_sample_size": r["weights_ess"], "n_years": res.n_years}


def enso_conditional(res: AnalysisResult) -> dict:
    """EP conditional on the simulated TC climate regime of each year (exact sub-setting, no weights)."""
    fm = res.ctx.freq.get("TC")
    if fm is None or not fm.regimes:
        return {"regimes": []}
    reg = res.ylt.regime[PERIL_INDEX["TC"]]
    ann = res.annual("gross")
    tc = res.annual("gross", "TC")
    out = []
    for i, g in enumerate(fm.regimes):
        m = reg == i
        if m.sum() < 50:
            continue
        x = np.sort(ann[m])
        out.append({"regime": g.name, "prob": g.prob, "multiplier": g.multiplier, "n_years": int(m.sum()),
                    "aal": float(ann[m].mean()), "tc_aal": float(tc[m].mean()),
                    "aep_100": quantile_at_rp(x, 100) if m.sum() >= 100 else None,
                    "aep_250": quantile_at_rp(x, 250) if m.sum() >= 250 else None,
                    "p_loss_gt_aep10": float((ann[m] > quantile_at_rp(np.sort(ann), 10)).mean())})
    return {"regimes": out, "unconditional_aal": float(ann.mean())}


MITIGATION_PRESETS = {
    "shutters_fl_pre2002": {"label": "Install hurricane shutters on FL personal homes built before 2002",
                            "filter": {"state": ["FL"], "lob": ["personal"], "year_built_max": 2001},
                            "changes": {"shutters": 1}},
    "hip_roofs_coastal": {"label": "Retrofit gable roofs to hip on coastal/open-terrain homes",
                          "filter": {"lob": ["personal"], "roof_shape": ["gable"], "terrain": ["coastal", "open"]},
                          "changes": {"roof_shape": "hip"}},
    "seismic_retrofit_ca_old": {"label": "Seismic retrofit (code upgrade) of pre-1975 CA buildings",
                                "filter": {"state": ["CA"], "year_built_max": 1974},
                                "changes": {"year_built": 1995}},
    "urm_retrofit": {"label": "Replace/retrofit unreinforced masonry to reinforced masonry",
                     "filter": {"construction": ["URM"]}, "changes": {"construction": "MASONRY"}},
}


def _mask(L, flt: dict) -> np.ndarray:
    m = np.ones(len(L), bool)
    for k, v in (flt or {}).items():
        if k == "year_built_max":
            m &= L["year_built"].to_numpy() <= int(v)
        elif k == "year_built_min":
            m &= L["year_built"].to_numpy() >= int(v)
        elif k in L.columns:
            m &= L[k].astype(str).isin([str(x) for x in (v if isinstance(v, list) else [v])]).to_numpy()
    return m


def mitigation(res: AnalysisResult, preset: str | None = None, flt: dict | None = None,
               changes: dict | None = None) -> dict:
    """What-if: modify building attributes, rebuild vulnerability, re-run ELT & YLT with CRN."""
    if preset:
        p = MITIGATION_PRESETS[preset]
        flt, changes, label = p["filter"], p["changes"], p["label"]
    else:
        label = "custom"
    ctx = res.ctx
    L = ctx.portfolio.locations
    mask = _mask(L, flt or {})
    if not mask.any():
        return {"label": label, "n_locations": 0, "message": "no locations match the filter"}
    pf2: Portfolio = ctx.portfolio.with_changes(mask, **(changes or {}))
    vt2 = build_tables(pf2.locations, ctx.sigma_within, perils=res.config.perils)
    elt2 = run_elt(ctx, res.config.elt_samples, vt=vt2)
    loc_aal2 = elt2[5]
    ann2 = rerun_ylt(res, vt=vt2)["annual"]
    b, a = base_metrics(res), _metrics(ann2)
    saved = res.loc_aal_gross - loc_aal2
    return {
        "label": label, "filter": flt, "changes": changes, "n_locations": int(mask.sum()),
        "tiv_affected": float(ctx.portfolio.tiv.sum(axis=1)[mask].sum()),
        "base": b, "mitigated": a,
        "aal_saving": float(saved.sum()), "aal_saving_affected_pct": float(saved[mask].sum() / max(res.loc_aal_gross[mask].sum(), 1e-300)),
        "aal_saving_per_location": float(saved[mask].mean()),
        "top_locations": [{"loc_id": str(L["loc_id"].iloc[i]), "saving": float(saved[i])}
                          for i in np.argsort(saved)[::-1][:15] if saved[i] > 0],
    }
