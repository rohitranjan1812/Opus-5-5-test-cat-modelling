"""Analysis orchestration: CatModel (catalogs + hazard caches) → AnalysisResult."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numba as nb
import numpy as np
import pandas as pd

from ..analytics.analytic import analytic_ep
from ..analytics.ep import RETURN_PERIODS, ep_curve, ep_table, summary_stats
from ..config import PERIL_INDEX, PERILS
from ..exposure.portfolio import Portfolio
from ..financial.reinsurance import Program, apply_program, layer_metrics
from ..financial.terms import FinancialArrays, build_financials
from ..geo import grid_cell_id
from ..hazard.base import EventCatalog, FrequencyModel
from ..hazard.earthquake import generate_eq_catalog
from ..hazard.footprint import Pairs, compute_pairs
from ..hazard.tropical_cyclone import generate_tc_catalog, intensity_rate_multiplier
from ..vulnerability.damage import BIN_HI, BIN_LO, VulnTables, build_tables
from .kernel import loss_kernel
from .ylt import ELT_KEY_BASE, YLT, rate_change_weights, simulate_ylt


@dataclass
class AnalysisConfig:
    name: str = "Analysis"
    perils: list[str] = field(default_factory=lambda: list(PERILS))
    n_years: int = 20000
    elt_samples: int = 24
    seed: int = 20240601
    damage_scale: dict[str, float] = field(default_factory=dict)
    rho_scale: float = 1.0
    sigma_between_scale: float = 1.0
    rate_multiplier: dict[str, float] = field(default_factory=dict)
    tc_intensity_scale: float = 1.0
    frequency: dict[str, dict] = field(default_factory=dict)
    per_risk: dict | None = None
    reinsurance: dict | None = None
    group_by: str = "state"
    allocation_rp: float = 250.0

    @classmethod
    def from_dict(cls, d: dict | None) -> AnalysisConfig:
        d = dict(d or {})
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed and v is not None})

    def to_dict(self) -> dict:
        return asdict(self)


class CatModel:
    """Holds stochastic catalogs and caches portfolio footprints."""

    def __init__(self, catalogs: dict[str, EventCatalog]):
        self.catalogs = catalogs
        self._pairs: dict[tuple[str, str], Pairs] = {}

    @classmethod
    def default(cls, tc_hurricanes: int = 4000, tc_storms: int = 800, tc_seed: int = 2024, eq_seed: int = 7,
                eq_density: float = 1.0) -> CatModel:
        return cls({"TC": generate_tc_catalog(tc_hurricanes, tc_storms, seed=tc_seed),
                    "EQ": generate_eq_catalog(seed=eq_seed, area_position_density=eq_density)})

    def pairs_for(self, portfolio: Portfolio, peril: str, progress=None) -> Pairs:
        key = (peril, portfolio.fingerprint())
        if key not in self._pairs:
            self._pairs[key] = compute_pairs(self.catalogs[peril], portfolio.sites(), progress=progress)
        elif progress:
            progress(1.0)
        return self._pairs[key]

    def invalidate(self, peril: str | None = None):
        self._pairs = {k: v for k, v in self._pairs.items() if peril is not None and k[0] != peril}

    def run(self, portfolio: Portfolio, config: AnalysisConfig | dict | None = None, progress=None) -> AnalysisResult:
        cfg = config if isinstance(config, AnalysisConfig) else AnalysisConfig.from_dict(config)
        return run_analysis(self, portfolio, cfg, progress=progress)


# ----------------------------------------------------------------------------------------------
@dataclass
class RunContext:
    """Everything the kernel needs — retained so what-ifs can be re-run with common random numbers."""

    portfolio: Portfolio
    config: AnalysisConfig
    events: pd.DataFrame  # global event table
    ev_ptr: np.ndarray
    pair_loc: np.ndarray
    pair_logi: np.ndarray
    ev_peril: np.ndarray
    ev_rate: np.ndarray
    ev_rate_base: np.ndarray
    relevant: np.ndarray
    vt: VulnTables
    fin: FinancialArrays
    loc_cell: np.ndarray
    loc_grp: np.ndarray
    grp_names: list[str]
    p_sig_b: np.ndarray
    p_rho_e: np.ndarray
    p_rho_c: np.ndarray
    p_dmg: np.ndarray
    pr_ret: float
    pr_lim: float
    freq: dict[str, FrequencyModel]
    max_pairs: int
    sigma_within: dict[str, float] = field(default_factory=dict)
    catalog_events: dict[str, pd.DataFrame] = field(default_factory=dict)

    def kernel(self, occ_event, occ_key, occ_w=None, n_grp=0, want_loc=False, detail_ptr=None, vt=None,
               p_dmg=None, p_rho_e=None, p_rho_c=None, p_sig_b=None, fin=None):
        vt = vt or self.vt
        fin = fin or self.fin
        occ_event = np.asarray(occ_event, np.int64)
        K = occ_event.size
        occ_w = np.ones(K) if occ_w is None else np.asarray(occ_w, float)
        n_chunks = int(min(max(nb.get_num_threads() * 8, 1), max(K, 1)))
        return loss_kernel(
            occ_event, np.asarray(occ_key, np.uint64), occ_w, np.uint64(self.config.seed),
            self.ev_ptr, self.pair_loc, self.pair_logi, self.ev_peril,
            self.p_sig_b if p_sig_b is None else p_sig_b, self.p_rho_e if p_rho_e is None else p_rho_e,
            self.p_rho_c if p_rho_c is None else p_rho_c, self.p_dmg if p_dmg is None else p_dmg,
            vt.loc_vuln, self.loc_cell, fin.tiv, fin.loc_ded, fin.loc_lim, fin.loc_pol, self.loc_grp,
            vt.cdf, vt.log_i0, vt.dlog, BIN_LO, BIN_HI, vt.cov,
            fin.pol_ded, fin.pol_lim, fin.pol_att, fin.pol_llim, fin.pol_share,
            float(self.pr_ret), float(self.pr_lim),
            int(n_grp), bool(want_loc),
            np.zeros(0, np.int64) if detail_ptr is None else np.asarray(detail_ptr, np.int64),
            int(max(self.max_pairs, 1)), n_chunks)


@dataclass
class AnalysisResult:
    id: str
    name: str
    portfolio_id: str
    config: AnalysisConfig
    created: str
    timings: dict
    events: pd.DataFrame
    elt: pd.DataFrame
    loc_aal_gross: np.ndarray
    loc_aal_gu: np.ndarray
    ylt: YLT
    occ_gu: np.ndarray
    occ_gross: np.ndarray
    occ_pr: np.ndarray
    occ_net: np.ndarray
    grp_occ: np.ndarray
    group_names: list[str]
    reinsurance: dict | None
    summary: dict
    ep: dict
    analytic: dict
    loc_cotvar: np.ndarray
    ctx: RunContext | None = None
    insights: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def n_years(self) -> int:
        return self.ylt.n_years

    def annual(self, basis: str = "gross", peril: str | None = None) -> np.ndarray:
        v = {"gu": self.occ_gu, "gross": self.occ_gross, "net": self.occ_net,
             "net_pre_cat": self.occ_gross - self.occ_pr}[basis]
        m = np.ones(v.shape, bool) if peril in (None, "all") else self.ylt.peril == PERIL_INDEX[peril]
        return np.bincount(self.ylt.year[m], weights=v[m], minlength=self.n_years)

    def annual_max(self, basis: str = "gross", peril: str | None = None) -> np.ndarray:
        v = {"gu": self.occ_gu, "gross": self.occ_gross, "net": self.occ_net,
             "net_pre_cat": self.occ_gross - self.occ_pr}[basis]
        m = np.ones(v.shape, bool) if peril in (None, "all") else self.ylt.peril == PERIL_INDEX[peril]
        mx = np.zeros(self.n_years)
        np.maximum.at(mx, self.ylt.year[m], v[m])
        return mx


# ----------------------------------------------------------------------------------------------
def _combine(model: CatModel, portfolio: Portfolio, cfg: AnalysisConfig, fin: FinancialArrays, progress):
    ev_frames, ptr_parts, loc_parts, li_parts, per_parts = [], [], [], [], []
    offset = 0
    perils = [p for p in PERILS if p in cfg.perils and p in model.catalogs]
    for n, peril in enumerate(perils):
        cat = model.catalogs[peril]

        def sub(f, n=n, peril=peril):
            if progress:
                progress(0.02 + 0.33 * (n + f) / len(perils), f"Footprints: {peril}")

        pairs = model.pairs_for(portfolio, peril, progress=sub)
        ev = cat.events
        rate = ev["rate"].to_numpy(float) * float(cfg.rate_multiplier.get(peril, 1.0))
        if peril == "TC" and abs(cfg.tc_intensity_scale - 1.0) > 1e-12:
            rate = rate * intensity_rate_multiplier(ev, cfg.tc_intensity_scale)
        ev_frames.append(pd.DataFrame({
            "event_uid": ev["event_id"].to_numpy(), "name": ev["name"].to_numpy(), "peril": peril,
            "cat_index": np.arange(len(ev)), "rate_base": ev["rate"].to_numpy(float), "rate": rate,
            "n_locs": np.diff(pairs.ev_ptr)}))
        ptr_parts.append(pairs.ev_ptr[1:] + offset)
        offset += pairs.n_pairs
        loc_parts.append(pairs.site)
        li_parts.append(pairs.log_i)
        per_parts.append(np.full(len(ev), PERIL_INDEX[peril], np.int64))
    events = pd.concat(ev_frames, ignore_index=True)
    ev_ptr = np.concatenate([[0]] + ptr_parts).astype(np.int64)
    pair_loc = np.concatenate(loc_parts).astype(np.int32)
    pair_logi = np.concatenate(li_parts).astype(np.float32)
    ev_peril = np.concatenate(per_parts)
    # sort pairs within each event by account so the kernel can aggregate accounts contiguously
    ev_of_pair = np.repeat(np.arange(len(events)), np.diff(ev_ptr))
    order = np.lexsort((pair_loc, fin.loc_pol[pair_loc], ev_of_pair))
    return events, ev_ptr, pair_loc[order], pair_logi[order], ev_peril


def build_context(model: CatModel, portfolio: Portfolio, cfg: AnalysisConfig, progress=None) -> RunContext:
    fin = build_financials(portfolio.locations)
    events, ev_ptr, pair_loc, pair_logi, ev_peril = _combine(model, portfolio, cfg, fin, progress)
    if progress:
        progress(0.36, "Vulnerability tables")
    sigw = {p: c.uncertainty.sigma_within for p, c in model.catalogs.items()}
    vt = build_tables(portfolio.locations, sigw, perils=cfg.perils)
    L = portfolio.locations
    loc_cell = grid_cell_id(L["lat"].to_numpy(), L["lon"].to_numpy(), 0.25)
    gcol = cfg.group_by if cfg.group_by in L.columns else "state"
    codes, names = pd.factorize(L[gcol].astype(str), sort=True)
    n_p = len(PERILS)
    p_sig_b, p_rho_e, p_rho_c, p_dmg = np.zeros(n_p), np.zeros(n_p), np.zeros(n_p), np.ones(n_p)
    freq = {}
    for p, c in model.catalogs.items():
        i = PERIL_INDEX[p]
        u = c.uncertainty
        p_sig_b[i] = u.sigma_between * cfg.sigma_between_scale
        re, rc = u.rho_event * cfg.rho_scale, u.rho_cell * cfg.rho_scale
        tot = re + rc
        if tot > 0.98:
            re, rc = re * 0.98 / tot, rc * 0.98 / tot
        p_rho_e[i], p_rho_c[i] = re, rc
        p_dmg[i] = float(cfg.damage_scale.get(p, 1.0))
        freq[p] = FrequencyModel.from_dict(cfg.frequency[p]) if p in cfg.frequency else c.frequency
    pr = cfg.per_risk or {}
    counts = np.diff(ev_ptr)
    rate = events["rate"].to_numpy(float)
    return RunContext(
        portfolio=portfolio, config=cfg, events=events, ev_ptr=ev_ptr, pair_loc=pair_loc, pair_logi=pair_logi,
        ev_peril=ev_peril, ev_rate=rate, ev_rate_base=events["rate_base"].to_numpy(float),
        relevant=(counts > 0) & (rate > 0), vt=vt, fin=fin, loc_cell=loc_cell.astype(np.int64),
        loc_grp=codes.astype(np.int32), grp_names=[str(x) for x in names], p_sig_b=p_sig_b, p_rho_e=p_rho_e,
        p_rho_c=p_rho_c, p_dmg=p_dmg, pr_ret=float(pr.get("retention", 0.0) or 0.0),
        pr_lim=float(pr.get("limit", 0.0) or 0.0), freq=freq, max_pairs=int(counts.max()) if counts.size else 1,
        sigma_within=sigw, catalog_events={p: model.catalogs[p].events for p in cfg.perils if p in model.catalogs})


def _event_caps(ctx: RunContext) -> np.ndarray:
    """Upper bound of each event's gross loss (policy terms applied to site maxima)."""
    fin = ctx.fin
    ev_of_pair = np.repeat(np.arange(len(ctx.events)), np.diff(ctx.ev_ptr))
    per = ctx.ev_peril[ev_of_pair]
    ttot = fin.tiv.sum(axis=1)
    j = ctx.pair_loc
    xmax = np.minimum(np.maximum(ttot[j] - fin.loc_ded[j, per], 0.0), fin.loc_lim[j, per])
    pol = fin.loc_pol[j]
    key = ev_of_pair.astype(np.int64) * max(fin.n_pol, 1) + pol
    uk, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=xmax)
    e_u = uk // max(fin.n_pol, 1)
    p_u = uk % max(fin.n_pol, 1)
    per_u = ctx.ev_peril[e_u]
    g1 = np.minimum(np.maximum(s - fin.pol_ded[p_u, per_u], 0.0), fin.pol_lim[p_u, per_u])
    g2 = np.minimum(np.maximum(g1 - fin.pol_att[p_u], 0.0), fin.pol_llim[p_u]) * fin.pol_share[p_u]
    return np.bincount(e_u, weights=g2, minlength=len(ctx.events))


def run_elt(ctx: RunContext, samples: int, **overrides):
    rel = np.nonzero(ctx.relevant)[0]
    S = int(max(samples, 2))
    occ_event = np.repeat(rel, S)
    occ_key = (ELT_KEY_BASE + occ_event.astype(np.uint64) * np.uint64(S)
               + np.tile(np.arange(S, dtype=np.uint64), rel.size)).astype(np.uint64)
    occ_w = np.repeat(ctx.ev_rate[rel] / S, S)
    gu, gross, pr, _, loc_g, loc_gu, _, _ = ctx.kernel(occ_event, occ_key, occ_w, want_loc=True, **overrides)
    return rel, S, gu.reshape(rel.size, S), gross.reshape(rel.size, S), pr.reshape(rel.size, S), loc_g, loc_gu


def run_analysis(model: CatModel, portfolio: Portfolio, cfg: AnalysisConfig, progress=None) -> AnalysisResult:
    t0 = time.time()
    timings = {}

    def prog(f, msg=""):
        if progress:
            progress(float(f), msg)

    ctx = build_context(model, portfolio, cfg, progress=prog)
    timings["hazard_and_setup_s"] = round(time.time() - t0, 3)

    # ---- ELT pass (event × S samples): event statistics + location AAL -----------------------
    prog(0.40, "Event loss table")
    t1 = time.time()
    rel, S, gu_s, gross_s, pr_s, loc_aal_g, loc_aal_gu = run_elt(ctx, cfg.elt_samples)
    caps = _event_caps(ctx)
    ev = ctx.events
    elt = pd.DataFrame({
        "event_uid": ev["event_uid"].to_numpy()[rel], "name": ev["name"].to_numpy()[rel],
        "peril": ev["peril"].to_numpy()[rel], "rate": ctx.ev_rate[rel], "n_locs": ev["n_locs"].to_numpy()[rel],
        "mean_gu": gu_s.mean(1), "sd_gu": gu_s.std(1, ddof=1), "mean": gross_s.mean(1), "sd": gross_s.std(1, ddof=1),
        "mean_net_pr": (gross_s - pr_s).mean(1), "p0": (gross_s <= 0).mean(1), "cap": caps[rel],
        "max_sample": gross_s.max(1), "global_index": rel,
    })
    elt["aal_contrib"] = elt["rate"] * elt["mean"]
    elt = elt.sort_values("aal_contrib", ascending=False).reset_index(drop=True)
    timings["elt_s"] = round(time.time() - t1, 3)

    # ---- YLT pass --------------------------------------------------------------------------
    prog(0.55, "Year loss table simulation")
    t2 = time.time()
    ylt = simulate_ylt(ctx.ev_rate, ctx.ev_peril, ctx.relevant, ctx.freq, cfg.n_years, cfg.seed)
    gu, gross, pr, grp, _, _, _, _ = ctx.kernel(ylt.event, ylt.key, n_grp=len(ctx.grp_names))
    timings["ylt_s"] = round(time.time() - t2, 3)

    # ---- reinsurance ---------------------------------------------------------------------------
    prog(0.78, "Reinsurance & metrics")
    subject = gross - pr
    program = Program.from_dict(cfg.reinsurance) if cfg.reinsurance else None
    reins = None
    occ_net = subject.copy()
    if program and program.contracts:
        applied = apply_program(program, ylt.year, ylt.peril, subject, ylt.n_years)
        occ_net = applied["net_occ"]
        gross_annual = np.bincount(ylt.year, weights=gross, minlength=ylt.n_years)
        reins = {"program": program.to_dict(), "metrics": layer_metrics(program, applied, gross_annual, ylt.n_years),
                 "annual_recoveries": {c["contract"].name: c["annual_recovery"] for c in applied["contracts"]}}

    res = AnalysisResult(
        id="an_" + uuid.uuid4().hex[:10], name=cfg.name, portfolio_id=portfolio.id, config=cfg,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"), timings=timings, events=ctx.events,
        elt=elt, loc_aal_gross=loc_aal_g, loc_aal_gu=loc_aal_gu, ylt=ylt, occ_gu=gu, occ_gross=gross, occ_pr=pr,
        occ_net=occ_net, grp_occ=grp, group_names=ctx.grp_names, reinsurance=reins, summary={}, ep={},
        analytic={}, loc_cotvar=np.zeros(portfolio.n), ctx=ctx)

    # ---- metrics ---------------------------------------------------------------------------
    t3 = time.time()
    res.ep = compute_ep(res)
    prog(0.86, "Analytic cross-check")
    elt_an = elt[["peril", "rate", "mean", "sd", "cap", "p0"]]
    try:
        res.analytic = analytic_ep(elt_an, ctx.freq)
    except Exception as exc:  # pragma: no cover - diagnostic only
        res.analytic = {"error": str(exc)}
    prog(0.90, "Tail allocation")
    res.loc_cotvar = tail_allocation(res, cfg.allocation_rp)
    res.summary = build_summary(res)
    timings["metrics_s"] = round(time.time() - t3, 3)
    timings["total_s"] = round(time.time() - t0, 3)
    from ..analytics.insights import generate_insights  # local import to avoid a cycle

    res.insights = generate_insights(res)
    prog(1.0, "Done")
    return res


# ----------------------------------------------------------------------------------------------
def compute_ep(res: AnalysisResult) -> dict:
    out = {}
    bases = ["gu", "gross"] + (["net"] if res.reinsurance or res.ctx.pr_lim > 0 else [])
    perils = ["all"] + [p for p in PERILS if p in res.config.perils]
    for b in bases:
        out[b] = {}
        for p in perils:
            ann = res.annual(b, p)
            mx = res.annual_max(b, p)
            out[b][p] = {"aep": ep_table(ann), "oep": ep_table(mx, n_boot=0), "stats": summary_stats(ann),
                         "aep_curve": ep_curve(ann), "oep_curve": ep_curve(mx)}
    return out


def tail_allocation(res: AnalysisResult, rp: float) -> np.ndarray:
    """Euler (co-TVaR) allocation of AEP-TVaR(rp) to locations, by exact re-simulation of tail years."""
    ann = res.annual("gross")
    n = ann.size
    k = max(int(np.floor(n / rp)), 1)
    tail_years = np.argsort(ann)[-k:]
    m = np.isin(res.ylt.year, tail_years)
    occ = np.nonzero(m)[0]
    if occ.size == 0:
        return np.zeros(res.ctx.portfolio.n)
    ev = res.ylt.event[occ]
    counts = np.diff(res.ctx.ev_ptr)[ev]
    dptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    _, gross, _, _, _, _, det_g, _ = res.ctx.kernel(ev, res.ylt.key[occ], detail_ptr=dptr)
    # map detail rows back to locations
    locs = np.concatenate([res.ctx.pair_loc[res.ctx.ev_ptr[e]:res.ctx.ev_ptr[e + 1]] for e in ev])
    cot = np.bincount(locs, weights=det_g, minlength=res.ctx.portfolio.n) / k
    res.extras["tail_check"] = {"tvar_direct": float(np.sort(ann)[-k:].mean()), "tvar_alloc_sum": float(cot.sum()),
                                "rp": rp, "n_tail_years": k,
                                "max_abs_occ_diff": float(np.max(np.abs(gross - res.occ_gross[occ])))}
    return cot


def build_summary(res: AnalysisResult) -> dict:
    pf = res.ctx.portfolio
    tiv = pf.tiv_total
    g = res.ep["gross"]["all"]
    s = {
        "id": res.id, "name": res.name, "portfolio_id": res.portfolio_id, "portfolio_name": pf.name,
        "created": res.created, "config": res.config.to_dict(), "timings": res.timings, "n_years": res.n_years,
        "n_occurrences": int(res.ylt.n), "n_events_relevant": int(len(res.elt)),
        "n_pairs": int(res.ctx.pair_loc.size), "tiv": tiv, "n_locations": pf.n,
        "aal": {b: res.ep[b]["all"]["stats"]["mean"] for b in res.ep},
        "aal_by_peril": {p: res.ep["gross"][p]["stats"]["mean"] for p in res.ep["gross"] if p != "all"},
        "sd": g["stats"]["sd"], "loss_cost_per_mille": 1000 * g["stats"]["mean"] / tiv if tiv else None,
        "elt_aal": float(res.elt["aal_contrib"].sum()),
        "location_aal_sum": float(res.loc_aal_gross.sum()),
        "rp_table": _rp_table(res),
        "analytic": res.analytic.get("table", []), "analytic_aal": res.analytic.get("aal"),
        "tail_check": res.extras.get("tail_check"),
        "reinsurance": _jsonable_reins(res.reinsurance),
    }
    return s


def _rp_table(res: AnalysisResult) -> list[dict]:
    rows = []
    g = res.ep["gross"]["all"]
    idx_a = {r["rp"]: r for r in g["aep"]}
    idx_o = {r["rp"]: r for r in g["oep"]}
    net = res.ep.get("net", {}).get("all")
    idx_n = {r["rp"]: r for r in net["aep"]} if net else {}
    idx_no = {r["rp"]: r for r in net["oep"]} if net else {}
    gu = {r["rp"]: r for r in res.ep["gu"]["all"]["aep"]}
    for rp in RETURN_PERIODS:
        if rp not in idx_a:
            continue
        a = idx_a[rp]
        row = {"rp": rp, "gross_aep": a["loss"], "gross_aep_lo": a.get("loss_lo"), "gross_aep_hi": a.get("loss_hi"),
               "gross_aep_tvar": a["tvar"], "gross_oep": idx_o[rp]["loss"], "gross_oep_tvar": idx_o[rp]["tvar"],
               "gu_aep": gu[rp]["loss"]}
        if idx_n:
            row["net_aep"] = idx_n[rp]["loss"]
            row["net_oep"] = idx_no[rp]["loss"]
        rows.append(row)
    return rows


def _jsonable_reins(reins: dict | None):
    if not reins:
        return None
    return {"program": reins["program"], "metrics": reins["metrics"]}


def rerun_ylt(res: AnalysisResult, **overrides) -> dict:
    """Re-run the YLT pass with kernel overrides (same YLT, same keys → common random numbers)."""
    ctx = res.ctx
    _, gross, pr, _, _, _, _, _ = ctx.kernel(res.ylt.event, res.ylt.key, **overrides)
    ann = np.bincount(res.ylt.year, weights=gross, minlength=res.n_years)
    mx = np.zeros(res.n_years)
    np.maximum.at(mx, res.ylt.year, gross)
    return {"annual": ann, "annual_max": mx, "occ_gross": gross}


def reweighted_metrics(res: AnalysisResult, new_rate: np.ndarray) -> dict:
    from ..analytics.ep import effective_sample_size, weighted_ep_table

    w = rate_change_weights(res.ylt, res.ctx.ev_rate, new_rate, res.ctx.relevant, res.ctx.ev_peril)
    ann = res.annual("gross")
    mx = res.annual_max("gross")
    return {"weights_ess": effective_sample_size(w), "aal": float(np.average(ann, weights=w)),
            "aep": weighted_ep_table(ann, w), "oep": weighted_ep_table(mx, w)}
