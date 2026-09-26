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
from ..physics.grf import (
    Matern,
    VecchiaField,
    build_vecchia,
    event_closures,
    maximin_order,
    ordered_neighbors,
    xyz_km,
)
from ..vulnerability.damage import BIN_HI, BIN_LO, SURGE_DEPTH, SURGE_DR, VulnTables, build_tables
from .fidelity import DEFAULTS as FIDELITY_DEFAULTS
from .fidelity import plan as fidelity_plan
from .fidelity import tail_participation as fidelity_participation
from .fidelity import tail_stats as fidelity_tail
from .fidelity import upgrade as fidelity_upgrade
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
    dependence: str = "grf"  # "grf" (Matérn random field, Vecchia) | "copula" (legacy two-level copula)
    grf: dict[str, dict] = field(default_factory=dict)  # per-peril overrides {"nu": .., "range_km": ..}
    vecchia_m: int = 30
    tc_surge: bool = True  # storm surge as part of the hurricane peril (multi-fidelity 2-D surge hazard)
    surge: dict = field(default_factory=dict)  # overrides of the calibration, e.g. {"sigma_scale": 1.5}
    # uncertainty-driven full-fidelity surge for the tail: {"budget": events, "tol": rel. TVaR sd, "rp": T}
    surge_fidelity: dict = field(default_factory=dict)

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
        self._surge: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

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

    def surge_for(self, portfolio: Portfolio, pairs: Pairs, cal: dict, progress=None):
        """Pair-aligned hurricane surge for this portfolio: (wet water level m above MSL, P(wet), x − x0).

        NaN/0 = none; x − x0 is the low-fidelity level relative to the event-error pivot.

        Low-fidelity 2-D surge runs (disk-cached per catalog) for the events that bring ≥30 m/s gusts to
        surge-reachable locations, site extraction, then the calibrated full-model correction: the wet
        level s(x) + γ·covariates + δ̂ (the node's site response) and the connectivity probability.
        """
        from ..hazard.surge import (
            catalog_surge,
            coastal_mask,
            event_error_chol,
            site_covariates,
            site_response,
            site_wse,
            surge_levels,
        )

        key = (portfolio.fingerprint(), tuple(sorted((k, str(v)) for k, v in cal.items())))
        if key in self._surge:
            return self._surge[key]
        L = portfolio.locations
        lat, lon = L["lat"].to_numpy(float), L["lon"].to_numpy(float)
        co = coastal_mask(lat, lon)
        n_ev = len(pairs.ev_ptr) - 1
        ev_of = np.repeat(np.arange(n_ev), np.diff(pairs.ev_ptr))
        rel = np.unique(ev_of[co[pairs.site] & (pairs.log_i >= np.log(30.0))])
        wse = np.full(pairs.n_pairs, np.nan, np.float32)
        pwet = np.zeros(pairs.n_pairs, np.float32)
        if rel.size:
            cells = catalog_surge(self.catalogs["TC"], events=rel, progress=progress)
            x, x_free = (site_wse(cells, pairs.ev_ptr, pairs.site, lat, lon, co, a, cal["r0_km"], cal["rmax_km"],
                                  ev_idx=rel) for a in (cal["alpha_m_per_km"], 0.0))
            _, zc, shore = site_covariates(lat, lon)
            lvl, pwet = surge_levels(x, x_free, zc[pairs.site], shore[pairs.site], cal)
            wse = lvl + site_response(lat, lon, cal)[0][pairs.site].astype(np.float32)
            xc = np.where(np.isfinite(x), x - event_error_chol(cal)[0], 0.0).astype(np.float32)
        else:
            xc = np.zeros(pairs.n_pairs, np.float32)
        self._surge[key] = (wse, pwet, xc)
        return wse, pwet, xc

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
    sigma_within: dict[str, float] = field(default_factory=dict)  # σ folded into vulnerability tables
    catalog_events: dict[str, pd.DataFrame] = field(default_factory=dict)
    grf: dict | None = None  # Vecchia arrays + per-event closures (GRF dependence)
    # storm surge (hurricane pairs): water level, building ground / first floor / modifier, error model
    pair_wse: np.ndarray | None = None
    loc_ground: np.ndarray | None = None
    loc_ffh: np.ndarray | None = None
    loc_smod: np.ndarray | None = None
    loc_ssig: np.ndarray | None = None  # per-location site σ: √(σ² + Var δ̂)
    pair_pwet: np.ndarray | None = None  # P(the pair's node connects to the surge)
    pair_xc: np.ndarray | None = None  # low-fidelity level minus the event-error pivot x0
    loc_node: np.ndarray | None = None  # dense 2′ node id per location (shared surge draws)
    node_kptr: np.ndarray | None = None  # correlated surge residual: node → lattice-knot CSR (process convolution)
    node_knot: np.ndarray | None = None
    node_kw: np.ndarray | None = None
    knot_key: np.ndarray | None = None
    ev_hf: np.ndarray | None = None  # 1 = this event's surge comes from the full 2′ model (fidelity allocation)
    p_surge: np.ndarray | None = None
    surge_cal: dict = field(default_factory=dict)
    elt_surge: np.ndarray | None = None  # (relevant events × samples) surge-attributed GU from the ELT pass

    def kernel(self, occ_event, occ_key, occ_w=None, n_grp=0, want_loc=False, detail_ptr=None, vt=None,
               p_dmg=None, p_rho_e=None, p_rho_c=None, p_sig_b=None, fin=None, grf=None, split=False):
        vt = vt or self.vt
        fin = fin or self.fin
        g = grf or self.grf or _NO_GRF
        occ_event = np.asarray(occ_event, np.int64)
        K = occ_event.size
        occ_w = np.ones(K) if occ_w is None else np.asarray(occ_w, float)
        n_chunks = int(min(max(nb.get_num_threads() * 8, 1), max(K, 1)))
        n_loc = fin.tiv.shape[0]
        wse = self.pair_wse if self.pair_wse is not None else np.full(self.pair_loc.size, np.nan, np.float32)
        ps = self.p_surge if self.p_surge is not None else np.zeros(len(PERILS), np.int8)
        from ..hazard.surge import event_error_chol

        cal = self.surge_cal or {"sigma_event_m": 0.0}
        sc = float(cal.get("sigma_scale", 1.0))
        _, l11, l21, l22 = event_error_chol(cal)
        ssig = self.loc_ssig if self.loc_ssig is not None else np.zeros(n_loc)
        node = self.loc_node if self.loc_node is not None else np.arange(n_loc, dtype=np.int64)
        out = loss_kernel(
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
            int(max(self.max_pairs, 1)), n_chunks,
            g["p_grf"], g["p_phi"], g["vptr"], g["vnbr"], g["vcoef"], g["vsd"], g["clo_ptr"], g["clo"],
            wse, self.pair_pwet if self.pair_pwet is not None else np.ones(self.pair_loc.size, np.float32),
            self.pair_xc if self.pair_xc is not None else np.zeros(self.pair_loc.size, np.float32),
            node,
            self.loc_ground if self.loc_ground is not None else np.ones(n_loc),
            self.loc_ffh if self.loc_ffh is not None else np.full(n_loc, 0.5),
            self.loc_smod if self.loc_smod is not None else np.ones(n_loc), ps,
            sc * l11, sc * l21, sc * l22, ssig, SURGE_DEPTH, SURGE_DR,
            self.node_kptr if self.node_kptr is not None else np.zeros(int(node.max(initial=-1)) + 2, np.int64),
            self.node_knot if self.node_knot is not None else np.zeros(0, np.int64),
            self.node_kw if self.node_kw is not None else np.zeros(0),
            self.knot_key if self.knot_key is not None else np.zeros(0, np.int64),
            self.ev_hf if self.ev_hf is not None else np.zeros(self.ev_ptr.size - 1, np.int8))
        return out if split else out[:8]


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
    ev_frames, ptr_parts, loc_parts, li_parts, per_parts, wse_parts, pw_parts, xc_parts = [], [], [], [], [], [], [], []
    offset = 0
    perils = [p for p in PERILS if p in cfg.perils and p in model.catalogs]
    for n, peril in enumerate(perils):
        cat = model.catalogs[peril]

        def sub(f, n=n, peril=peril):
            if progress:
                progress(0.02 + 0.25 * (n + f) / len(perils), f"Footprints: {peril}")

        pairs = model.pairs_for(portfolio, peril, progress=sub)
        if peril == "TC" and cfg.tc_surge:
            def ssub(f):
                if progress:
                    progress(0.27 + 0.08 * f, "Storm surge: low-fidelity 2-D runs (cached per catalog)")

            w, pw, xc = model.surge_for(portfolio, pairs, surge_calibration(cfg), progress=ssub)
            wse_parts.append(w)
            pw_parts.append(pw)
            xc_parts.append(xc)
        else:
            wse_parts.append(np.full(pairs.n_pairs, np.nan, np.float32))
            pw_parts.append(np.zeros(pairs.n_pairs, np.float32))
            xc_parts.append(np.zeros(pairs.n_pairs, np.float32))
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
    pair_wse = np.concatenate(wse_parts).astype(np.float32)
    pair_pwet = np.concatenate(pw_parts).astype(np.float32)
    pair_xc = np.concatenate(xc_parts).astype(np.float32)
    # sort pairs within each event by account so the kernel can aggregate accounts contiguously
    ev_of_pair = np.repeat(np.arange(len(events)), np.diff(ev_ptr))
    order = np.lexsort((pair_loc, fin.loc_pol[pair_loc], ev_of_pair))
    return events, ev_ptr, pair_loc[order], pair_logi[order], ev_peril, pair_wse[order], pair_pwet[order], pair_xc[order]


def surge_summary(ctx: RunContext, ylt: YLT, occ_gu: np.ndarray, occ_surge: np.ndarray) -> dict:
    """Surge share of hurricane ground-up loss, and hurricane AEP with vs without surge."""
    from ..analytics.ep import quantile_at_rp
    from ..hazard.surge import site_response

    tc = ylt.peril == PERIL_INDEX["TC"]
    n = ylt.n_years
    ann_tc = np.bincount(ylt.year[tc], weights=occ_gu[tc], minlength=n)
    ann_s = np.bincount(ylt.year[tc], weights=occ_surge[tc], minlength=n)
    a_all, a_wind = np.sort(ann_tc), np.sort(ann_tc - ann_s)
    rows = [{"rp": rp, "tc_gu": quantile_at_rp(a_all, rp), "tc_gu_wind_only": quantile_at_rp(a_wind, rp),
             "surge_gu": quantile_at_rp(np.sort(ann_s), rp)} for rp in (10, 25, 50, 100, 250, 500, 1000) if rp <= n]
    for r in rows:
        r["uplift"] = r["tc_gu"] / r["tc_gu_wind_only"] - 1.0 if r["tc_gu_wind_only"] > 0 else None
    wse = ctx.pair_wse if ctx.pair_wse is not None else np.zeros(0)
    cal = ctx.surge_cal
    reached = np.unique(ctx.pair_loc[np.isfinite(wse)]) if wse.size else np.zeros(0, int)
    L = ctx.portfolio.locations
    m_seen = site_response(L["lat"].to_numpy(float)[reached], L["lon"].to_numpy(float)[reached], cal)[2]
    return {
        "aal_gu": float(ann_s.mean()), "tc_aal_gu": float(ann_tc.mean()),
        "share_of_tc_aal_gu": float(ann_s.mean() / ann_tc.mean()) if ann_tc.mean() > 0 else 0.0,
        "aep": rows, "n_pairs_with_water": int(np.isfinite(wse).sum()), "n_locations_reached": int(reached.size),
        "n_locations_measured_ground": int(np.isfinite(ctx.portfolio.locations["ground_elev_m"]).sum())
        if "ground_elev_m" in ctx.portfolio.locations else 0,
        "n_locations_with_site_response": int(np.sum(m_seen > 0)),
        "mean_p_wet": float(np.mean(ctx.pair_pwet[np.isfinite(wse)])) if ctx.pair_pwet is not None and np.isfinite(wse).any() else None,
        "model": {k: cal.get(k) for k in ("model", "alpha_m_per_km", "rmax_km", "sigma_event_m", "sigma_site_m", "event_error",
                                           "tau_site_response_m", "cv_depth_rmse_m", "cv_depth_bias", "cv_hit_rate",
                                           "cv_false_alarm_ratio", "n_events", "n_nodes", "source")},
    }


def surge_calibration(cfg: AnalysisConfig) -> dict:
    from ..hazard.surge import calibration

    cal = {**calibration(), **{k: v for k, v in cfg.surge.items() if k != "sigma_scale"}}
    s = float(cfg.surge.get("sigma_scale", 1.0))
    cal["sigma_scale"] = s
    return cal


def surge_nodes(portfolio: Portfolio) -> tuple[np.ndarray, np.ndarray]:
    """(sorted unique 2′ node keys, dense node id per location): buildings in one node share their surge draws."""
    from ..hazard.surge import node_key

    L = portfolio.locations
    keys, inv = np.unique(node_key(L["lat"].to_numpy(float), L["lon"].to_numpy(float)), return_inverse=True)
    return keys, inv.astype(np.int64)


def surge_field(portfolio: Portfolio, cal: dict, keys: np.ndarray, loc_node: np.ndarray) -> dict:
    """Correlated surge residual field tables for the portfolio's surge-reachable nodes."""
    from ..hazard.surge import coastal_mask, field_tables

    L = portfolio.locations
    co = coastal_mask(L["lat"].to_numpy(float), L["lon"].to_numpy(float))
    active = np.zeros(keys.size, bool)
    active[loc_node[co]] = True
    kp, kn, kw, kk = field_tables(keys, active, cal, float(cal.get("sigma_scale", 1.0)))
    return {"node_kptr": kp, "node_knot": kn, "node_kw": kw, "knot_key": kk}


def surge_site_sigma(portfolio: Portfolio, cal: dict) -> np.ndarray:
    """Per-location independent site σ = √(σ²·nugget + Var δ̂); the correlated part is the field.

    Nodes the calibration never saw carry the full τ².
    """
    from ..hazard.surge import site_response

    L = portfolio.locations
    pv = site_response(L["lat"].to_numpy(float), L["lon"].to_numpy(float), cal)[1]
    fld = cal.get("field") or {}
    nug = float(fld.get("nugget_frac", 1.0)) if fld.get("scales") else 1.0
    return float(cal.get("sigma_scale", 1.0)) * np.sqrt(float(cal["sigma_site_m"]) ** 2 * nug + pv)


def surge_site_arrays(portfolio: Portfolio):
    """Per-location building ground (measured, else the 2′ DEM floored at 1 m), first floor and surge modifier."""
    from ..physics.dem import elevation
    from ..vulnerability.damage import surge_site_params

    L = portfolio.locations
    lat, lon = L["lat"].to_numpy(float), L["lon"].to_numpy(float)
    ground = np.maximum(elevation(lat, lon, fill=1.0), 1.0)
    if "ground_elev_m" in L:
        g = L["ground_elev_m"].to_numpy(float)
        ground = np.where(np.isfinite(g), g, ground)
    ffh = L["first_floor_height_m"].to_numpy(float) if "first_floor_height_m" in L else None
    ff, mod = surge_site_params(L["construction"].to_numpy(), L["occupancy"].to_numpy(), L["stories"].to_numpy(),
                                L["year_built"].to_numpy(), ffh)
    return ground.astype(np.float64), ff, mod


_NO_GRF = {"p_grf": np.zeros(len(PERILS), np.int8), "p_phi": np.zeros(len(PERILS)),
           "vptr": np.zeros((len(PERILS), 2), np.int64), "vnbr": np.zeros(1, np.int64), "vcoef": np.zeros(1),
           "vsd": np.ones((len(PERILS), 1)), "clo_ptr": np.zeros(2, np.int64), "clo": np.zeros(1, np.int64)}
_VECCHIA_CACHE: dict[tuple, object] = {}


def _cached(key, fn):
    if key not in _VECCHIA_CACHE:
        if len(_VECCHIA_CACHE) > 24:
            _VECCHIA_CACHE.pop(next(iter(_VECCHIA_CACHE)))
        _VECCHIA_CACHE[key] = fn()
    return _VECCHIA_CACHE[key]


def grf_models(model_or_catalogs, cfg: AnalysisConfig, range_scale: float = 1.0) -> dict[str, Matern]:
    cats = model_or_catalogs.catalogs if hasattr(model_or_catalogs, "catalogs") else model_or_catalogs
    out = {}
    for p in cfg.perils:
        if p not in cats:
            continue
        u = cats[p].uncertainty
        o = cfg.grf.get(p, {})
        out[p] = Matern(float(o.get("nu", u.grf_nu)), float(o.get("range_km", u.grf_range_km)) * range_scale)
    return out


def build_grf(portfolio: Portfolio, models: dict[str, Matern], sigma_w: dict[str, float], m: int,
              ev_ptr: np.ndarray, pair_loc: np.ndarray, ev_peril: np.ndarray) -> dict:
    """Vecchia factors per peril (shared maximin ordering / neighbour search) + per-event closures."""
    L = portfolio.locations
    lat, lon = L["lat"].to_numpy(float), L["lon"].to_numpy(float)
    fp = portfolio.fingerprint()
    n = lat.size
    order = _cached(("order", fp), lambda: maximin_order(lat, lon))
    nbrs = _cached(("nbr", fp, m), lambda: ordered_neighbors(xyz_km(lat, lon), order, m))
    n_p = len(PERILS)
    p_grf = np.zeros(n_p, np.int8)
    p_phi = np.zeros(n_p)
    vptr = np.zeros((n_p, n + 1), np.int64)
    vsd = np.ones((n_p, n))
    rank2 = np.zeros((n_p, n), np.int64)
    nb_parts, cf_parts, off = [], [], 0
    fields: dict[str, VecchiaField] = {}
    for p, mod in models.items():
        i = PERIL_INDEX[p]
        V = _cached(("vecchia", fp, m, mod.nu, round(mod.range_km, 6)),
                    lambda mod=mod: build_vecchia(lat, lon, mod, m=m, order=order, neighbors=nbrs))
        fields[p] = V
        p_grf[i], p_phi[i] = 1, float(sigma_w[p])
        vptr[i] = V.ptr + off
        vsd[i] = V.cond_sd
        rank2[i] = V.rank
        nb_parts.append(V.nbr)
        cf_parts.append(V.coef)
        off += V.nbr.size
    vnbr = np.concatenate(nb_parts) if nb_parts else np.zeros(1, np.int64)
    vcoef = np.concatenate(cf_parts) if cf_parts else np.zeros(1)
    ev_field = np.where(p_grf[ev_peril] == 1, ev_peril, -1).astype(np.int64)
    clo_ptr, clo = event_closures(ev_ptr, pair_loc.astype(np.int64), ev_field, vptr, vnbr, rank2, n)
    return {"p_grf": p_grf, "p_phi": p_phi, "vptr": vptr, "vnbr": vnbr, "vcoef": vcoef, "vsd": vsd,
            "clo_ptr": clo_ptr, "clo": clo if clo.size else np.zeros(1, np.int64),
            "models": {p: mo.to_dict() for p, mo in models.items()}, "m": m,
            "closure_overhead": float(clo.size / max(pair_loc.size, 1)),
            "nnz_per_site": {p: float(V.nbr.size / max(n, 1)) for p, V in fields.items()}}


def build_context(model: CatModel, portfolio: Portfolio, cfg: AnalysisConfig, progress=None) -> RunContext:
    fin = build_financials(portfolio.locations)
    events, ev_ptr, pair_loc, pair_logi, ev_peril, pair_wse, pair_pwet, pair_xc = _combine(model, portfolio, cfg, fin,
                                                                                          progress)
    ground, ffh, smod = surge_site_arrays(portfolio)
    scal = surge_calibration(cfg) if cfg.tc_surge else {}
    node_keys, loc_node = surge_nodes(portfolio)
    p_surge = np.zeros(len(PERILS), np.int8)
    if cfg.tc_surge and "TC" in cfg.perils:
        p_surge[PERIL_INDEX["TC"]] = 1
    if progress:
        progress(0.36, "Vulnerability tables")
    use_grf = cfg.dependence == "grf"
    # in GRF mode the within-event residual is sampled explicitly, so tables carry no σ_w
    sigw = {p: (0.0 if use_grf else c.uncertainty.sigma_within) for p, c in model.catalogs.items()}
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
        if use_grf:  # copula now only couples vulnerability residuals; rho_scale scales the field range
            re, rc = u.dmg_rho_event, u.dmg_rho_cell
        else:
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
    grf = None
    if use_grf:
        if progress:
            progress(0.37, "Spatial random field (Vecchia)")
        grf = build_grf(portfolio, grf_models(model, cfg, cfg.rho_scale),
                        {p: c.uncertainty.sigma_within for p, c in model.catalogs.items()}, cfg.vecchia_m,
                        ev_ptr, pair_loc, ev_peril)
    return RunContext(
        portfolio=portfolio, config=cfg, events=events, ev_ptr=ev_ptr, pair_loc=pair_loc, pair_logi=pair_logi,
        ev_peril=ev_peril, ev_rate=rate, ev_rate_base=events["rate_base"].to_numpy(float),
        relevant=(counts > 0) & (rate > 0), vt=vt, fin=fin, loc_cell=loc_cell.astype(np.int64),
        loc_grp=codes.astype(np.int32), grp_names=[str(x) for x in names], p_sig_b=p_sig_b, p_rho_e=p_rho_e,
        p_rho_c=p_rho_c, p_dmg=p_dmg, pr_ret=float(pr.get("retention", 0.0) or 0.0),
        pr_lim=float(pr.get("limit", 0.0) or 0.0), freq=freq, max_pairs=int(counts.max()) if counts.size else 1,
        sigma_within=sigw, catalog_events={p: model.catalogs[p].events for p in cfg.perils if p in model.catalogs},
        grf=grf, pair_wse=pair_wse, loc_ground=ground, loc_ffh=ffh, loc_smod=smod, p_surge=p_surge,
        surge_cal=scal, loc_ssig=surge_site_sigma(portfolio, scal) if cfg.tc_surge else None,
        pair_pwet=pair_pwet, pair_xc=pair_xc, loc_node=loc_node, ev_hf=np.zeros(len(events), np.int8),
        **(surge_field(portfolio, scal, node_keys, loc_node) if cfg.tc_surge else {}))


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


def _elt_occurrences(ctx: RunContext, events: np.ndarray, S: int):
    """ELT occurrences of ``events`` × S samples: (event, key, weight λ/S). Keys depend on (event, sample) only."""
    occ_event = np.repeat(events, S)
    occ_key = (ELT_KEY_BASE + occ_event.astype(np.uint64) * np.uint64(S)
               + np.tile(np.arange(S, dtype=np.uint64), events.size)).astype(np.uint64)
    return occ_event, occ_key, np.repeat(ctx.ev_rate[events] / S, S)


def run_elt(ctx: RunContext, samples: int, **overrides):
    rel = np.nonzero(ctx.relevant)[0]
    S = int(max(samples, 2))
    occ_event, occ_key, occ_w = _elt_occurrences(ctx, rel, S)
    gu, gross, pr, _, loc_g, loc_gu, _, _, sg = ctx.kernel(occ_event, occ_key, occ_w, want_loc=True, split=True,
                                                           **overrides)
    ctx.elt_surge = sg.reshape(rel.size, S)
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
    timings["elt_s"] = round(time.time() - t1, 3)

    # ---- YLT pass --------------------------------------------------------------------------
    prog(0.55, "Year loss table simulation")
    t2 = time.time()
    ylt = simulate_ylt(ctx.ev_rate, ctx.ev_peril, ctx.relevant, ctx.freq, cfg.n_years, cfg.seed)
    gu, gross, pr, grp, _, _, _, _, occ_surge = ctx.kernel(ylt.event, ylt.key, n_grp=len(ctx.grp_names), split=True)
    timings["ylt_s"] = round(time.time() - t2, 3)

    # ---- surge fidelity allocation: full 2′ model where it reduces the tail error most -------------
    fid = None
    fcfg = {**FIDELITY_DEFAULTS, **(cfg.surge_fidelity or {})}
    if cfg.tc_surge and "TC" in cfg.perils and int(fcfg["budget"]) > 0:
        t4 = time.time()
        prog(0.62, "Surge fidelity allocation")
        annual = np.bincount(ylt.year, weights=gu, minlength=ylt.n_years)
        a_tail = fidelity_participation(ylt.year, ylt.event, annual, float(fcfg["rp"]), ctx.ev_ptr.size - 1,
                                        float(fcfg["taper"]))[rel]
        p = fidelity_plan(ctx.ev_rate[rel], gu_s, ctx.elt_surge, annual, ctx.ev_peril[rel] == PERIL_INDEX["TC"],
                          int(fcfg["budget"]), float(fcfg["tol"]), float(fcfg["rp"]), float(fcfg["rho"]), a=a_tail)
        rows = p["rows"]
        fid = {k: p[k] for k in ("u_before", "u_after", "tol_met", "n_candidates", "u_independent", "u_coherent")}
        fid["u_before_rel"], fid["u_after_rel"] = p["u_before"] / p["tvar"], p["u_after"] / p["tvar"]
        fid.update({"rp": float(fcfg["rp"]), "tol": float(fcfg["tol"]), "budget": int(fcfg["budget"]),
                    "rho": float(fcfg["rho"]), "upgraded": []})
        if rows.size:
            up = rel[rows]
            oe, ok, ow = _elt_occurrences(ctx, up, S)
            old = ctx.kernel(oe, ok, ow, want_loc=True)
            fid.update(fidelity_upgrade(model, ctx, up, progress=lambda f, m: prog(0.62 + 0.1 * f, m)))
            new = ctx.kernel(oe, ok, ow, want_loc=True, split=True)
            surge_before = ctx.elt_surge[rows].mean(1)
            gu_s[rows], gross_s[rows], pr_s[rows] = (new[i].reshape(rows.size, S) for i in range(3))
            ctx.elt_surge[rows] = new[8].reshape(rows.size, S)
            loc_aal_g = loc_aal_g + new[4] - old[4]  # exact: same keys, only these events changed
            loc_aal_gu = loc_aal_gu + new[5] - old[5]
            idx = np.nonzero(np.isin(ylt.event, up))[0]
            o = ctx.kernel(ylt.event[idx], ylt.key[idx], n_grp=len(ctx.grp_names), split=True)
            gu[idx], gross[idx], pr[idx], occ_surge[idx] = o[0], o[1], o[2], o[8]
            if len(ctx.grp_names):
                grp[idx] = o[3]
            names = ctx.events["name"].to_numpy()
            fid["upgraded"] = [{"event": int(e), "name": str(names[e]), "tail_sensitivity": float(p["a"][r]),
                                "sd_surge_gu": float(np.sqrt(p["v"][r])), "share": float(p["lin"][r] / p["lin"].sum()),
                                "mean_surge_gu_before": float(sb), "mean_surge_gu_after": float(ctx.elt_surge[r].mean())}
                               for e, r, sb in zip(up, rows, surge_before)]
        after = np.bincount(ylt.year, weights=gu, minlength=ylt.n_years)
        (fid["var_before"], fid["tvar_before"]), (fid["var_after"], fid["tvar_after"]) = (
            fidelity_tail(annual, fid["rp"]), fidelity_tail(after, fid["rp"]))
        timings["surge_fidelity_s"] = round(time.time() - t4, 3)

    caps = _event_caps(ctx)
    ev = ctx.events
    elt = pd.DataFrame({
        "event_uid": ev["event_uid"].to_numpy()[rel], "name": ev["name"].to_numpy()[rel],
        "peril": ev["peril"].to_numpy()[rel], "rate": ctx.ev_rate[rel], "n_locs": ev["n_locs"].to_numpy()[rel],
        "mean_gu": gu_s.mean(1), "sd_gu": gu_s.std(1, ddof=1), "mean": gross_s.mean(1), "sd": gross_s.std(1, ddof=1),
        "mean_net_pr": (gross_s - pr_s).mean(1), "p0": (gross_s <= 0).mean(1), "cap": caps[rel],
        "max_sample": gross_s.max(1), "global_index": rel,
    })
    elt["mean_surge_gu"] = ctx.elt_surge.mean(1)
    elt["surge_full_fidelity"] = ctx.ev_hf[rel].astype(bool) if ctx.ev_hf is not None else False
    elt["aal_contrib"] = elt["rate"] * elt["mean"]
    elt = elt.sort_values("aal_contrib", ascending=False).reset_index(drop=True)

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
    if cfg.tc_surge and "TC" in cfg.perils:
        res.extras["surge"] = surge_summary(ctx, ylt, gu, occ_surge)
        res.extras["surge"]["fidelity"] = fid

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
        "surge": res.extras.get("surge"),
        "dependence": {"mode": res.config.dependence,
                       **({k: v for k, v in res.ctx.grf.items() if k in ("models", "m", "closure_overhead", "nnz_per_site")}
                          if res.ctx.grf else {})},
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
