"""Reinsurance programme engine operating on the simulated year-occurrence loss table.

Contracts are applied in *inuring stages*: all contracts in a stage see the same subject loss
(the net of all earlier stages), so a cat-XL tower is one stage and a quota share inuring to it is
an earlier stage.  Occurrences must be sorted by (year, day) so that aggregate features
(annual aggregate deductible/limit, reinstatements, stop-loss) are path-correct:

    x_k   = min((S_k - A)₊, L)                                  occurrence layer loss
    C_k   = Σ_{i≤k, same year} x_i                              running annual layer loss
    R_k   = min((C_k - AAD)₊, AAL) - min((C_{k-1} - AAD)₊, AAL)  recovery of occurrence k

An aggregate stop-loss is the special case A=0, L=∞ with AAD/AAL as its retention/limit.
Reinstatement premium is pro rata as to amount:  RIP_y = P · r · min(R_y, n·L) / L.
Technical pricing solves  P (1 + r E[min(R, nL)]/L) = E[R] + load  in closed form.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import PERIL_INDEX


@dataclass
class Contract:
    name: str
    type: str = "cat_xl"  # "cat_xl" | "quota_share" | "agg_xl"
    stage: int = 1
    attachment: float = 0.0
    limit: float = 0.0
    reinstatements: int = 0
    reinstatement_rate: float = 1.0
    aad: float = 0.0
    aal: float | None = None
    cession: float = 0.0
    event_limit: float | None = None
    placed: float = 1.0
    perils: list[str] | None = None
    premium: float | None = None

    @property
    def agg_limit(self) -> float:
        if self.type == "agg_xl":
            return float(self.limit) if self.limit > 0 else np.inf
        if self.aal is not None and self.aal > 0:
            return float(self.aal)
        return float(self.limit) * (1 + int(self.reinstatements)) if self.limit > 0 else np.inf

    @classmethod
    def from_dict(cls, d: dict) -> Contract:
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class Program:
    contracts: list[Contract] = field(default_factory=list)
    pricing_method: str = "stdev"  # "stdev" | "coc"
    pricing_load: float = 0.30  # stdev multiple, or cost-of-capital rate for "coc"
    coc_alpha: float = 0.99

    @classmethod
    def from_dict(cls, d: dict | None) -> Program:
        if not d:
            return cls()
        return cls(contracts=[Contract.from_dict(c) for c in d.get("contracts", [])],
                   pricing_method=d.get("pricing_method", "stdev"), pricing_load=float(d.get("pricing_load", 0.30)),
                   coc_alpha=float(d.get("coc_alpha", 0.99)))

    def to_dict(self) -> dict:
        return {"contracts": [asdict(c) for c in self.contracts], "pricing_method": self.pricing_method,
                "pricing_load": self.pricing_load, "coc_alpha": self.coc_alpha}


def _year_cumsum(x: np.ndarray, year: np.ndarray, n_years: int) -> np.ndarray:
    """Running sum of x within each year (x sorted by year)."""
    if x.size == 0:
        return x
    g = np.cumsum(x)
    tot = np.bincount(year, weights=x, minlength=n_years)
    start = np.concatenate([[0.0], np.cumsum(tot)[:-1]])
    return g - start[year]


def _xl_recoveries(subject, year, n_years, att, lim, aad, aal):
    occ = np.minimum(np.maximum(subject - att, 0.0), lim)
    c = _year_cumsum(occ, year, n_years)
    after = np.minimum(np.maximum(c - aad, 0.0), aal)
    before = np.minimum(np.maximum(c - occ - aad, 0.0), aal)
    return after - before


def apply_program(program: Program, occ_year: np.ndarray, occ_peril: np.ndarray, subject: np.ndarray,
                  n_years: int) -> dict:
    """Apply the programme. Returns per-contract occurrence & annual recoveries and net losses."""
    net = np.asarray(subject, float).copy()
    out = {"contracts": []}
    for stage in sorted({c.stage for c in program.contracts}):
        stage_subject = net.copy()
        stage_rec = np.zeros_like(net)
        for c in [c for c in program.contracts if c.stage == stage]:
            mask = np.ones(net.shape, bool)
            if c.perils:
                mask = np.isin(occ_peril, [PERIL_INDEX[p] for p in c.perils])
            subj = np.where(mask, stage_subject, 0.0)
            if c.type == "quota_share":
                rec = c.cession * subj
                if c.event_limit:
                    rec = np.minimum(rec, c.event_limit)
                rec_gross_share = rec
            elif c.type == "cat_xl":
                rec_gross_share = _xl_recoveries(subj, occ_year, n_years, c.attachment,
                                                 c.limit if c.limit > 0 else np.inf, c.aad, c.agg_limit)
                rec = rec_gross_share * c.placed
            elif c.type == "agg_xl":
                rec_gross_share = _xl_recoveries(subj, occ_year, n_years, 0.0, np.inf, c.attachment, c.agg_limit)
                rec = rec_gross_share * c.placed
            else:
                raise ValueError(f"unknown contract type {c.type}")
            annual = np.bincount(occ_year, weights=rec, minlength=n_years)
            annual_100 = np.bincount(occ_year, weights=rec_gross_share, minlength=n_years)
            stage_rec += rec
            out["contracts"].append({"contract": c, "occ_recovery": rec, "annual_recovery": annual,
                                     "annual_recovery_100": annual_100})
        net = np.maximum(net - stage_rec, 0.0)
    out["net_occ"] = net
    out["net_annual"] = np.bincount(occ_year, weights=net, minlength=n_years)
    return out


def _tvar(x: np.ndarray, alpha: float) -> float:
    if x.size == 0:
        return 0.0
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    return float(tail.mean()) if tail.size else float(q)


def layer_metrics(program: Program, applied: dict, gross_annual: np.ndarray, n_years: int) -> dict:
    """Pricing and risk metrics for every contract plus gross-vs-net capital view."""
    rows = []
    total_prem, total_rec = 0.0, 0.0
    for item in applied["contracts"]:
        c: Contract = item["contract"]
        R = item["annual_recovery"]
        R100 = item["annual_recovery_100"]
        el = float(R.mean())
        sd = float(R.std())
        if c.type == "quota_share":
            premium = c.premium if c.premium is not None else el  # proportional: ceded premium ~ ceded loss + margin
            rip = 0.0
            exp_reinst = 0.0
        else:
            if program.pricing_method == "coc":
                load = program.pricing_load * max(_tvar(R, program.coc_alpha) - el, 0.0)
            else:
                load = program.pricing_load * sd
            reinstated = np.minimum(R100, c.limit * c.reinstatements) if c.limit > 0 else np.zeros_like(R100)
            exp_reinst = float(reinstated.mean()) * c.placed
            ratio = c.reinstatement_rate * exp_reinst / (c.limit * c.placed) if c.limit > 0 else 0.0
            premium = c.premium if c.premium is not None else (el + load) / (1.0 + ratio)
            rip = premium * ratio
        lim_eff = (c.limit * c.placed) if c.limit > 0 else np.nan
        agg = c.agg_limit * c.placed if np.isfinite(c.agg_limit) else np.inf
        rows.append({
            "name": c.name, "type": c.type, "stage": c.stage, "attachment": c.attachment, "limit": c.limit,
            "placed": c.placed, "reinstatements": c.reinstatements, "expected_loss": el, "sd": sd,
            "prob_attach": float((R > 0).mean()),
            "prob_exhaust": float((R >= agg * (1 - 1e-9)).mean()) if np.isfinite(agg) else 0.0,
            "premium": float(premium), "expected_reinstatement_premium": float(rip),
            "rate_on_line": float(premium / lim_eff) if np.isfinite(lim_eff) and lim_eff > 0 else None,
            "loss_on_line": float(el / lim_eff) if np.isfinite(lim_eff) and lim_eff > 0 else None,
            "expected_margin": float(premium + rip - el), "expected_reinstated": exp_reinst,
        })
        total_prem += premium + rip
        total_rec += el
    net = applied["net_annual"]

    def cap(x):
        return {"aal": float(x.mean()), "sd": float(x.std()), "var_99_5": float(np.quantile(x, 0.995)),
                "var_99": float(np.quantile(x, 0.99)), "tvar_99": _tvar(x, 0.99)}

    g, n = cap(gross_annual), cap(net)
    relief = g["var_99_5"] - n["var_99_5"]
    cost = total_prem - total_rec
    return {"contracts": rows, "gross": g, "net": n, "total_premium": total_prem, "total_expected_recovery": total_rec,
            "net_cost": cost, "capital_relief_var995": relief,
            "relief_per_cost": float(relief / cost) if cost > 0 else None}


def optimise_cat_xl(occ_year, occ_peril, subject, n_years, gross_oep_at, attach_rps=(5, 10, 15, 20, 25, 35, 50, 75),
                    exhaust_rps=(100, 150, 200, 250, 350, 500), reinstatements: int = 1, pricing_load: float = 0.30,
                    pricing_method: str = "stdev") -> dict:
    """Grid search over single cat-XL layers → cost vs net 1-in-200 AEP efficient frontier."""
    gross_annual = np.bincount(occ_year, weights=subject, minlength=n_years)
    pts = []
    for ra in attach_rps:
        for rx in exhaust_rps:
            if rx <= ra:
                continue
            A, X = gross_oep_at(ra), gross_oep_at(rx)
            if X <= A or A <= 0:
                continue
            prog = Program([Contract(name="layer", attachment=A, limit=X - A, reinstatements=reinstatements)],
                           pricing_method=pricing_method, pricing_load=pricing_load)
            applied = apply_program(prog, occ_year, occ_peril, subject, n_years)
            m = layer_metrics(prog, applied, gross_annual, n_years)
            c = m["contracts"][0]
            pts.append({"attach_rp": ra, "exhaust_rp": rx, "attachment": A, "limit": X - A, "premium": c["premium"],
                        "expected_loss": c["expected_loss"], "net_cost": m["net_cost"],
                        "net_var_99_5": m["net"]["var_99_5"], "net_tvar_99": m["net"]["tvar_99"],
                        "capital_relief": m["capital_relief_var995"], "rate_on_line": c["rate_on_line"]})
    # Pareto frontier: minimise (net_cost, net_var_99_5)
    order = sorted(range(len(pts)), key=lambda i: (pts[i]["net_cost"], pts[i]["net_var_99_5"]))
    best = np.inf
    for i in order:
        pts[i]["efficient"] = pts[i]["net_var_99_5"] < best - 1e-6
        if pts[i]["efficient"]:
            best = pts[i]["net_var_99_5"]
    return {"points": pts, "gross_var_99_5": float(np.quantile(gross_annual, 0.995))}
