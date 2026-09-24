"""Risk allocation & segmentation: AAL and Euler co-TVaR by any exposure dimension."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PERIL_INDEX, PERILS

DIMENSIONS = ("state", "construction", "occupancy", "lob", "year_band", "terrain", "acc_id", "loc_id", "stories_band",
              "peril")


def _year_band(y):
    return pd.cut(y, [0, 1949, 1974, 1994, 2001, 3000], labels=["<1950", "1950-74", "1975-94", "1995-01", "2002+"]
                  ).astype(str)


def location_frame(res) -> pd.DataFrame:
    L = res.ctx.portfolio.locations
    df = L[["loc_id", "acc_id", "lat", "lon", "state", "construction", "occupancy", "lob", "terrain", "year_built",
            "stories"]].copy()
    df["tiv"] = res.ctx.portfolio.tiv.sum(axis=1)
    df["aal"] = res.loc_aal_gross
    df["aal_gu"] = res.loc_aal_gu
    df["cotvar"] = res.loc_cotvar
    df["loss_cost_pm"] = np.where(df["tiv"] > 0, 1000 * df["aal"] / df["tiv"], 0.0)
    df["year_band"] = _year_band(df["year_built"])
    df["stories_band"] = np.where(df["stories"] <= 2, "1-2", np.where(df["stories"] <= 7, "3-7", "8+"))
    return df


def peril_tail_split(res, rp: float | None = None) -> dict:
    rp = rp or res.config.allocation_rp
    ann = res.annual("gross")
    k = max(int(np.floor(ann.size / rp)), 1)
    tail = np.argsort(ann)[-k:]
    out = {}
    for p in PERILS:
        if p not in res.config.perils:
            continue
        a = res.annual("gross", p)
        out[p] = {"aal": float(a.mean()), "cotvar": float(a[tail].mean())}
    return out


def allocate(res, dimension: str = "state", top: int | None = None) -> dict:
    if dimension == "peril":
        split = peril_tail_split(res)
        tot_aal = sum(v["aal"] for v in split.values()) or 1.0
        tot_tv = sum(v["cotvar"] for v in split.values()) or 1.0
        rows = [{"key": p, "aal": v["aal"], "aal_share": v["aal"] / tot_aal, "cotvar": v["cotvar"],
                 "cotvar_share": v["cotvar"] / tot_tv, "tiv": None, "tiv_share": None, "n": None,
                 "loss_cost_pm": None} for p, v in split.items()]
        return {"dimension": dimension, "rp": res.config.allocation_rp, "rows": rows}
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    df = location_frame(res)
    g = df.groupby(dimension).agg(n=("loc_id", "size"), tiv=("tiv", "sum"), aal=("aal", "sum"),
                                  aal_gu=("aal_gu", "sum"), cotvar=("cotvar", "sum"))
    tot = g.sum()
    g["aal_share"] = g["aal"] / max(tot["aal"], 1e-300)
    g["cotvar_share"] = g["cotvar"] / max(tot["cotvar"], 1e-300)
    g["tiv_share"] = g["tiv"] / max(tot["tiv"], 1e-300)
    g["loss_cost_pm"] = np.where(g["tiv"] > 0, 1000 * g["aal"] / g["tiv"], 0.0)
    g["tail_leverage"] = np.where(g["aal_share"] > 0, g["cotvar_share"] / g["aal_share"], 0.0)
    g = g.sort_values("cotvar", ascending=False)
    if top:
        g = g.head(top)
    rows = [{"key": str(k), **{c: float(v) for c, v in r.items()}} for k, r in g.iterrows()]
    return {"dimension": dimension, "rp": res.config.allocation_rp, "totals": {k: float(v) for k, v in tot.items()},
            "rows": rows}


def segment_ep(res, rps=(100, 250)) -> dict:
    """Stand-alone AEP per segment (res.group_names) vs. contribution → diversification benefit."""
    G = len(res.group_names)
    ann_grp = np.zeros((res.n_years, G))
    for g in range(G):
        ann_grp[:, g] = np.bincount(res.ylt.year, weights=res.grp_occ[:, g], minlength=res.n_years)
    total = ann_grp.sum(axis=1)
    out = {"dimension": res.config.group_by, "rows": [], "portfolio": {}}
    for rp in rps:
        k = max(int(np.floor(res.n_years / rp)), 1)
        tail = np.argsort(total)[-k:]
        q = 1 - 1 / rp
        out["portfolio"][str(rp)] = {"var": float(np.quantile(total, q)), "tvar": float(np.sort(total)[-k:].mean())}
        for g in range(G):
            sa_tvar = float(np.sort(ann_grp[:, g])[-k:].mean())
            if len(out["rows"]) <= g:
                out["rows"].append({"key": res.group_names[g], "aal": float(ann_grp[:, g].mean())})
            out["rows"][g][f"standalone_var_{rp}"] = float(np.quantile(ann_grp[:, g], q))
            out["rows"][g][f"standalone_tvar_{rp}"] = sa_tvar
            out["rows"][g][f"cotvar_{rp}"] = float(ann_grp[tail, g].mean())
    for rp in rps:
        s = sum(r[f"standalone_tvar_{rp}"] for r in out["rows"])
        p = out["portfolio"][str(rp)]["tvar"]
        out["portfolio"][str(rp)]["sum_standalone_tvar"] = s
        out["portfolio"][str(rp)]["diversification_benefit"] = 1 - p / s if s > 0 else 0.0
    out["rows"].sort(key=lambda r: -r["aal"])
    return out


def peril_index(p: str) -> int:
    return PERIL_INDEX[p]
