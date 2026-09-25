"""Portfolio (exposure) data model with OED-inspired CSV import/export and validation."""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import CONSTRUCTION_CLASSES, OCCUPANCIES, ROOF_SHAPES, TERRAINS
from ..hazard.footprint import Sites
from ..hazard.tropical_cyclone import TERRAIN_GUST_FACTOR

# canonical column -> (dtype, default)
SCHEMA: dict[str, tuple[str, object]] = {
    "loc_id": ("str", None),
    "acc_id": ("str", None),
    "lat": ("float", None),
    "lon": ("float", None),
    "state": ("str", "NA"),
    "lob": ("str", "personal"),
    "construction": ("str", "WOOD"),
    "occupancy": ("str", "RES_SF"),
    "year_built": ("int", 1985),
    "stories": ("int", 1),
    "tiv_building": ("float", None),
    "tiv_contents": ("float", 0.0),
    "tiv_bi": ("float", 0.0),
    "ded_tc": ("float", 0.0),
    "ded_eq": ("float", 0.0),
    "loc_limit": ("float", 0.0),
    "terrain": ("str", "suburban"),
    "vs30": ("float", 400.0),
    "roof_shape": ("str", "unknown"),
    "shutters": ("int", 0),
    "acc_ded": ("float", 0.0),
    "acc_limit": ("float", 0.0),
    "acc_attach": ("float", 0.0),
    "acc_layer_limit": ("float", 0.0),
    "acc_share": ("float", 1.0),
    # physical attributes (OED-style; NaN = unknown, the models fall back to their defaults)
    "ground_elev_m": ("float", None),
    "first_floor_height_m": ("float", None),
    "building_height_m": ("float", None),
    "floor_area_m2": ("float", None),
}
REQUIRED = ("lat", "lon", "tiv_building")

# OED-style aliases (case-insensitive) -> canonical
ALIASES = {
    "locnumber": "loc_id", "accnumber": "acc_id", "latitude": "lat", "longitude": "lon",
    "areacode": "state", "statecode": "state", "constructioncode": "construction", "occupancycode": "occupancy",
    "yearbuilt": "year_built", "numberofstoreys": "stories", "numberofstories": "stories",
    "buildingtiv": "tiv_building", "contentstiv": "tiv_contents", "bitiv": "tiv_bi",
    "locdedwind": "ded_tc", "locdedeq": "ded_eq", "loclimit": "loc_limit", "accded": "acc_ded",
    "acclimit": "acc_limit", "accattach": "acc_attach", "acclayerlimit": "acc_layer_limit",
    "accshare": "acc_share", "roofshape": "roof_shape", "lineofbusiness": "lob",
    "groundelevation": "ground_elev_m", "elevation": "ground_elev_m", "firstfloorheight": "first_floor_height_m",
    "buildingheight": "building_height_m", "floorarea": "floor_area_m2",
}


@dataclass
class Portfolio:
    name: str
    locations: pd.DataFrame
    id: str = field(default_factory=lambda: "pf_" + uuid.uuid4().hex[:10])
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- construction / IO
    @classmethod
    def from_frame(cls, df: pd.DataFrame, name: str = "portfolio", **kw) -> Portfolio:
        clean, warnings = validate(df)
        pf = cls(name=name, locations=clean, **kw)
        pf.warnings = warnings
        return pf

    @classmethod
    def from_csv(cls, data: str | bytes | io.IOBase, name: str = "uploaded", **kw) -> Portfolio:
        if isinstance(data, bytes):
            data = io.StringIO(data.decode("utf-8-sig"))
        elif isinstance(data, str) and "\n" in data:
            data = io.StringIO(data)
        return cls.from_frame(pd.read_csv(data), name=name, **kw)

    def to_csv(self) -> str:
        return self.locations.to_csv(index=False)

    # ---------------------------------------------------------------- derived views
    @property
    def n(self) -> int:
        return len(self.locations)

    @property
    def tiv(self) -> np.ndarray:
        L = self.locations
        return L[["tiv_building", "tiv_contents", "tiv_bi"]].to_numpy(float)

    @property
    def tiv_total(self) -> float:
        return float(self.tiv.sum())

    def sites(self) -> Sites:
        L = self.locations
        k = L["terrain"].map(TERRAIN_GUST_FACTOR).fillna(TERRAIN_GUST_FACTOR["suburban"]).to_numpy(float)
        return Sites(L["lat"].to_numpy(float), L["lon"].to_numpy(float), k, L["vs30"].to_numpy(float))

    def fingerprint(self) -> str:
        """Content hash used to key hazard caches."""
        L = self.locations
        h = pd.util.hash_pandas_object(L[["lat", "lon", "terrain", "vs30"]], index=False).to_numpy()
        return f"{len(L)}-{int(np.bitwise_xor.reduce(h * np.arange(1, len(h) + 1, dtype=np.uint64))):x}"

    def summary(self) -> dict:
        L = self.locations.copy()
        L["tiv_total"] = self.tiv.sum(axis=1)

        def by(col):
            g = L.groupby(col).agg(n=("loc_id", "size"), tiv=("tiv_total", "sum")).sort_values("tiv", ascending=False)
            return [{"key": str(k), "n": int(r.n), "tiv": float(r.tiv)} for k, r in g.iterrows()]

        L["year_band"] = pd.cut(L["year_built"], [0, 1949, 1974, 1994, 2001, 3000],
                                labels=["<1950", "1950-74", "1975-94", "1995-01", "2002+"]).astype(str)
        return {
            "id": self.id, "name": self.name, "created": self.created, "n_locations": self.n,
            "n_accounts": int(L["acc_id"].nunique()), "tiv_total": self.tiv_total,
            "tiv_by_coverage": dict(zip(["building", "contents", "bi"], self.tiv.sum(axis=0).tolist())),
            "bbox": [float(L["lon"].min()), float(L["lat"].min()), float(L["lon"].max()), float(L["lat"].max())],
            "by_state": by("state"), "by_construction": by("construction"), "by_occupancy": by("occupancy"),
            "by_lob": by("lob"), "by_year_band": by("year_band"), "warnings": self.warnings[:50],
            "meta": {k: v for k, v in self.meta.items() if k != "quality"},  # per-location QA: /quality endpoint
        }

    def with_changes(self, mask: np.ndarray, **changes) -> Portfolio:
        """Copy of the portfolio with attributes changed on ``mask`` (for mitigation what-ifs)."""
        L = self.locations.copy()
        for k, v in changes.items():
            L.loc[mask, k] = v
        return Portfolio(name=self.name + " (modified)", locations=L, meta=dict(self.meta))


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map aliases, coerce types, apply defaults and check domain constraints."""
    warnings: list[str] = []
    rename = {}
    for c in df.columns:
        key = str(c).strip()
        low = key.lower().replace(" ", "").replace("_", "")
        if key in SCHEMA:
            continue
        if low in ALIASES:
            rename[c] = ALIASES[low]
        else:
            for canon in SCHEMA:
                if canon.replace("_", "") == low:
                    rename[c] = canon
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    out = pd.DataFrame(index=df.index)
    for col, (typ, default) in SCHEMA.items():
        if col in df.columns:
            s = df[col]
        else:
            s = pd.Series([default] * len(df), index=df.index)
        if typ == "float":
            s = pd.to_numeric(s, errors="coerce")
            if default is not None:
                s = s.fillna(default)
            out[col] = s.astype(float)
        elif typ == "int":
            out[col] = pd.to_numeric(s, errors="coerce").fillna(default).round().astype(np.int64)
        else:
            out[col] = s.astype("string").fillna("" if default is None else str(default)).astype(str)
    n = len(out)
    if (out["loc_id"] == "").any() or "loc_id" not in df.columns:
        out["loc_id"] = [f"L{i + 1:06d}" for i in range(n)]
    dup = out["loc_id"].duplicated()
    if dup.any():
        warnings.append(f"{int(dup.sum())} duplicate loc_id values were suffixed")
        out.loc[dup, "loc_id"] = out.loc[dup, "loc_id"] + "_" + np.arange(int(dup.sum())).astype(str)
    blank_acc = out["acc_id"] == ""
    out.loc[blank_acc, "acc_id"] = "A-" + out.loc[blank_acc, "loc_id"]

    bad = out["lat"].isna() | out["lon"].isna() | (out["lat"].abs() > 90) | (out["lon"].abs() > 180)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have missing/invalid coordinates (first: row {int(np.argmax(bad.values))})")
    if (out["tiv_building"].isna()).any():
        raise ValueError("tiv_building has missing values")
    for c in ("tiv_building", "tiv_contents", "tiv_bi", "ded_tc", "ded_eq", "loc_limit", "acc_ded", "acc_limit",
              "acc_attach", "acc_layer_limit"):
        neg = out[c] < 0
        if neg.any():
            raise ValueError(f"{c} has {int(neg.sum())} negative value(s)")
    if ((out["acc_share"] <= 0) | (out["acc_share"] > 1)).any():
        raise ValueError("acc_share must be in (0, 1]")

    for col, allowed, default in (("construction", CONSTRUCTION_CLASSES, "WOOD"), ("occupancy", OCCUPANCIES, "RES_SF"),
                                  ("terrain", TERRAINS, "suburban"), ("roof_shape", ROOF_SHAPES, "unknown")):
        up = out[col].str.strip()
        up = up.str.upper() if col in ("construction", "occupancy") else up.str.lower()
        unknown = ~up.isin(list(allowed))
        if unknown.any():
            vals = sorted(set(up[unknown]))[:5]
            warnings.append(f"{int(unknown.sum())} unknown {col} value(s) {vals} mapped to {default}")
            up[unknown] = default
        out[col] = up
    out["lob"] = np.where(out["lob"].str.lower().str.startswith("comm"), "commercial", "personal")
    out["stories"] = out["stories"].clip(1, 120)
    out["year_built"] = out["year_built"].clip(1800, 2030)
    out["vs30"] = out["vs30"].clip(150.0, 1500.0)
    out["shutters"] = (out["shutters"] > 0).astype(np.int64)
    return out.reset_index(drop=True), warnings
