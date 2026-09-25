"""Calibrate the stochastic engine's low-fidelity surge against the full-resolution 2-D model.

    python scripts/calibrate_surge.py collect [--events 96] [--seed 7]   # ≈ 12 s per event (full model)
    python scripts/calibrate_surge.py fit                                # seconds, re-runnable

**collect.** For a stratified design set of catalog hurricanes (region × intensity class), plus the
historical analogs, it runs both models:

- the full model (2′, track-following domain, −18 h…+12 h);
- the low-fidelity model (``hazard.surge.lf_event_cells``).

It then records every coastal node of the 2′ grid inside the low-fidelity box: land nodes, plus
shoreline water nodes, which is where waterfront buildings geocode on a 2′ DEM. For each node it keeps:

- the full model's peak water surface in the node's 3×3 stencil (the rule the engine uses for
  buildings), or NaN when the stencil stayed dry;
- the stencil's lowest ground + H_dry;
- the node's ground.

The low-fidelity cells are saved as they are. Nodes are portfolio-independent, so the calibration and
the site-response map cover the whole coast rather than one book.

**fit.** Runs ``hazard.surge_calib.calibrate`` (two-part hurdle model with crossed random effects; see
there) over a grid of penetration parameters (α, R). It keeps the pair with the best event-grouped
cross-validated depth error, then writes:

- ``catforge/data/surge_calibration.json`` (a, b, σ_e, σ_s, τ, diagnostics);
- ``catforge/data/surge_site_response.npz`` (the shrunk per-node site term δ on the 2′ grid).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from catforge.engine.model import CatModel
from catforge.hazard.surge import LF, _tracks, cache_dir, coastal_mask, landfall_point, lf_event_cells
from catforge.physics.surge2d import H_DRY, run_surge
from catforge.physics.tc_dynamics import track_arrays
from catforge.scenario import ANALOGS, build_event

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "catforge" / "data"
RAW = cache_dir() / "calibration_pairs.npz"


def design_events(cat, n, rng):
    ev = cat.events
    vmax = ev["vmax"].to_numpy(float) if "vmax" in ev else np.full(len(ev), 40.0)
    lat = ev["landfall_lat"].to_numpy(float)
    lon = ev["landfall_lon"].to_numpy(float)
    cls = np.digitize(vmax, [33, 43, 50, 58])  # TS, Cat1, Cat2, Cat3, Cat4+
    region = np.where(lon < -94, 0, np.where(lon < -88, 1, np.where(lon < -82.5, 2,
                      np.where(lat < 27.5, 3, np.where(lat < 33, 4, np.where(lat < 37.5, 5, 6))))))
    keys = sorted({(r, c) for r, c in zip(region, cls) if c >= 1})
    per = max(1, n // len(keys))
    picks = []
    for r, c in keys:
        idx = np.nonzero((region == r) & (cls == c))[0]
        picks += list(rng.choice(idx, size=min(per, idx.size), replace=False))
    return np.array(picks[:n])


def coastal_nodes(res, la, lo, lf=LF):
    """2′ nodes a building could stand on inside the LF box, with the full model's stencil water level."""
    lat, lon, z, em, dm = res["lat"], res["lon"], res["z"], res["eta_max"], res["depth_max"]
    iy = np.nonzero(np.abs(lat - la) <= lf["half_lat"])[0]
    ix = np.nonzero(np.abs(lon - lo) <= lf["half_lon"])[0]
    if iy.size < 3 or ix.size < 3:
        return None
    Y, X = np.meshgrid(iy, ix, indexing="ij")
    Y, X = Y.ravel(), X.ravel()
    ny, nx = z.shape
    off = np.array([-1, 0, 1])

    def stencil(Y, X):
        sy = np.clip(Y[:, None, None] + off[None, :, None], 0, ny - 1)
        sx = np.clip(X[:, None, None] + off[None, None, :], 0, nx - 1)
        sy, sx = np.broadcast_arrays(sy, sx)
        return sy.reshape(Y.size, 9), sx.reshape(Y.size, 9)

    zn = z[Y, X]
    sy, sx = stencil(Y, X)
    # land nodes, plus shoreline water nodes (land in the stencil): where waterfront buildings geocode on a 2′ DEM
    cand = ((zn > -1.0) & (zn < 15.0)) | ((zn > -10.0) & (z[sy, sx].max(axis=1) > 0.0))
    Y, X = Y[cand], X[cand]
    keep = coastal_mask(lat[Y], lon[X])
    Y, X = Y[keep], X[keep]
    sy, sx = stencil(Y, X)
    wet = dm[sy, sx] > H_DRY
    wse = np.where(wet, em[sy, sx], -np.inf).max(axis=1)
    cz = z[sy, sx].min(axis=1) + H_DRY  # a dry stencil means water stayed below its lowest ground
    return {"lat": lat[Y].astype(np.float32), "lon": lon[X].astype(np.float32), "z": z[Y, X].astype(np.float32),
            "hf": np.where(np.isfinite(wse), wse, np.nan).astype(np.float32), "cz": cz.astype(np.float32)}


def collect(args):
    rng = np.random.default_rng(args.seed)
    cat = CatModel.default().catalogs["TC"]
    tracks = [("cat", str(int(i)), _tracks(cat, int(i))) for i in design_events(cat, args.events, rng)]
    tracks += [("analog", a, track_arrays(build_event("TC", v["params"]))) for a, v in ANALOGS.items() if v["peril"] == "TC"]
    out = {k: [] for k in ("lat", "lon", "z", "hf", "cz", "cell_lat", "cell_lon", "cell_eta")}
    nptr, cptr, names = [0], [0], []
    t0 = time.time()
    for k, (kind, name, tr) in enumerate(tracks):
        la, lo = landfall_point(tr)
        try:
            res = run_surge(tr, -18.0, 12.0, keep_frames=False, max_cells=400_000)
        except ValueError:
            continue
        nodes = coastal_nodes(res, la, lo)
        if nodes is None or not nodes["lat"].size:
            continue
        cl, cn, ce = lf_event_cells(tr, LF)
        for key in ("lat", "lon", "z", "hf", "cz"):
            out[key].append(nodes[key])
        out["cell_lat"].append(cl)
        out["cell_lon"].append(cn)
        out["cell_eta"].append(ce)
        nptr.append(nptr[-1] + nodes["lat"].size)
        cptr.append(cptr[-1] + cl.size)
        names.append(f"{kind}:{name}")
        print(f"{k + 1}/{len(tracks)} {kind} {name}: {nodes['lat'].size} nodes, HF wet {np.isfinite(nodes['hf']).sum()}, "
              f"LF cells {cl.size}, stride {res['stride_deg'] * 60:.0f}′, {time.time() - t0:.0f}s", flush=True)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW, **{k: np.concatenate(v) for k, v in out.items()}, nptr=np.array(nptr),
                        cptr=np.array(cptr), names=np.array(names), lf=json.dumps(LF))
    print(f"wrote {RAW} ({len(names)} events, {nptr[-1]:,} node-events)")


def fit(args):
    from catforge.hazard import surge_calib as C

    d = dict(np.load(RAW))
    res = C.calibrate(d, alphas=args.alphas, rmaxs=args.rmaxs, r0=args.r0, progress=print)
    cal, site = res["calibration"], res["site_response"]
    cal["lf"] = json.loads(str(d["lf"]))
    (DATA / "surge_calibration.json").write_text(json.dumps(cal, indent=1))
    np.savez_compressed(DATA / "surge_site_response.npz", **site)
    print(json.dumps(cal, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--events", type=int, default=96)
    c.add_argument("--seed", type=int, default=7)
    f = sub.add_parser("fit")
    f.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3, 0.4])
    f.add_argument("--rmaxs", type=float, nargs="+", default=[15.0, 20.0, 25.0, 30.0])
    f.add_argument("--r0", type=float, default=3.5)
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else fit(args)


if __name__ == "__main__":
    main()
