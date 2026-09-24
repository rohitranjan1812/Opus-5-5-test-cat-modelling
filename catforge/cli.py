"""Command-line interface.

    catforge serve [--port 8000] [--data-dir ./catforge_data] [--no-demo]
    catforge portfolio --n 10000 --states FL TX --out book.csv
    catforge run --portfolio book.csv --years 50000 --reinsurance program.json --out results/
    catforge hazard-curve --peril TC --lat 25.77 --lon -80.19
    catforge scenario --analog andrew_1992 --portfolio book.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _fmt(x: float) -> str:
    return f"{x / 1e6:12,.2f}m"


def cmd_serve(a):
    import uvicorn

    from .api.app import create_app

    app = create_app(data_dir=a.data_dir, demo=not a.no_demo, demo_locations=a.demo_locations, demo_years=a.demo_years)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


def cmd_portfolio(a):
    from .exposure import generate_portfolio

    pf = generate_portfolio(a.n, seed=a.seed, states=a.states, commercial_share=a.commercial_share)
    Path(a.out).write_text(pf.to_csv())
    print(f"wrote {pf.n:,} locations, TIV ${pf.tiv_total / 1e9:,.2f}bn → {a.out}")


def _load_portfolio(path_or_n):
    from .exposure import Portfolio, generate_portfolio

    p = Path(str(path_or_n))
    if p.exists():
        return Portfolio.from_csv(p.read_bytes(), name=p.stem)
    return generate_portfolio(int(path_or_n))


def cmd_run(a):
    from .analytics.allocation import location_frame
    from .engine import AnalysisConfig, CatModel

    pf = _load_portfolio(a.portfolio)
    reins = json.loads(Path(a.reinsurance).read_text()) if a.reinsurance else None
    cfg = AnalysisConfig(n_years=a.years, elt_samples=a.samples, seed=a.seed, perils=a.perils, reinsurance=reins)
    model = CatModel.default()
    t0 = time.time()

    def progress(f, msg=""):
        sys.stderr.write(f"\r[{f * 100:5.1f}%] {msg:40s}")
        sys.stderr.flush()

    res = model.run(pf, cfg, progress=progress)
    sys.stderr.write("\n")
    s = res.summary
    print(f"\nPortfolio: {pf.name}  ({pf.n:,} locations, TIV ${pf.tiv_total / 1e9:,.2f}bn)  run {time.time() - t0:.1f}s")
    print(f"AAL  gross {_fmt(s['aal']['gross'])}   ground-up {_fmt(s['aal']['gu'])}")
    print(f"{'RP':>6} {'AEP gross':>14} {'OEP gross':>14} {'TVaR (AEP)':>14}" + (f" {'AEP net':>14}" if reins else ""))
    for r in s["rp_table"]:
        line = f"{r['rp']:>6} {_fmt(r['gross_aep'])} {_fmt(r['gross_oep'])} {_fmt(r['gross_aep_tvar'])}"
        if reins:
            line += f" {_fmt(r.get('net_aep', 0))}"
        print(line)
    print("\nInsights:")
    for i in res.insights:
        print(f"  [{i['severity']}] {i['title']}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        from .api.app import clean

        (out / "summary.json").write_text(json.dumps(clean({**s, "insights": res.insights}), indent=2))
        res.elt.to_csv(out / "elt.csv", index=False)
        location_frame(res).to_csv(out / "locations.csv", index=False)
        import pandas as pd

        pd.DataFrame({"year": range(res.n_years), "gross": res.annual("gross"), "gu": res.annual("gu"),
                      "net": res.annual("net")}).to_csv(out / "ylt.csv", index=False)
        print(f"\nresults written to {out}/")


def cmd_hazard_curve(a):
    from .engine import CatModel
    from .hazard.footprint import hazard_curve

    m = CatModel.default()
    h = hazard_curve(m.catalogs[a.peril], a.lat, a.lon)
    print(f"{a.peril} hazard at ({a.lat}, {a.lon}) [{h['unit']}]")
    for rp, v in h["return_period_intensity"].items():
        print(f"  {rp:>5}-yr: {'—' if v is None else f'{v:.3f}'}")


def cmd_scenario(a):
    from .engine import CatModel
    from .scenario import ANALOGS, run_scenario

    pf = _load_portfolio(a.portfolio)
    an = ANALOGS[a.analog]
    s = run_scenario(pf, an["peril"], an["params"], n_samples=a.samples, base_model=CatModel.default(),
                     with_footprint=False)
    print(an["label"])
    print(f"  locations affected: {s['n_locations_affected']:,}")
    for k in ("mean", "p5", "p50", "p95", "p99"):
        print(f"  gross {k:>4}: {_fmt(s['gross'][k])}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="catforge", description="CatForge catastrophe modelling platform")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the API + UI server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data-dir", default="catforge_data")
    s.add_argument("--no-demo", action="store_true")
    s.add_argument("--demo-locations", type=int, default=5000)
    s.add_argument("--demo-years", type=int, default=20000)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("portfolio", help="generate a synthetic portfolio CSV")
    s.add_argument("--n", type=int, default=5000)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--states", nargs="*")
    s.add_argument("--commercial-share", type=float, default=0.18)
    s.add_argument("--out", default="portfolio.csv")
    s.set_defaults(fn=cmd_portfolio)
    s = sub.add_parser("run", help="run an analysis from the command line")
    s.add_argument("--portfolio", default="5000", help="CSV path or number of synthetic locations")
    s.add_argument("--years", type=int, default=20000)
    s.add_argument("--samples", type=int, default=24)
    s.add_argument("--seed", type=int, default=20240601)
    s.add_argument("--perils", nargs="*", default=["TC", "EQ"])
    s.add_argument("--reinsurance", help="JSON file with a programme {contracts: [...]}")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("hazard-curve", help="site hazard curve")
    s.add_argument("--peril", default="TC", choices=["TC", "EQ"])
    s.add_argument("--lat", type=float, required=True)
    s.add_argument("--lon", type=float, required=True)
    s.set_defaults(fn=cmd_hazard_curve)
    s = sub.add_parser("scenario", help="run a historical analog against a portfolio")
    s.add_argument("--analog", required=True)
    s.add_argument("--portfolio", default="5000")
    s.add_argument("--samples", type=int, default=1000)
    s.set_defaults(fn=cmd_scenario)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
