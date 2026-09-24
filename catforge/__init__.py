"""CatForge — a multi-peril catastrophe modelling platform.

Quick start::

    from catforge import CatModel, AnalysisConfig, generate_portfolio
    model = CatModel.default()
    pf = generate_portfolio(5000)
    res = model.run(pf, AnalysisConfig(n_years=20000))
    res.summary["rp_table"]
"""

__version__ = "0.1.0"

from .engine import AnalysisConfig, AnalysisResult, CatModel  # noqa: E402
from .exposure import Portfolio, generate_portfolio  # noqa: E402

__all__ = ["__version__", "AnalysisConfig", "AnalysisResult", "CatModel", "Portfolio", "generate_portfolio"]
