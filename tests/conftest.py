import pytest

from catforge.engine import AnalysisConfig, CatModel
from catforge.exposure import generate_portfolio
from catforge.hazard import generate_eq_catalog, generate_tc_catalog


@pytest.fixture(scope="session")
def small_model():
    return CatModel({"TC": generate_tc_catalog(700, 150, seed=5),
                     "EQ": generate_eq_catalog(seed=3, area_position_density=0.3)})


@pytest.fixture(scope="session")
def small_portfolio():
    return generate_portfolio(500, seed=21, states=["FL", "CA", "TX", "LA", "NC"])


@pytest.fixture(scope="session")
def small_result(small_model, small_portfolio):
    cfg = AnalysisConfig(n_years=4000, elt_samples=16, seed=7, reinsurance={"contracts": [
        {"name": "L1", "type": "cat_xl", "attachment": 5e6, "limit": 2e7, "reinstatements": 1}]})
    return small_model.run(small_portfolio, cfg)
