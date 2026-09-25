"""Typed request models (these drive the OpenAPI schema at /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Peril = Literal["TC", "EQ"]


class SyntheticPortfolioRequest(BaseModel):
    n_locations: int = Field(5000, ge=10, le=250_000, description="Number of locations to generate")
    seed: int = 11
    states: list[str] | None = Field(None, description="Restrict to these state codes, e.g. ['FL','CA']")
    commercial_share: float = Field(0.18, ge=0.0, le=1.0)
    name: str | None = None


class ContractModel(BaseModel):
    name: str
    type: Literal["cat_xl", "quota_share", "agg_xl"] = "cat_xl"
    stage: int = Field(1, ge=1, le=10, description="Inuring stage; same-stage contracts share a subject loss")
    attachment: float = Field(0.0, ge=0)
    limit: float = Field(0.0, ge=0, description="Occurrence limit (cat_xl) or aggregate limit (agg_xl); 0 = unlimited")
    reinstatements: int = Field(0, ge=0, le=20)
    reinstatement_rate: float = Field(1.0, ge=0)
    aad: float = Field(0.0, ge=0, description="Annual aggregate deductible")
    aal: float | None = Field(None, ge=0, description="Annual aggregate limit (default limit × (1 + reinstatements))")
    cession: float = Field(0.0, ge=0, le=1)
    event_limit: float | None = Field(None, ge=0)
    placed: float = Field(1.0, gt=0, le=1)
    perils: list[Peril] | None = None
    premium: float | None = Field(None, ge=0, description="Upfront premium; priced technically if omitted")


class ProgramModel(BaseModel):
    contracts: list[ContractModel] = Field(default_factory=list)
    pricing_method: Literal["stdev", "coc"] = "stdev"
    pricing_load: float = Field(0.30, ge=0)
    coc_alpha: float = Field(0.99, gt=0.5, lt=1)


class PerRiskModel(BaseModel):
    retention: float = Field(0.0, ge=0)
    limit: float = Field(0.0, ge=0)


class AnalysisConfigModel(BaseModel):
    name: str = "Analysis"
    perils: list[Peril] = Field(default_factory=lambda: ["TC", "EQ"])
    n_years: int = Field(20000, ge=200, le=1_000_000)
    elt_samples: int = Field(24, ge=2, le=512)
    seed: int = 20240601
    damage_scale: dict[str, float] = Field(default_factory=dict)
    rho_scale: float = Field(1.0, ge=0, le=3)
    sigma_between_scale: float = Field(1.0, ge=0, le=3)
    rate_multiplier: dict[str, float] = Field(default_factory=dict)
    tc_intensity_scale: float = Field(1.0, gt=0.5, lt=1.5)
    frequency: dict[str, dict] = Field(default_factory=dict)
    per_risk: PerRiskModel | None = None
    reinsurance: ProgramModel | None = None
    group_by: Literal["state", "construction", "occupancy", "lob", "terrain"] = "state"
    allocation_rp: float = Field(250.0, ge=2, le=10000)


class AnalysisRequest(BaseModel):
    portfolio_id: str
    config: AnalysisConfigModel = Field(default_factory=AnalysisConfigModel)


class ReinsuranceRequest(BaseModel):
    program: ProgramModel


class OptimizeRequest(BaseModel):
    attach_rps: list[float] = Field(default_factory=lambda: [5, 10, 15, 20, 25, 35, 50, 75])
    exhaust_rps: list[float] = Field(default_factory=lambda: [100, 150, 200, 250, 350, 500])
    reinstatements: int = Field(1, ge=0, le=10)
    pricing_load: float = Field(0.30, ge=0)
    pricing_method: Literal["stdev", "coc"] = "stdev"


class ClimateRequest(BaseModel):
    tc_frequency: float = Field(1.0, gt=0, le=3)
    tc_intensity: float = Field(1.0, gt=0.7, lt=1.3)
    eq_frequency: float = Field(1.0, gt=0, le=3)


class MitigationRequest(BaseModel):
    preset: str | None = None
    filter: dict | None = None
    changes: dict | None = None


class MarginalRequest(BaseModel):
    acc_ids: list[str] = Field(..., min_length=1, max_length=5000)


class ScenarioRequest(BaseModel):
    portfolio_id: str
    analog: str | None = Field(None, description="Key of a historical analog (see /api/scenarios/analogs)")
    peril: Peril | None = None
    params: dict | None = None
    n_samples: int = Field(1000, ge=50, le=20000)
    seed: int = 99


class CatalogRebuildRequest(BaseModel):
    seed: int | None = None
    n_hurricanes: int | None = Field(None, ge=100, le=50000)
    n_tropical_storms: int | None = Field(None, ge=0, le=20000)
    area_position_density: float | None = Field(None, gt=0, le=10)


class DevelopRequest(BaseModel):
    portfolio_id: str | None = Field(None, description="Portfolio whose buildings are tracked through the event")
    analog: str | None = Field(None, description="Historical analog key")
    peril: Peril | None = None
    event_id: int | None = Field(None, description="Stochastic catalog event id (with peril)")
    params: dict | None = Field(None, description="Custom event parameters (with peril)")
    seed: int = Field(1, description="Realization seed (hazard residual field, slip, damage uncertainty)")
    surge: bool = Field(True, description="Run the 2-D storm-surge model (hurricanes)")


class SeismogramRequest(BaseModel):
    analog: str | None = None
    peril: Peril | None = "EQ"
    event_id: int | None = None
    params: dict | None = None
    seed: int = 1
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    vs30: float = Field(400.0, ge=150, le=1500)
