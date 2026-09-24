"""Direct insurance terms compiled to dense arrays for the loss kernel.

Per occurrence, for location j and account (policy) a:
    GU_j   = Σ_c TIV_jc · g_c(D_j)                              (coverage damage functions)
    X_j    = min( (GU_j - ded_j)₊ , lim_j )                      (site all-coverage deductible & limit)
    S_a    = Σ_{j∈a} X_j
    G_a    = share_a · min( (min((S_a - accded_a)₊, acclim_a) - attach_a)₊ , layerlim_a )
Deductibles ≤ 1 are interpreted as a fraction of the site's total TIV; zero limits mean unlimited.
Account-level gross is allocated back to sites pro rata to X_j (additive, Euler-consistent).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PERILS


@dataclass
class FinancialArrays:
    tiv: np.ndarray  # (n_loc, 3)
    loc_ded: np.ndarray  # (n_loc, n_perils) amounts
    loc_lim: np.ndarray  # (n_loc, n_perils)
    loc_pol: np.ndarray  # int32 (n_loc,)
    pol_ids: np.ndarray  # (n_pol,) account ids
    pol_ded: np.ndarray  # (n_pol, n_perils)
    pol_lim: np.ndarray  # (n_pol, n_perils)
    pol_att: np.ndarray  # (n_pol,)
    pol_llim: np.ndarray  # (n_pol,)
    pol_share: np.ndarray  # (n_pol,)

    @property
    def n_pol(self) -> int:
        return int(self.pol_ids.shape[0])

    def max_gross_by_loc(self, peril_idx: int) -> np.ndarray:
        """Upper bound of site gross (pre-account terms) for exposure/beta fitting."""
        t = self.tiv.sum(axis=1)
        return np.minimum(np.maximum(t - self.loc_ded[:, peril_idx], 0.0), self.loc_lim[:, peril_idx])


def _amount(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.where((x > 0) & (x <= 1.0), x * base, x)


def build_financials(locations: pd.DataFrame) -> FinancialArrays:
    L = locations
    tiv = L[["tiv_building", "tiv_contents", "tiv_bi"]].to_numpy(float)
    ttot = tiv.sum(axis=1)
    lim = L["loc_limit"].to_numpy(float)
    lim = np.where(lim > 0, lim, np.inf)
    loc_ded = np.zeros((len(L), len(PERILS)))
    loc_lim = np.zeros((len(L), len(PERILS)))
    for pi, peril in enumerate(PERILS):
        loc_ded[:, pi] = _amount(L[f"ded_{peril.lower()}"].to_numpy(float), ttot)
        loc_lim[:, pi] = lim
    codes, uniq = pd.factorize(L["acc_id"], sort=False)
    first = pd.Series(np.arange(len(L))).groupby(codes).first().to_numpy()
    acc = L.iloc[first]
    acc_tiv = np.bincount(codes, weights=ttot, minlength=len(uniq))
    pol_ded_1 = _amount(acc["acc_ded"].to_numpy(float), acc_tiv)
    pol_lim_1 = acc["acc_limit"].to_numpy(float)
    pol_lim_1 = np.where(pol_lim_1 > 0, pol_lim_1, np.inf)
    llim = acc["acc_layer_limit"].to_numpy(float)
    return FinancialArrays(
        tiv=tiv, loc_ded=loc_ded, loc_lim=loc_lim, loc_pol=codes.astype(np.int32), pol_ids=np.asarray(uniq),
        pol_ded=np.repeat(pol_ded_1[:, None], len(PERILS), axis=1),
        pol_lim=np.repeat(pol_lim_1[:, None], len(PERILS), axis=1),
        pol_att=acc["acc_attach"].to_numpy(float), pol_llim=np.where(llim > 0, llim, np.inf),
        pol_share=acc["acc_share"].to_numpy(float),
    )


def apply_terms_scalar(gu: np.ndarray, ded: float, lim: float) -> np.ndarray:
    """Reference (numpy) implementation of a deductible/limit layer — used in tests."""
    return np.minimum(np.maximum(np.asarray(gu, float) - ded, 0.0), lim)
