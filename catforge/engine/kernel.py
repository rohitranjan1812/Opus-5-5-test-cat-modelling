"""The Monte Carlo loss kernel.

For each occurrence k of event e (peril p) and each affected site j:

    η_k   = σ_b(p) Φ⁻¹(u(k, ETA))                       inter-event hazard residual (shared)
    x_kj  = √ρ_e Z_k + √ρ_c Z_{k,cell(j)} + √(1-ρ_e-ρ_c) ε_kj     two-level Gaussian copula
    U_kj  = Φ(x_kj)
    D_kj  = F⁻¹_eff(U_kj | ln m_ej + η_k)                 inverse CDF of the convolved damage table
    → coverage losses → site terms → account terms → per-risk → occurrence totals

All normals come from the stateless counter RNG (``catforge.rng``): draws depend only on
(seed, occurrence key, site / cell id), so results are thread-count independent and any subset of
occurrences can be re-simulated exactly (tail allocation, common-random-number what-ifs).
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ..rng import TAG_CELL_BASE, TAG_ETA, TAG_LOC_BASE, TAG_Z, norm_cdf, stream, uniform
from ..rng import norm_ppf_fast as norm_ppf


@nb.njit(inline="always", cache=True)
def _sample_damage(cdf, v, li, li0, dli, u, bin_lo, bin_hi):
    n_grid = cdf.shape[1]
    nbins = cdf.shape[2]
    g = (li - li0) / dli
    if g <= 0.0:
        i0 = 0
        w = 0.0
    elif g >= n_grid - 1:
        i0 = n_grid - 2
        w = 1.0
    else:
        i0 = int(g)
        w = g - i0
    w0 = 1.0 - w
    lo = 0
    hi = nbins - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if w0 * cdf[v, i0, mid] + w * cdf[v, i0 + 1, mid] >= u:
            hi = mid
        else:
            lo = mid + 1
    b = lo
    if bin_hi[b] <= bin_lo[b]:
        return bin_lo[b]
    cb = w0 * cdf[v, i0, b] + w * cdf[v, i0 + 1, b]
    cp = 0.0
    if b > 0:
        cp = w0 * cdf[v, i0, b - 1] + w * cdf[v, i0 + 1, b - 1]
    frac = 0.5
    if cb > cp:
        frac = (u - cp) / (cb - cp)
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
    return bin_lo[b] + frac * (bin_hi[b] - bin_lo[b])


@nb.njit(parallel=True, cache=True)
def loss_kernel(occ_event, occ_key, occ_w, seed,
                ev_ptr, pair_loc, pair_logi, ev_peril,
                p_sig_b, p_rho_e, p_rho_c, p_dmg_scale,
                loc_vuln, loc_cell, tiv, loc_ded, loc_lim, loc_pol, loc_grp,
                cdf, v_li0, v_dli, bin_lo, bin_hi, cov,
                pol_ded, pol_lim, pol_att, pol_llim, pol_share,
                pr_ret, pr_lim,
                n_grp, want_loc, detail_ptr, max_pairs, n_chunks):
    K = occ_event.shape[0]
    n_loc = tiv.shape[0]
    gu_out = np.zeros(K)
    gross_out = np.zeros(K)
    pr_out = np.zeros(K)
    grp_rows = K if n_grp > 0 else 1
    grp_out = np.zeros((grp_rows, max(n_grp, 1)))
    loc_cols = n_loc if want_loc else 0
    acc_gross = np.zeros((n_chunks, loc_cols))
    acc_gu = np.zeros((n_chunks, loc_cols))
    want_detail = detail_ptr.shape[0] > 0
    n_det = detail_ptr[K] if want_detail else 0
    det_gross = np.zeros(n_det)
    det_gu = np.zeros(n_det)
    chunk = (K + n_chunks - 1) // n_chunks
    useed = np.uint64(seed)
    for c in nb.prange(n_chunks):
        xbuf = np.empty(max_pairs)
        gbuf = np.empty(max_pairs)
        k_end = min((c + 1) * chunk, K)
        for k in range(c * chunk, k_end):
            e = occ_event[k]
            per = ev_peril[e]
            h = stream(useed, np.uint64(occ_key[k]))
            eta = p_sig_b[per] * norm_ppf(uniform(h, TAG_ETA))
            z = norm_ppf(uniform(h, TAG_Z))
            re = p_rho_e[per]
            rc = p_rho_c[per]
            sre = math.sqrt(re)
            src = math.sqrt(rc)
            sri = math.sqrt(max(1.0 - re - rc, 0.0))
            dsc = p_dmg_scale[per]
            a = ev_ptr[e]
            b = ev_ptr[e + 1]
            gu_tot = 0.0
            gross_tot = 0.0
            pr_tot = 0.0
            p = a
            while p < b:
                pol = loc_pol[pair_loc[p]]
                q = p
                s_x = 0.0
                while q < b and loc_pol[pair_loc[q]] == pol:
                    j = pair_loc[q]
                    v = loc_vuln[j, per]
                    zc = norm_ppf(uniform(h, TAG_CELL_BASE + loc_cell[j]))
                    eps = norm_ppf(uniform(h, TAG_LOC_BASE + j))
                    u = norm_cdf(sre * z + src * zc + sri * eps)
                    d = _sample_damage(cdf, v, pair_logi[q] + eta, v_li0[v], v_dli[v], u, bin_lo, bin_hi)
                    d = min(1.0, d * dsc)
                    gu = tiv[j, 0] * d
                    if tiv[j, 1] > 0.0:
                        gu += tiv[j, 1] * min(1.0, cov[v, 0] * d ** cov[v, 1])
                    if tiv[j, 2] > 0.0:
                        gu += tiv[j, 2] * min(1.0, (d / cov[v, 2]) ** cov[v, 3])
                    xl = min(max(gu - loc_ded[j, per], 0.0), loc_lim[j, per])
                    xbuf[q - p] = xl
                    gbuf[q - p] = gu
                    s_x += xl
                    gu_tot += gu
                    q += 1
                g1 = min(max(s_x - pol_ded[pol, per], 0.0), pol_lim[pol, per])
                g2 = min(max(g1 - pol_att[pol], 0.0), pol_llim[pol])
                gp = g2 * pol_share[pol]
                gross_tot += gp
                if pr_lim > 0.0:
                    pr_tot += min(max(gp - pr_ret, 0.0), pr_lim)
                if n_grp > 0 or want_loc or want_detail:
                    f = gp / s_x if s_x > 0.0 else 0.0
                    for t in range(q - p):
                        j = pair_loc[p + t]
                        gj = xbuf[t] * f
                        if n_grp > 0:
                            grp_out[k, loc_grp[j]] += gj
                        if want_loc:
                            acc_gross[c, j] += occ_w[k] * gj
                            acc_gu[c, j] += occ_w[k] * gbuf[t]
                        if want_detail:
                            det_gross[detail_ptr[k] + (p - a) + t] = gj
                            det_gu[detail_ptr[k] + (p - a) + t] = gbuf[t]
                p = q
            gu_out[k] = gu_tot
            gross_out[k] = gross_tot
            pr_out[k] = pr_tot
    loc_gross = np.zeros(loc_cols)
    loc_gu = np.zeros(loc_cols)
    for c in range(n_chunks):
        for j in range(loc_cols):
            loc_gross[j] += acc_gross[c, j]
            loc_gu[j] += acc_gu[c, j]
    return gu_out, gross_out, pr_out, grp_out, loc_gross, loc_gu, det_gross, det_gu
