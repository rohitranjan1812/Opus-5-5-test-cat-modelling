// Payloads of POST /api/develop and POST /api/develop/seismogram (see catforge/physics/develop.py).
import type { B64, GridSpec, Track } from './physics'

export interface City { name: string; state: string; lat: number; lon: number; weight: number }
export interface FieldPayload extends GridSpec { h_km: number; values: B64 }
export interface RealizationPayload { eta: number; sigma_w: number; grf: { nu: number; range_km: number }; field?: FieldPayload }

export interface TcDev {
  peril: 'TC'; name: string; seed: number; t0: number; t1: number; frame_h: number; frames: number[]
  bbox: [number, number, number, number]; constants: Record<string, unknown>
  track: Track; realization: RealizationPayload & { field: FieldPayload }; event: Record<string, unknown>
  land: B64
  probes: { lat: number; lon: number; t: number; gust: number }[]
  rain?: GridSpec & { t: number[]; frames: B64; max_mm: number; model: string }
  surge?: (GridSpec & {
    t: number[]; frames_cm: B64; z: B64; peak_m: number; peak_lat: number; peak_lon: number; inundated_km2: number; model: string
    max: GridSpec & { eta_cm: B64; land_depth_cm: B64 }
  }) | { error: string }
  sites: null | {
    n: number; loc_id: string[]; lat: number[]; lon: number[]; tiv: number[]; construction: string[]; occupancy: string[]
    elev: number[]; gust: B64; damage: B64; surge_depth: B64; gu: number[]; damage_final: number[]
  }
  loss: null | { t: number[]; gu: number[]; gu_wind_only: number[]; n_damaged: number[]; tiv_affected: number }
  cities: City[]
  stats: { vmax_landfall: number; min_pressure_hpa: number; peak_surge_m: number | null; max_rain_mm: number | null; final_gu: number | null; n_sites: number }
}

export interface EqDev {
  peril: 'EQ'; name: string; seed: number; bbox: [number, number, number, number]; event: Record<string, unknown>
  fault: {
    nL: number; nW: number; L_km: number; W_km: number; dip: number; ztor: number; mech: string; M0: number; Mw: number
    D_mean: number; D_max: number; vr_kms: number; duration_s: number
    hypo: { lat: number; lon: number; depth: number; along: number }
    corner_lat: number[][]; corner_lon: number[][]; corner_depth_km: number[][]; slip_m: number[][]; t_rupture_s: number[][]
    ring: { lat: number[]; lon: number[]; trace_lat?: number[]; trace_lon?: number[] }
  }
  grid: GridSpec & { t_p: B64; t_s: B64; t_end: B64; t_peak: B64; pga: B64; pga_median: B64; elev: B64 }
  realization: RealizationPayload
  waves: { alpha_kms: number; beta_kms: number }
  t_max: number
  sites: null | {
    n: number; loc_id: string[]; lat: number[]; lon: number[]; tiv: number[]; vs30: number[]; construction: string[]
    elev: number[]; pga: number[]; pga_median: number[]; t_p: number[]; t_s: number[]; t_peak: number[]; damage: number[]; gu: number[]
  }
  loss: null | { t: number[]; gu: number[]; n_damaged: number[]; tiv_affected: number }
  cities: City[]
  stats: { Mw: number; M0: number; rupture_duration_s: number; max_pga_g: number; final_gu: number | null; n_sites: number }
}

export type DevPayload = TcDev | EqDev

export interface Seismogram {
  lat: number; lon: number; vs30: number; dt: number; acc_g: number[]; vel_cms: number[]; pga_g: number; pgv_cms: number
  t_p: number; t_s: number; periods: number[]; psa_g: number[]; freq: number[]; fas: number[]
  gmpe_median_g: number; gmpe_lo_g: number; gmpe_hi_g: number; n_subfaults: number; epicentral_km: number
  method: string; params: Record<string, number | string>; Mw: number
}
