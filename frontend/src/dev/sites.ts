// Peril-agnostic view of the exposed buildings in a development payload (what the building-level
// layers, the exposure table and the street-view card need).
export interface SiteAccess {
  n: number
  lat: number[]; lon: number[]; elev: number[]
  id: string[]; cls: string[]; tiv: number[]
  finalLoss: number[]; finalDamage: number[]
  /** damage ratio reached by time t */
  damage: (i: number, t: number) => number
  /** inundation depth (m) at time t — hurricanes; 0 for earthquakes */
  depth: (i: number, t: number) => number
  /** live hazard intensity for labels, and a 0..1 normalised value for halos */
  intensity: (i: number, t: number) => { text: string; x: number }
}
