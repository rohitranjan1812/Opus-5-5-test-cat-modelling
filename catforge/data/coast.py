"""Coarse US Gulf & Atlantic coastline (Rio Grande → Maine) used as landfall "gates", and a land mask.

Vertices are (lon, lat, region) traversed so that land is always on the LEFT of the direction of
travel; the landward normal of a segment is therefore ``bearing - 90°``.  Resolution is ~30-80 km,
adequate for landfall sampling and over-land decay; it is not a cartographic product.

Regional climatology (``REGIONS``) is an *illustrative* calibration to the order of magnitude of
HURDAT2-era US landfall statistics: annual hurricane landfall rate, climatological storm heading at
landfall, and a scale factor on the landfall-intensity distribution (cooler SSTs further north).
"""

from __future__ import annotations

COASTLINE: list[tuple[float, float, str]] = [
    # --- Texas (south) ---
    (-97.15, 25.96, "TX_S"), (-97.28, 26.55, "TX_S"), (-97.22, 27.10, "TX_S"), (-97.05, 27.83, "TX_S"),
    (-96.40, 28.43, "TX_N"),
    # --- Texas (north / upper coast) ---
    (-95.95, 28.60, "TX_N"), (-95.60, 28.75, "TX_N"), (-95.30, 28.93, "TX_N"), (-94.78, 29.28, "TX_N"),
    (-94.40, 29.55, "TX_N"),
    # --- Louisiana ---
    (-93.87, 29.70, "LA"), (-93.32, 29.77, "LA"), (-92.60, 29.60, "LA"), (-91.85, 29.48, "LA"),
    (-91.30, 29.25, "LA"), (-90.66, 29.24, "LA"), (-90.00, 29.20, "LA"), (-89.42, 28.93, "LA"),
    (-89.15, 29.30, "LA"), (-89.50, 29.65, "LA"), (-89.60, 30.05, "LA"),
    # --- Mississippi / Alabama ---
    (-89.35, 30.30, "MS"), (-89.09, 30.36, "MS"), (-88.90, 30.39, "MS"), (-88.55, 30.34, "MS"),
    (-88.11, 30.25, "AL"), (-87.70, 30.25, "AL"),
    # --- Florida panhandle ---
    (-87.15, 30.33, "FL_NW"), (-86.50, 30.38, "FL_NW"), (-85.80, 30.17, "FL_NW"), (-85.42, 29.94, "FL_NW"),
    (-85.35, 29.68, "FL_NW"), (-84.98, 29.72, "FL_NW"), (-84.55, 29.90, "FL_NW"),
    # --- Big Bend ---
    (-84.20, 30.07, "FL_BB"), (-83.40, 29.67, "FL_BB"), (-83.03, 29.14, "FL_BB"), (-82.70, 28.90, "FL_BB"),
    # --- Southwest Florida ---
    (-82.80, 28.15, "FL_SW"), (-82.83, 27.97, "FL_SW"), (-82.74, 27.72, "FL_SW"), (-82.72, 27.50, "FL_SW"),
    (-82.58, 27.30, "FL_SW"), (-82.45, 27.10, "FL_SW"), (-82.26, 26.75, "FL_SW"), (-82.10, 26.45, "FL_SW"),
    (-81.95, 26.44, "FL_SW"), (-81.81, 26.14, "FL_SW"), (-81.72, 25.93, "FL_SW"), (-81.40, 25.85, "FL_SW"),
    # --- South-east Florida ---
    (-81.10, 25.12, "FL_SE"), (-80.70, 25.15, "FL_SE"), (-80.40, 25.20, "FL_SE"), (-80.33, 25.45, "FL_SE"),
    (-80.13, 25.77, "FL_SE"), (-80.10, 26.12, "FL_SE"), (-80.07, 26.35, "FL_SE"), (-80.03, 26.70, "FL_SE"),
    # --- East Florida ---
    (-80.07, 26.95, "FL_E"), (-80.17, 27.20, "FL_E"), (-80.30, 27.45, "FL_E"), (-80.37, 27.65, "FL_E"),
    (-80.55, 28.05, "FL_E"), (-80.58, 28.45, "FL_E"), (-80.90, 29.00, "FL_E"), (-81.00, 29.22, "FL_E"),
    (-81.28, 29.90, "FL_E"), (-81.39, 30.29, "FL_E"),
    # --- Georgia ---
    (-81.43, 30.68, "GA"), (-81.37, 31.15, "GA"), (-81.20, 31.45, "GA"),
    # --- South Carolina ---
    (-80.84, 32.00, "SC"), (-80.70, 32.18, "SC"), (-80.30, 32.50, "SC"), (-79.90, 32.73, "SC"),
    (-79.37, 33.00, "SC"), (-79.15, 33.35, "SC"), (-78.85, 33.70, "SC"),
    # --- North Carolina ---
    (-78.55, 33.85, "NC"), (-77.95, 33.85, "NC"), (-77.80, 34.20, "NC"), (-77.55, 34.45, "NC"),
    (-76.53, 34.60, "NC"), (-75.95, 35.10, "NC"), (-75.53, 35.23, "NC"), (-75.62, 35.95, "NC"),
    (-75.80, 36.40, "NC"),
    # --- Mid-Atlantic ---
    (-75.87, 36.55, "MA"), (-75.97, 36.85, "MA"), (-75.97, 37.12, "MA"), (-75.35, 37.90, "MA"),
    (-75.08, 38.35, "MA"), (-75.07, 38.72, "MA"),
    # --- New Jersey ---
    (-75.09, 38.80, "NJ"), (-74.95, 38.93, "NJ"), (-74.42, 39.36, "NJ"), (-74.10, 39.76, "NJ"),
    # --- New York ---
    (-74.00, 40.45, "NY"), (-73.85, 40.57, "NY"), (-73.65, 40.58, "NY"), (-73.20, 40.63, "NY"),
    (-72.65, 40.80, "NY"),
    # --- New England ---
    (-71.86, 41.07, "NE"), (-71.90, 41.30, "NE"), (-71.49, 41.36, "NE"), (-71.30, 41.45, "NE"),
    (-70.67, 41.52, "NE"), (-69.95, 41.67, "NE"), (-69.97, 41.93, "NE"), (-70.10, 42.07, "NE"),
    (-70.62, 41.95, "NE"), (-70.95, 42.32, "NE"), (-70.60, 42.65, "NE"), (-70.80, 42.80, "NE"),
    # --- Maine ---
    (-70.70, 43.07, "ME"), (-70.20, 43.62, "ME"), (-69.50, 43.85, "ME"), (-68.80, 44.10, "ME"),
    (-68.20, 44.33, "ME"), (-67.50, 44.55, "ME"), (-66.98, 44.90, "ME"),
]

# Closure of the land polygon: inland rim and the Mexican Gulf coast.
_MEXICO = [(-105.0, 20.0), (-96.9, 20.0), (-97.4, 21.0), (-97.85, 22.3), (-97.75, 23.8), (-97.5, 25.0)]
_INLAND = [(-66.98, 47.5), (-105.0, 47.5)]


def land_polygon() -> tuple[list[float], list[float]]:
    pts = _MEXICO + [(lo, la) for lo, la, _ in COASTLINE] + _INLAND
    return [p[0] for p in pts], [p[1] for p in pts]


# region: (annual hurricane landfall rate, climatological heading at landfall [deg], intensity scale,
#          display name)
REGIONS: dict[str, tuple[float, float, float, str]] = {
    "TX_S": (0.13, 300.0, 1.00, "South Texas"),
    "TX_N": (0.18, 330.0, 1.00, "Upper Texas"),
    "LA": (0.28, 355.0, 1.00, "Louisiana"),
    "MS": (0.08, 5.0, 1.00, "Mississippi"),
    "AL": (0.09, 10.0, 1.00, "Alabama"),
    "FL_NW": (0.16, 10.0, 1.00, "Florida Panhandle"),
    "FL_BB": (0.04, 30.0, 0.90, "Florida Big Bend"),
    "FL_SW": (0.13, 35.0, 1.00, "Southwest Florida"),
    "FL_SE": (0.19, 285.0, 1.05, "Southeast Florida"),
    "FL_E": (0.06, 300.0, 0.95, "East Florida"),
    "GA": (0.03, 320.0, 0.90, "Georgia"),
    "SC": (0.13, 335.0, 0.95, "South Carolina"),
    "NC": (0.23, 10.0, 0.90, "North Carolina"),
    "MA": (0.05, 15.0, 0.80, "Mid-Atlantic"),
    "NJ": (0.02, 350.0, 0.75, "New Jersey"),
    "NY": (0.06, 10.0, 0.78, "New York"),
    "NE": (0.09, 15.0, 0.75, "New England"),
    "ME": (0.025, 20.0, 0.60, "Maine"),
}
