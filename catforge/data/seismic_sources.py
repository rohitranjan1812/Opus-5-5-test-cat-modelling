"""Illustrative seismic source model: named fault sources + rectangular background area sources.

Rates follow truncated Gutenberg-Richter recurrence ``N(M >= m)`` between ``mmin`` and ``mmax``.
Geometry and rates are *order-of-magnitude* representations of well-known US sources (UCERF-like
ballpark values), intended for demonstration and methodology development — not for pricing.
"""

# Fault geometry follows the right-hand rule: the fault dips to the RIGHT of the trace direction.
# dip [deg], ztor = depth to top of rupture [km], zbot = base of seismogenic zone [km],
# mech: SS strike-slip, RV reverse/thrust, NM normal, SUB subduction interface.
# h is the GMPE pseudo-depth (km); rate is the annual rate of M >= mmin.
FAULTS: list[dict] = [
    {"name": "San Andreas (South)", "trace": [(-115.60, 33.30), (-116.50, 33.95), (-117.40, 34.25),
                                               (-118.50, 34.80), (-119.50, 35.10), (-120.40, 35.90)],
     "mmin": 6.5, "mmax": 8.0, "rate": 0.030, "b": 0.6, "h": 4.5, "region": "Southern California",
     "dip": 90.0, "ztor": 0.0, "zbot": 15.0, "mech": "SS"},
    {"name": "San Andreas (North)", "trace": [(-121.50, 36.85), (-122.50, 37.70), (-123.00, 38.10),
                                               (-123.70, 38.95), (-124.10, 40.20)],
     "mmin": 6.5, "mmax": 7.9, "rate": 0.015, "b": 0.6, "h": 4.5, "region": "Northern California",
     "dip": 90.0, "ztor": 0.0, "zbot": 12.0, "mech": "SS"},
    {"name": "Hayward-Rodgers Creek", "trace": [(-121.80, 37.45), (-122.10, 37.75), (-122.30, 37.95),
                                                 (-122.70, 38.35)],
     "mmin": 6.2, "mmax": 7.3, "rate": 0.012, "b": 0.7, "h": 4.5, "region": "Northern California",
     "dip": 90.0, "ztor": 0.0, "zbot": 12.0, "mech": "SS"},
    {"name": "Calaveras", "trace": [(-121.40, 36.90), (-121.80, 37.40), (-121.95, 37.75)],
     "mmin": 6.0, "mmax": 7.0, "rate": 0.010, "b": 0.8, "h": 4.5, "region": "Northern California",
     "dip": 90.0, "ztor": 0.0, "zbot": 12.0, "mech": "SS"},
    {"name": "San Jacinto", "trace": [(-115.90, 32.90), (-116.60, 33.60), (-117.30, 34.10)],
     "mmin": 6.2, "mmax": 7.5, "rate": 0.012, "b": 0.7, "h": 4.5, "region": "Southern California",
     "dip": 90.0, "ztor": 0.0, "zbot": 16.0, "mech": "SS"},
    {"name": "Newport-Inglewood", "trace": [(-117.90, 33.60), (-118.40, 34.05)],
     "mmin": 6.0, "mmax": 7.2, "rate": 0.006, "b": 0.8, "h": 4.5, "region": "Southern California",
     "dip": 90.0, "ztor": 0.0, "zbot": 15.0, "mech": "SS"},
    {"name": "Puente Hills (blind thrust)", "trace": [(-117.90, 34.00), (-118.30, 34.00)],
     "mmin": 6.0, "mmax": 7.1, "rate": 0.005, "b": 0.8, "h": 6.0, "region": "Southern California",
     "dip": 27.0, "ztor": 3.0, "zbot": 15.0, "mech": "RV"},
    {"name": "Sierra Madre-Cucamonga", "trace": [(-117.55, 34.15), (-118.00, 34.20), (-118.50, 34.28)],
     "mmin": 6.0, "mmax": 7.3, "rate": 0.006, "b": 0.8, "h": 5.0, "region": "Southern California",
     "dip": 45.0, "ztor": 0.0, "zbot": 15.0, "mech": "RV"},
    {"name": "Hollywood-Santa Monica", "trace": [(-118.25, 34.10), (-118.55, 34.03)],
     "mmin": 6.0, "mmax": 7.0, "rate": 0.004, "b": 0.8, "h": 5.0, "region": "Southern California",
     "dip": 70.0, "ztor": 0.0, "zbot": 15.0, "mech": "RV"},
    {"name": "Cascadia Subduction Zone", "trace": [(-124.70, 40.30), (-125.00, 42.00), (-125.10, 44.00),
                                                    (-125.30, 46.00), (-125.80, 47.50), (-126.30, 48.60)],
     "mmin": 8.0, "mmax": 9.1, "rate": 0.008, "b": 0.5, "h": 20.0, "region": "Pacific Northwest",
     "dip": 11.0, "ztor": 5.0, "zbot": 30.0, "mech": "SUB"},
    {"name": "Seattle Fault", "trace": [(-122.80, 47.55), (-122.20, 47.60)],
     "mmin": 6.3, "mmax": 7.3, "rate": 0.001, "b": 0.8, "h": 5.0, "region": "Pacific Northwest",
     "dip": 45.0, "ztor": 0.0, "zbot": 20.0, "mech": "RV"},
    {"name": "Wasatch (Salt Lake City)", "trace": [(-111.95, 41.30), (-111.85, 40.80), (-111.80, 40.30)],
     "mmin": 6.3, "mmax": 7.3, "rate": 0.003, "b": 0.7, "h": 5.0, "region": "Intermountain West",
     "dip": 50.0, "ztor": 0.0, "zbot": 15.0, "mech": "NM"},
    {"name": "New Madrid", "trace": [(-90.00, 35.80), (-89.60, 36.30), (-89.30, 36.80)],
     "mmin": 6.5, "mmax": 7.7, "rate": 0.004, "b": 0.7, "h": 6.0, "region": "Central US",
     "dip": 90.0, "ztor": 0.0, "zbot": 15.0, "mech": "SS"},
    {"name": "Charleston", "trace": [(-80.35, 32.85), (-80.05, 33.15)],
     "mmin": 6.3, "mmax": 7.3, "rate": 0.0015, "b": 0.8, "h": 6.0, "region": "South Carolina",
     "dip": 90.0, "ztor": 0.0, "zbot": 15.0, "mech": "SS"},
]

# name, (lon_min, lat_min, lon_max, lat_max), mmin, mmax, rate(M>=mmin), b, h
AREAS: list[dict] = [
    {"name": "Southern California background", "bbox": (-120.5, 32.5, -114.8, 35.5),
     "mmin": 5.0, "mmax": 7.0, "rate": 1.40, "b": 1.0, "h": 6.0, "region": "Southern California"},
    {"name": "Northern California background", "bbox": (-124.3, 35.5, -119.8, 40.5),
     "mmin": 5.0, "mmax": 7.0, "rate": 0.90, "b": 1.0, "h": 6.0, "region": "Northern California"},
    {"name": "Pacific Northwest background", "bbox": (-124.3, 42.0, -121.0, 49.0),
     "mmin": 5.0, "mmax": 7.0, "rate": 0.15, "b": 1.0, "h": 12.0, "region": "Pacific Northwest"},
    {"name": "Intermountain background", "bbox": (-119.5, 36.0, -111.0, 42.0),
     "mmin": 5.0, "mmax": 7.0, "rate": 0.30, "b": 1.0, "h": 6.0, "region": "Intermountain West"},
    {"name": "Central US background", "bbox": (-91.5, 35.0, -88.5, 38.0),
     "mmin": 5.0, "mmax": 6.5, "rate": 0.02, "b": 1.0, "h": 8.0, "region": "Central US"},
]
