"""Illustrative seismic source model: named fault sources + rectangular background area sources.

Rates follow truncated Gutenberg-Richter recurrence ``N(M >= m)`` between ``mmin`` and ``mmax``.
Geometry and rates are *order-of-magnitude* representations of well-known US sources (UCERF-like
ballpark values), intended for demonstration and methodology development — not for pricing.
"""

# name, trace [(lon, lat), ...], mmin, mmax, annual rate of M>=mmin, b-value, pseudo-depth h (km), region
FAULTS: list[dict] = [
    {"name": "San Andreas (South)", "trace": [(-115.60, 33.30), (-116.50, 33.95), (-117.40, 34.25),
                                               (-118.50, 34.80), (-119.50, 35.10), (-120.40, 35.90)],
     "mmin": 6.5, "mmax": 8.0, "rate": 0.030, "b": 0.6, "h": 4.5, "region": "Southern California"},
    {"name": "San Andreas (North)", "trace": [(-121.50, 36.85), (-122.50, 37.70), (-123.00, 38.10),
                                               (-123.70, 38.95), (-124.10, 40.20)],
     "mmin": 6.5, "mmax": 7.9, "rate": 0.015, "b": 0.6, "h": 4.5, "region": "Northern California"},
    {"name": "Hayward-Rodgers Creek", "trace": [(-121.80, 37.45), (-122.10, 37.75), (-122.30, 37.95),
                                                 (-122.70, 38.35)],
     "mmin": 6.2, "mmax": 7.3, "rate": 0.012, "b": 0.7, "h": 4.5, "region": "Northern California"},
    {"name": "Calaveras", "trace": [(-121.40, 36.90), (-121.80, 37.40), (-121.95, 37.75)],
     "mmin": 6.0, "mmax": 7.0, "rate": 0.010, "b": 0.8, "h": 4.5, "region": "Northern California"},
    {"name": "San Jacinto", "trace": [(-115.90, 32.90), (-116.60, 33.60), (-117.30, 34.10)],
     "mmin": 6.2, "mmax": 7.5, "rate": 0.012, "b": 0.7, "h": 4.5, "region": "Southern California"},
    {"name": "Newport-Inglewood", "trace": [(-117.90, 33.60), (-118.40, 34.05)],
     "mmin": 6.0, "mmax": 7.2, "rate": 0.006, "b": 0.8, "h": 4.5, "region": "Southern California"},
    {"name": "Puente Hills (blind thrust)", "trace": [(-118.30, 34.00), (-117.90, 34.00)],
     "mmin": 6.0, "mmax": 7.1, "rate": 0.005, "b": 0.8, "h": 6.0, "region": "Southern California"},
    {"name": "Sierra Madre-Cucamonga", "trace": [(-118.50, 34.28), (-118.00, 34.20), (-117.55, 34.15)],
     "mmin": 6.0, "mmax": 7.3, "rate": 0.006, "b": 0.8, "h": 5.0, "region": "Southern California"},
    {"name": "Hollywood-Santa Monica", "trace": [(-118.55, 34.03), (-118.25, 34.10)],
     "mmin": 6.0, "mmax": 7.0, "rate": 0.004, "b": 0.8, "h": 5.0, "region": "Southern California"},
    {"name": "Cascadia Subduction Zone", "trace": [(-123.90, 40.50), (-124.00, 42.50), (-123.80, 45.00),
                                                    (-124.00, 47.50), (-124.20, 48.50)],
     "mmin": 8.0, "mmax": 9.1, "rate": 0.008, "b": 0.5, "h": 20.0, "region": "Pacific Northwest"},
    {"name": "Seattle Fault", "trace": [(-122.80, 47.55), (-122.20, 47.60)],
     "mmin": 6.3, "mmax": 7.3, "rate": 0.001, "b": 0.8, "h": 5.0, "region": "Pacific Northwest"},
    {"name": "Wasatch (Salt Lake City)", "trace": [(-111.80, 40.30), (-111.85, 40.80), (-111.95, 41.30)],
     "mmin": 6.3, "mmax": 7.3, "rate": 0.003, "b": 0.7, "h": 5.0, "region": "Intermountain West"},
    {"name": "New Madrid", "trace": [(-90.00, 35.80), (-89.60, 36.30), (-89.30, 36.80)],
     "mmin": 6.5, "mmax": 7.7, "rate": 0.004, "b": 0.7, "h": 6.0, "region": "Central US"},
    {"name": "Charleston", "trace": [(-80.35, 32.85), (-80.05, 33.15)],
     "mmin": 6.3, "mmax": 7.3, "rate": 0.0015, "b": 0.8, "h": 6.0, "region": "South Carolina"},
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
