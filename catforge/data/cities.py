"""Exposure seeds for the synthetic portfolio generator.

(name, state, lat, lon, relative weight, spread_km).  Weights are loosely proportional to
insured-property concentration in hazard-prone metros (not population), chosen so that a
generated national portfolio has realistic hurricane/earthquake accumulations.
"""

CITIES: list[tuple[str, str, float, float, float, float]] = [
    # Florida
    ("Miami", "FL", 25.76, -80.19, 9.0, 18), ("Fort Lauderdale", "FL", 26.12, -80.14, 6.0, 14),
    ("West Palm Beach", "FL", 26.71, -80.05, 4.5, 14), ("Tampa", "FL", 27.95, -82.46, 6.0, 20),
    ("St. Petersburg", "FL", 27.77, -82.64, 3.5, 10), ("Orlando", "FL", 28.54, -81.38, 5.0, 25),
    ("Jacksonville", "FL", 30.33, -81.66, 3.5, 22), ("Tallahassee", "FL", 30.44, -84.28, 1.0, 12),
    ("Pensacola", "FL", 30.42, -87.22, 1.2, 12), ("Fort Myers", "FL", 26.64, -81.87, 2.8, 14),
    ("Naples", "FL", 26.14, -81.79, 2.2, 10), ("Sarasota", "FL", 27.34, -82.53, 2.5, 12),
    ("Daytona Beach", "FL", 29.21, -81.02, 1.4, 10), ("Key West", "FL", 24.56, -81.78, 0.4, 4),
    ("Panama City", "FL", 30.16, -85.66, 0.8, 10), ("Port St. Lucie", "FL", 27.27, -80.35, 1.8, 12),
    # Texas
    ("Houston", "TX", 29.76, -95.37, 8.0, 30), ("Galveston", "TX", 29.30, -94.80, 1.0, 8),
    ("Corpus Christi", "TX", 27.80, -97.40, 1.5, 12), ("Brownsville", "TX", 25.90, -97.50, 0.8, 10),
    ("Beaumont", "TX", 30.08, -94.13, 0.8, 12), ("San Antonio", "TX", 29.42, -98.49, 2.5, 20),
    ("Austin", "TX", 30.27, -97.74, 2.0, 18),
    # Gulf states
    ("New Orleans", "LA", 29.95, -90.07, 4.0, 15), ("Baton Rouge", "LA", 30.45, -91.15, 1.8, 14),
    ("Lake Charles", "LA", 30.23, -93.22, 0.8, 10), ("Lafayette", "LA", 30.22, -92.02, 1.0, 12),
    ("Gulfport", "MS", 30.37, -89.09, 0.8, 10), ("Biloxi", "MS", 30.40, -88.89, 0.6, 8),
    ("Mobile", "AL", 30.69, -88.04, 1.4, 14), ("Gulf Shores", "AL", 30.25, -87.70, 0.5, 6),
    # South Atlantic
    ("Savannah", "GA", 32.08, -81.09, 1.2, 12), ("Atlanta", "GA", 33.75, -84.39, 4.0, 30),
    ("Charleston", "SC", 32.78, -79.93, 2.2, 14), ("Myrtle Beach", "SC", 33.69, -78.89, 1.3, 12),
    ("Columbia", "SC", 34.00, -81.03, 1.0, 14), ("Hilton Head", "SC", 32.22, -80.75, 0.7, 6),
    ("Wilmington", "NC", 34.23, -77.94, 1.2, 12), ("Raleigh", "NC", 35.78, -78.64, 2.5, 20),
    ("Outer Banks", "NC", 35.90, -75.63, 0.5, 8), ("Norfolk", "VA", 36.85, -76.29, 1.8, 12),
    ("Virginia Beach", "VA", 36.85, -75.98, 1.5, 10), ("Richmond", "VA", 37.54, -77.44, 1.5, 15),
    # North-east
    ("Washington", "DC", 38.91, -77.04, 3.0, 18), ("Baltimore", "MD", 39.29, -76.61, 2.0, 15),
    ("Ocean City", "MD", 38.34, -75.08, 0.4, 5), ("Atlantic City", "NJ", 39.36, -74.42, 1.0, 10),
    ("Philadelphia", "PA", 39.95, -75.17, 3.5, 20), ("New York", "NY", 40.71, -74.01, 10.0, 22),
    ("Long Island", "NY", 40.79, -73.13, 3.5, 25), ("Providence", "RI", 41.82, -71.41, 1.2, 12),
    ("Boston", "MA", 42.36, -71.06, 4.0, 18), ("Cape Cod", "MA", 41.68, -70.20, 0.8, 12),
    ("Hartford", "CT", 41.76, -72.67, 1.5, 15), ("Portland", "ME", 43.66, -70.26, 0.6, 10),
    # California
    ("Los Angeles", "CA", 34.05, -118.24, 12.0, 28), ("San Diego", "CA", 32.72, -117.16, 4.5, 18),
    ("San Francisco", "CA", 37.77, -122.42, 5.0, 8), ("Oakland", "CA", 37.80, -122.27, 3.0, 12),
    ("San Jose", "CA", 37.34, -121.89, 4.0, 14), ("Sacramento", "CA", 38.58, -121.49, 2.5, 16),
    ("Fresno", "CA", 36.74, -119.79, 1.2, 12), ("Riverside", "CA", 33.95, -117.40, 3.0, 20),
    ("Santa Barbara", "CA", 34.42, -119.70, 0.8, 8), ("Bakersfield", "CA", 35.37, -119.02, 0.9, 12),
    ("Palm Springs", "CA", 33.83, -116.55, 0.8, 12), ("Santa Rosa", "CA", 38.44, -122.71, 0.9, 10),
    # Pacific North-west & intermountain
    ("Seattle", "WA", 47.61, -122.33, 4.0, 16), ("Tacoma", "WA", 47.25, -122.44, 1.2, 10),
    ("Olympia", "WA", 47.04, -122.90, 0.5, 8), ("Portland", "OR", 45.52, -122.68, 2.8, 15),
    ("Eugene", "OR", 44.05, -123.09, 0.6, 8), ("Salt Lake City", "UT", 40.76, -111.89, 1.8, 15),
    # Central US (New Madrid)
    ("Memphis", "TN", 35.15, -90.05, 1.5, 15), ("St. Louis", "MO", 38.63, -90.20, 2.0, 18),
    ("Nashville", "TN", 36.16, -86.78, 1.5, 16),
]
