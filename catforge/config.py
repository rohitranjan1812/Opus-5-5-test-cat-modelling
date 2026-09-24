"""Global enumerations shared by all modules."""

PERILS = ("TC", "EQ")
PERIL_INDEX = {p: i for i, p in enumerate(PERILS)}
PERIL_NAMES = {"TC": "Tropical cyclone (wind)", "EQ": "Earthquake (shake)"}

CONSTRUCTION_CLASSES = {
    "WOOD": "Wood frame",
    "MASONRY": "Reinforced masonry / CBS",
    "URM": "Unreinforced masonry",
    "RC": "Reinforced concrete",
    "STEEL": "Steel frame",
    "LIGHT_METAL": "Light metal",
    "MOBILE_HOME": "Manufactured / mobile home",
}

OCCUPANCIES = {
    "RES_SF": "Single-family residential",
    "RES_MF": "Multi-family residential",
    "COMMERCIAL": "Commercial",
    "INDUSTRIAL": "Industrial",
}

TERRAINS = ("coastal", "open", "suburban", "urban")
ROOF_SHAPES = ("hip", "gable", "flat", "unknown")
LINES_OF_BUSINESS = ("personal", "commercial")
