"""The 10 stations frozen in SPEC section 2. Coordinates are the ACIS station coordinates."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    sid: str      # ACIS id and ASOS id
    city: str
    office: str   # NWS forecast office that writes the Area Forecast Discussion (AFD<office>)
    lat: float
    lon: float
    tz: str


STATIONS = [
    Station("NYC", "New York", "OKX", 40.779, -73.969, "America/New_York"),
    Station("BOS", "Boston", "BOX", 42.361, -71.010, "America/New_York"),
    Station("ATL", "Atlanta", "FFC", 33.630, -84.442, "America/New_York"),
    Station("MIA", "Miami", "MFL", 25.788, -80.317, "America/New_York"),
    Station("ORD", "Chicago", "LOT", 41.960, -87.932, "America/Chicago"),
    Station("DFW", "Dallas-Fort Worth", "FWD", 32.897, -97.022, "America/Chicago"),
    Station("DEN", "Denver", "BOU", 39.847, -104.656, "America/Denver"),
    Station("PHX", "Phoenix", "PSR", 33.428, -112.004, "America/Phoenix"),
    Station("LAX", "Los Angeles", "LOX", 33.938, -118.387, "America/Los_Angeles"),
    Station("SEA", "Seattle", "SEW", 47.445, -122.314, "America/Los_Angeles"),
]
BY_SID = {s.sid: s for s in STATIONS}
