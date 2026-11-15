"""
csp/appliances.py - Appliance data models and default specifications.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Appliance:
    name: str
    power_kw: float
    duration_hours: int
    earliest_start: int
    latest_start: int
    circuit: int
    predecessor: Optional[str] = None
    profile: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.profile:
            self.profile = [self.power_kw] * self.duration_hours

def get_default_appliances() -> Dict[str, Appliance]:
    """Returns the standard 5-appliance test suite."""
    return {
        "washer": Appliance(
            name="washer",
            power_kw=1.5,
            duration_hours=2,
            earliest_start=7,
            latest_start=20,
            circuit=1
        ),
        "dryer": Appliance(
            name="dryer",
            power_kw=2.2,
            duration_hours=2,
            earliest_start=8,
            latest_start=21,
            circuit=1,
            predecessor="washer"
        ),
        "dishwasher": Appliance(
            name="dishwasher",
            power_kw=1.8,
            duration_hours=2,
            earliest_start=12,
            latest_start=22,
            circuit=2
        ),
        "ev_charger": Appliance(
            name="ev_charger",
            power_kw=3.0,
            duration_hours=3,
            earliest_start=0,
            latest_start=20,
            circuit=3
        ),
        "oven": Appliance(
            name="oven",
            power_kw=2.5,
            duration_hours=1,
            earliest_start=17,
            latest_start=21,
            circuit=2  # Shares circuit 2 with dishwasher
        )
    }

def get_scaled_appliances(n: int = 5) -> Dict[str, Appliance]:
    """Generates 3, 5, or 8 appliances for scale experiments."""
    base = get_default_appliances()
    if n == 3:
        return {
            "washer": base["washer"],
            "dryer": base["dryer"],
            "dishwasher": base["dishwasher"]
        }
    elif n == 5:
        return base
    elif n == 8:
        # Add 3 more appliances for scaling experiments
        res = dict(base)
        res["heat_pump"] = Appliance(
            name="heat_pump",
            power_kw=2.0,
            duration_hours=3,
            earliest_start=10,
            latest_start=18,
            circuit=4
        )
        res["pool_pump"] = Appliance(
            name="pool_pump",
            power_kw=1.2,
            duration_hours=4,
            earliest_start=9,
            latest_start=16,
            circuit=5
        )
        res["vacuum_robot"] = Appliance(
            name="vacuum_robot",
            power_kw=0.8,
            duration_hours=2,
            earliest_start=8,
            latest_start=19,
            circuit=1
        )
        return res
    else:
        return base
