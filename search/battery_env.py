"""
search/battery_env.py - Environment and state-transition model for battery storage optimization.
Includes efficiency losses, SoC bounds, grid power limit, and admissible heuristic.
"""

import math
from typing import List, Tuple, Dict, Any, Optional

class BatteryEnv:
    def __init__(self,
                 capacity_kwh: float = 10.0,
                 max_charge_kw: float = 3.0,
                 max_discharge_kw: float = 3.0,
                 charge_eff: float = 0.95,
                 discharge_eff: float = 0.95,
                 min_soc_kwh: float = 1.0,
                 max_soc_kwh: float = 10.0,
                 initial_soc_kwh: float = 2.0,
                 soc_step_kwh: float = 0.5,
                 power_limit_kw: float = 5.0,
                 penalty_rate: float = 100.0):
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        self.roundtrip_eff = charge_eff * discharge_eff
        self.min_soc_kwh = min_soc_kwh
        self.max_soc_kwh = max_soc_kwh
        self.initial_soc_kwh = initial_soc_kwh
        self.soc_step_kwh = soc_step_kwh
        self.power_limit_kw = power_limit_kw
        self.penalty_rate = penalty_rate

    def get_initial_state(self) -> Tuple[int, float]:
        """Returns initial state (hour=0, initial_soc)."""
        soc = round(round(self.initial_soc_kwh / self.soc_step_kwh) * self.soc_step_kwh, 2)
        return (0, soc)

    def is_goal(self, state: Tuple[int, float]) -> bool:
        """Goal is reached when hour == 24."""
        return state[0] >= 24

    def get_actions(self, state: Tuple[int, float], demand: float) -> List[float]:
        """
        Returns allowed battery power actions in kW for current state.
        Positive: charging (draws power from grid into battery).
        Negative: discharging (delivers power from battery to reduce demand).
        Zero: idle.
        """
        t, soc = state
        actions = [0.0]  # Idle is always an action
        
        # Discretized power levels in steps of 0.5 kW or 1.0 kW
        step = 0.5
        
        # 1. Discharging actions: p_dis in (step .. max_discharge)
        # Power delivered cannot exceed household demand (no grid export in standard residential)
        max_possible_dis = min(self.max_discharge_kw, demand)
        p = step
        while p <= max_possible_dis + 1e-5:
            # Energy needed from battery: p / discharge_eff
            soc_needed = p / self.discharge_eff
            if soc - soc_needed >= self.min_soc_kwh - 1e-5:
                actions.append(-round(p, 2))
            p += step
            
        # 2. Charging actions: p_chg in (step .. max_charge)
        p = step
        while p <= self.max_charge_kw + 1e-5:
            # Energy stored into battery: p * charge_eff
            soc_added = p * self.charge_eff
            if soc + soc_added <= self.max_soc_kwh + 1e-5:
                actions.append(round(p, 2))
            p += step
            
        return actions

    def step(self, state: Tuple[int, float], action: float, demand: float, price: float) -> Tuple[Tuple[int, float], float, Dict[str, Any]]:
        """
        Executes action from state (hour, soc).
        Returns: (next_state, step_cost, info)
        """
        t, soc = state
        p_batt = action
        
        if p_batt > 0:
            # Charging
            delta_soc = p_batt * self.charge_eff
            new_soc = min(self.max_soc_kwh, soc + delta_soc)
            grid_power = demand + p_batt
        elif p_batt < 0:
            # Discharging
            delivered = -p_batt
            delta_soc = delivered / self.discharge_eff
            new_soc = max(self.min_soc_kwh, soc - delta_soc)
            grid_power = max(0.0, demand - delivered)
        else:
            # Idle
            new_soc = soc
            grid_power = demand
            
        # Discretize new_soc to keep state space clean
        new_soc = round(round(new_soc / self.soc_step_kwh) * self.soc_step_kwh, 2)
        next_state = (t + 1, new_soc)
        
        # Simulated cost: price * grid electricity bought
        base_cost = grid_power * price
        
        # Violation penalty if grid power exceeds limit
        violation = max(0.0, grid_power - self.power_limit_kw)
        step_cost = base_cost + violation * self.penalty_rate
        
        info = {
            "hour": t,
            "demand": demand,
            "action": p_batt,
            "grid_power": grid_power,
            "price": price,
            "base_cost": base_cost,
            "violation": violation,
            "new_soc": new_soc
        }
        return next_state, step_cost, info

    def heuristic(self, state: Tuple[int, float], demands: List[float], prices: List[float]) -> float:
        """
        Admissible and consistent heuristic for A*.
        h(t, b) = max(0, remaining_demand - usable_battery_energy) * min_future_price
        Guaranteed to never overestimate because electricity cannot be purchased for less
        than the cheapest remaining price, and cannot require less grid power than what the
        battery physically cannot cover.
        """
        t, soc = state
        if t >= 24:
            return 0.0
            
        rem_demand = sum(demands[t:])
        usable_battery_kwh = max(0.0, soc - self.min_soc_kwh) * self.discharge_eff
        
        uncovered_demand = max(0.0, rem_demand - usable_battery_kwh)
        min_future_price = min(prices[t:])
        
        return uncovered_demand * min_future_price
