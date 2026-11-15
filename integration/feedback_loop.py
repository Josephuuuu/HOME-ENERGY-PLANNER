"""
integration/feedback_loop.py - Integrated orchestrator coordinating NN Forecast -> CSP -> Battery Search.
Implements the 5-loop feedback mechanism:
If Battery Search cannot stay strictly under the grid limit, the clashing appliance schedule
is forbidden and the CSP is re-executed to find an alternative schedule.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from csp.appliances import Appliance, get_default_appliances
from csp.csp_solver import ApplianceCSP
from search.battery_env import BatteryEnv
from search.astar import AStarSearcher
from search.ucs import UCSSearcher

class EnergyPlannerPipeline:
    def __init__(self,
                 appliances: Optional[Dict[str, Appliance]] = None,
                 battery_env: Optional[BatteryEnv] = None,
                 power_limit_kw: float = 5.0,
                 max_feedback_loops: int = 5,
                 search_algorithm: str = "AStar"):
        self.appliances = appliances or get_default_appliances()
        self.power_limit_kw = power_limit_kw
        self.max_feedback_loops = max_feedback_loops
        self.search_algorithm = search_algorithm
        self.battery_env = battery_env or BatteryEnv(power_limit_kw=power_limit_kw)
        
    def plan_day(self,
                 forecasted_baseline: List[float],
                 actual_baseline: Optional[List[float]] = None,
                 prices: Optional[List[float]] = None,
                 csp_method: str = "BT_FC_AC3",
                 force_feedback_test: bool = False) -> Dict[str, Any]:
        """
        Executes the integrated planning loop:
        1. CSP schedules shiftable appliances on top of forecasted baseline
        2. Battery Search optimizes charging/discharging
        3. If search incurs a limit violation, forbid current appliance schedule and re-run CSP
        """
        if prices is None:
            # Default 3-tier TOU pricing
            prices = [
                0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
                0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18,
                0.35, 0.35, 0.35, 0.35, 0.35,
                0.10, 0.10
            ]
            
        forbidden_schedules: List[Dict[str, int]] = []
        feedback_history: List[Dict[str, Any]] = []
        
        loop = 0
        final_solution = None
        
        while loop < self.max_feedback_loops:
            loop += 1
            # Step 1: Run CSP
            csp = ApplianceCSP(
                self.appliances,
                forecasted_baseline,
                power_limit_kw=self.power_limit_kw,
                forbidden_assignments=forbidden_schedules
            )
            appliance_schedule = csp.solve(method=csp_method)
            
            if appliance_schedule is None:
                # Trigger CSP relaxation if no schedule can be found
                diag = csp.diagnose_and_relax(max_steps=10, method=csp_method)
                if not diag["success"]:
                    return {
                        "success": False,
                        "error": "CSP failed to find valid schedule even with relaxation.",
                        "diagnostics": diag,
                        "feedback_loops": loop,
                        "history": feedback_history
                    }
                appliance_schedule = diag["solution"]
                effective_limit = diag["power_limit_kw"]
            else:
                effective_limit = self.power_limit_kw
                
            # Compute aggregate demand (forecasted baseline + scheduled appliances)
            _, total_demand = csp.check_hourly_limit(appliance_schedule)
            
            # Step 2: Battery Search
            # For demonstration of feedback loop in test mode, trigger a synthetic limit check on loop 1
            if force_feedback_test and loop == 1:
                # Force violation on loop 1 to prove feedback mechanism works
                violation_detected = True
                max_viol_hour = 18
                max_viol_val = 1.2
                search_res = {
                    "optimal_cost": 12.50,
                    "actions": [0.0] * 24,
                    "infos": [{"grid_power": total_demand[h], "violation": (1.2 if h == 18 else 0.0), "base_cost": 0.5, "new_soc": 2.0} for h in range(24)],
                    "nodes_expanded": 100,
                    "algorithm": self.search_algorithm
                }
            else:
                if self.search_algorithm == "UCS":
                    searcher = UCSSearcher(self.battery_env)
                else:
                    searcher = AStarSearcher(self.battery_env)
                    
                search_res = searcher.search(total_demand, prices)
                
                # Check if battery plan stayed strictly under limit
                violation_detected = False
                max_viol_hour = -1
                max_viol_val = 0.0
                for info in search_res.get("infos", []):
                    if info["violation"] > 1e-4:
                        violation_detected = True
                        if info["violation"] > max_viol_val:
                            max_viol_val = info["violation"]
                            max_viol_hour = info["hour"]
                            
            feedback_history.append({
                "loop": loop,
                "appliance_schedule": dict(appliance_schedule),
                "violation_detected": violation_detected,
                "max_violation_hour": max_viol_hour,
                "max_violation_kw": round(max_viol_val, 2),
                "simulated_cost": search_res.get("optimal_cost", 0.0),
                "nodes_expanded": search_res.get("nodes_expanded", 0)
            })
            
            if not violation_detected:
                # Success! Battery plan stayed under limit
                final_solution = {
                    "success": True,
                    "feedback_triggered": (loop > 1 or force_feedback_test),
                    "total_loops": loop,
                    "appliance_schedule": appliance_schedule,
                    "search_result": search_res,
                    "forecasted_baseline": forecasted_baseline,
                    "total_demand": total_demand,
                    "actual_baseline": actual_baseline,
                    "prices": prices,
                    "history": feedback_history,
                    "power_limit_kw": effective_limit
                }
                break
            else:
                # Limit violation encountered! Forbid current appliance schedule and re-run CSP
                forbidden_schedules.append(dict(appliance_schedule))
                
        if final_solution is None:
            return {
                "success": False,
                "error": f"Failed to eliminate grid violations after {self.max_feedback_loops} feedback loops.",
                "history": feedback_history
            }
            
        # If actual ground truth baseline is provided, evaluate actual outcome
        if actual_baseline is not None:
            # Reconstruct load with actual baseline + planned appliances + planned battery actions
            actual_grid_loads = []
            actual_costs = []
            for h in range(24):
                app_load = total_demand[h] - forecasted_baseline[h]
                batt_act = final_solution["search_result"]["actions"][h]
                
                if batt_act > 0:
                    grid_p = actual_baseline[h] + app_load + batt_act
                elif batt_act < 0:
                    deliv = min(-batt_act, actual_baseline[h] + app_load)
                    grid_p = max(0.0, actual_baseline[h] + app_load - deliv)
                else:
                    grid_p = actual_baseline[h] + app_load
                    
                cost = grid_p * prices[h]
                actual_grid_loads.append(round(grid_p, 3))
                actual_costs.append(round(cost, 4))
                
            actual_violations = [max(0.0, p - self.power_limit_kw) for p in actual_grid_loads]
            final_solution["actual_evaluation"] = {
                "actual_grid_loads": actual_grid_loads,
                "total_simulated_cost": round(sum(actual_costs), 2),
                "max_grid_load_kw": round(max(actual_grid_loads), 2),
                "total_violations_kwh": round(sum(actual_violations), 3),
                "num_violation_hours": sum(1 for v in actual_violations if v > 1e-4)
            }
            
        return final_solution
