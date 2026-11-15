"""
search/dp_check.py - Exact Dynamic Programming solver for the battery scheduling problem.
Used as ground-truth oracle to verify UCS and A* search correctness and optimality.
"""

from typing import List, Tuple, Dict, Any
from search.battery_env import BatteryEnv

class DPSolver:
    def __init__(self, env: BatteryEnv):
        self.env = env
        
    def solve(self, demands: List[float], prices: List[float]) -> Dict[str, Any]:
        """
        Backward dynamic programming from t = 24 down to t = 0.
        V(t, soc) = min_a [ step_cost(t, soc, a) + V(t+1, next_soc) ]
        """
        # All valid discrete soc levels
        socs = []
        s = self.env.min_soc_kwh
        while s <= self.env.max_soc_kwh + 1e-5:
            socs.append(round(s, 2))
            s += self.env.soc_step_kwh
            
        # V[t][soc] = min cost to go from (t, soc) to end
        # Best action: policy[t][soc]
        V: List[Dict[float, float]] = [{} for _ in range(25)]
        policy: List[Dict[float, float]] = [{} for _ in range(25)]
        
        # Base case: at t=24, cost-to-go is 0.0
        for s in socs:
            V[24][s] = 0.0
            
        # Backward pass
        for t in range(23, -1, -1):
            demand = demands[t]
            price = prices[t]
            for s in socs:
                state = (t, s)
                allowed_actions = self.env.get_actions(state, demand)
                best_cost = float('inf')
                best_action = 0.0
                
                for act in allowed_actions:
                    next_state, step_cost, _ = self.env.step(state, act, demand, price)
                    next_soc = next_state[1]
                    
                    # Look up V[t+1][next_soc]
                    future_cost = V[t + 1].get(next_soc, float('inf'))
                    total_cost = step_cost + future_cost
                    
                    if total_cost < best_cost - 1e-7:
                        best_cost = total_cost
                        best_action = act
                        
                V[t][s] = best_cost
                policy[t][s] = best_action
                
        # Forward reconstruction from initial state
        init_state = self.env.get_initial_state()
        curr_soc = init_state[1]
        trajectory = [init_state]
        actions = []
        infos = []
        total_cost = 0.0
        
        for t in range(24):
            demand = demands[t]
            price = prices[t]
            act = policy[t][curr_soc]
            next_state, step_cost, info = self.env.step((t, curr_soc), act, demand, price)
            total_cost += step_cost
            actions.append(act)
            infos.append(info)
            trajectory.append(next_state)
            curr_soc = next_state[1]
            
        return {
            "optimal_cost": round(total_cost, 4),
            "trajectory": trajectory,
            "actions": actions,
            "infos": infos,
            "algorithm": "Dynamic Programming (DP)"
        }
