"""
search/ucs.py - Uniform Cost Search (UCS) for optimal battery scheduling.
Tracks nodes expanded and generated, returns optimal plan, trajectory, and cost.
"""

import heapq
import time
from typing import List, Tuple, Dict, Any
from search.battery_env import BatteryEnv

class UCSSearcher:
    def __init__(self, env: BatteryEnv):
        self.env = env
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.runtime_ms = 0.0

    def search(self, demands: List[float], prices: List[float]) -> Dict[str, Any]:
        """
        Executes Uniform Cost Search.
        State: (hour, soc)
        Priority: path cost g(n)
        """
        start_time = time.perf_counter()
        self.nodes_expanded = 0
        self.nodes_generated = 0
        
        initial_state = self.env.get_initial_state()
        
        # Priority queue stores tuples: (g_cost, counter, state, path, actions, infos)
        tie_breaker = 0
        pq = [(0.0, tie_breaker, initial_state, [initial_state], [], [])]
        
        # Best cost to reach a state (t, soc)
        best_g = {initial_state: 0.0}
        self.nodes_generated += 1
        
        goal_result = None
        
        while pq:
            g, _, curr_state, path, actions, infos = heapq.heappop(pq)
            
            # Prune if we have already found a strictly cheaper way to curr_state
            if g > best_g.get(curr_state, float('inf')) + 1e-7:
                continue
                
            self.nodes_expanded += 1
            
            # Goal test
            if self.env.is_goal(curr_state):
                goal_result = {
                    "optimal_cost": round(g, 4),
                    "trajectory": path,
                    "actions": actions,
                    "infos": infos,
                    "nodes_expanded": self.nodes_expanded,
                    "nodes_generated": self.nodes_generated,
                    "runtime_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                    "algorithm": "Uniform Cost Search (UCS)"
                }
                break
                
            t, soc = curr_state
            hour_demand = demands[t]
            hour_price = prices[t]
            
            allowed_actions = self.env.get_actions(curr_state, hour_demand)
            for act in allowed_actions:
                next_state, step_cost, info = self.env.step(curr_state, act, hour_demand, hour_price)
                new_g = g + step_cost
                
                if new_g < best_g.get(next_state, float('inf')) - 1e-7:
                    best_g[next_state] = new_g
                    tie_breaker += 1
                    self.nodes_generated += 1
                    heapq.heappush(pq, (
                        new_g,
                        tie_breaker,
                        next_state,
                        path + [next_state],
                        actions + [act],
                        infos + [info]
                    ))
                    
        self.runtime_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        if goal_result is None:
            return {"error": "No solution found", "nodes_expanded": self.nodes_expanded}
        return goal_result
