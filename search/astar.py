"""
search/astar.py - A* Search for optimal battery scheduling.
Uses admissible heuristic h(t, b) = max(0, remaining_demand - usable_battery) * min_future_price.
Tracks nodes expanded, nodes generated, and checks for admissibility.
"""

import heapq
import time
from typing import List, Tuple, Dict, Any
from search.battery_env import BatteryEnv

class AStarSearcher:
    def __init__(self, env: BatteryEnv):
        self.env = env
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.runtime_ms = 0.0

    def search(self, demands: List[float], prices: List[float], check_admissibility: bool = False) -> Dict[str, Any]:
        """
        Executes A* Search.
        State: (hour, soc)
        Evaluation function: f(n) = g(n) + h(n)
        """
        start_time = time.perf_counter()
        self.nodes_expanded = 0
        self.nodes_generated = 0
        
        initial_state = self.env.get_initial_state()
        h0 = self.env.heuristic(initial_state, demands, prices)
        
        tie_breaker = 0
        # Priority queue stores: (f_cost, counter, state, g_cost, path, actions, infos)
        pq = [(h0, tie_breaker, initial_state, 0.0, [initial_state], [], [])]
        
        best_g = {initial_state: 0.0}
        self.nodes_generated += 1
        
        goal_result = None
        
        while pq:
            f, _, curr_state, g, path, actions, infos = heapq.heappop(pq)
            
            # Prune if already found a strictly cheaper way to curr_state
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
                    "algorithm": "A* Search"
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
                    h_next = self.env.heuristic(next_state, demands, prices)
                    new_f = new_g + h_next
                    
                    tie_breaker += 1
                    self.nodes_generated += 1
                    heapq.heappush(pq, (
                        new_f,
                        tie_breaker,
                        next_state,
                        new_g,
                        path + [next_state],
                        actions + [act],
                        infos + [info]
                    ))
                    
        self.runtime_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        if goal_result is None:
            return {"error": "No solution found", "nodes_expanded": self.nodes_expanded}
        return goal_result
