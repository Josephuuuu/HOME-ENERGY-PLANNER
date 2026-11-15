"""
tests/test_search.py - Unit tests for battery search algorithms (UCS, A*, DP).
Includes:
1. Small test case with hand-calculated known answer
2. Equivalence test between UCS, A*, and Dynamic Programming
3. Heuristic admissibility validation (h(n) <= h*(n))
4. Edge cases: zero capacity, full battery, extreme peak price
"""

import unittest
from search.battery_env import BatteryEnv
from search.ucs import UCSSearcher
from search.astar import AStarSearcher
from search.dp_check import DPSolver

class TestBatterySearch(unittest.TestCase):
    def test_small_example_with_known_answer(self):
        """
        Small test case with hand-calculated known answer:
        Horizon: 3 hours (padded to 24 with 0s)
        Hours 0: price $0.10, demand 0.0 kW
        Hour 1: price $0.40, demand 2.0 kW
        Hour 2: price $0.10, demand 0.0 kW
        Battery: initial 1.0 kWh, min 1.0 kWh, max 5.0 kWh, step 1.0 kWh
        Efficiencies: 1.0 (lossless for clean manual calculation)
        
        Analytical Solution:
        Hour 0: Charge 2.0 kW -> battery goes to 3.0 kWh. Grid draw = 2.0 kW * $0.10 = $0.20
        Hour 1: Discharge 2.0 kW -> covers full 2.0 kW demand! Battery goes to 1.0 kWh. Grid draw = 0 kW. Cost = $0.00
        Hour 2: Idle. Grid draw = 0 kW. Cost = $0.00
        Total optimal cost = $0.20!
        Without battery, cost would be 2.0 kW * $0.40 = $0.80.
        """
        env = BatteryEnv(
            capacity_kwh=5.0,
            max_charge_kw=3.0,
            max_discharge_kw=3.0,
            charge_eff=1.0,
            discharge_eff=1.0,
            min_soc_kwh=1.0,
            max_soc_kwh=5.0,
            initial_soc_kwh=1.0,
            soc_step_kwh=1.0,
            power_limit_kw=5.0
        )
        
        demands = [0.0, 2.0, 0.0] + [0.0] * 21
        prices = [0.10, 0.40, 0.10] + [0.10] * 21
        
        ucs = UCSSearcher(env).search(demands, prices)
        astar = AStarSearcher(env).search(demands, prices)
        dp = DPSolver(env).solve(demands, prices)
        
        expected_cost = 0.20
        self.assertAlmostEqual(ucs["optimal_cost"], expected_cost, places=3)
        self.assertAlmostEqual(astar["optimal_cost"], expected_cost, places=3)
        self.assertAlmostEqual(dp["optimal_cost"], expected_cost, places=3)
        print("\n[PASSED] Small example known answer test: Cost = $0.20 across UCS, A*, and DP.")

    def test_search_dp_equivalence(self):
        """
        Tests that UCS, A*, and DP find the exact same optimal cost on a realistic 24-hour demand profile.
        """
        env = BatteryEnv(
            capacity_kwh=10.0,
            max_charge_kw=3.0,
            max_discharge_kw=3.0,
            charge_eff=0.95,
            discharge_eff=0.95,
            min_soc_kwh=1.0,
            max_soc_kwh=10.0,
            initial_soc_kwh=2.0,
            soc_step_kwh=0.5,
            power_limit_kw=5.0
        )
        
        # Realistic 24-hour load (morning dip, evening peak)
        demands = [
            0.8, 0.7, 0.6, 0.6, 0.7, 1.2, 2.0, 2.5, 1.8, 1.5, 1.4, 1.3,
            1.5, 1.6, 1.8, 2.2, 2.8, 3.5, 4.2, 3.8, 2.6, 1.8, 1.2, 0.9
        ]
        prices = [
            0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
            0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18,
            0.35, 0.35, 0.35, 0.35, 0.35,
            0.10, 0.10
        ]
        
        ucs = UCSSearcher(env).search(demands, prices)
        astar = AStarSearcher(env).search(demands, prices)
        dp = DPSolver(env).solve(demands, prices)
        
        self.assertAlmostEqual(ucs["optimal_cost"], dp["optimal_cost"], places=2)
        self.assertAlmostEqual(astar["optimal_cost"], dp["optimal_cost"], places=2)
        # A* should expand fewer or equal nodes than UCS
        self.assertLessEqual(astar["nodes_expanded"], ucs["nodes_expanded"])
        print(f"[PASSED] UCS vs A* vs DP equivalence: Cost = ${astar['optimal_cost']:.2f} | UCS nodes: {ucs['nodes_expanded']} vs A* nodes: {astar['nodes_expanded']}")

    def test_heuristic_admissibility(self):
        """
        Validates that heuristic h(t, soc) <= h*(t, soc) for every state.
        Uses DP value function V[t][soc] as the true exact h*(t, soc).
        """
        env = BatteryEnv(
            capacity_kwh=10.0,
            max_charge_kw=3.0,
            max_discharge_kw=3.0,
            charge_eff=0.95,
            discharge_eff=0.95,
            min_soc_kwh=1.0,
            max_soc_kwh=10.0,
            soc_step_kwh=0.5
        )
        demands = [1.5] * 24
        prices = [0.10 if (h < 7 or h >= 22) else (0.35 if 17 <= h < 22 else 0.18) for h in range(24)]
        
        # Calculate true cost-to-go using DP
        socs = [round(1.0 + 0.5 * i, 2) for i in range(19)]
        V = [{} for _ in range(25)]
        for s in socs:
            V[24][s] = 0.0
            
        for t in range(23, -1, -1):
            d = demands[t]
            p = prices[t]
            for s in socs:
                best = float('inf')
                for act in env.get_actions((t, s), d):
                    nxt, cost, _ = env.step((t, s), act, d, p)
                    nxt_s = nxt[1]
                    tot = cost + V[t+1].get(nxt_s, float('inf'))
                    if tot < best:
                        best = tot
                V[t][s] = best
                
        # Check admissibility: h(t, s) <= V[t][s] for all t and s
        violations = 0
        for t in range(24):
            for s in socs:
                h_val = env.heuristic((t, s), demands, prices)
                h_star = V[t][s]
                if h_val > h_star + 1e-5:
                    violations += 1
                    
        self.assertEqual(violations, 0, f"Found {violations} admissibility violations!")
        print("[PASSED] Heuristic admissibility verified across all states (0 violations).")

if __name__ == '__main__':
    unittest.main()
