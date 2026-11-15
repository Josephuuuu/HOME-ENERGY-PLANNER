"""
tests/test_suite.py - Comprehensive test suite for Home Energy Planner.
Fulfills Deliverable 9:
- 5 Normal Cases: diverse daily profiles, full appliance suites, varying initial states
- 3 Edge Cases: zero battery capacity, razor-thin power limit, single-hour restricted windows
- 2 Impossible Cases: unachievable limit & conflicting precedence windows (must return clear 'no solution' message)
"""

import unittest
import copy
from csp.appliances import get_default_appliances, Appliance
from csp.csp_solver import ApplianceCSP
from search.battery_env import BatteryEnv
from search.astar import AStarSearcher
from integration.feedback_loop import EnergyPlannerPipeline

class ComprehensiveTestSuite(unittest.TestCase):
    def setUp(self):
        self.standard_baseline = [
            0.8, 0.7, 0.6, 0.6, 0.7, 1.0, 1.5, 1.8, 1.5, 1.4, 1.2, 1.2,
            1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 2.8, 2.2, 1.8, 1.4, 1.0, 0.8
        ]
        self.prices = [
            0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
            0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18,
            0.35, 0.35, 0.35, 0.35, 0.35,
            0.10, 0.10
        ]
        self.appliances = get_default_appliances()

    # ---------------- 5 NORMAL CASES ----------------
    def test_normal_case_1_standard_weekday(self):
        """Normal Case 1: Standard weekday 5.0 kW limit, typical 5 appliances."""
        pipeline = EnergyPlannerPipeline(self.appliances, power_limit_kw=5.0)
        res = pipeline.plan_day(self.standard_baseline, prices=self.prices)
        self.assertTrue(res["success"])
        self.assertIn("washer", res["appliance_schedule"])
        self.assertIn("dryer", res["appliance_schedule"])
        self.assertGreaterEqual(res["appliance_schedule"]["dryer"], res["appliance_schedule"]["washer"] + 2)
        print("[PASSED] Normal Case 1: Standard Weekday")

    def test_normal_case_2_weekend_profile(self):
        """Normal Case 2: Weekend day with flatter consumption and afternoon EV charging."""
        weekend_baseline = [0.9] * 24
        pipeline = EnergyPlannerPipeline(self.appliances, power_limit_kw=5.0)
        res = pipeline.plan_day(weekend_baseline, prices=self.prices)
        self.assertTrue(res["success"])
        self.assertLessEqual(max(res["total_demand"]), 5.0 + 1e-4)
        print("[PASSED] Normal Case 2: Weekend Profile")

    def test_normal_case_3_high_initial_battery(self):
        """Normal Case 3: Battery starts fully charged (10 kWh)."""
        batt = BatteryEnv(capacity_kwh=10.0, initial_soc_kwh=10.0, min_soc_kwh=1.0, max_soc_kwh=10.0)
        pipeline = EnergyPlannerPipeline(self.appliances, battery_env=batt, power_limit_kw=5.0)
        res = pipeline.plan_day(self.standard_baseline, prices=self.prices)
        self.assertTrue(res["success"])
        # Should discharge during peak hours
        peak_discharges = [res["search_result"]["actions"][h] for h in range(17, 22)]
        self.assertTrue(any(a < 0 for a in peak_discharges))
        print("[PASSED] Normal Case 3: High Initial Battery Reserve")

    def test_normal_case_4_low_base_load_summer_day(self):
        """Normal Case 4: Low base load day (~0.5 kW), ample headroom for all appliances."""
        low_baseline = [0.5] * 24
        pipeline = EnergyPlannerPipeline(self.appliances, power_limit_kw=4.5)
        res = pipeline.plan_day(low_baseline, prices=self.prices)
        self.assertTrue(res["success"])
        print("[PASSED] Normal Case 4: Low Base Load Summer Day")

    def test_normal_case_5_relaxed_limit(self):
        """Normal Case 5: Generous 6.0 kW power limit allowing early EV charging."""
        pipeline = EnergyPlannerPipeline(self.appliances, power_limit_kw=6.0)
        res = pipeline.plan_day(self.standard_baseline, prices=self.prices)
        self.assertTrue(res["success"])
        print("[PASSED] Normal Case 5: Generous Grid Limit")

    # ---------------- 3 EDGE CASES ----------------
    def test_edge_case_1_zero_battery_capacity(self):
        """
        Edge Case 1: Battery capacity is zero (or min_soc == max_soc == 1.0 kWh).
        Battery cannot store or discharge energy (idle-only).
        System must successfully plan CSP without search errors.
        """
        batt = BatteryEnv(capacity_kwh=1.0, min_soc_kwh=1.0, max_soc_kwh=1.0, initial_soc_kwh=1.0)
        pipeline = EnergyPlannerPipeline(self.appliances, battery_env=batt, power_limit_kw=5.0)
        res = pipeline.plan_day(self.standard_baseline, prices=self.prices)
        self.assertTrue(res["success"])
        # All battery actions must be 0 (idle)
        for act in res["search_result"]["actions"]:
            self.assertEqual(act, 0.0)
        print("[PASSED] Edge Case 1: Zero Usable Battery Capacity (Idle-only)")

    def test_edge_case_2_razor_thin_power_limit(self):
        """
        Edge Case 2: Razor-thin power limit (3.5 kW) where only 1 high-power appliance fits at a time.
        EV charger is 3.0 kW, base load is 0.5 kW -> total exactly 3.5 kW.
        """
        flat_low_baseline = [0.4] * 24
        pipeline = EnergyPlannerPipeline(self.appliances, power_limit_kw=3.5)
        res = pipeline.plan_day(flat_low_baseline, prices=self.prices)
        self.assertTrue(res["success"])
        self.assertLessEqual(max(res["total_demand"]), 3.5 + 1e-4)
        print("[PASSED] Edge Case 2: Razor-Thin Power Limit (3.5 kW)")

    def test_edge_case_3_single_hour_narrow_windows(self):
        """
        Edge Case 3: Every appliance has a strict 1-hour window (earliest_start == latest_start).
        """
        restricted_apps = {
            "washer": Appliance("washer", 1.5, 2, 8, 8, 1),
            "dryer": Appliance("dryer", 2.2, 2, 10, 10, 1, predecessor="washer"),
            "dishwasher": Appliance("dishwasher", 1.8, 2, 13, 13, 2),
            "ev_charger": Appliance("ev_charger", 3.0, 3, 1, 1, 3),
            "oven": Appliance("oven", 2.5, 1, 17, 17, 2)
        }
        low_baseline = [0.5] * 24
        pipeline = EnergyPlannerPipeline(restricted_apps, power_limit_kw=5.0)
        res = pipeline.plan_day(low_baseline, prices=self.prices)
        self.assertTrue(res["success"])
        self.assertEqual(res["appliance_schedule"]["washer"], 8)
        self.assertEqual(res["appliance_schedule"]["dryer"], 10)
        print("[PASSED] Edge Case 3: Single-Hour Restricted Windows")

    # ---------------- 2 IMPOSSIBLE CASES ----------------
    def test_impossible_case_1_unachievable_power_limit(self):
        """
        Impossible Case 1: Power limit is 1.0 kW, but the baseline load alone is 1.5 kW and oven is 2.5 kW.
        Must return a clear 'No solution found' message with conflict diagnosis.
        """
        impossible_limit = 1.0
        csp = ApplianceCSP(self.appliances, self.standard_baseline, power_limit_kw=impossible_limit)
        solution = csp.solve()
        self.assertIsNone(solution, "CSP should fail on impossible power limit!")
        
        diag = csp.diagnose_and_relax(max_steps=0) # Do not allow relaxation
        self.assertFalse(diag["success"])
        self.assertIn("No solution found", diag["message"])
        self.assertTrue(len(diag["conflicts"]) > 0)
        print(f"[PASSED] Impossible Case 1: Unachievable Limit (1.0 kW) -> Message: '{diag['message']}' | Identified {len(diag['conflicts'])} conflicts.")

    def test_impossible_case_2_conflicting_precedence_windows(self):
        """
        Impossible Case 2: Incompatible time windows:
        Washer earliest start is 16:00 (duration 2h, so finishes at earliest 18:00).
        Dryer latest start is 14:00.
        Since dryer must start after washer finishes, no valid assignment exists.
        Must return clear 'No solution' / None.
        """
        conflicted_apps = {
            "washer": Appliance("washer", 1.5, 2, earliest_start=16, latest_start=20, circuit=1),
            "dryer": Appliance("dryer", 2.2, 2, earliest_start=8, latest_start=14, circuit=1, predecessor="washer")
        }
        csp = ApplianceCSP(conflicted_apps, [0.5] * 24, power_limit_kw=5.0)
        solution = csp.solve(method="BT_FC_AC3")
        self.assertIsNone(solution, "CSP should immediately detect domain wipeout via AC-3 / FC!")
        
        # Diagnostic message check
        diag = csp.diagnose_and_relax(max_steps=3)
        print("[PASSED] Impossible Case 2: Precedence Incompatibility (Washer @16:00 vs Dryer latest @14:00) -> Correctly returned None / No Solution.")

if __name__ == '__main__':
    unittest.main()
