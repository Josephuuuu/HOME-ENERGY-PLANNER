"""
tests/test_csp.py - Unit tests for Appliance CSP solver.
Tests:
1. Precedence constraint (dryer after washer)
2. Circuit mutual exclusion (dishwasher and oven on circuit 2)
3. Hourly power limit compliance
4. Equivalence across BT, BT_FC, and BT_FC_AC3
5. Limit relaxation on over-constrained instances
"""

import unittest
from csp.appliances import get_default_appliances, Appliance
from csp.csp_solver import ApplianceCSP

class TestApplianceCSP(unittest.TestCase):
    def setUp(self):
        self.appliances = get_default_appliances()
        # Typical baseline profile with values around 1.0 - 2.0 kW
        self.baseline = [
            0.8, 0.7, 0.6, 0.6, 0.7, 1.0, 1.5, 1.8, 1.5, 1.4, 1.2, 1.2,
            1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 2.8, 2.2, 1.8, 1.4, 1.0, 0.8
        ]
        self.power_limit = 5.0

    def test_csp_all_methods(self):
        """Tests that BT, BT_FC, and BT_FC_AC3 all find valid solutions satisfying all constraints."""
        for method in ["BT", "BT_FC", "BT_FC_AC3"]:
            csp = ApplianceCSP(self.appliances, self.baseline, self.power_limit)
            sol = csp.solve(method=method)
            self.assertIsNotNone(sol, f"{method} failed to find a valid solution!")
            
            # 1. Precedence: dryer after washer
            washer_end = sol["washer"] + self.appliances["washer"].duration_hours
            self.assertGreaterEqual(sol["dryer"], washer_end, f"Precedence violated in {method}: dryer={sol['dryer']}, washer_end={washer_end}")
            
            # 2. Circuit 2: dishwasher and oven cannot overlap
            dw_start = sol["dishwasher"]
            dw_end = dw_start + self.appliances["dishwasher"].duration_hours
            ov_start = sol["oven"]
            ov_end = ov_start + self.appliances["oven"].duration_hours
            no_overlap = (dw_end <= ov_start) or (ov_end <= dw_start)
            self.assertTrue(no_overlap, f"Circuit 2 overlap in {method}: DW [{dw_start}, {dw_end}) vs Oven [{ov_start}, {ov_end})")
            
            # 3. Hourly power limit
            ok, total_load = csp.check_hourly_limit(sol)
            self.assertTrue(ok, f"Power limit violated in {method}!")
            self.assertLessEqual(max(total_load), self.power_limit + 1e-4)
            
            print(f"[PASSED] CSP {method:<10} | Sol: {sol} | Assignments: {csp.assignments_count} | Backtracks: {csp.backtracks_count}")

    def test_relaxation_on_impossible_instance(self):
        """Tests that an overconstrained limit triggers the relaxation mechanism."""
        # Unrealistic tight limit of 2.0 kW (oven alone is 2.5 kW!)
        tight_limit = 2.0
        csp = ApplianceCSP(self.appliances, self.baseline, power_limit_kw=tight_limit)
        
        diag = csp.diagnose_and_relax(max_steps=10)
        self.assertTrue(diag["relaxed"])
        self.assertGreater(diag["power_limit_kw"], tight_limit)
        self.assertIsNotNone(diag["solution"])
        print(f"[PASSED] CSP relaxation triggered: Original {tight_limit} kW -> Relaxed to {diag['power_limit_kw']} kW (+{diag['relaxation_percentage']}%)")

if __name__ == '__main__':
    unittest.main()
