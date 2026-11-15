"""
csp/csp_solver.py - Constraint Satisfaction Problem (CSP) solver for appliance scheduling.
Supports three switchable inference modes:
1. 'BT': Pure Backtracking
2. 'BT_FC': Backtracking with Forward Checking
3. 'BT_FC_AC3': Backtracking with Forward Checking and AC-3 Arc Consistency

Includes counters (assignments, backtracks, constraint checks) and limit relaxation.
"""

import time
import copy
from typing import Dict, List, Tuple, Optional, Any, Set
from csp.appliances import Appliance

class ApplianceCSP:
    def __init__(self,
                 appliances: Dict[str, Appliance],
                 baseline_load: List[float],
                 power_limit_kw: float = 5.0,
                 forbidden_assignments: Optional[List[Dict[str, int]]] = None):
        self.appliances = appliances
        self.baseline_load = baseline_load
        self.power_limit_kw = power_limit_kw
        self.forbidden_assignments = forbidden_assignments or []
        
        # Build initial domains: for each appliance, hours in [earliest_start, latest_start]
        self.initial_domains: Dict[str, List[int]] = {}
        for name, app in appliances.items():
            valid_hours = []
            for h in range(app.earliest_start, app.latest_start + 1):
                if h + app.duration_hours <= 24:
                    valid_hours.append(h)
            self.initial_domains[name] = valid_hours
            
        # Counters
        self.assignments_count = 0
        self.backtracks_count = 0
        self.constraint_checks = 0
        self.runtime_ms = 0.0

    def reset_counters(self):
        self.assignments_count = 0
        self.backtracks_count = 0
        self.constraint_checks = 0
        self.runtime_ms = 0.0

    # ---------------- Constraint Verification ----------------
    def satisfies_binary_constraint(self, var1: str, val1: int, var2: str, val2: int) -> bool:
        """Checks precedence and circuit non-overlap between two assigned appliances."""
        self.constraint_checks += 1
        app1 = self.appliances[var1]
        app2 = self.appliances[var2]
        
        # Precedence: dryer after washer
        if app2.predecessor == var1:
            if val2 < val1 + app1.duration_hours:
                return False
        if app1.predecessor == var2:
            if val1 < val2 + app2.duration_hours:
                return False
                
        # Circuit non-overlap: cannot run simultaneously on same circuit
        if app1.circuit == app2.circuit:
            # Check interval overlap [val1, val1 + d1) and [val2, val2 + d2)
            if not (val1 + app1.duration_hours <= val2 or val2 + app2.duration_hours <= val1):
                return False
                
        return True

    def check_hourly_limit(self, assignment: Dict[str, int]) -> Tuple[bool, List[float]]:
        """Calculates total load for each hour given current partial/full assignment."""
        total_loads = list(self.baseline_load)
        for var, start_h in assignment.items():
            app = self.appliances[var]
            for dh in range(app.duration_hours):
                h = start_h + dh
                if h < 24:
                    total_loads[h] += app.profile[dh]
                    self.constraint_checks += 1
                    if total_loads[h] > self.power_limit_kw + 1e-5:
                        return False, total_loads
        return True, total_loads

    def check_forbidden_feedback(self, assignment: Dict[str, int]) -> bool:
        """Ensures the assignment does not match any forbidden combination from the feedback loop."""
        for forbidden in self.forbidden_assignments:
            # Check if assignment is a superset of or matches the forbidden dictionary
            if all(assignment.get(k) == v for k, v in forbidden.items()):
                return False
        return True

    def is_consistent(self, var: str, val: int, current_assignment: Dict[str, int]) -> bool:
        """Tests whether adding var=val to current_assignment is valid."""
        # 1. Binary checks against already assigned variables
        for other_var, other_val in current_assignment.items():
            if not self.satisfies_binary_constraint(var, val, other_var, other_val):
                return False
                
        # 2. Hourly power limit check
        tentative = dict(current_assignment)
        tentative[var] = val
        ok, _ = self.check_hourly_limit(tentative)
        if not ok:
            return False
            
        # 3. Forbidden assignment check
        if len(tentative) == len(self.appliances):
            if not self.check_forbidden_feedback(tentative):
                return False
                
        return True

    # ---------------- AC-3 Inference ----------------
    def ac3(self, domains: Dict[str, List[int]]) -> bool:
        """
        Executes standard AC-3 algorithm on binary constraints (order and circuit overlap).
        Returns False if any domain is wiped out, True otherwise.
        """
        # Build binary arcs between all pairs that share a constraint
        queue: List[Tuple[str, str]] = []
        vars_list = list(self.appliances.keys())
        for i in range(len(vars_list)):
            for j in range(len(vars_list)):
                if i != j:
                    v1, v2 = vars_list[i], vars_list[j]
                    app1, app2 = self.appliances[v1], self.appliances[v2]
                    # Arcs exist if precedence or shared circuit
                    if app1.predecessor == v2 or app2.predecessor == v1 or app1.circuit == app2.circuit:
                        queue.append((v1, v2))
                        
        while queue:
            xi, xj = queue.pop(0)
            if self.revise(domains, xi, xj):
                if len(domains[xi]) == 0:
                    return False
                # Add all neighbors xk of xi (except xj) back to queue
                for xk in vars_list:
                    if xk != xi and xk != xj:
                        app_k, app_i = self.appliances[xk], self.appliances[xi]
                        if app_k.predecessor == xi or app_i.predecessor == xk or app_k.circuit == app_i.circuit:
                            if (xk, xi) not in queue:
                                queue.append((xk, xi))
        return True

    def revise(self, domains: Dict[str, List[int]], xi: str, xj: str) -> bool:
        """Prunes values from domains[xi] that have no valid support in domains[xj]."""
        revised = False
        to_remove = []
        for x in domains[xi]:
            has_support = False
            for y in domains[xj]:
                if self.satisfies_binary_constraint(xi, x, xj, y):
                    has_support = True
                    break
            if not has_support:
                to_remove.append(x)
                revised = True
        for val in to_remove:
            domains[xi].remove(val)
        return revised

    # ---------------- Forward Checking Inference ----------------
    def forward_check(self, var: str, val: int, current_assignment: Dict[str, int], domains: Dict[str, List[int]]) -> Optional[Dict[str, List[int]]]:
        """
        Prunes domains of unassigned variables after assigning var=val.
        Removes values violating binary constraints or hourly limit.
        Returns pruned domains, or None if domain wipeout occurs.
        """
        new_domains = {v: list(d) for v, d in domains.items()}
        new_domains[var] = [val]
        
        tentative_assignment = dict(current_assignment)
        tentative_assignment[var] = val
        
        for other_var, other_domain in new_domains.items():
            if other_var in tentative_assignment:
                continue
                
            pruned_domain = []
            for other_val in other_domain:
                # 1. Binary check against assigned variables
                binary_ok = True
                for a_var, a_val in tentative_assignment.items():
                    if not self.satisfies_binary_constraint(other_var, other_val, a_var, a_val):
                        binary_ok = False
                        break
                if not binary_ok:
                    continue
                    
                # 2. Hourly limit check with other_var=other_val added
                temp = dict(tentative_assignment)
                temp[other_var] = other_val
                limit_ok, _ = self.check_hourly_limit(temp)
                if limit_ok:
                    pruned_domain.append(other_val)
                    
            if len(pruned_domain) == 0:
                return None  # Domain wipeout
            new_domains[other_var] = pruned_domain
            
        return new_domains

    # ---------------- Recursive Search Solvers ----------------
    def solve(self, method: str = "BT_FC_AC3") -> Optional[Dict[str, int]]:
        """
        Solves CSP using specified method:
        'BT': Pure Backtracking
        'BT_FC': Backtracking with Forward Checking
        'BT_FC_AC3': Backtracking with FC and AC-3
        """
        self.reset_counters()
        start_time = time.perf_counter()
        
        # Deep copy initial domains
        domains = {k: list(v) for k, v in self.initial_domains.items()}
        
        # Initial AC-3 pruning if enabled
        if method == "BT_FC_AC3":
            if not self.ac3(domains):
                self.runtime_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                return None
                
        var_order = list(self.appliances.keys())
        
        # Variable ordering: prefer predecessor first, then highest power
        var_order.sort(key=lambda v: (0 if self.appliances[v].predecessor is None else 1, -self.appliances[v].power_kw))
        
        solution = self._backtrack({}, domains, var_order, method)
        self.runtime_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return solution

    def _backtrack(self, assignment: Dict[str, int], domains: Dict[str, List[int]], var_order: List[str], method: str) -> Optional[Dict[str, int]]:
        if len(assignment) == len(self.appliances):
            if self.check_forbidden_feedback(assignment):
                return assignment
            return None
            
        # Select next unassigned variable (MRV heuristic: Minimum Remaining Values)
        unassigned = [v for v in var_order if v not in assignment]
        if method in ("BT_FC", "BT_FC_AC3"):
            var = min(unassigned, key=lambda v: len(domains[v]))
        else:
            var = unassigned[0]
            
        for val in domains[var]:
            self.assignments_count += 1
            if self.is_consistent(var, val, assignment):
                new_assignment = dict(assignment)
                new_assignment[var] = val
                
                if method == "BT":
                    res = self._backtrack(new_assignment, domains, var_order, method)
                    if res is not None:
                        return res
                elif method == "BT_FC":
                    pruned_domains = self.forward_check(var, val, assignment, domains)
                    if pruned_domains is not None:
                        res = self._backtrack(new_assignment, pruned_domains, var_order, method)
                        if res is not None:
                            return res
                elif method == "BT_FC_AC3":
                    pruned_domains = self.forward_check(var, val, assignment, domains)
                    if pruned_domains is not None:
                        domains_copy = {k: list(v) for k, v in pruned_domains.items()}
                        if self.ac3(domains_copy):
                            res = self._backtrack(new_assignment, domains_copy, var_order, method)
                            if res is not None:
                                return res
                                
            self.backtracks_count += 1
            
        return None

    # ---------------- Relaxation & Diagnostic Analysis ----------------
    def diagnose_and_relax(self, max_steps: int = 10, method: str = "BT_FC_AC3") -> Dict[str, Any]:
        """
        Diagnoses failures, lists conflicting appliance windows, and relaxes limit in 10% steps.
        """
        initial_solution = self.solve(method=method)
        if initial_solution is not None:
            _, total_load = self.check_hourly_limit(initial_solution)
            return {
                "success": True,
                "solution": initial_solution,
                "power_limit_kw": self.power_limit_kw,
                "relaxed": False,
                "total_load": total_load,
                "conflicts": []
            }
            
        # If no solution exists: Identify conflicting hours and windows
        conflicts = []
        for name, app in self.appliances.items():
            valid_hours = []
            for h in range(app.earliest_start, app.latest_start + 1):
                if h + app.duration_hours <= 24:
                    exceeds = False
                    for dh in range(app.duration_hours):
                        if self.baseline_load[h + dh] + app.profile[dh] > self.power_limit_kw:
                            exceeds = True
                            conflicts.append({
                                "appliance": name,
                                "start_hour": h,
                                "violating_hour": h + dh,
                                "projected_load_kw": round(self.baseline_load[h + dh] + app.profile[dh], 2),
                                "limit_kw": self.power_limit_kw
                            })
                            break
                            
        # Progressively relax limit in 10% steps
        current_limit = self.power_limit_kw
        for step in range(1, max_steps + 1):
            relaxed_limit = round(current_limit * (1.0 + 0.10 * step), 2)
            temp_csp = ApplianceCSP(
                self.appliances,
                self.baseline_load,
                power_limit_kw=relaxed_limit,
                forbidden_assignments=self.forbidden_assignments
            )
            sol = temp_csp.solve(method=method)
            if sol is not None:
                _, total_load = temp_csp.check_hourly_limit(sol)
                return {
                    "success": True,
                    "solution": sol,
                    "original_limit_kw": self.power_limit_kw,
                    "power_limit_kw": relaxed_limit,
                    "relaxed": True,
                    "relaxation_steps": step,
                    "relaxation_percentage": step * 10,
                    "total_load": total_load,
                    "conflicts": conflicts[:10]
                }
                
        return {
            "success": False,
            "message": "No solution found even after maximum limit relaxations.",
            "conflicts": conflicts
        }
