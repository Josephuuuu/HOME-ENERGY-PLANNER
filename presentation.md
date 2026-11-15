# Presentation & Live Demo Guide: Home Energy Planner

**Session Format**: 10-Minute Slide Presentation + 5-Minute Interactive Live Demo  
**Project**: P2 — Home Energy Planner (Search + CSP + Neural Network)  
**Author**: Yousuf Rehab

---

# Part I: 10-Minute Presentation Slide Deck

### Slide 1: Title & Real-World Motivation (1:00)
- **Title**: Home Energy Planner: Multi-Agent Demand-Side Management via Neural Forecasting, CSP Scheduling, and Graph Search
- **Motivation**: Rising evening residential electricity peaks (EV chargers, induction cooking, HVAC) strain distribution transformers and trip circuit breakers.
- **Goal**: Minimize simulated household electricity cost under dynamic Time-Of-Use (TOU) tariffs while guaranteeing total load stays under the 5.0 kW breaker limit.
- **Key Triad**:
  - *Neural Network*: Predicts 24-hour background non-shiftable electrical load.
  - *CSP*: Schedules shiftable appliances (washer, dryer, dishwasher, EV) respecting circuit & precedence rules.
  - *Graph Search (A\*)*: Dispatches home battery storage (charging cheap, discharging peak) with efficiency losses.
- *Speaker Note*: "Welcome everyone. Today I am presenting the Home Energy Planner, an integrated AI system that solves the residential peak-load and energy-cost dilemma by uniting forecasting, constraint satisfaction, and optimal heuristic search into a closed-loop system."

---

### Slide 2: Tri-Partite System Architecture (1:00)
- **System Diagram**:
  - NN Forecaster $\rightarrow$ Base Load $\hat{L}(t)$
  - CSP Scheduler $\rightarrow$ Shiftable Start Hours $\mathbf{S}$ under $5.0$ kW limit
  - A\* Battery Search $\rightarrow$ Battery charging/discharging trajectory
  - Closed-Loop Feedback: If battery cannot cover peak, forbid clashing appliance schedule and re-solve CSP.
- *Speaker Note*: "Instead of solving a massive, brittle mixed-integer non-linear program, my architecture decomposes the problem into three modular agents connected by an iterative feedback loop."

---

### Slide 3: Dataset & Preprocessing Pipeline (1:00)
- **Data Source**: UCI Individual Household Electric Power Consumption (2,075,259 one-minute readings near Paris, France).
- **Preprocessing**: Resampled to 1-hour means; missing gaps $\le 3$ hours linearly interpolated; days with extensive missing data removed.
- **Chronological Split**:
  - *Train*: Dec 2006 – Sep 2009 (1,012 days).
  - *Validation*: Oct 2009 – Dec 2009 (92 days, last 3 months).
  - *Test*: Full year 2010 (329 days).
- **Data Integrity**: Real electricity measurements vs. Simulated pricing tariffs, battery specs, and appliance ratings.
- *Speaker Note*: "To ensure complete academic rigor, I enforced a strict chronological split preventing data leakage. Real empirical data drives the load, while simulated TOU tariffs and physical appliance profiles evaluate the optimization."

---

### Slide 4: Neural Network 24-Hour Day-Ahead Load Forecasting (1:00)
- **Input Features (53)**: 24 yesterday hourly loads + 24 same day last week hourly loads + 5 calendar features (`is_weekend`, cyclical sine/cosine encodings of month and day-of-week).
- **Architecture**: Deep MLP with residual skip-connections, LayerNorm, and ReLU non-negativity output.
- **Test Set Results (Year 2010, 329 days)**:
  - *NN MLP*: **0.4556 kW MAE** | **0.6403 kW RMSE**
  - *Yesterday Baseline*: 0.5501 kW MAE | 0.8191 kW RMSE (**NN is 17.2% more accurate**)
  - *Last Week Baseline*: 0.5919 kW MAE | 0.8479 kW RMSE (**NN is 23.0% more accurate**)
- *Speaker Note*: "The neural forecaster learns weekly routines and seasonal shifts, significantly beating persistence baselines on unseen test data."

---

### Slide 5: Constraint Satisfaction Problem for Appliance Scheduling (1:00)
- **Variables**: Start hours for Washer, Dryer, Dishwasher, EV Charger, Oven.
- **Hard Constraints**:
  1. *Precedence*: Dryer must start after Washer finishes: $\text{start}_{\text{dryer}} \ge \text{start}_{\text{washer}} + 2$.
  2. *Circuit Mutual Exclusion*: Dishwasher and Oven share Circuit 2 (no simultaneous run).
  3. *Hourly Power Limit*: $\hat{L}_{\text{base}}(t) + \sum P_{\text{appliances}}(t) \le 5.0\text{ kW}$.
- **Inference Engines**: Switchable Pure Backtracking (`BT`), Forward Checking (`BT_FC`), and Arc Consistency (`BT_FC_AC3`).
- *Speaker Note*: "The CSP guarantees that high-power devices never trip shared circuits or exceed household limits, pruning invalid start times ahead of search."

---

### Slide 6: A\* Battery Graph Search & Admissible Heuristic (1:00)
- **State-Space**: $(t, \text{SoC})$ with $t \in [0, 24]$, $\text{SoC} \in [1.0, 10.0]$ kWh in 0.5 kWh steps.
- **Physical Reality**: Round-trip efficiency of 90.25% ($\eta_c = 0.95, \eta_d = 0.95$).
- **Admissible Heuristic**:
  $$h(t, b) = \max\left(0, \sum_{k=t}^{23} D_k - (b - b_{\min}) \eta_d \right) \times \min_{k \ge t} \text{price}_k$$
- **Correctness**: Validated against an exact backward Dynamic Programming oracle: $\text{Cost}_{\text{A*}} \equiv \text{Cost}_{\text{DP}}$ with 0 heuristic overestimations across 3,600 verified states.
- *Speaker Note*: "Because electricity cannot be bought cheaper than the minimum future tariff, the heuristic is mathematically guaranteed never to overestimate, ensuring optimal search."

---

### Slide 7: Closed-Loop Multi-Agent Feedback Mechanism (1:00)
- **Feedback Protocol**:
  - If Battery Search detects unavoidable peak violation ($P_{\text{grid}} > 5.0\text{ kW}$), current appliance combination is tagged as forbidden no-good.
  - CSP re-executes with new constraint (up to 5 loops).
- **Empirical Evidence**: In this demo, Loop 1 encountered an overage (+1.2 kW at 18:00); feedback was triggered, forbidding the schedule; Loop 2 rescheduled the Dryer from 09:00 to 10:00, resolving the peak with 0 violations!
- *Speaker Note*: "This closed-loop feedback provides robustness against tight physical battery discharge bottlenecks."

---

### Slide 8: Key Experimental Findings (1:00)
- **Experiment 1 (UCS vs A\*)**: A\* prunes node expansions across 30 days and 5 seeds with 0 admissibility violations.
- **Experiment 2 (CSP Scaling)**: At 8 appliances, pure BT backtracks explode to **123.52**, while `BT_FC_AC3` requires only **0.20 backtracks** (a **99.8% reduction**).
- **Experiment 3 (NN vs Yesterday Baseline on Real Loads)**: Planning with NN forecast saves simulated electricity cost by **23.1%** compared to unplanned baseline, while avoiding breaker overloads.
- *Speaker Note*: "The experiments rigorously prove the computational superiority of AC-3 and the economic benefits of neural planning."

---

### Slide 9: Edge Cases, Impossible Scenarios & Diagnostics (1:00)
- **3 Edge Cases**: Handled zero battery capacity (idle-only), razor-thin 3.5 kW limit, and single-hour fixed windows.
- **2 Impossible Cases**:
  - *Unachievable Limit (1.0 kW)*: Clear "No solution found" message with diagnostic list of 65 conflicting windows and 10% relaxation offers.
  - *Incompatible Precedence*: Washer $\ge 16:00$ vs Dryer $\le 14:00$ immediately pruned via AC-3 domain wipeout.
- *Speaker Note*: "The planner fails gracefully and informatively, offering actionable diagnostic relaxations when problems are physically infeasible."

---

### Slide 10: Summary & Conclusions (1:00)
- **Delivered**: Source code, data pipeline, trained NN, verified search with DP, switchable CSP, 10 unit tests, 3 experiments (5 seeds), 4 publication plots, technical report.
- **Impact**: Demonstrates that combining predictive intelligence (NN), structural logic (CSP), and decision optimization (Search) creates a complete, production-grade demand management framework.
- *Speaker Note*: "Thank you. I will now transition immediately to the live 5-minute interactive demonstration."

---

# Part II: 5-Minute Live Demo Script

### Preparation
Open PowerShell in `scratch/home_energy_planner` and have Python ready.

```powershell
cd "C:\Users\Yousuf Rehab\.gemini\antigravity\scratch\home_energy_planner"
```

---

### Demo Segment 1: Search Trace & Admissibility (1.5 Minutes)
**Action**: Run the search test and trace verification.
```powershell
& "D:\python.exe" -m unittest tests/test_search.py
```
**Talking Points**:
- "Notice the first test: a small 3-hour analytical example with known answer of \$0.20. UCS, A\*, and Dynamic Programming all return exactly \$0.20."
- "Next, observe the 24-hour test: A\* expands fewer nodes than UCS while achieving the exact same \$7.57 optimal cost as DP."
- "Finally, notice the admissibility test: all 3,600 visited states are checked against true cost-to-go, confirming zero heuristic violations."

---

### Demo Segment 2: CSP Pruning With vs. Without FC and AC-3 (1.5 Minutes)
**Action**: Run the CSP scaling benchmark live.
```powershell
& "D:\python.exe" -c "from csp.appliances import get_scaled_appliances; from csp.csp_solver import ApplianceCSP; apps = get_scaled_appliances(8); baseline = [1.2]*24; [print(f'Method: {m:<10} | Assignments: {ApplianceCSP(apps, baseline, 5.0).solve(m) and ApplianceCSP(apps, baseline, 5.0).assignments_count} | Backtracks: {ApplianceCSP(apps, baseline, 5.0).backtracks_count}') for m in ['BT', 'BT_FC', 'BT_FC_AC3']]"
```
**Talking Points**:
- "Watch what happens when scaling to 8 appliances: Pure Backtracking (`BT`) must search through multiple blind paths."
- "When Forward Checking (`BT_FC`) is enabled, backtracks plummet."
- "When activating **AC-3 (`BT_FC_AC3`)**, arc consistency eliminates conflicting start hours before the search even starts, resulting in near-zero backtracks and instant convergence!"

---

### Demo Segment 3: Integrated System with NN Forecast vs. Yesterday Baseline (2 Minutes)
**Action**: Run the full end-to-end demo script.
```powershell
& "D:\python.exe" run_demo.py
```
**Talking Points**:
- "Step 1 & 2: The neural network forecasts the next day's 24 hourly loads with an MAE of 0.27 kW, a 78.5% improvement over the yesterday baseline."
- "Step 3: Here is the **Feedback Loop in action**! In Loop 1, the battery search detected an evening peak limit violation (+1.2 kW at 18:00). It immediately forbade that schedule. In Loop 2, the CSP rescheduled the Dryer, and the battery search succeeded with zero violations."
- "Step 4 & 5: Look at the final performance table:
  - Unplanned cost: \$9.41 with a 5.16 kW peak violation.
  - Planned cost: \$7.24 with peak shaved to 3.94 kW (a **23.1% simulated cost savings** and **zero limit violations**)."
- "Finally, the 4-panel cooperation plot has been generated and saved to `results/plots/demo_integration_plot.png`."
