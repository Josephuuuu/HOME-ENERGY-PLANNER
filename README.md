# Home Energy Planner: Multi-Agent Neural-Search-CSP System

> **Project P2: Home Energy Planner**  
> An autonomous decision-support system for residential demand-side management. It forecasts next-day non-shiftable electrical load using a Neural Network, schedules flexible domestic appliances using a Constraint Satisfaction Problem (CSP) solver, and optimizes home battery energy storage using Graph Search (UCS and A*), coordinated via an iterative 5-loop conflict feedback mechanism.

---

## 1. System Architecture & Cooperation

```
                  +--------------------------------+
                  |  Historical Load & Time Feats  |
                  +---------------+----------------+
                                  |
                                  v
                   [ Neural Network Forecaster ]  (PyTorch MLP)
                     Predicts 24h Base Load L(t)
                                  |
                                  v
                   [ Appliance CSP Scheduler ]    (BT / FC / AC-3)
                     Variables: Start Hours
                     Rules: Precedence, Circuit, Peak Limit
                                  |
       +------------------------->+
       |                          |
       |                          v
       |           [ Battery Search Optimizer ]   (UCS / A* / DP)
       |             State: (t, SoC)
       |             Actions: Charge, Discharge, Idle
       |             Objective: Minimize TOU Grid Cost
       |                          |
[ Feedback Loop ]                 v
If Peak Limit Violated  ---> [ Limit Check ] ---> [ Optimal Plan ]
(Up to 5 Iterations)                               - Grid Load Profile
                                                    - Dispatched Battery
                                                    - Shifted Appliances
```

---

## 2. Directory Structure

```
home_energy_planner/
├── data/
│   ├── make_data.py          # Data download, resampling, and chronological splitting
│   ├── data_dictionary.md    # Complete metadata dictionary (real vs generated)
│   ├── raw/                  # UCI raw minute-level measurements
│   └── processed/            # train.csv, val.csv, test.csv, config.json
├── nn/
│   ├── model.py              # PyTorch 24-hour load forecaster MLP
│   ├── train.py              # Training script with learning rate scheduling
│   ├── baseline.py           # "Same hour yesterday" and "same hour last week" baselines
│   └── evaluate.py           # Evaluation script and predictor interface
├── search/
│   ├── battery_env.py        # Battery storage dynamics, efficiency, and heuristic
│   ├── ucs.py                # Uniform Cost Search with node counters
│   ├── astar.py              # A* Search with admissible heuristic
│   └── dp_check.py           # Exact Dynamic Programming oracle
├── csp/
│   ├── appliances.py         # Appliance profiles, durations, circuits, precedence
│   └── csp_solver.py         # Backtracking, Forward Checking, AC-3 & Relaxation
├── integration/
│   ├── feedback_loop.py      # 5-loop orchestrator (NN -> CSP -> Search -> Feedback)
│   └── planner.py            # Unified interface
├── eval/
│   ├── experiments.py        # 3 Experiments across 5 seeds with mean +/- std
│   └── metrics.py            # Cost, violation, and search efficiency metrics
├── tests/
│   ├── test_search.py        # Small analytical test, search-DP equivalence, admissibility
│   ├── test_csp.py           # CSP constraints, inference pruning, relaxation
│   └── test_suite.py         # 5 Normal, 3 Edge, 2 Impossible test cases
├── results/
│   ├── experiments_summary.json  # Raw experiment logs across all seeds
│   └── plots/
│       ├── demo_integration_plot.png
│       ├── exp1_ucs_vs_astar.png
│       ├── exp2_csp_scaling.png
│       └── exp3_nn_vs_baseline.png
├── run_demo.py               # End-to-end integrated demonstration script
├── requirements.txt          # Pinned dependency requirements
├── report.md                 # 12-page comprehensive technical report
├── presentation.md           # 10-minute presentation slides & 5-minute live demo guide
└── contributions.md          # Individual contribution statements and git log
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.12 (64-bit)
- Standard system dependencies

### Installation
Clone or navigate to the project root and install the pinned dependencies:

```powershell
pip install -r requirements.txt
```

*(Note: PyTorch CPU wheel can be obtained directly via `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`)*

---

## 4. Exact Execution Commands

### Step 1: Preprocess Dataset Reproducibly
Downloads/extracts and preprocesses the UCI dataset with a fixed seed (`seed=42`), producing `train.csv`, `val.csv`, `test.csv`, and `config.json`:

```powershell
python data/make_data.py
```

### Step 2: Train Neural Network Forecaster & Evaluate Baselines
Trains the MLP load forecaster and validates on the test set against "same hour yesterday" and "same hour last week":

```powershell
python -m nn.train --epochs 40 --batch_size 32 --lr 0.001 --seed 42
```

### Step 3: Run Unit Tests
Run individual module tests or the complete test suite:

- **Battery Search Tests** (small analytical example, DP equivalence, heuristic admissibility):
  ```powershell
  python -m unittest tests/test_search.py
  ```

- **CSP Solver Tests** (precedence, circuit mutual exclusion, AC-3 consistency, relaxation):
  ```powershell
  python -m unittest tests/test_csp.py
  ```

- **Full Project Test Suite** (5 Normal, 3 Edge, 2 Impossible cases):
  ```powershell
  python -m unittest tests/test_suite.py
  ```

### Step 4: Run the Integrated End-to-End Demo
Executes the full cooperation pipeline (NN Forecast -> CSP Scheduler -> Battery Search -> Feedback Loop), prints the complete solution with simulated costs, and saves a 4-panel plot:

```powershell
python run_demo.py
```
*Generated plot will be saved to: `results/plots/demo_integration_plot.png`*

### Step 5: Execute the Three Core Experiments
Runs all 3 experiments across 5 seeds (42, 101, 2024, 777, 999), reports mean and standard deviation, and saves high-resolution figures:

```powershell
python -m eval.experiments
```

---

## 5. Key Experimental Findings

1. **Forecasting**:
   - NN MLP achieves **0.4556 kW MAE** and **0.6403 kW RMSE**, significantly outperforming the "Same Hour Yesterday" baseline (**0.5501 kW MAE**, a 17.2% error reduction) and "Same Hour Last Week" (**0.5919 kW MAE**, a 23.0% reduction).
2. **Search Efficiency**:
   - A* with our admissible heuristic $h(t, b) = \max(0, E_{\text{rem}} - E_{\text{batt}}) \cdot c_{\min}$ expands fewer nodes than UCS while achieving mathematically identical cost to Dynamic Programming.
   - 0 admissibility violations across 3,600 state evaluations.
3. **CSP Pruning**:
   - As problem scale increases from 3 to 8 appliances, pure Backtracking (BT) experiences exponential explosion (from 1.32 to 123.52 backtracks).
   - Backtracking with Forward Checking and AC-3 (`BT_FC_AC3`) maintains search efficiency with only **0.20 backtracks** at 8 appliances (a **99.8% reduction in search backtracking**).
4. **Economic & Grid Impact**:
   - Active appliance shifting combined with battery energy storage reduces daily simulated electricity costs by **23.1%** and completely eliminates grid limit violations during peak hours.

---

## 6. Disclaimers
- **Simulated Economics**: All monetary figures, electricity pricing tiers, and cost savings are simulated models based on synthetic Time-of-Use tariffs and do not represent actual customer billing.
- **Data Source**: Real electricity measurements are derived from the public UCI Individual Household Electric Power Consumption dataset.
