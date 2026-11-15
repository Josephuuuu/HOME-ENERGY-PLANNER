"""
eval/experiments.py - Implementation of the three required project experiments:
Experiment 1: UCS vs A* (nodes, runtime, cost, heuristic admissibility check) over 30 days, 5 seeds.
Experiment 2: BT vs BT+FC vs BT+FC+AC-3 with 3, 5, 8 appliances (assignments, backtracks, runtime), 5 seeds.
Experiment 3: NN Forecast vs Yesterday-Baseline evaluated on actual real loads (cost, limit violations), 5 seeds.

Outputs formatted tables with mean +/- standard deviation and generates publication-grade plots.
NOTE: All costs and savings are SIMULATED.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Set matplotlib cache directory
os.environ['MPLCONFIGDIR'] = os.path.abspath(".mpl")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nn.evaluate import LoadForecaster
from csp.appliances import get_scaled_appliances, get_default_appliances
from csp.csp_solver import ApplianceCSP
from search.battery_env import BatteryEnv
from search.ucs import UCSSearcher
from search.astar import AStarSearcher
from search.dp_check import DPSolver
from integration.feedback_loop import EnergyPlannerPipeline

SEEDS = [42, 101, 2024, 777, 999]

def run_experiment_1(test_df: pd.DataFrame, config: dict, num_days: int = 30) -> dict:
    """
    Experiment 1: UCS vs A* nodes and runtime over 30 days, checking heuristic admissibility.
    Repeated over 5 seeds/runs to compute mean and standard deviation.
    """
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 1: UCS vs A* Battery Search (30 Days x 5 Seeds)")
    print("=" * 70)
    
    prices = config["pricing_tariff"]["hourly_prices"]
    power_limit = config["power_limit_kw"]
    
    env = BatteryEnv(
        capacity_kwh=config["battery"]["capacity_kwh"],
        max_charge_kw=config["battery"]["max_charge_kw"],
        max_discharge_kw=config["battery"]["max_discharge_kw"],
        charge_eff=config["battery"]["charge_efficiency"],
        discharge_eff=config["battery"]["discharge_efficiency"],
        min_soc_kwh=config["battery"]["min_soc_kwh"],
        max_soc_kwh=config["battery"]["max_soc_kwh"],
        initial_soc_kwh=config["battery"]["initial_soc_kwh"],
        soc_step_kwh=config["battery"]["discretization_step_kwh"],
        power_limit_kw=power_limit
    )
    
    ucs_searcher = UCSSearcher(env)
    astar_searcher = AStarSearcher(env)
    
    seed_results = []
    total_admissibility_violations = 0
    total_states_checked = 0
    
    for s_idx, seed in enumerate(SEEDS):
        np.random.seed(seed)
        # Sample 30 random days from test set
        sampled_indices = np.random.choice(len(test_df), size=num_days, replace=False)
        
        ucs_nodes_list = []
        astar_nodes_list = []
        ucs_gen_list = []
        astar_gen_list = []
        ucs_time_list = []
        astar_time_list = []
        cost_diff_list = []
        
        for idx in sampled_indices:
            row = test_df.iloc[idx]
            demands = [float(row[f"target_h{h}"]) for h in range(24)]
            
            res_ucs = ucs_searcher.search(demands, prices)
            res_astar = astar_searcher.search(demands, prices)
            
            ucs_nodes_list.append(res_ucs["nodes_expanded"])
            astar_nodes_list.append(res_astar["nodes_expanded"])
            ucs_gen_list.append(res_ucs["nodes_generated"])
            astar_gen_list.append(res_astar["nodes_generated"])
            ucs_time_list.append(res_ucs["runtime_ms"])
            astar_time_list.append(res_astar["runtime_ms"])
            
            # Check cost optimality equivalence
            cost_diff = abs(res_ucs["optimal_cost"] - res_astar["optimal_cost"])
            cost_diff_list.append(cost_diff)
            
            # Admissibility check on trajectory states
            for st in res_astar["trajectory"][:-1]:
                h_val = env.heuristic(st, demands, prices)
                # Remainder cost from st using ucs
                rem_demands = demands[st[0]:] + [0.0] * st[0]
                rem_prices = prices[st[0]:] + [0.0] * st[0]
                env_sub = BatteryEnv(initial_soc_kwh=st[1], capacity_kwh=env.capacity_kwh)
                # True remaining cost cannot be less than h_val
                # Analytical lower bound: uncovered remaining energy * min price
                rem_energy = max(0.0, sum(demands[st[0]:]) - (st[1] - env.min_soc_kwh) * env.discharge_eff)
                h_star_lower = rem_energy * min(prices[st[0]:])
                total_states_checked += 1
                if h_val > h_star_lower + 1e-4:
                    total_admissibility_violations += 1
                    
        seed_results.append({
            "seed": seed,
            "ucs_nodes": np.mean(ucs_nodes_list),
            "astar_nodes": np.mean(astar_nodes_list),
            "ucs_gen": np.mean(ucs_gen_list),
            "astar_gen": np.mean(astar_gen_list),
            "ucs_time": np.mean(ucs_time_list),
            "astar_time": np.mean(astar_time_list),
            "max_cost_diff": np.max(cost_diff_list)
        })
        print(f"  Seed {seed} ({s_idx+1}/5) | UCS nodes: {np.mean(ucs_nodes_list):.1f} | A* nodes: {np.mean(astar_nodes_list):.1f} | A* speedup: {(np.mean(ucs_nodes_list)-np.mean(astar_nodes_list))/np.mean(ucs_nodes_list)*100:.1f}%")
        
    # Aggregate across 5 seeds
    ucs_nodes_seeds = [r["ucs_nodes"] for r in seed_results]
    astar_nodes_seeds = [r["astar_nodes"] for r in seed_results]
    ucs_gen_seeds = [r["ucs_gen"] for r in seed_results]
    astar_gen_seeds = [r["astar_gen"] for r in seed_results]
    ucs_time_seeds = [r["ucs_time"] for r in seed_results]
    astar_time_seeds = [r["astar_time"] for r in seed_results]
    
    summary = {
        "UCS_nodes_expanded": f"{np.mean(ucs_nodes_seeds):.2f} +/- {np.std(ucs_nodes_seeds):.2f}",
        "AStar_nodes_expanded": f"{np.mean(astar_nodes_seeds):.2f} +/- {np.std(astar_nodes_seeds):.2f}",
        "UCS_nodes_generated": f"{np.mean(ucs_gen_seeds):.2f} +/- {np.std(ucs_gen_seeds):.2f}",
        "AStar_nodes_generated": f"{np.mean(astar_gen_seeds):.2f} +/- {np.std(astar_gen_seeds):.2f}",
        "UCS_runtime_ms": f"{np.mean(ucs_time_seeds):.2f} +/- {np.std(ucs_time_seeds):.2f}",
        "AStar_runtime_ms": f"{np.mean(astar_time_seeds):.2f} +/- {np.std(astar_time_seeds):.2f}",
        "node_reduction_pct": f"{(np.mean(ucs_nodes_seeds) - np.mean(astar_nodes_seeds)) / np.mean(ucs_nodes_seeds) * 100:.2f}%",
        "admissibility_violations": total_admissibility_violations,
        "states_checked": total_states_checked,
        "is_strictly_admissible": (total_admissibility_violations == 0),
        "seed_details": seed_results
    }
    
    print("\n--- EXPERIMENT 1 SUMMARY (Mean +/- Std Dev across 5 seeds) ---")
    print(f"Nodes Expanded:   UCS = {summary['UCS_nodes_expanded']} vs A* = {summary['AStar_nodes_expanded']} ({summary['node_reduction_pct']} pruned)")
    print(f"Nodes Generated:  UCS = {summary['UCS_nodes_generated']} vs A* = {summary['AStar_nodes_generated']}")
    print(f"Runtime (ms):     UCS = {summary['UCS_runtime_ms']} ms vs A* = {summary['AStar_runtime_ms']} ms")
    print(f"Admissibility:    {total_admissibility_violations} violations across {total_states_checked} evaluated states (Admissible: {summary['is_strictly_admissible']})")
    
    return summary

def run_experiment_2(test_df: pd.DataFrame, config: dict) -> dict:
    """
    Experiment 2: BT vs BT+FC vs BT+FC+AC-3 with 3, 5, 8 appliances.
    Repeated over 5 seeds to compute mean and standard deviation for assignments, backtracks, runtime.
    """
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 2: CSP Pruning Scaling (BT vs FC vs AC-3 for 3, 5, 8 Apps)")
    print("=" * 70)
    
    power_limit = config["power_limit_kw"]
    scales = [3, 5, 8]
    methods = ["BT", "BT_FC", "BT_FC_AC3"]
    
    results_by_scale = {}
    
    for n in scales:
        apps = get_scaled_appliances(n)
        print(f"\nEvaluating Scale: {n} Appliances...")
        scale_data = {m: {"assignments": [], "backtracks": [], "runtime_ms": []} for m in methods}
        
        for seed in SEEDS:
            np.random.seed(seed)
            # Pick 10 representative baseline profiles
            sampled_indices = np.random.choice(len(test_df), size=10, replace=False)
            
            for idx in sampled_indices:
                row = test_df.iloc[idx]
                baseline = [float(row[f"target_h{h}"]) for h in range(24)]
                
                for m in methods:
                    csp = ApplianceCSP(apps, baseline, power_limit_kw=power_limit)
                    sol = csp.solve(method=m)
                    # If tighter limit needed relaxation, use diagnose_and_relax
                    if sol is None:
                        diag = csp.diagnose_and_relax(max_steps=5, method=m)
                        
                    scale_data[m]["assignments"].append(csp.assignments_count)
                    scale_data[m]["backtracks"].append(csp.backtracks_count)
                    scale_data[m]["runtime_ms"].append(csp.runtime_ms)
                    
        scale_summary = {}
        for m in methods:
            scale_summary[m] = {
                "assignments": f"{np.mean(scale_data[m]['assignments']):.2f} +/- {np.std(scale_data[m]['assignments']):.2f}",
                "backtracks": f"{np.mean(scale_data[m]['backtracks']):.2f} +/- {np.std(scale_data[m]['backtracks']):.2f}",
                "runtime_ms": f"{np.mean(scale_data[m]['runtime_ms']):.2f} +/- {np.std(scale_data[m]['runtime_ms']):.2f}",
                "raw_means": {
                    "assignments": float(np.mean(scale_data[m]['assignments'])),
                    "backtracks": float(np.mean(scale_data[m]['backtracks'])),
                    "runtime_ms": float(np.mean(scale_data[m]['runtime_ms']))
                }
            }
            print(f"  [{m:<10}] {n} apps | Assign: {scale_summary[m]['assignments']} | Backtracks: {scale_summary[m]['backtracks']} | Time: {scale_summary[m]['runtime_ms']} ms")
        results_by_scale[f"{n}_appliances"] = scale_summary
        
    return results_by_scale

def run_experiment_3(test_df: pd.DataFrame, config: dict, num_days: int = 30) -> dict:
    """
    Experiment 3: NN forecast vs yesterday-baseline: cost and limit violations using the real load.
    Repeated over 5 seeds to compute mean and standard deviation.
    """
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 3: NN Forecast vs Yesterday-Baseline on Real Load")
    print("=" * 70)
    
    forecaster = LoadForecaster(model_dir="nn")
    appliances = get_default_appliances()
    prices = config["pricing_tariff"]["hourly_prices"]
    power_limit = config["power_limit_kw"]
    
    battery_env = BatteryEnv(
        capacity_kwh=config["battery"]["capacity_kwh"],
        max_charge_kw=config["battery"]["max_charge_kw"],
        max_discharge_kw=config["battery"]["max_discharge_kw"],
        charge_eff=config["battery"]["charge_efficiency"],
        discharge_eff=config["battery"]["discharge_efficiency"],
        min_soc_kwh=config["battery"]["min_soc_kwh"],
        max_soc_kwh=config["battery"]["max_soc_kwh"],
        initial_soc_kwh=config["battery"]["initial_soc_kwh"],
        soc_step_kwh=config["battery"]["discretization_step_kwh"],
        power_limit_kw=power_limit
    )
    
    pipeline = EnergyPlannerPipeline(appliances, battery_env, power_limit_kw=power_limit)
    
    seed_metrics = []
    
    for s_idx, seed in enumerate(SEEDS):
        np.random.seed(seed)
        sampled_indices = np.random.choice(len(test_df), size=num_days, replace=False)
        
        nn_costs, yest_costs = [], []
        nn_viol_hours, yest_viol_hours = [], []
        nn_viol_kwh, yest_viol_kwh = [], []
        
        for idx in sampled_indices:
            row = test_df.iloc[idx]
            actual_load = [float(row[f"target_h{h}"]) for h in range(24)]
            yest_load = [float(row[f"yest_h{h}"]) for h in range(24)]
            nn_forecast = list(forecaster.predict_row(row))
            
            # Plan with NN Forecast
            res_nn = pipeline.plan_day(forecasted_baseline=nn_forecast, actual_baseline=actual_load, prices=prices)
            eval_nn = res_nn["actual_evaluation"]
            nn_costs.append(eval_nn["total_simulated_cost"])
            nn_viol_hours.append(eval_nn["num_violation_hours"])
            nn_viol_kwh.append(eval_nn["total_violations_kwh"])
            
            # Plan with Yesterday Baseline Forecast
            res_yest = pipeline.plan_day(forecasted_baseline=yest_load, actual_baseline=actual_load, prices=prices)
            eval_yest = res_yest["actual_evaluation"]
            yest_costs.append(eval_yest["total_simulated_cost"])
            yest_viol_hours.append(eval_yest["num_violation_hours"])
            yest_viol_kwh.append(eval_yest["total_violations_kwh"])
            
        seed_metrics.append({
            "seed": seed,
            "nn_cost": np.mean(nn_costs),
            "yest_cost": np.mean(yest_costs),
            "nn_viol_hours": np.mean(nn_viol_hours),
            "yest_viol_hours": np.mean(yest_viol_hours),
            "nn_viol_kwh": np.mean(nn_viol_kwh),
            "yest_viol_kwh": np.mean(yest_viol_kwh)
        })
        print(f"  Seed {seed} ({s_idx+1}/5) | NN Cost: ${np.mean(nn_costs):.2f} vs Yest Cost: ${np.mean(yest_costs):.2f} | NN Viol Hours: {np.mean(nn_viol_hours):.2f} vs Yest: {np.mean(yest_viol_hours):.2f}")
        
    summary = {
        "NN_cost_simulated": f"${np.mean([m['nn_cost'] for m in seed_metrics]):.2f} +/- ${np.std([m['nn_cost'] for m in seed_metrics]):.2f}",
        "Yesterday_cost_simulated": f"${np.mean([m['yest_cost'] for m in seed_metrics]):.2f} +/- ${np.std([m['yest_cost'] for m in seed_metrics]):.2f}",
        "NN_violation_hours": f"{np.mean([m['nn_viol_hours'] for m in seed_metrics]):.2f} +/- {np.std([m['nn_viol_hours'] for m in seed_metrics]):.2f}",
        "Yesterday_violation_hours": f"{np.mean([m['yest_viol_hours'] for m in seed_metrics]):.2f} +/- {np.std([m['yest_viol_hours'] for m in seed_metrics]):.2f}",
        "NN_violation_energy_kwh": f"{np.mean([m['nn_viol_kwh'] for m in seed_metrics]):.3f} +/- {np.std([m['nn_viol_kwh'] for m in seed_metrics]):.3f} kWh",
        "Yesterday_violation_energy_kwh": f"{np.mean([m['yest_viol_kwh'] for m in seed_metrics]):.3f} +/- {np.std([m['yest_viol_kwh'] for m in seed_metrics]):.3f} kWh",
        "violation_hour_reduction_pct": f"{(np.mean([m['yest_viol_hours'] for m in seed_metrics]) - np.mean([m['nn_viol_hours'] for m in seed_metrics])) / max(1e-4, np.mean([m['yest_viol_hours'] for m in seed_metrics])) * 100:.1f}%",
        "seed_details": seed_metrics
    }
    
    print("\n--- EXPERIMENT 3 SUMMARY (Mean +/- Std Dev across 5 seeds) ---")
    print(f"Simulated Cost:     NN = {summary['NN_cost_simulated']} vs Yesterday = {summary['Yesterday_cost_simulated']}")
    print(f"Violation Hours:    NN = {summary['NN_violation_hours']} vs Yesterday = {summary['Yesterday_violation_hours']} ({summary['violation_hour_reduction_pct']} reduction)")
    print(f"Violation Energy:   NN = {summary['NN_violation_energy_kwh']} vs Yesterday = {summary['Yesterday_violation_energy_kwh']}")
    
    return summary

def generate_experiment_plots(exp1: dict, exp2: dict, exp3: dict):
    """Generates and saves visual plots for each experiment."""
    print("\nGenerating publication figures for all 3 experiments...")
    os.makedirs("results/plots", exist_ok=True)
    
    # Plot 1: Exp 1 - UCS vs A*
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ucs_n = [r["ucs_nodes"] for r in exp1["seed_details"]]
    astar_n = [r["astar_nodes"] for r in exp1["seed_details"]]
    ucs_t = [r["ucs_time"] for r in exp1["seed_details"]]
    astar_t = [r["astar_time"] for r in exp1["seed_details"]]
    
    ax1.bar(["UCS", "A* Search"], [np.mean(ucs_n), np.mean(astar_n)],
            yerr=[np.std(ucs_n), np.std(astar_n)], capsize=8, color=['#4a7bb0', '#3b9a67'], alpha=0.85)
    ax1.set_ylabel("Nodes Expanded", fontsize=11, fontweight='bold')
    ax1.set_title("Exp 1: Search Node Expansions (Mean +/- Std)", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.bar(["UCS", "A* Search"], [np.mean(ucs_t), np.mean(astar_t)],
            yerr=[np.std(ucs_t), np.std(astar_t)], capsize=8, color=['#4a7bb0', '#3b9a67'], alpha=0.85)
    ax2.set_ylabel("Runtime (ms)", fontsize=11, fontweight='bold')
    ax2.set_title("Exp 1: Search Runtime (ms)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/plots/exp1_ucs_vs_astar.png", dpi=300)
    plt.close()
    
    # Plot 2: Exp 2 - CSP Pruning Scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    scales = [3, 5, 8]
    methods = ["BT", "BT_FC", "BT_FC_AC3"]
    colors = {'BT': '#e74c3c', 'BT_FC': '#f39c12', 'BT_FC_AC3': '#27ae60'}
    
    for m in methods:
        assign_means = [exp2[f"{n}_appliances"][m]["raw_means"]["assignments"] for n in scales]
        bt_means = [exp2[f"{n}_appliances"][m]["raw_means"]["backtracks"] for n in scales]
        ax1.plot(scales, assign_means, marker='o', linewidth=2, label=m, color=colors[m])
        ax2.plot(scales, bt_means, marker='s', linewidth=2, label=m, color=colors[m])
        
    ax1.set_xlabel("Number of Shiftable Appliances", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Assignments Tried", fontsize=11, fontweight='bold')
    ax1.set_title("Exp 2: CSP Variable Assignments vs Scale", fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.set_xlabel("Number of Shiftable Appliances", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Backtracks Count", fontsize=11, fontweight='bold')
    ax2.set_title("Exp 2: CSP Backtracks vs Scale", fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/plots/exp2_csp_scaling.png", dpi=300)
    plt.close()
    
    # Plot 3: Exp 3 - NN vs Yesterday Baseline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    nn_c = [m["nn_cost"] for m in exp3["seed_details"]]
    yest_c = [m["yest_cost"] for m in exp3["seed_details"]]
    nn_v = [m["nn_viol_hours"] for m in exp3["seed_details"]]
    yest_v = [m["yest_viol_hours"] for m in exp3["seed_details"]]
    
    ax1.bar(["Yesterday Baseline", "NN Forecast"], [np.mean(yest_c), np.mean(nn_c)],
            yerr=[np.std(yest_c), np.std(nn_c)], capsize=8, color=['#d9534f', '#20b2aa'], alpha=0.85)
    ax1.set_ylabel("Daily Simulated Cost (USD)", fontsize=11, fontweight='bold')
    ax1.set_title("Exp 3: Electricity Cost on Real Load (Simulated)", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.bar(["Yesterday Baseline", "NN Forecast"], [np.mean(yest_v), np.mean(nn_v)],
            yerr=[np.std(yest_v), np.std(nn_v)], capsize=8, color=['#d9534f', '#20b2aa'], alpha=0.85)
    ax2.set_ylabel("Limit Violation Hours per Day", fontsize=11, fontweight='bold')
    ax2.set_title("Exp 3: Grid Limit Violations (> 5 kW)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/plots/exp3_nn_vs_baseline.png", dpi=300)
    plt.close()
    
    print("  -> Saved results/plots/exp1_ucs_vs_astar.png")
    print("  -> Saved results/plots/exp2_csp_scaling.png")
    print("  -> Saved results/plots/exp3_nn_vs_baseline.png")

def main():
    test_path = "data/processed/test.csv"
    config_path = "data/processed/config.json"
    
    test_df = pd.read_csv(test_path)
    with open(config_path, "r") as f:
        config = json.load(f)
        
    exp1 = run_experiment_1(test_df, config, num_days=30)
    exp2 = run_experiment_2(test_df, config)
    exp3 = run_experiment_3(test_df, config, num_days=30)
    
    all_results = {
        "experiment_1_search": exp1,
        "experiment_2_csp": exp2,
        "experiment_3_integration": exp3
    }
    
    def convert_types(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open("results/experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=convert_types)
        
    generate_experiment_plots(exp1, exp2, exp3)
    print("\nAll experiments successfully completed and recorded to results/experiments_summary.json!")

if __name__ == '__main__':
    main()
