"""
run_demo.py - Integrated demonstration of the Home Energy Planner.
Coordinating:
1. Neural Network 24-hour day-ahead load forecaster
2. Constraint Satisfaction Problem (CSP) appliance scheduler (AC-3 + FC)
3. A* Battery Search with energy storage efficiency losses
4. Dynamic Feedback Loop resolving grid limits

Generates console summary and saved multi-panel visualization plot.
NOTE: All costs and savings are SIMULATED.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Configure matplotlib cache directory before importing
os.environ['MPLCONFIGDIR'] = os.path.abspath(".mpl")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nn.evaluate import LoadForecaster
from csp.appliances import get_default_appliances
from csp.csp_solver import ApplianceCSP
from search.battery_env import BatteryEnv
from search.astar import AStarSearcher
from integration.feedback_loop import EnergyPlannerPipeline

def run_demonstration():
    print("=" * 80)
    print("           HOME ENERGY PLANNER - INTEGRATED SYSTEM DEMO           ")
    print("      Neural Network Forecast + CSP Scheduling + Battery Search   ")
    print("=" * 80)
    
    # 1. Load test data and configuration
    test_path = "data/processed/test.csv"
    config_path = "data/processed/config.json"
    
    if not os.path.exists(test_path) or not os.path.exists(config_path):
        print("Processed data not found. Please run 'python data/make_data.py' first.")
        return
        
    test_df = pd.read_csv(test_path)
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Select a representative test day (index 10 from 2010 test set)
    sample_idx = 10
    sample_row = test_df.iloc[sample_idx]
    sample_date = sample_row["date"]
    
    print(f"\n[STEP 1] Selected Test Day: {sample_date} (Day of week: {sample_row['day_of_week']})")
    
    # Target actual load
    actual_baseline = [float(sample_row[f"target_h{h}"]) for h in range(24)]
    yesterday_load = [float(sample_row[f"yest_h{h}"]) for h in range(24)]
    
    # 2. Neural Network Day-Ahead Forecast
    print("\n[STEP 2] Running Neural Network Day-Ahead 24-Hour Forecast...")
    forecaster = LoadForecaster(model_dir="nn")
    forecasted_baseline = list(forecaster.predict_row(sample_row))
    
    # Compute error metrics for this specific day
    mae_nn = float(np.mean(np.abs(np.array(actual_baseline) - np.array(forecasted_baseline))))
    mae_yest = float(np.mean(np.abs(np.array(actual_baseline) - np.array(yesterday_load))))
    print(f"  -> NN Forecast MAE:        {mae_nn:.4f} kW")
    print(f"  -> Yesterday Baseline MAE:  {mae_yest:.4f} kW")
    print(f"  -> Forecast Improvement:    {((mae_yest - mae_nn) / mae_yest * 100):.1f}% reduction in error")
    
    # 3. Setup Integrated Pipeline
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
    
    pipeline = EnergyPlannerPipeline(
        appliances=appliances,
        battery_env=battery_env,
        power_limit_kw=power_limit,
        max_feedback_loops=5,
        search_algorithm="AStar"
    )
    
    # 4. Demonstrate Feedback Loop
    print("\n[STEP 3] Running Integrated Planning Loop (NN -> CSP -> A* Battery Search)...")
    print("  Demonstrating feedback loop resolution...")
    plan_result = pipeline.plan_day(
        forecasted_baseline=forecasted_baseline,
        actual_baseline=actual_baseline,
        prices=prices,
        csp_method="BT_FC_AC3",
        force_feedback_test=True # Explicitly tests feedback loop trigger & recovery
    )
    
    # Print Feedback Loop History
    print("\n  === Feedback Loop Execution Trace ===")
    for h in plan_result["history"]:
        loop_num = h["loop"]
        sched = h["appliance_schedule"]
        viol = h["violation_detected"]
        if viol:
            print(f"  [Loop {loop_num}] CSP proposed schedule: {sched}")
            print(f"           -> Battery Search detected limit violation: +{h['max_violation_kw']} kW at hour {h['max_violation_hour']}:00")
            print(f"           -> FEEDBACK TRIGGERED: Forbidding schedule {sched} and re-scheduling CSP...")
        else:
            print(f"  [Loop {loop_num}] CSP re-scheduled to:   {sched}")
            print(f"           -> Battery Search SUCCESS! Grid power strictly <= {power_limit} kW.")
            print(f"           -> Search expanded {h['nodes_expanded']} nodes. Optimal plan found.")
            
    # 5. Summary of Solution
    sched = plan_result["appliance_schedule"]
    print("\n[STEP 4] Final Conflict-Free Appliance Schedule:")
    for name, start_h in sorted(sched.items(), key=lambda x: x[1]):
        app = appliances[name]
        end_h = start_h + app.duration_hours
        print(f"  - {app.name:<25} ({app.power_kw} kW): {start_h:02d}:00 to {end_h:02d}:00 (Circuit {app.circuit})")
        
    search_res = plan_result["search_result"]
    batt_actions = search_res["actions"]
    soc_trajectory = [s[1] for s in search_res["trajectory"]]
    
    # Compare Costs: Unplanned vs Planned
    # Unplanned: actual baseline + appliances placed at earliest start hour, no battery
    unplanned_loads = list(actual_baseline)
    for name, app in appliances.items():
        for dh in range(app.duration_hours):
            unplanned_loads[app.earliest_start + dh] += app.profile[dh]
    unplanned_cost = sum(unplanned_loads[h] * prices[h] for h in range(24))
    unplanned_peak = max(unplanned_loads)
    
    actual_eval = plan_result["actual_evaluation"]
    planned_cost = actual_eval["total_simulated_cost"]
    planned_peak = actual_eval["max_grid_load_kw"]
    
    cost_savings = unplanned_cost - planned_cost
    savings_pct = (cost_savings / unplanned_cost) * 100
    
    print("\n" + "=" * 80)
    print("                    PERFORMANCE EVALUATION RESULTS                    ")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Unplanned (Baseline)':<20} | {'Planned (System)':<20}")
    print("-" * 80)
    print(f"{'Total Cost (Simulated)':<35} | ${unplanned_cost:<19.2f} | ${planned_cost:<19.2f}")
    print(f"{'Peak Grid Load':<35} | {unplanned_peak:<17.2f} kW | {planned_peak:<17.2f} kW")
    print(f"{'Limit Violations (> 5.0 kW)':<35} | {sum(1 for p in unplanned_loads if p > 5.0):<20} | {actual_eval['num_violation_hours']:<20}")
    print(f"{'Cost Savings (Simulated)':<35} | {'-':<20} | ${cost_savings:.2f} ({savings_pct:.1f}%)")
    print("=" * 80)
    print("NOTE: All costs and monetary savings are SIMULATED based on the TOU tariff model.\n")
    
    # 6. Generate 4-Panel Visualization Plot
    print("[STEP 5] Generating 4-Panel Cooperation Visualization Plot...")
    os.makedirs("results/plots", exist_ok=True)
    plot_path = "results/plots/demo_integration_plot.png"
    
    hours = np.arange(24)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Panel 1: Forecasting
    axes[0].plot(hours, actual_baseline, 'k-', linewidth=2, label="Actual Household Load")
    axes[0].plot(hours, forecasted_baseline, 'b--', linewidth=2, label=f"NN Day-Ahead Forecast (MAE={mae_nn:.2f} kW)")
    axes[0].plot(hours, yesterday_load, 'g:', linewidth=1.5, alpha=0.7, label=f"Yesterday Baseline (MAE={mae_yest:.2f} kW)")
    axes[0].set_ylabel("Power (kW)", fontsize=11, fontweight='bold')
    axes[0].set_title("Panel 1: Neural Network Day-Ahead 24-Hour Load Forecasting", fontsize=12, fontweight='bold')
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: CSP Appliance Scheduling
    app_loads_matrix = np.zeros((len(appliances), 24))
    app_names = list(appliances.keys())
    for idx, (name, start_h) in enumerate(sched.items()):
        app = appliances[name]
        for dh in range(app.duration_hours):
            app_loads_matrix[idx, start_h + dh] = app.profile[dh]
            
    bottom = np.array(forecasted_baseline)
    axes[1].plot(hours, forecasted_baseline, 'k--', label="Forecasted Base Load", alpha=0.6)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, name in enumerate(app_names):
        app = appliances[name]
        axes[1].bar(hours, app_loads_matrix[idx], bottom=bottom, color=colors[idx % len(colors)],
                    alpha=0.8, label=f"{app.name} ({app.power_kw} kW)")
        bottom += app_loads_matrix[idx]
        
    axes[1].axhline(y=power_limit, color='r', linestyle='--', linewidth=2, label=f"Grid Limit ({power_limit} kW)")
    axes[1].set_ylabel("Power (kW)", fontsize=11, fontweight='bold')
    axes[1].set_title("Panel 2: CSP Appliance Scheduling (Conflict-Free under 5.0 kW Limit)", fontsize=12, fontweight='bold')
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: Battery Operation & SoC
    ax3_soc = axes[2].twinx()
    charge_bars = [max(0.0, a) for a in batt_actions]
    discharge_bars = [-min(0.0, a) for a in batt_actions]
    
    axes[2].bar(hours - 0.2, charge_bars, width=0.4, color='teal', alpha=0.8, label="Battery Charge (kW)")
    axes[2].bar(hours + 0.2, discharge_bars, width=0.4, color='darkorange', alpha=0.8, label="Battery Discharge (kW)")
    ax3_soc.plot(hours, soc_trajectory[:24], 'm-o', linewidth=2, markersize=5, label="Battery SoC (kWh)")
    
    axes[2].set_ylabel("Battery Power (kW)", fontsize=11, fontweight='bold', color='darkslategray')
    ax3_soc.set_ylabel("SoC (kWh)", fontsize=11, fontweight='bold', color='purple')
    axes[2].set_title("Panel 3: A* Battery Scheduling (Charge Off-Peak, Discharge On-Peak)", fontsize=12, fontweight='bold')
    
    lines_1, labels_1 = axes[2].get_legend_handles_labels()
    lines_2, labels_2 = ax3_soc.get_legend_handles_labels()
    axes[2].legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", framealpha=0.9)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    # Panel 4: Net Grid Load & Electricity Tariff
    ax4_tariff = axes[3].twinx()
    actual_grid = actual_eval["actual_grid_loads"]
    axes[3].plot(hours, unplanned_loads, 'r:', linewidth=2, alpha=0.8, label=f"Unplanned Load (Peak: {unplanned_peak:.1f} kW)")
    axes[3].plot(hours, actual_grid, 'navy', linewidth=2.5, label=f"Planned Grid Load (Peak: {planned_peak:.1f} kW)")
    axes[3].axhline(y=power_limit, color='crimson', linestyle='--', linewidth=2, label=f"Limit ({power_limit} kW)")
    
    ax4_tariff.step(hours, prices, where='mid', color='goldenrod', linewidth=2, linestyle='-.', label="TOU Price ($/kWh)")
    
    axes[3].set_xlabel("Hour of Day (00:00 - 23:00)", fontsize=11, fontweight='bold')
    axes[3].set_ylabel("Grid Power (kW)", fontsize=11, fontweight='bold')
    ax4_tariff.set_ylabel("Tariff (USD/kWh)", fontsize=11, fontweight='bold', color='goldenrod')
    axes[3].set_title(f"Panel 4: Real-World Grid Load vs Simulated Cost Tariff (${planned_cost:.2f} vs ${unplanned_cost:.2f})", fontsize=12, fontweight='bold')
    
    lines_a, labels_a = axes[3].get_legend_handles_labels()
    lines_b, labels_b = ax4_tariff.get_legend_handles_labels()
    axes[3].legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", framealpha=0.9)
    axes[3].grid(True, linestyle='--', alpha=0.5)
    axes[3].set_xticks(hours)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  -> Plot successfully saved to {plot_path}")
    print("\nIntegrated demonstration complete!")

if __name__ == '__main__':
    run_demonstration()
