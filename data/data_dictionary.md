# Data Dictionary - Home Energy Planner

This document provides a comprehensive description of every variable, column, and configuration parameter used across the Home Energy Planner project, distinguishing between empirical real-world measurements and simulated generated parameters.

---

## 1. Raw Dataset: UCI Individual Household Electric Power Consumption

- **Origin / Link**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Source**: One single-family home located in Sceaux, near Paris, France.
- **Duration**: December 16, 2006 to November 26, 2010 (47 months, ~2,075,259 measurements).
- **Sampling Frequency**: 1-minute intervals.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Missing Values**: Represented by `?` in raw CSV (approx 1.25% of rows).

| Column Name | Type | Unit | Nature | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Date` | String | `DD/MM/YYYY` | Real | Calendar date of observation. |
| `Time` | String | `HH:MM:SS` | Real | Time of observation (24-hour format). |
| `Global_active_power` | Float | Kilowatts (kW) | **Real** | Household global minute-averaged active power. |
| `Global_reactive_power`| Float | Kilowatts (kW) | Real | Household global minute-averaged reactive power. |
| `Voltage` | Float | Volts (V) | Real | Minute-averaged voltage. |
| `Global_intensity` | Float | Amperes (A) | Real | Minute-averaged current intensity. |
| `Sub_metering_1` | Float | Watt-hours (Wh)| Real | Kitchen appliances (dishwasher, oven, microwave). |
| `Sub_metering_2` | Float | Watt-hours (Wh)| Real | Laundry room appliances (washer, dryer, fridge, lights). |
| `Sub_metering_3` | Float | Watt-hours (Wh)| Real | Electric water-heater and air-conditioner. |

---

## 2. Processed Daily Forecasting Dataset (`train.csv`, `val.csv`, `test.csv`)

Preprocessing performed by `data/make_data.py`:
1. Resampled 1-minute measurements to 1-hour arithmetic means.
2. Linear interpolation for missing gaps $\le 3$ hours (longer gaps dropped or forward/backward filled at boundaries).
3. Supervised window extraction for each day $d$ with a 24-hour planning horizon ($h \in [0, 23]$).

| Column Name | Type | Unit | Nature | Origin / Generation Method |
| :--- | :--- | :--- | :--- | :--- |
| `date` | String | `YYYY-MM-DD`| Real | Target forecast day date. |
| `month` | Integer | 1 - 12 | Real | Calendar month of target day. |
| `day_of_week` | Integer | 0 - 6 | Real | Day of week (0 = Monday, 6 = Sunday). |
| `is_weekend` | Integer | 0 or 1 | Real | Binary indicator: 1 if `day_of_week` $\ge 5$, else 0. |
| `month_sin` | Float | [-1, 1] | Generated | Cyclical encoding: $\sin(2\pi \cdot \text{month} / 12)$. |
| `month_cos` | Float | [-1, 1] | Generated | Cyclical encoding: $\cos(2\pi \cdot \text{month} / 12)$. |
| `dow_sin` | Float | [-1, 1] | Generated | Cyclical encoding: $\sin(2\pi \cdot \text{day\_of\_week} / 7)$. |
| `dow_cos` | Float | [-1, 1] | Generated | Cyclical encoding: $\cos(2\pi \cdot \text{day\_of\_week} / 7)$. |
| `target_h0` ... `target_h23` | Float | kW | **Real** | Actual hourly active power consumption for day $d$ (24 values, NN regression target). |
| `yest_h0` ... `yest_h23` | Float | kW | **Real** | Hourly active power consumption for day $d-1$ (yesterday's 24 hours, NN input). |
| `lastweek_h0` ... `lastweek_h23` | Float | kW | **Real** | Hourly active power consumption for day $d-7$ (same day last week, NN input). |

---

## 3. Simulated Parameters & Configuration (`config.json`)

*All cost, pricing, appliance flexibility, and battery parameters are simulated models designed to evaluate the optimization algorithms.*

### A. Electricity Pricing Tariff (Time-of-Use / TOU)
Generated based on realistic 3-tier residential dynamic tariff schemes:
- **Currency**: USD (\$) per kWh (Simulated).
- **Off-Peak** ($0.10 / kWh): Hours 00:00–06:59 and 22:00–23:59.
- **Mid-Peak** ($0.18 / kWh): Hours 07:00–16:59.
- **On-Peak** ($0.35 / kWh): Hours 17:00–21:59 (high demand evening peak).

### B. Grid Power Constraint
- **`power_limit_kw`**: **5.0 kW**. Simulates the maximum allowable continuous grid power drawn by the household main circuit breaker before tripping or triggering demand penalties.

### C. Home Battery Energy Storage System (BESS)
- **`capacity_kwh`**: 10.0 kWh (nominal capacity).
- **`max_charge_kw`**: 3.0 kW (max charge power rate).
- **`max_discharge_kw`**: 3.0 kW (max discharge power rate).
- **`charge_efficiency`**: 95% ($\eta_c = 0.95$).
- **`discharge_efficiency`**: 95% ($\eta_d = 0.95$).
- **`roundtrip_efficiency`**: $90.25\%$ ($\eta_{rt} = \eta_c \cdot \eta_d$).
- **`min_soc_kwh`**: 1.0 kWh (10% depth-of-discharge reserve for battery life preservation).
- **`max_soc_kwh`**: 10.0 kWh (100% state of charge).
- **`initial_soc_kwh`**: 2.0 kWh (starting reserve at hour 00:00).
- **`discretization_step_kwh`**: 0.5 kWh step size for search state-space discretization.

### D. Shiftable Household Appliances
Modeled after standard residential electrical appliances:
1. **Washing Machine (`washer`)**:
   - Power: 1.5 kW | Duration: 2 hours | Earliest start: 07:00 | Latest start: 20:00 | Circuit: 1
2. **Tumble Dryer (`dryer`)**:
   - Power: 2.2 kW | Duration: 2 hours | Earliest start: 08:00 | Latest start: 21:00 | Circuit: 1
   - **Precedence Constraint**: Must run strictly after `washer` has completed.
3. **Dishwasher (`dishwasher`)**:
   - Power: 1.8 kW | Duration: 2 hours | Earliest start: 12:00 | Latest start: 22:00 | Circuit: 2
4. **Electric Vehicle Charger (`ev_charger`)**:
   - Power: 3.0 kW | Duration: 3 hours | Earliest start: 00:00 | Latest start: 20:00 | Circuit: 3
5. **Electric Oven (`oven`)**:
   - Power: 2.5 kW | Duration: 1 hour | Earliest start: 17:00 | Latest start: 21:00 | Circuit: 2
   - **Mutual Exclusion Constraint**: Shares Circuit 2 with `dishwasher`; cannot run simultaneously.

---

## 4. Warnings and Domain Limitations
- **Single Household**: Data represents a single French household; cross-household generalization is simulated.
- **No Direct Holiday / Temperature Features**: Public dataset lacks temperature and localized calendar holiday records.
- **Simulated Economic Impact**: All financial savings ($) and peak reduction metrics reflect algorithmic simulation models.
