"""
make_data.py - Reproducible data processing pipeline for Home Energy Planner.
Preprocesses UCI Individual Household Electric Power Consumption dataset:
- Minute-to-hourly aggregation of Global Active Power (kW)
- Linear interpolation for missing gaps up to 3 hours (180 mins)
- Chronological train/validation/test split
- Constructs 24h lag, 168h lag (same day last week), and calendar features
- Creates simulated parameters: TOU pricing tariffs, appliance profiles, and battery configs
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Process UCI Household Power Consumption dataset.")
    parser.add_argument("--raw_path", type=str, default="data/raw/household_power_consumption.txt",
                        help="Path to raw household_power_consumption.txt")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save processed datasets and metadata")
    parser.add_argument("--max_gap_hours", type=int, default=3,
                        help="Maximum gap length in hours to linearly interpolate")
    return parser.parse_args()

def process_data(raw_path: str, output_dir: str, max_gap_hours: int = 3):
    print(f"Loading raw data from: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Read raw CSV
    # Delimiter is ';', missing values are '?'
    # We load Date, Time, and Global_active_power
    print("Reading CSV (Global_active_power)...")
    df = pd.read_csv(
        raw_path,
        sep=';',
        usecols=['Date', 'Time', 'Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
        na_values=['?'],
        dtype={
            'Global_active_power': 'float32',
            'Sub_metering_1': 'float32',
            'Sub_metering_2': 'float32',
            'Sub_metering_3': 'float32'
        },
        low_memory=False
    )

    print(f"Loaded {len(df):,} minute-level rows.")

    # 2. Parse DateTime index
    print("Parsing datetime index...")
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.set_index('Datetime')
    df = df.drop(columns=['Date', 'Time'])
    df = df.sort_index()

    # 3. Resample to hourly averages
    print("Resampling minute data to hourly means...")
    # Resample with mean
    hourly_df = df.resample('1h').agg({
        'Global_active_power': 'mean',
        'Sub_metering_1': 'mean',
        'Sub_metering_2': 'mean',
        'Sub_metering_3': 'mean'
    })

    print(f"Hourly dataset length: {len(hourly_df):,} hours ({hourly_df.index.min()} to {hourly_df.index.max()})")

    # 4. Handle missing values:
    # Interpolate gaps up to max_gap_hours (e.g., 3 hours)
    initial_nan_count = hourly_df['Global_active_power'].isna().sum()
    print(f"Initial missing hourly records: {initial_nan_count} ({initial_nan_count / len(hourly_df) * 100:.2f}%)")

    hourly_df['Global_active_power'] = hourly_df['Global_active_power'].interpolate(
        method='linear', limit=max_gap_hours
    )

    remaining_nan = hourly_df['Global_active_power'].isna().sum()
    print(f"Remaining missing hourly records after <= {max_gap_hours}h interpolation: {remaining_nan}")

    # Forward fill / backward fill any boundary edge cases
    hourly_df['Global_active_power'] = hourly_df['Global_active_power'].ffill().bfill()

    # 5. Build daily structured samples
    # For day D:
    # Target: 24 hourly loads for day D [h0..h23]
    # Input features:
    # - Yesterday loads: Day D-1 [h0..h23] (24 values)
    # - Same day last week loads: Day D-7 [h0..h23] (24 values)
    # - Time features: month (1-12), day of week (0-6), day of year
    # - Cyclical sine/cosine encodings for month and day of week
    print("Constructing 24-hour daily input-target pairs...")

    daily_dates = pd.date_range(
        start=hourly_df.index.min().floor('D') + pd.Timedelta(days=7), # Start after 7 days for lag
        end=hourly_df.index.max().floor('D'),
        freq='D'
    )

    samples = []
    for current_day in daily_dates:
        # Target day: current_day 00:00 to 23:00 (24 hours)
        target_hours = hourly_df.loc[current_day : current_day + pd.Timedelta(hours=23), 'Global_active_power']
        if len(target_hours) != 24:
            continue

        # Day D-1 (yesterday)
        yest_day = current_day - pd.Timedelta(days=1)
        yest_hours = hourly_df.loc[yest_day : yest_day + pd.Timedelta(hours=23), 'Global_active_power']
        if len(yest_hours) != 24:
            continue

        # Day D-7 (same day last week)
        last_week_day = current_day - pd.Timedelta(days=7)
        last_week_hours = hourly_df.loc[last_week_day : last_week_day + pd.Timedelta(hours=23), 'Global_active_power']
        if len(last_week_hours) != 24:
            continue

        row = {
            'date': current_day.strftime('%Y-%m-%d'),
            'month': current_day.month,
            'day_of_week': current_day.dayofweek,
            'is_weekend': int(current_day.dayofweek >= 5),
            'month_sin': np.sin(2 * np.pi * current_day.month / 12.0),
            'month_cos': np.cos(2 * np.pi * current_day.month / 12.0),
            'dow_sin': np.sin(2 * np.pi * current_day.dayofweek / 7.0),
            'dow_cos': np.cos(2 * np.pi * current_day.dayofweek / 7.0),
        }

        # Add 24 target values
        for h in range(24):
            row[f'target_h{h}'] = float(target_hours.iloc[h])

        # Add 24 yesterday values
        for h in range(24):
            row[f'yest_h{h}'] = float(yest_hours.iloc[h])

        # Add 24 last week values
        for h in range(24):
            row[f'lastweek_h{h}'] = float(last_week_hours.iloc[h])

        samples.append(row)

    dataset_df = pd.DataFrame(samples)
    print(f"Total valid daily samples constructed: {len(dataset_df)}")

    # 6. Chronological split:
    # Test: Final year (2010)
    # Validation: Last 3 months of pre-2010 (approx Oct 1, 2009 to Dec 31, 2009)
    # Train: Everything prior (late Dec 2006 / Jan 2007 through Sep 30, 2009)
    dataset_df['dt'] = pd.to_datetime(dataset_df['date'])

    train_df = dataset_df[dataset_df['dt'] < '2009-10-01'].copy()
    val_df = dataset_df[(dataset_df['dt'] >= '2009-10-01') & (dataset_df['dt'] < '2010-01-01')].copy()
    test_df = dataset_df[dataset_df['dt'] >= '2010-01-01'].copy()

    train_df = train_df.drop(columns=['dt'])
    val_df = val_df.drop(columns=['dt'])
    test_df = test_df.drop(columns=['dt'])

    print(f"Train samples: {len(train_df)} ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Val samples:   {len(val_df)} ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"Test samples:  {len(test_df)} ({test_df['date'].min()} to {test_df['date'].max()})")

    # Save CSVs
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # 7. Create configuration for simulated parameters
    # Real: Electricity use
    # Generated: TOU Electricity Tariffs, Appliance Profiles, Battery Parameters, Grid Power Limit
    config = {
        "pricing_tariff": {
            "name": "Three-Tier Time-Of-Use (TOU)",
            "currency": "USD/kWh",
            "type": "simulated",
            "rates": {
                "off_peak": {"price": 0.10, "hours": [0, 1, 2, 3, 4, 5, 6, 22, 23]},
                "mid_peak": {"price": 0.18, "hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
                "on_peak":  {"price": 0.35, "hours": [17, 18, 19, 20, 21]}
            },
            "hourly_prices": [
                0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  # 00:00 - 06:00
                0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, # 07:00 - 16:00
                0.35, 0.35, 0.35, 0.35, 0.35,              # 17:00 - 21:00
                0.10, 0.10                                 # 22:00 - 23:00
            ]
        },
        "power_limit_kw": 5.0,
        "battery": {
            "capacity_kwh": 10.0,
            "max_charge_kw": 3.0,
            "max_discharge_kw": 3.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
            "roundtrip_efficiency": 0.9025,
            "initial_soc_kwh": 2.0,
            "min_soc_kwh": 1.0,
            "max_soc_kwh": 10.0,
            "discretization_step_kwh": 0.5,
            "type": "simulated"
        },
        "appliances": {
            "washer": {
                "name": "Washing Machine",
                "power_kw": 1.5,
                "duration_hours": 2,
                "earliest_start": 7,
                "latest_start": 20,
                "circuit": 1,
                "profile": [1.5, 1.5],
                "type": "simulated"
            },
            "dryer": {
                "name": "Tumble Dryer",
                "power_kw": 2.2,
                "duration_hours": 2,
                "earliest_start": 8,
                "latest_start": 21,
                "circuit": 1,
                "predecessor": "washer",
                "profile": [2.2, 2.2],
                "type": "simulated"
            },
            "dishwasher": {
                "name": "Dishwasher",
                "power_kw": 1.8,
                "duration_hours": 2,
                "earliest_start": 12,
                "latest_start": 22,
                "circuit": 2,
                "profile": [1.8, 1.8],
                "type": "simulated"
            },
            "ev_charger": {
                "name": "Electric Vehicle Charger",
                "power_kw": 3.0,
                "duration_hours": 3,
                "earliest_start": 0,
                "latest_start": 20,
                "circuit": 3,
                "profile": [3.0, 3.0, 3.0],
                "type": "simulated"
            },
            "oven": {
                "name": "Electric Oven",
                "power_kw": 2.5,
                "duration_hours": 1,
                "earliest_start": 17,
                "latest_start": 21,
                "circuit": 2,
                "profile": [2.5],
                "type": "simulated"
            }
        }
    }

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration written to {config_path}")
    print("Dataset processing complete!")

if __name__ == '__main__':
    args = parse_args()
    process_data(args.raw_path, args.output_dir, args.max_gap_hours)
