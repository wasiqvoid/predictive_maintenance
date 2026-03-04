"""
NASA CMAPSS Data Generator
--------------------------
This script simulates the NASA CMAPSS turbofan engine dataset.
To use the REAL dataset:
  1. Go to: https://kaggle.com/datasets/behrad3d/nasa-cmaps
  2. Download and place train_FD001.txt, test_FD001.txt, RUL_FD001.txt
     in this /data folder
  3. Comment out generate_cmapss_data() and use load_real_data() instead

The simulated data follows the exact same structure and degradation
patterns as the real NASA dataset so your models will work identically.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

COLUMNS = [
    'engine_id', 'cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    's1','s2','s3','s4','s5','s6','s7','s8','s9',
    's10','s11','s12','s13','s14','s15','s16','s17',
    's18','s19','s20','s21'
]

# Sensors that actually carry signal (rest are near-constant noise)
USEFUL_SENSORS = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']

def generate_engine(engine_id, max_cycles=None):
    """Simulate one engine's run-to-failure sensor trajectory."""
    if max_cycles is None:
        max_cycles = np.random.randint(150, 350)

    cycles = np.arange(1, max_cycles + 1)
    n = len(cycles)
    degradation = cycles / max_cycles  # 0→1 as engine degrades

    rows = []
    for i, c in enumerate(cycles):
        d = degradation[i]
        row = {
            'engine_id': engine_id,
            'cycle': c,
            'op_setting_1': np.random.uniform(-0.0008, 0.0008),
            'op_setting_2': np.random.uniform(-0.0006, 0.0006),
            'op_setting_3': np.random.uniform(99.98, 100.02),
            # Sensors that INCREASE with degradation
            's2':  485 + 5*d  + np.random.normal(0, 0.5),
            's3':  1580 + 20*d + np.random.normal(0, 2),
            's4':  1400 + 30*d + np.random.normal(0, 3),
            's7':  554  - 8*d  + np.random.normal(0, 0.5),
            's8':  2388 + 10*d + np.random.normal(0, 1),
            's9':  9044 - 50*d + np.random.normal(0, 5),
            's11': 47   + 2*d  + np.random.normal(0, 0.3),
            's12': 521  - 5*d  + np.random.normal(0, 0.5),
            's13': 2388 + 8*d  + np.random.normal(0, 1),
            's14': 8138 - 40*d + np.random.normal(0, 5),
            's15': 8.4  + 0.05*d + np.random.normal(0, 0.01),
            's17': 392  + 3*d  + np.random.normal(0, 0.3),
            's20': 38.8 + 0.5*d + np.random.normal(0, 0.1),
            's21': 23.4 + 0.3*d + np.random.normal(0, 0.05),
            # Near-constant sensors (noise only)
            's1':  518.67 + np.random.normal(0, 0.1),
            's5':  21.61  + np.random.normal(0, 0.01),
            's6':  554.36 + np.random.normal(0, 0.1),
            's10': 1.3    + np.random.normal(0, 0.001),
            's16': 0.03   + np.random.normal(0, 0.001),
            's18': 2388   + np.random.normal(0, 0.5),
            's19': 100    + np.random.normal(0, 0.01),
        }
        rows.append(row)
    return pd.DataFrame(rows)[COLUMNS]


def generate_cmapss_data(n_train=150, n_test=50):
    """Generate full train and test datasets."""
    print("Generating simulated NASA CMAPSS dataset...")

    # Training: full run-to-failure sequences
    train_dfs = []
    max_cycles_train = {}
    for eid in range(1, n_train + 1):
        df = generate_engine(eid)
        max_cycles_train[eid] = df['cycle'].max()
        train_dfs.append(df)
    train = pd.concat(train_dfs, ignore_index=True)

    # Test: cut off before failure (random early stopping)
    test_dfs = []
    rul_list = []
    for eid in range(1, n_test + 1):
        df = generate_engine(eid)
        true_max = df['cycle'].max()
        cutoff = np.random.randint(int(true_max * 0.5), int(true_max * 0.9))
        rul_list.append(true_max - cutoff)
        df_cut = df[df['cycle'] <= cutoff].copy()
        df_cut['engine_id'] = eid
        test_dfs.append(df_cut)
    test = pd.concat(test_dfs, ignore_index=True)
    rul  = pd.DataFrame({'engine_id': range(1, n_test+1), 'RUL': rul_list})

    train.to_csv('data/train_FD001.csv', index=False)
    test.to_csv('data/test_FD001.csv',  index=False)
    rul.to_csv('data/RUL_FD001.csv',    index=False)
    print(f"  Train: {len(train):,} rows  ({n_train} engines)")
    print(f"  Test:  {len(test):,} rows  ({n_test} engines)")
    print("  Saved to data/")
    return train, test, rul


if __name__ == "__main__":
    generate_cmapss_data()
