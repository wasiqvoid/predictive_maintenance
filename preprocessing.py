"""
preprocessing.py
----------------
Feature engineering and preprocessing pipeline for the
NASA CMAPSS predictive maintenance project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle, os

USEFUL_SENSORS = [
    's2','s3','s4','s7','s8','s9',
    's11','s12','s13','s14','s15','s17','s20','s21'
]
WINDOW_SIZES   = [5, 10, 20]
FAIL_THRESHOLD = 30   # "will fail within N cycles?"


def compute_rul(df):
    """Add RUL column: cycles remaining before engine failure."""
    max_cycles = df.groupby('engine_id')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    return df


def add_rolling_features(df, sensors=USEFUL_SENSORS, windows=WINDOW_SIZES):
    """Add rolling mean & std for each sensor at each window size."""
    new_cols = {}
    for sensor in sensors:
        for w in windows:
            new_cols[f'{sensor}_mean_{w}'] = (df.groupby('engine_id')[sensor]
                                               .transform(lambda x: x.rolling(w, min_periods=1).mean()))
            new_cols[f'{sensor}_std_{w}']  = (df.groupby('engine_id')[sensor]
                                               .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0)))
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_lag_features(df, sensors=USEFUL_SENSORS, lags=[1, 3, 5]):
    """Add lagged sensor values (rate of change signals)."""
    new_cols = {}
    for sensor in sensors:
        for lag in lags:
            new_cols[f'{sensor}_lag_{lag}'] = (df.groupby('engine_id')[sensor]
                                                 .transform(lambda x: x.shift(lag).bfill()))
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_cycle_features(df):
    """Add normalized cycle position within each engine's life."""
    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    df['cycle_norm']   = df['cycle'] / max_cycles
    df['cycle_sqrt']   = np.sqrt(df['cycle'])
    df['cycle_log']    = np.log1p(df['cycle'])
    return df


def clip_rul(df, max_rul=130):
    """
    Clip RUL at max_rul — standard practice for CMAPSS.
    Early in life, all engines are equally healthy so
    predicting exact RUL=300 vs RUL=250 doesn't matter.
    """
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    return df


def add_binary_label(df, threshold=FAIL_THRESHOLD):
    """Add will_fail_soon: 1 if RUL <= threshold, 0 otherwise."""
    df['will_fail_soon'] = (df['RUL'] <= threshold).astype(int)
    return df


def get_feature_cols(df):
    """Return all feature columns (excludes metadata and targets)."""
    exclude = {'engine_id','cycle','RUL','will_fail_soon','max_cycle'}
    return [c for c in df.columns if c not in exclude]


def build_features(df, is_train=True, scaler=None):
    """Full feature engineering pipeline."""
    df = df.copy()
    if is_train:
        df = compute_rul(df)
        df = clip_rul(df)
        df = add_binary_label(df)
    df = add_cycle_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df.fillna(0, inplace=True)

    feature_cols = get_feature_cols(df)

    if is_train:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        os.makedirs('models', exist_ok=True)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('models/feature_cols.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
    else:
        if scaler is None:
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, feature_cols, scaler


def prepare_test_data(test_df, rul_df):
    """Prepare test set — use last cycle per engine + true RUL."""
    last_cycles = test_df.groupby('engine_id').last().reset_index()
    last_cycles = last_cycles.merge(rul_df, on='engine_id')
    last_cycles['will_fail_soon'] = (last_cycles['RUL'] <= FAIL_THRESHOLD).astype(int)
    return last_cycles


if __name__ == "__main__":
    train = pd.read_csv('data/train_FD001.csv')
    test  = pd.read_csv('data/test_FD001.csv')
    rul   = pd.read_csv('data/RUL_FD001.csv')

    print("Building training features...")
    train_feat, feat_cols, scaler = build_features(train, is_train=True)
    print(f"  Shape: {train_feat.shape}  |  Features: {len(feat_cols)}")

    test_last = prepare_test_data(test, rul)
    print("Building test features...")
    test_feat, _, _ = build_features(test_last, is_train=False, scaler=scaler)
    print(f"  Shape: {test_feat.shape}")

    train_feat.to_csv('data/train_features.csv', index=False)
    test_feat.to_csv('data/test_features.csv',   index=False)
    print("Feature files saved to data/")
