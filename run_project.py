"""
run_project.py
--------------
One-click script that runs the entire predictive maintenance pipeline:
  1. Generate / load data
  2. Build features
  3. Train all models
  4. Print results summary
  5. Instructions for launching dashboard
"""

import os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("""
╔══════════════════════════════════════════════════════════════╗
║     PREDICTIVE MAINTENANCE — Equipment Failure Prediction    ║
║     NASA CMAPSS Dataset  |  Built for HelixIntel Portfolio   ║
╚══════════════════════════════════════════════════════════════╝
""")

# Step 1: Generate data
print("─" * 60)
print("STEP 1/3  Generate Data")
print("─" * 60)
from data.generate_data import generate_cmapss_data
import pandas as pd

if not os.path.exists('data/train_FD001.csv'):
    train_raw, test_raw, rul_raw = generate_cmapss_data()
else:
    print("  Data already exists — skipping generation")
    train_raw = pd.read_csv('data/train_FD001.csv')
    test_raw  = pd.read_csv('data/test_FD001.csv')
    rul_raw   = pd.read_csv('data/RUL_FD001.csv')

print(f"  Train engines: {train_raw['engine_id'].nunique()}")
print(f"  Test engines:  {test_raw['engine_id'].nunique()}")
print(f"  Total rows:    {len(train_raw):,}")

# Step 2: Feature engineering
print("\n" + "─" * 60)
print("STEP 2/3  Feature Engineering")
print("─" * 60)
from preprocessing import build_features, prepare_test_data

t0 = time.time()
train_feat, feat_cols, scaler = build_features(train_raw, is_train=True)
print(f"  Training features: {train_feat.shape}  ({time.time()-t0:.1f}s)")

test_last = prepare_test_data(test_raw, rul_raw)
test_feat, _, _ = build_features(test_last, is_train=False, scaler=scaler)
print(f"  Test features:     {test_feat.shape}")
print(f"  Total features:    {len(feat_cols)}")

train_feat.to_csv('data/train_features.csv', index=False)
test_feat.to_csv('data/test_features.csv',   index=False)

# Step 3: Train models
print("\n" + "─" * 60)
print("STEP 3/3  Train Models")
print("─" * 60)

import subprocess
result = subprocess.run([sys.executable, 'train_models.py'], capture_output=False)

# Final instructions
print("""
╔══════════════════════════════════════════════════════════════╗
║  ✅  PROJECT COMPLETE!                                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OUTPUT FILES:                                               ║
║    data/train_features.csv   — engineered training data      ║
║    data/test_features.csv    — engineered test data          ║
║    models/reg_*.pkl          — regression models             ║
║    models/clf_*.pkl          — classification models         ║
║    models/scaler.pkl         — fitted scaler                 ║
║    assets/model_report.json  — full metrics report           ║
║    assets/feature_importance.csv                             ║
║                                                              ║
║  LAUNCH DASHBOARD:                                           ║
║    streamlit run dashboard/app.py                            ║
║                                                              ║
║  DEPLOY FREE:                                                ║
║    1. Push this folder to GitHub                             ║
║    2. Go to share.streamlit.io                               ║
║    3. Connect your repo → done!                              ║
╚══════════════════════════════════════════════════════════════╝
""")
