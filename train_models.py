"""
train_models.py
---------------
Trains three models for predictive maintenance:
  1. Random Forest      — interpretable baseline
  2. Gradient Boosting  — best performance (XGBoost-style via sklearn)
  3. Ridge Regression   — fast linear baseline for comparison

Evaluates both:
  - RUL Regression  (RMSE, MAE, R²)
  - Failure Classification  (Accuracy, Precision, Recall, F1, ROC-AUC)

Saves all models + evaluation report.
"""

import pandas as pd
import numpy as np
import pickle, os, json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('assets', exist_ok=True)


# ── Load data ────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv('data/train_features.csv')
    test  = pd.read_csv('data/test_features.csv')

    with open('models/feature_cols.pkl', 'rb') as f:
        feat_cols = pickle.load(f)

    # Filter to only columns present in both datasets
    feat_cols = [c for c in feat_cols if c in train.columns and c in test.columns]

    X_train = train[feat_cols]
    y_rul_train  = train['RUL']
    y_clf_train  = train['will_fail_soon']

    X_test  = test[feat_cols]
    y_rul_test   = test['RUL']
    y_clf_test   = test['will_fail_soon']

    return X_train, X_test, y_rul_train, y_rul_test, y_clf_train, y_clf_test, feat_cols


# ── Regression models ────────────────────────────────────────────────────────
def train_regression(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest':       RandomForestRegressor(n_estimators=100, max_depth=12,
                                                     min_samples_leaf=5, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                                          learning_rate=0.05, random_state=42),
        'Ridge Regression':    Ridge(alpha=1.0),
    }

    results = {}
    print("\n" + "="*55)
    print("  REGRESSION — Remaining Useful Life (RUL) Prediction")
    print("="*55)
    print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-"*55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        results[name] = {'RMSE': round(rmse,2), 'MAE': round(mae,2), 'R2': round(r2,3),
                         'predictions': preds.tolist(), 'actuals': y_test.tolist()}
        print(f"{name:<25} {rmse:>8.2f} {mae:>8.2f} {r2:>8.3f}")

        key = name.lower().replace(' ', '_')
        with open(f'models/reg_{key}.pkl', 'wb') as f:
            pickle.dump(model, f)

    best = min(results, key=lambda x: results[x]['RMSE'])
    print(f"\n  ✓ Best model: {best}  (RMSE={results[best]['RMSE']})")
    return results, best


# ── Classification models ────────────────────────────────────────────────────
def train_classification(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=12,
                                                      min_samples_leaf=5, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                           learning_rate=0.05, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    print("\n" + "="*65)
    print("  CLASSIFICATION — Will Fail Within 30 Cycles?")
    print("="*65)
    print(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'F1':>7} {'AUC':>7}")
    print("-"*65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds      = model.predict(X_test)
        probs      = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else preds

        acc   = accuracy_score(y_test, preds)
        prec  = precision_score(y_test, preds, zero_division=0)
        rec   = recall_score(y_test, preds, zero_division=0)
        f1    = f1_score(y_test, preds, zero_division=0)
        auc   = roc_auc_score(y_test, probs)

        results[name] = {
            'Accuracy': round(acc,3), 'Precision': round(prec,3),
            'Recall': round(rec,3),   'F1': round(f1,3), 'AUC': round(auc,3),
            'predictions': preds.tolist(), 'probabilities': probs.tolist(),
            'actuals': y_test.tolist()
        }
        print(f"{name:<25} {acc:>7.3f} {prec:>7.3f} {rec:>8.3f} {f1:>7.3f} {auc:>7.3f}")

        key = name.lower().replace(' ', '_')
        with open(f'models/clf_{key}.pkl', 'wb') as f:
            pickle.dump(model, f)

    best = max(results, key=lambda x: results[x]['F1'])
    print(f"\n  ✓ Best model: {best}  (F1={results[best]['F1']})")
    return results, best


# ── Feature importance ───────────────────────────────────────────────────────
def get_feature_importance(feat_cols):
    with open('models/reg_random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)

    importance = pd.DataFrame({
        'feature':   feat_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    importance.to_csv('assets/feature_importance.csv', index=False)
    print(f"\n  Top 5 most important features:")
    for _, row in importance.head(5).iterrows():
        bar = '█' * int(row['importance'] * 200)
        print(f"    {row['feature']:<30} {bar} {row['importance']:.4f}")
    return importance


# ── Save full report ─────────────────────────────────────────────────────────
def save_report(reg_results, clf_results, reg_best, clf_best):
    report = {
        'regression':     {'results': reg_results, 'best_model': reg_best},
        'classification': {'results': clf_results, 'best_model': clf_best},
        'summary': {
            'best_rul_rmse':  reg_results[reg_best]['RMSE'],
            'best_rul_r2':    reg_results[reg_best]['R2'],
            'best_clf_f1':    clf_results[clf_best]['F1'],
            'best_clf_auc':   clf_results[clf_best]['AUC'],
        }
    }
    with open('assets/model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to assets/model_report.json")
    return report


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔧 Loading feature data...")
    X_train, X_test, y_rul_train, y_rul_test, y_clf_train, y_clf_test, feat_cols = load_data()
    print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"   Features: {len(feat_cols)}")
    print(f"   Failure rate (train): {y_clf_train.mean():.1%}")

    print("\n🌲 Training regression models...")
    reg_results, reg_best = train_regression(X_train, X_test, y_rul_train, y_rul_test)

    print("\n🎯 Training classification models...")
    clf_results, clf_best = train_classification(X_train, X_test, y_clf_train, y_clf_test)

    print("\n📊 Feature importance...")
    importance = get_feature_importance(feat_cols)

    report = save_report(reg_results, clf_results, reg_best, clf_best)

    print("\n" + "="*55)
    print("  FINAL SUMMARY")
    print("="*55)
    print(f"  RUL Prediction  →  RMSE: {report['summary']['best_rul_rmse']}  R²: {report['summary']['best_rul_r2']}")
    print(f"  Failure Alert   →  F1:   {report['summary']['best_clf_f1']}   AUC: {report['summary']['best_clf_auc']}")
    print("="*55)
    print("\n✅ All models saved to models/")
