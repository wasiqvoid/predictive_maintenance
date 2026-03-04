# 🔧 Predictive Maintenance — Equipment Failure Prediction

**NASA CMAPSS Turbofan Engine Dataset | Machine Learning | Streamlit Dashboard**

> Built by Wasiq Bakhsh — MS Data Science, University at Buffalo (Fall 2026)
> Portfolio project for HelixIntel PropertyOS internship application

---

## 🎯 Project Overview

This project builds a predictive maintenance system that predicts **when industrial equipment will fail** before it actually does — the exact problem HelixIntel's PropertyOS platform solves for maintenance teams.

**Two ML tasks:**
- **RUL Regression** — predict exact remaining useful life in cycles
- **Failure Classification** — alert when failure is imminent (within 30 cycles)

---

## 📊 Results

| Task | Model | Score |
|------|-------|-------|
| RUL Prediction | Gradient Boosting | RMSE ~18 cycles |
| Failure Alert | Random Forest | F1 ~0.91, AUC ~0.96 |

---

## 🏗️ Project Structure

```
predictive_maintenance/
│
├── data/
│   ├── generate_data.py      ← Simulates NASA CMAPSS dataset
│   ├── train_FD001.csv       ← Training data (150 engines)
│   └── test_FD001.csv        ← Test data (50 engines)
│
├── preprocessing.py          ← Feature engineering pipeline
├── train_models.py           ← Model training + evaluation
├── run_project.py            ← One-click full pipeline runner
│
├── dashboard/
│   └── app.py                ← Streamlit interactive dashboard
│
├── models/                   ← Saved model files (.pkl)
├── assets/                   ← Reports + feature importance
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/predictive-maintenance
cd predictive-maintenance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (data → features → models)
python run_project.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

---

## 🔬 Technical Approach

### Dataset
NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
- 100–200 engines per subset
- 21 sensor readings per cycle
- Run-to-failure trajectories
- Real dataset: [Kaggle](https://kaggle.com/datasets/behrad3d/nasa-cmaps)

### Feature Engineering
- **Rolling statistics** — mean & std at windows 5, 10, 20 cycles
- **Lag features** — rate of change signals (lags 1, 3, 5)
- **Cycle normalization** — position within engine's expected life
- **RUL clipping** — capped at 130 cycles (industry standard)
- **MinMax scaling** — all features normalized to [0,1]

### Models
| Model | Type | Why |
|-------|------|-----|
| Random Forest | Regression + Classification | Interpretable, strong baseline |
| Gradient Boosting | Regression + Classification | Best performance |
| Ridge / Logistic | Regression + Classification | Fast linear baseline |

### Dashboard Features
- 📊 Fleet health overview with KPI cards
- 🔍 Individual engine deep-dive with sensor plots
- 🚨 Alert board ranked by risk score
- 📈 Model performance comparison
- 🔑 Feature importance visualization
  
---

## 📬 Contact

**Wasiq Bakhsh**
MS Data Science — University at Buffalo
[LinkedIn](https://linkedin.com/in/wasiqcyber) | wasiqcyber@gmail.com
# predictive_maintenance
