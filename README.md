# ESI Triage Classification Models

## Objective
Classify patients into **5 Emergency Severity Index (ESI) levels** using presentation-time vital signs from the **PhysioNet 2019 Sepsis dataset**.  
A total of **11 machine learning models**, including a **stacking ensemble**, were trained and evaluated.

---

## Proxy ESI Labeling
Since true ESI labels are not available in the dataset, a proxy labeling strategy was used:

**MEWS + Shock Index + O2Sat thresholds → ESI levels (1–5)**

See `data_loader.py` for the complete heuristic implementation.

---

## Dataset Information

| Attribute | Value |
|----------|------|
| Total Valid Patients Extracted | 40,336 |
| Training Set | 32,268 |
| Testing Set | 8,068 |
| Feature Set | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 |
| Processing | 80/20 stratified split, median imputation, standard scaling |

---

## Model Performance Summary

| Model | Accuracy | Weighted F1 |
|------|---------|-------------|
| LightGBM (Tuned) | **98.36%** | **0.9835** |
| Stacking (Tuned) | 98.31% | 0.9830 |
| CatBoost | 98.09% | 0.9809 |
| DART (LightGBM) | 97.81% | 0.9780 |
| XGBoost | 97.72% | 0.9771 |
| HistGradientBoosting | 97.20% | 0.9719 |
| RandomForest | 97.12% | 0.9712 |
| StochasticGB | 96.18% | 0.9615 |
| GradientBoosting | 94.96% | 0.9492 |
| ExtraTrees | 93.85% | 0.9377 |
| NGBoost | 70.41% | 0.6980 |
| AdaBoost | 56.16% | 0.4642 |
| AdaptiveGB | 48.23% | 0.3529 |

**Note:**  
Hyperparameter tuning using **Optuna** improved the performance of top-tier models, with **LightGBM achieving the highest accuracy (98.36%)**.

