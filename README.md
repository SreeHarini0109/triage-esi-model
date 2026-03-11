# ESI Triage Classification Models

 **Live Demo:** https://triage-esi-model.onrender.com/

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

# Top 3 Models — Detailed Metrics

## 1. LightGBM
**Accuracy:** 98.33%  
**Weighted Quadratic Cohen’s Kappa:** 0.9911  

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|---------|--------|
| ESI 1 | 0.99 | 0.97 | 0.98 | 266 |
| ESI 2 | 0.98 | 0.96 | 0.97 | 785 |
| ESI 3 | 0.98 | 0.99 | 0.99 | 3174 |
| ESI 4 | 0.98 | 0.99 | 0.98 | 2421 |
| ESI 5 | 0.99 | 0.96 | 0.98 | 1422 |

**Overall Accuracy:** 0.98  
**Total Samples:** 8068  

### Confusion Matrix

| Actual \ Predicted | ESI1 | ESI2 | ESI3 | ESI4 | ESI5 |
|--------------------|------|------|------|------|------|
| **ESI1** | 259 | 7 | 0 | 0 | 0 |
| **ESI2** | 3 | 757 | 25 | 0 | 0 |
| **ESI3** | 0 | 11 | 3156 | 5 | 2 |
| **ESI4** | 0 | 0 | 24 | 2389 | 8 |
| **ESI5** | 0 | 0 | 0 | 50 | 1372 |

---

## 2. Stacking (LightGBM + CatBoost + XGBoost + RandomForest)

**Accuracy:** 98.30%  
**Weighted Quadratic Cohen’s Kappa:** 0.9908  

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|---------|--------|
| ESI 1 | 0.98 | 0.97 | 0.98 | 266 |
| ESI 2 | 0.97 | 0.97 | 0.97 | 785 |
| ESI 3 | 0.99 | 0.99 | 0.99 | 3174 |
| ESI 4 | 0.98 | 0.99 | 0.98 | 2421 |
| ESI 5 | 0.99 | 0.97 | 0.98 | 1422 |

**Overall Accuracy:** 0.98  
**Total Samples:** 8068  

### Confusion Matrix

| Actual \ Predicted | ESI1 | ESI2 | ESI3 | ESI4 | ESI5 |
|--------------------|------|------|------|------|------|
| **ESI1** | 258 | 8 | 0 | 0 | 0 |
| **ESI2** | 4 | 759 | 22 | 0 | 0 |
| **ESI3** | 0 | 12 | 3149 | 10 | 3 |
| **ESI4** | 0 | 0 | 24 | 2391 | 6 |
| **ESI5** | 0 | 0 | 0 | 48 | 1374 |

---

## 3. CatBoost

**Accuracy:** 98.09%  
**Weighted Quadratic Cohen’s Kappa:** 0.9890  

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|---------|--------|
| ESI 1 | 1.00 | 0.96 | 0.98 | 266 |
| ESI 2 | 0.97 | 0.97 | 0.97 | 785 |
| ESI 3 | 0.98 | 0.99 | 0.99 | 3174 |
| ESI 4 | 0.97 | 0.99 | 0.98 | 2421 |
| ESI 5 | 0.99 | 0.96 | 0.98 | 1422 |

**Overall Accuracy:** 0.98  
**Total Samples:** 8068  

### Confusion Matrix

| Actual \ Predicted | ESI1 | ESI2 | ESI3 | ESI4 | ESI5 |
|--------------------|------|------|------|------|------|
| **ESI1** | 255 | 11 | 0 | 0 | 0 |
| **ESI2** | 1 | 758 | 26 | 0 | 0 |
| **ESI3** | 0 | 12 | 3143 | 13 | 6 |
| **ESI4** | 0 | 0 | 28 | 2389 | 4 |
| **ESI5** | 0 | 0 | 1 | 52 | 1369 |

---

