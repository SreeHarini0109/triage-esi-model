import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, cohen_kappa_score
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, 
    HistGradientBoostingClassifier, 
    GradientBoostingClassifier, 
    RandomForestClassifier, 
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from ngboost import NGBClassifier
from ngboost.distns import k_categorical

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv('/Users/sreehariniganishkaa/triagemodel/presentation_vitals.csv')
    
    # We use these inputs specifically requested by the user: Vital signs
    features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    X = df[features]
    
    # Target is ESI
    y = df['ESI']
    
    # Needs re-mapping from 1-5 to 0-4 for some classifiers (XGBoost requires 0-indexed)
    y_mapped = y - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Many methods benefit from scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print("\n--- Dataset Information ---")
    print(f"Features: {features}")
    print(f"Target Variable: ESI (Mapped internally from 0 to 4)")
    print(f"Total rows in dataset: {len(X)}")
    print(f"Training set rows: {len(X_train)}")
    print(f"Testing set rows:  {len(X_test)}")
    print("---------------------------\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features

def main():
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    
    # ---- Individual Models ----
    models = {
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "DART": LGBMClassifier(boosting_type='dart', random_state=42, verbose=-1),
        "NGBoost": NGBClassifier(Dist=k_categorical(5), random_state=42, verbose=False),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "StochasticGB": GradientBoostingClassifier(subsample=0.8, random_state=42),
        "AdaptiveGB": AdaBoostClassifier(learning_rate=0.1, random_state=42)
    }
    
    # ---- Stacking Classifier ----
    stacking_estimators = [
        ('lgbm', LGBMClassifier(random_state=42, verbose=-1)),
        ('catboost', CatBoostClassifier(random_state=42, verbose=0)),
        ('xgboost', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ]
    models["Stacking"] = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )

    results = {}
    predictions = {}
    
    print("\n--- Training Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {"Accuracy": acc, "F1-Score": f1}
        predictions[name] = y_pred
        
    print("\n" + "=" * 60)
    print("--- Evaluation Results (All Models) ---")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # ---- Detailed Metrics for Top 3 Models ----
    sorted_models = sorted(results.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
    top3 = [name for name, _ in sorted_models[:3]]
    
    target_names = ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5']
    
    for name in top3:
        y_pred = predictions[name]
        
        print("\n" + "=" * 60)
        print(f"  DETAILED METRICS: {name}")
        print("=" * 60)
        
        # Classification Report
        print(f"\nClassification Report ({name}):")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix ({name}):")
        print(pd.DataFrame(cm, index=target_names, columns=target_names).to_string())
        
        # Weighted Kappa (Cohen's Kappa with quadratic weights)
        kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
        print(f"\nWeighted (Quadratic) Cohen's Kappa ({name}): {kappa:.4f}")

if __name__ == "__main__":
    main()
