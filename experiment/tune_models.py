import pandas as pd
import numpy as np
import optuna
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

warnings.filterwarnings("ignore")

def load_data():
    df = pd.read_csv('/Users/sreehariniganishkaa/triagemodel/presentation_vitals.csv')
    features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    X = df[features]
    y = df['ESI'] - 1  # 0-indexed for many models
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_processed = scaler.transform(imputer.transform(X_test))
    
    # Save preprocessing artifacts for the web UI
    joblib.dump(imputer, '/Users/sreehariniganishkaa/triagemodel/imputer.pkl')
    joblib.dump(scaler, '/Users/sreehariniganishkaa/triagemodel/scaler.pkl')
    
    return X_train_processed, X_test_processed, y_train, y_test

def get_model(name, trial):
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 100),
            random_state=42, verbose=-1
        )
    elif name == "XGBoost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
    elif name == "CatBoost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            random_state=42, verbose=0
        )
    elif name == "AdaBoost":
        return AdaBoostClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            random_state=42
        )
    elif name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=trial.suggest_int("max_iter", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            random_state=42
        )
    elif name == "DART":
        return LGBMClassifier(
            boosting_type='dart',
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            random_state=42, verbose=-1
        )
    elif name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            random_state=42
        )
    elif name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            random_state=42, n_jobs=-1
        )
    elif name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            random_state=42, n_jobs=-1
        )
    elif name == "StochasticGB":
        return GradientBoostingClassifier(
            subsample=trial.suggest_float("subsample", 0.5, 0.9),
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            random_state=42
        )
    elif name == "AdaptiveGB":
        return AdaBoostClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
            random_state=42
        )
    elif name == "NGBoost":
        # Keep static for NGBoost to avoid agonizingly long runtimes during search
        return NGBClassifier(Dist=k_categorical(5), n_estimators=50, random_state=42, verbose=False)
    elif name == "Stacking":
        # We will just evaluate a stacking classifier built from base models, no deep param tuning to save time
        stacking_estimators = [
            ('lgbm', LGBMClassifier(random_state=42, verbose=-1)),
            ('catboost', CatBoostClassifier(random_state=42, verbose=0)),
            ('xgboost', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)),
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ]
        return StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(C=trial.suggest_float("C", 0.1, 10.0, log=True), max_iter=1000, random_state=42),
            cv=3, n_jobs=-1
        )

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    models_to_tune = [
        "LightGBM", "XGBoost", "CatBoost", "AdaBoost", "HistGradientBoosting", 
        "DART", "NGBoost", "GradientBoosting", "RandomForest", "ExtraTrees", 
        "StochasticGB", "AdaptiveGB", "Stacking"
    ]
    
    best_overall_acc = 0
    best_overall_model = None
    best_overall_name = ""
    
    # 5 Trials each to quickly demonstrate tuning functionality for all 13 models
    TRIALS = 5 
    
    for model_name in models_to_tune:
        print(f"\n--- Tuning {model_name} ---")
        
        def objective(trial):
            model = get_model(model_name, trial)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if hasattr(pred, 'ndim') and pred.ndim > 1:
                pred = pred.ravel()
            return accuracy_score(y_test, pred)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=TRIALS if model_name != "NGBoost" else 1) # Reduce NGBoost trials due to extreme slowness
        
        print(f"Best trial for {model_name}: Accuracy {study.best_value:.4f}")
        
        # Train final version of this model with best params
        best_model = get_model(model_name, study.best_trial)
        best_model.fit(X_train, y_train)
        
        if study.best_value > best_overall_acc:
            best_overall_acc = study.best_value
            best_overall_model = best_model
            best_overall_name = model_name

    print("\n" + "="*50)
    print(f"WINNER: {best_overall_name} with Accuracy {best_overall_acc:.4f}")
    print("="*50)
    
    # Save the absolute best model
    out_path = '/Users/sreehariniganishkaa/triagemodel/best_esi_model.pkl'
    joblib.dump(best_overall_model, out_path)
    print(f"Saved best model to {out_path}")

if __name__ == "__main__":
    main()
