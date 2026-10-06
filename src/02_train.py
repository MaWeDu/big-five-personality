from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== Model Training & Hyperparameter Tuning ===")
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_FILE = BASE_DIR / "DATA" / "data_clean.csv"
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_FILE}. Please run EDA first.")

    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    question_cols = [
        'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10',
        'E1', 'E3', 'E4', 'E5', 'E7', 'E9', 'E10', 'C4', 'A4'
    ]
    numeric_cols = question_cols + ['age']
    categorical_cols = ['gender', 'hand']

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]) 

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]) 

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    pipe_logreg = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=5000, random_state=42))])
    pipe_knn = Pipeline([("preprocessor", preprocessor), ("classifier", KNeighborsClassifier())])
    pipe_rf = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))])
    pipe_hgb = Pipeline([("preprocessor", preprocessor), ("classifier", HistGradientBoostingClassifier(random_state=42))])

    models_config = {
        "Logistic Regression": {
            "pipeline": pipe_logreg,
            "params": {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
        },
        "K-Nearest Neighbors": {
            "pipeline": pipe_knn,
            "params": {"classifier__n_neighbors": [3, 5, 7, 11], "classifier__weights": ["uniform", "distance"]}
        },
        "Random Forest": {
            "pipeline": pipe_rf,
            "params": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 10, 20], "classifier__min_samples_split": [2, 5]}
        },
        "Gradient Boosting": {
            "pipeline": pipe_hgb,
            "params": {"classifier__learning_rate": [0.01, 0.1, 0.2], "classifier__max_iter": [100, 200], "classifier__l2_regularization": [0.0, 0.1, 1.0]}
        }
    }

    results = []
    print("\nTraining models with GridSearchCV (Metric: f1_macro)...")
    for name, config in models_config.items():
        print(f" -> Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        test_score = best_model.score(X_test, y_test)
        results.append({
            "name": name,
            "model": best_model,
            "test_score": test_score
        })
        print(f"    Test F1-Macro: {test_score:.3f}")

    best_overall = max(results, key=lambda x: x["test_score"])
    best_pipeline = best_overall["model"]

    print(f"\n🏆 Best Model: {best_overall['name']} (F1-Macro: {best_overall['test_score']:.3f})")

    model_filename = BASE_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_filename)
    print(f"✓ Winning pipeline successfully saved as '{model_filename}'.")

if __name__ == "__main__":
    main()