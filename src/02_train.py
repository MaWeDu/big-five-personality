"""Train, evaluate, and save the personality type classifier."""

import importlib.util
import subprocess
import sys
from pathlib import Path

PACKAGES = {"catboost": "catboost", "scikit-learn": "sklearn", "joblib": "joblib"}
for package, module in PACKAGES.items():
    if importlib.util.find_spec(module) is None:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package], check=True, timeout=120)

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE, N_SPLITS, TEST_SIZE = 42, 5, 0.2
DATA_FILE = Path("data") / "data_clean.csv"
MODEL_FILE = Path("models") / "personality_pipeline.joblib"
QUESTION_COLUMNS = [
    "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10",
    "E1", "E3", "E4", "E5", "E7", "E9", "E10", "C4", "A4",
]

if not DATA_FILE.exists():
    raise FileNotFoundError(f"{DATA_FILE} not found. Run eda.py first.")
df = pd.read_csv(DATA_FILE)
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), QUESTION_COLUMNS + ["age"]),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["gender", "hand"]),
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False)),
])

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Rows: {len(df):,}; features: {X.shape[1]}")
print(f"Cross-validation macro F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Test macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
print(classification_report(y_test, y_pred, digits=3))

pipeline.fit(X, y)
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_FILE)
print(f"Saved: {MODEL_FILE}")
