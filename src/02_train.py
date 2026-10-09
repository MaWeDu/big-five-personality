    "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10",
    "E1", "E3", "E4", "E5", "E7", "E9", "E10", "C4", "A4",
]

if not DATA_FILE.exists():
    raise FileNotFoundError(f"{DATA_FILE} not found. Run eda.py first.")

df = pd.read_csv(DATA_FILE)
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), QUESTION_COLUMNS + ["age"]),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), ["gender", "hand"]),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "CatBoost": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False),
}
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
results = []

for name, classifier in models.items():
    pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", classifier)])
    started_at = time.perf_counter()
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    seconds = time.perf_counter() - started_at
    results.append({"name": name, "pipeline": pipeline, "mean_f1": scores.mean(), "std_f1": scores.std(), "seconds": seconds})
    print(f"{name}: macro F1 {scores.mean():.3f} ± {scores.std():.3f}; {seconds:.1f}s")

champion = max(results, key=lambda result: result["mean_f1"])


def tuning_grid(model_name):
    """Return a small, model-specific grid for the baseline winner."""
    grids = {
        "Logistic Regression": {"classifier__C": [0.01, 0.1, 1, 10]},
        "K-Nearest Neighbors": {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
        },
        "HistGradientBoosting": {
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_iter": [100, 200],
        },
        "CatBoost": [
            {
                "classifier__iterations": [300],
                "classifier__depth": [4, 6],
                "classifier__learning_rate": [0.1],
            },
            {
                "classifier__iterations": [300],
                "classifier__depth": [4, 6],
                "classifier__learning_rate": [0.1],
                "classifier__auto_class_weights": ["SqrtBalanced", "Balanced"],
            },
        ],
    }
    return grids[model_name]


print(f"Baseline winner: {champion['name']}")
started_at = time.perf_counter()
grid_search = GridSearchCV(
    estimator=clone(champion["pipeline"]),
    param_grid=tuning_grid(champion["name"]),
    scoring="f1_macro",
    cv=cv,
    refit=True,
)
grid_search.fit(X_train, y_train)
tuning_seconds = time.perf_counter() - started_at

tuned_pipeline = grid_search.best_estimator_
y_pred = tuned_pipeline.predict(X_test)

print(f"Rows: {len(df):,}; features: {X.shape[1]}")
print(f"Selected model: {champion['name']}")
print(f"Baseline macro F1: {champion['mean_f1']:.3f} ± {champion['std_f1']:.3f}")
print(f"Tuned macro F1: {grid_search.best_score_:.3f}")
print(f"Tuning time: {tuning_seconds:.1f}s")
print(f"Best settings: {grid_search.best_params_}")
print(f"Test macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
print(classification_report(y_test, y_pred, digits=3))

final_pipeline = clone(tuned_pipeline)
final_pipeline.fit(X, y)
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_pipeline, MODEL_FILE)
print(f"Saved: {MODEL_FILE}")
