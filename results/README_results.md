## Results

**Champion model: HistGradientBoosting, tuned with grid search**

Selected by **macro F1** under 5-fold stratified cross-validation on the training set. Macro F1 treats every personality type equally, which matters here because the classes are imbalanced (Moderate 43%, Undercontroller 12%).

- Cross-validation macro F1: **0.804**
- Held-out test set macro F1: **0.799**
- Held-out test set accuracy: **83.5%**
- Baseline (always predicts Moderate): 0.150 macro F1, 0.429 accuracy

Champion settings: `class_weight: balanced, learning_rate: 0.1, max_leaf_nodes: 31`

| Model | Tuning | CV F1 macro | Test F1 macro |
|---|---|---|---|
| HistGradientBoosting | grid search | 0.804 | 0.799 |
| HistGradientBoosting | hyperopt | 0.802 | 0.796 |
| HistGradientBoosting | none | 0.795 | — |
| Random Forest | grid search | 0.786 | 0.795 |
| Random Forest | hyperopt | 0.785 | 0.793 |
| Logistic Regression | hyperopt | 0.785 | 0.780 |
| Logistic Regression | grid search | 0.785 | 0.780 |
| Logistic Regression | none | 0.766 | — |
| K-Nearest Neighbors | hyperopt | 0.741 | 0.750 |
| K-Nearest Neighbors | grid search | 0.740 | 0.743 |
| Random Forest | none | 0.740 | — |
| K-Nearest Neighbors | none | 0.730 | — |

The champion pipeline (preprocessing + classifier) was refit on the full dataset and saved as `models/personality_pipeline.joblib` for the Streamlit app.