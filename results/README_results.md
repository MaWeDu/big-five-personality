## Results

**Champion model: HistGradientBoosting (tuned with grid search)**

Selected by **macro F1** under 5-fold stratified cross-validation — the metric that treats every personality type equally, which matters because the classes are imbalanced (Moderate 43%, Undercontroller 12%).

- Champion macro F1: **0.807**
- Baseline (always predicts Moderate): 0.150 macro F1

| Model | Tuning | Macro F1 |
|---|---|---|
| HistGradientBoosting | grid search | 0.807 |
| HistGradientBoosting | hyperopt | 0.805 |
| HistGradientBoosting | none | 0.801 |
| LogisticRegression | hyperopt | 0.783 |
| LogisticRegression | grid search | 0.783 |
| LogisticRegression | none | 0.767 |
| RandomForest | none | 0.743 |
| KNN | none | 0.728 |

Champion settings: `{'classifier__class_weight': 'balanced', 'classifier__learning_rate': 0.1, 'classifier__max_leaf_nodes': 31}`

Undercontroller is the hardest class for every model: it is defined by low Conscientiousness and low Agreeableness, but only two of the 19 statements measure those traits, so most of the signal needed to identify it is not in the data.