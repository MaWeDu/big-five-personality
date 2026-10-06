# Personality Type Predictor 🌊

Predicts a person's OCEAN Big Five personality type from a short questionnaire, and serves the
prediction through an ocean-themed Streamlit web app.

---

## 1. Overview

**What this project does.** A user answers **19 short statements** (1 = Disagree to
5 = Agree) and gives their age, gender and writing hand. A trained scikit-learn pipeline
predicts one of four personality types and the app shows the result instantly, together
with the probability the model assigns to each type.

**The problem.** This is a **supervised multi-class classification** task: given 22 features
(19 questionnaire answers + 3 demographics), predict one of four classes. The classes are
imbalanced, and that decides how the models have to be measured.

**The dataset.** The data comes from the **IPIP Big Five (OCEAN)** personality test published
on Kaggle. The original version has 50 statements answered by about 19,000 people; this
project uses a prepared version with the 19 most informative items plus three demographics and
the target.

| Column group | What it holds |
|---|---|
| 19 item columns (`N1`–`N10`, `E1`, `E3`, `E4`, `E5`, `E7`, `E9`, `E10`, `C4`, `A4`) | The raw answer to one statement, 1–5 |
| `age`, `gender`, `hand` | Age in years; Female/Male/Other; Right/Left/Both |
| `target` | The personality type to predict |

One row is one person. The letter in a column name is the trait the statement belongs to:
**N** = Neuroticism (10 items), **E** = Extraversion (7), **C** = Conscientiousness (1),
**A** = Agreeableness (1).

**The four target classes:**

| Type | In a nutshell | Share of the data |
|---|---|---|
| **Moderate** | Balanced, no extreme traits | ~43% |
| **Resilient** | Emotionally stable, calm under pressure | ~31% |
| **Overcontroller** | Anxious and introverted | ~14% |
| **Undercontroller** | Impulsive, less concerned with rules | ~12% |

**The approach.** The project covers the full workflow: EDA and cleaning → preprocessing
pipeline → model comparison with cross-validation → hyperparameter tuning with grid search and
Hyperopt → a Streamlit app that loads the champion model.

1. **EDA and cleaning** (`eda.ipynb`) explores the data and writes a cleaned dataset. It
   removes rows with a missing or `0` value in `gender` or `hand`, rows with a `0` in the
   question items (outside the 1–5 scale) and rows with impossible ages (three-digit ages and
   other invalid values, up to 999,000,000). Some people typed in their birth year instead of
   their age. Unlike the stricter `JT_EDA_Personality_Type_prediction.ipynb`, which deletes
   these rows, `eda.ipynb` keeps the plausible birth years and converts them to ages, using
   2017 as the base year of the survey data. In total 131 rows (0.66%) are removed, leaving 19,588 of the 19,719 rows.
2. **One preprocessing pipeline**, a `ColumnTransformer` that imputes and scales the numeric
   columns and imputes and one-hot encodes the categorical ones, is shared by every model, so
   the comparison is fair.
3. **Four models** (Logistic Regression, K-Nearest Neighbors, Random Forest,
   HistGradientBoosting), each in its own pipeline, are compared with 5-fold stratified
   cross-validation and then **each tuned twice**: with `GridSearchCV` and with `Hyperopt`.
4. The **champion is saved with joblib** as the whole pipeline, preprocessing included, in
   `models/personality_pipeline.joblib`.
5. The **Streamlit app** loads that file and serves predictions. It never trains anything.

**The result.**

> **Champion model: HistGradientBoosting (`HistGradientBoostingClassifier`), tuned with grid
> search.** Selected by **macro F1** under 5-fold stratified cross-validation:
> **0.804 CV**, confirmed with **0.799** on a held-out 20% test set that no model saw during
> training or tuning (test accuracy **83.5%**). The baseline that always predicts "Moderate"
> scores 0.150 macro F1.
>
> Winning settings: `class_weight='balanced'`, `learning_rate=0.1`, `max_leaf_nodes=31`.

Macro F1 rather than accuracy, because the classes are imbalanced: always predicting
"Moderate" already gives about 43% accuracy while learning nothing.

**How to use the app.** Answer the 19 statements by clicking 1–5, set your age and gender,
slide the writing-hand control, and press the button. You get your predicted type, the sea
creature that matches it, an explanation, the probability for all four types, and how
your four trait scores compare with the average.

---

## 2. Setup

Everything below assumes you have just cloned this repository and have nothing else.
Python 3.11 or newer.

### Step 1: Get the data

The dataset is **not** in this repository. Simply running the eda.ipynb will create a DATA folder and download the file in it.

Alternative: Download it from the project Drive folder:

<https://drive.google.com/drive/folders/1KhwTPAG07EdaENW_XX9nVvKhC-DP1Ags?usp=sharing>

Create a `data/` folder in the project root and put the file there, so you have:

```
data/data.csv
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install the dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the notebooks, in this order

```bash
jupyter notebook
```

| Order | Notebook | What it does | What it writes |
|---|---|---|---|
| 1 | `eda.ipynb` | Explores the data and cleans it | `data/data_clean.csv` |
| 2 | `modeling.ipynb` | Compares and tunes four models, picks the champion | `models/personality_pipeline.joblib` |

Run each one top to bottom. The modeling notebook takes about 5 minutes.

The modeling notebook creates the `models/` folder and saves the trained champion pipeline in
it. The file is named `personality_pipeline.joblib` to match the project brief, and it is the
file `app.py` loads:

```python
joblib.dump(best_pipeline, "models/personality_pipeline.joblib")
```

Every step that involves randomness uses `random_state=42`, so these notebooks regenerate
exactly the same model file on any machine.

### Step 5: Run the Streamlit app

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>. If the app reports that no model was found, notebook 2
has not been run yet.

---

## 3. Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                                        # the Streamlit app
├── .streamlit/
│   └── config.toml                               # dark ocean theme for the app
├── eda.ipynb                                     # notebook 1: EDA and cleaning
├── modeling.ipynb                                # notebook 2: modeling and tuning
├── data/                                         # NOT committed, see step 1
│   ├── data.csv                                  #   downloaded from Drive
│   └── data_clean.csv                            #   written by notebook 1
├── models/                                       # NOT committed, written by notebook 2
│   └── personality_pipeline.joblib
└── results/                                      # written by notebook 2
    ├── model_comparison.csv                      #   the full leaderboard
    └── README_results.md                         #   the results block for this README
```

Neither the dataset nor the trained model is committed, as the project brief requires. Both
are regenerated by following the setup steps.

The repository also contains the team members' individual exploration notebooks (for example
`JT_EDA_Personality_Type_prediction.ipynb`, which tests a stricter cleaning that deletes the
birth-year ages instead of converting them). They document our process but are **not needed**
to reproduce the model: only the two notebooks in step 4 are.

---

## 4. Model comparison

All four models were tuned with both methods. Selection metric: macro F1 under 5-fold
stratified cross-validation on the training set. "Test" is the held-out 20%.

| Model | Tuning | CV F1 macro | Test F1 macro |
|---|---|---|---|
| **HistGradientBoosting** | **grid search** | **0.804** | **0.799** |
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

**Undercontroller is the hardest class for every model.** It is defined by low
Conscientiousness *and* low Agreeableness, but only two of the 19 statements (`C4` and `A4`)
measure those traits, so most of the signal needed to identify it was removed when the original
50-item questionnaire was reduced to 19. `class_weight='balanced'`, which both searches chose,
lifts its recall from 0.42 to 0.66 at the cost of a little overall accuracy. That is the
trade-off macro F1 is meant to reward.

---

## 5. Data source

Big Five (OCEAN) personality test data, published on Kaggle and prepared for this course.
Download link in step 1 above.

*This is a student project for a machine-learning course. The model recognizes patterns in questionnaire answers and is not a psychological or diagnostic tool.*
