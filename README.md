# Personality Profile Predictor 🌊

Predicts a person's Big Five personality profile from a short questionnaire, and serves the
prediction through an ocean-themed Streamlit web app.

---

## 1. Overview

**What this project does.** A user answers **19 short statements** (1 = Disagree to
5 = Agree) and gives their age, gender and writing hand. A trained scikit-learn pipeline
predicts one of four personality profiles and the app shows the result instantly, together
with the probability the model assigns to each profile.

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
| `target` | The personality profile to predict |

One row is one person. The letter in a column name is the trait the statement belongs to:
**N** = Neuroticism (10 items), **E** = Extraversion (7), **C** = Conscientiousness (1),
**A** = Agreeableness (1).

**The four target classes:**

| Profile | In a nutshell | Share of the data |
|---|---|---|
| **Moderate** | Balanced, no extreme traits | ~43% |
| **Resilient** | Emotionally stable, calm under pressure | ~31% |
| **Overcontroller** | Anxious and introverted | ~14% |
| **Undercontroller** | Impulsive, less concerned with rules | ~12% |

**The approach.** EDA → preprocessing pipeline → modeling and tuning → save the best model →
Streamlit app.

1. **EDA** (`eda.ipynb`) explores the data and writes a cleaned dataset. It removes rows with a
   missing or `0` value in `gender` or `hand`, rows with a `0` in the question items (outside
   the 1–5 scale) and rows with impossible ages (three-digit ages and other invalid values,
   up to 1,000,000,000). Ages that were typed in as a birth year are kept and converted to an
   age, using 2017 as the survey year. In total 131 rows (0.66%) are removed, leaving
   19,588 of the 19,719 rows.
2. **One preprocessing pipeline**, a `ColumnTransformer` that imputes and scales the numeric
   columns and imputes and one-hot encodes the categorical ones, is shared by every model, so
   the comparison is fair.
3. **Four models** (Logistic Regression, K-Nearest Neighbors, Random Forest,
   HistGradientBoosting), each in its own pipeline, are compared with 5-fold stratified
   cross-validation and then **each tuned twice**: with `GridSearchCV` and with Hyperopt.
4. The **champion is saved with joblib** as the whole pipeline, preprocessing included.
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
slide the writing-hand control, and press the button. You get your predicted profile, the sea
creature that matches it, an explanation, the probability for all four profiles, and how
your four trait scores compare with the average.

---

## 2. Setup

These steps start from zero. We used **Python 3.14** and **VS Code**. Every command below is typed
in a terminal **inside the project folder**.

Get the project first:

```bash
git clone https://github.com/MaWeDu/big-five-personality.git
cd big-five-personality
```

No Git? On the GitHub page, click **Code → Download ZIP** and unzip it.

### Step 1: Get the data

The dataset is **not** in this repository, and you don't need to download it yourself:
`eda.ipynb` (see below step 4) creates a `data/` folder and downloads the file into it.

Alternative: download it from the project Drive folder and save it in the project as `data/data.csv`:

<https://drive.google.com/drive/folders/1KhwTPAG07EdaENW_XX9nVvKhC-DP1Ags?usp=sharing>

### Step 2: Install Python and VS Code, then create the virtual environment

1. **Python 3.14:** download it from <https://www.python.org/downloads/> and install it.
   On Windows, tick **"Add python.exe to PATH"** in the first window of the installer.
2. **VS Code:** download it from <https://code.visualstudio.com/> and install it. Then add the
   **Python** and **Jupyter** extensions (Extensions icon in the left sidebar).
3. In VS Code, open the project folder (**File → Open Folder…**) and open a terminal
   (**Terminal → New Terminal**). The terminal starts in the project folder.
4. Create the virtual environment (`venv`) and activate it:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

When it is active, the terminal line starts with `(venv)`.

> **Windows:** if `python` is not found, use `py -3.14 -m venv venv`. If PowerShell blocks the
> activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

### Step 3: Install the dependencies

In the same terminal, with `(venv)` shown and still in the project folder:

```bash
pip install -r requirements.txt
```

This installs the exact package versions we used (pandas, seaborn, scikit-learn, CatBoost, Hyperopt,
Streamlit, joblib, …) into the virtual environment. It takes a few minutes.

### Step 4: Run the notebooks in the virtual environment, in this order

1. Open `eda.ipynb` in VS Code. Click **Select Kernel** (top right) → **Python Environments** →
   **venv**, then click **Run All**.
2. Open `modeling.ipynb`, select the same **venv** kernel and click **Run All**.

| Order | Notebook | What it does | What it writes |
|---|---|---|---|
| 1 | `eda.ipynb` | Downloads the data, explores it and cleans it | `data/data_clean.csv` |
| 2 | `modeling.ipynb` | Compares and tunes five models, picks the champion, saves and checks it | `models/personality_pipeline.joblib` |

The modeling notebook takes about 5 to 10 minutes. It saves the complete trained pipeline
(preprocessing + tuned CatBoost) under the file name from the project brief, the file `app.py` loads:

```python
joblib.dump(best_pipeline, "models/personality_pipeline.joblib")
```

Every step that involves randomness uses `random_state=42`, so the notebooks recreate exactly
the same model file on any machine.

*Without VS Code:* run `jupyter notebook` in the activated terminal and open the two notebooks
in the browser that opens.

### Step 5: Run the Streamlit app

In the VS Code terminal (or any terminal), with `(venv)` active and inside the project folder:

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>; if no browser window opens, copy that address into your
browser. Stop the app with **Ctrl + C** in the terminal. If the app reports that no model was
found, notebook 2 has not been run yet.

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

*This is a student project for a machine-learning course. The model recognises patterns in
questionnaire answers and is not a psychological or diagnostic tool.*
