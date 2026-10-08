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
   their age. As a group, we combined the best ideas from everyone into one consolidated `eda.ipynb`;
   the individual folders with notebooks keep slightly different versions, tests and assumptions.
   For example, unlike the stricter `JT_EDA_Personality_Type_prediction.ipynb`, which deletes
   these rows, `eda.ipynb` keeps the plausible birth years and converts them to ages, using
   2017 as the base year of the survey data. In total 131 rows (0.66%) are removed, leaving 19,588 of the 19,719 rows.
3. **One preprocessing pipeline**, a `ColumnTransformer` that imputes and scales the numeric
   columns and imputes and one-hot encodes the categorical ones, is shared by every model, so
   the comparison is fair.
4. **Five models** (Logistic Regression, K-Nearest Neighbors, Random Forest,
   HistGradientBoosting and CatBoost), each in its own pipeline, are first compared with their
   default settings (untuned) under 5-fold stratified cross-validation and then **tuned**: all five with `GridSearchCV`, the four faster
   ones also with `Hyperopt` (CatBoost is left out of Hyperopt because it trains much slower).
5. The **champion is saved with joblib** as the whole pipeline, preprocessing included, in
   `models/personality_pipeline.joblib`.
6. The **Streamlit app** loads that file and serves predictions. It never trains anything.

**The result.**

> **Champion model: CatBoost (`CatBoostClassifier`), tuned with grid search.** Selected by
> **macro F1** under 5-fold stratified cross-validation: **0.819 CV**, the highest of all 15
> runs, confirmed with **0.823** on a held-out 20% test set that no model saw during training
> or tuning (test accuracy **86.9%**). The baseline that always predicts "Moderate" scores
> 0.150 macro F1.
>
> Winning settings: `auto_class_weights='SqrtBalanced'`, `depth=4` (+0.013 macro F1 over
> CatBoost's default settings).

Macro F1 rather than accuracy, because the classes are imbalanced: always predicting
"Moderate" already gives about 43% accuracy while learning nothing.

**How to use the app.** Answer the 19 statements by clicking 1–5, set your age and gender,
slide the writing-hand control, and press the button. You get your predicted type, the sea
creature that matches it, an explanation, the probability for all four types, and how
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
The same folder with the notebooks has to have a folder called .streamlit/ that contains the config.toml, so it has the ocean background of the app before it can run properly.   
It opens at <http://localhost:8501>; if no browser window opens, copy that address into your
browser. Stop the app with **Ctrl + C** in the terminal. If the app reports that no model was
found, notebook 2 has not been run yet.

### One more thing: one-click launch

> On Windows, there is an easier way.
> Double-click **Start Personality Predictor** (or `start.bat`) in the project folder.
> On its first run, the launcher creates the virtual environment, installs the dependencies, prepares the data, trains the model, and starts the Streamlit app.
> 
> After that, it is simply one click to return to the ocean.
>
> The manual steps below remain available if you prefer to run the notebooks yourself, inspect the workflow, or change the model.

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
├── venv/                                         # NOT committed, created in step 2
├── data/                                         # NOT committed, see step 1
│   ├── data.csv                                  #   downloaded from Drive
│   └── data_clean.csv                            #   written by notebook 1
└── models/                                       # NOT committed, written by notebook 2
    └── personality_pipeline.joblib
```

Neither the dataset nor the trained model is committed, as the project brief requires. Both
are regenerated by following the setup steps.

---

## 4. Model comparison

All five models were tuned with grid search (HistGradientBoosting with two different grids),
the four faster ones also with Hyperopt. Every run is scored on the same five folds. Selection
metric: macro F1 under 5-fold stratified cross-validation on the training set. "Test" is the
held-out 20%.

| Model | Tuning | CV F1 macro | Test F1 macro |
|---|---|---|---|
| **CatBoost** | **grid search** | **0.819** | **0.823** |
| CatBoost | untuned | 0.806 | 0.811 |
| HistGradientBoosting (balanced) | grid search | 0.803 | 0.800 |
| HistGradientBoosting | hyperopt | 0.798 | 0.805 |
| HistGradientBoosting | grid search | 0.798 | 0.807 |
| HistGradientBoosting | untuned | 0.797 | 0.805 |
| Logistic Regression | grid search | 0.767 | 0.780 |
| Logistic Regression | hyperopt | 0.767 | 0.779 |
| Logistic Regression | untuned | 0.767 | 0.779 |
| Random Forest | grid search | 0.741 | 0.761 |
| Random Forest | untuned | 0.741 | 0.760 |
| Random Forest | hyperopt | 0.740 | 0.760 |
| K-Nearest Neighbors | grid search | 0.739 | 0.742 |
| K-Nearest Neighbors | hyperopt | 0.731 | 0.735 |
| K-Nearest Neighbors | untuned | 0.726 | 0.716 |

**Undercontroller is the hardest class for every model.** It is defined by low
Conscientiousness *and* low Agreeableness, but only two of the 19 statements (`C4` and `A4`)
measure those traits, so most of the signal needed to identify it was removed when the original
50-item questionnaire was reduced to 19. Class weights help: CatBoost with its default settings
finds 44% of the Undercontrollers in the test set, the champion's mild `SqrtBalanced` weighting
58%, while the overall accuracy stays practically the same (86.9% against 87.2%). That is the
trade-off macro F1 is meant to reward.

---

## 5. Data source

Big Five (OCEAN) personality test data, published on Kaggle and prepared for this course.
Download link in step 1 above.

*This is a student project for a machine-learning course. The model recognizes patterns in questionnaire answers and is not a psychological or diagnostic tool.*
