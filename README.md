# big-five-personality
Machine-learning project that predicts personality types from 19 Big Five questionnaire responses, age, gender, and writing hand. It covers data exploration, preprocessing, model comparison, hyperparameter tuning, MLflow experiment tracking, and a Streamlit app for instant predictions.


Gelöscht wird jede Zeile wo:
- Hand leer
- Geschlecht leer
- Werte mit 0, da außerhalb der erlaubten Bereichs von 1-5
- Alter über 100, falsche Werte (Geburtsjahr umrechnen?)

- ## About the data

The data comes from the **IPIP Big Five (OCEAN)** personality test, published on Kaggle.
The original version has 50 statements answered by about 19,000 people; for this project a
streamlined version was prepared that keeps the **19 most informative items** plus a few
demographics.

Each row is one person:

| Columns | What they are |
|---|---|
| 19 item columns (`N1`–`N10`, `E1`–`E10`, `C4`, `A4`) | The raw answer to one statement, on a scale of **1 = Disagree** to **5 = Agree** |
| `age`, `gender`, `hand` | Age in years; Male/Female/Other; Right/Left/Both |
| `target` | The personality type we want to predict |

The letter in a column name is the OCEAN trait the statement belongs to:
**N** = Neuroticism (10 items), **E** = Extraversion (7), **C** = Conscientiousness (1),
**A** = Agreeableness (1). For example `N1` is *"I get stressed out easily"* and
`E1` is *"I am the life of the party"*.

**The four personality types** were derived from the OCEAN trait scores with fixed rules:

| Type | In a nutshell | Rule |
|---|---|---|
| Moderate | Balanced, no extreme traits | default, when no other rule matches |
| Resilient | Emotionally stable, calm under pressure | N < −0.5 |
| Overcontroller | Anxious and introverted | E < −0.5 and N > 0.5 |
| Undercontroller | Impulsive, less concerned with rules | C < −0.2 and A < −0.2 |

**Our task:** a supervised, multi-class classification problem — predict `target` from the
19 answers plus the three demographic columns.


CHANGE