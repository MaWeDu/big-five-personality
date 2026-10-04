# big-five-personality
This notebook is part of a complete supervised machine-learning project: predicting a person's personality type from a short questionnaire. It covers data exploration, preprocessing, model comparison, hyperparameter tuning, and a Streamlit app for instant predictions.

## About the data

The data comes from the **IPIP Big Five (OCEAN)** personality test, published on Kaggle.
The original version has 50 statements answered by about 19,000 people; for this project a
streamlined version was prepared that keeps the **19 most informative items** plus a few
demographics.

Each row is one person:

| Columns | What they are |
|---|---|
| 19 item columns | The raw answer to one statement: **1 = Disagree, 2 = Slightly Disagree, 3 = Neutral, 4 = Slightly Agree, 5 = Agree** |
| `age`, `gender`, `hand` | Age in years; Male/Female/Other; Right/Left/Both |
| `target` | The personality type we want to predict |

### The OCEAN traits

The letter in a column name says which trait the statement belongs to:

| Trait | Letter | High scores | Low scores | Items here |
|---|---|---|---|---|
| Neuroticism | **N** | More emotional — easily stressed, upset or irritated | Calm and emotionally stable | 10 |
| Extraversion | **E** | Social, outgoing, comfortable with attention | Prefers to work alone and stay in the background | 7 |
| Conscientiousness | **C** | Follows the rules, prefers order | Tends to be disorganized | 1 |
| Agreeableness | **A** | Accommodating, sympathetic to others | Direct | 1 |
| Openness to Experience | **O** | "Dreams with their eyes open" | "Feet on the ground" | 0 — not in this dataset |

### The 19 personality items

Values are the **raw** answer to the statement. Items marked ↺ are worded in the opposite
direction, so a high answer means *less* of that trait.

| Item | Statement | | Item | Statement |
|---|---|---|---|---|
| `N1` | I get stressed out easily. | | `E1` | I am the life of the party. |
| `N2` ↺ | I am relaxed most of the time. | | `E3` | I feel comfortable around people. |
| `N3` | I worry about things. | | `E4` ↺ | I keep in the background. |
| `N4` ↺ | I seldom feel blue. | | `E5` | I start conversations. |
| `N5` | I am easily disturbed. | | `E7` | I talk to a lot of different people at parties. |
| `N6` | I get upset easily. | | `E9` | I don't mind being the center of attention. |
| `N7` | I change my mood a lot. | | `E10` ↺ | I am quiet around strangers. |
| `N8` | I have frequent mood swings. | | `C4` ↺ | I make a mess of things. |
| `N9` | I get irritated easily. | | `A4` | I sympathize with others' feelings. |
| `N10` | I often feel blue. | | | |

### The four personality types

The `target` was derived from the OCEAN trait scores (z-scores) with fixed rules:

| Personality type | Rule | What that means |
|---|---|---|
| **Resilient** | N < −0.5 | Emotionally stable, calm under pressure |
| **Overcontroller** | E < −0.5 **and** N > 0.5 | Anxious and introverted |
| **Undercontroller** | C < −0.2 **and** A < −0.2 | Impulsive, disorganized, less concerned with rules |
| **Moderate** | default, when no other rule matches | Balanced, no extreme traits |

The codebook also defines a fifth type, **Reserved** (E < −0.5 and N < −0.5), but it does not
appear in a single row of our data.

**Our task:** a supervised, multi-class classification problem — predict `target` from the
19 answers plus the three demographic columns.