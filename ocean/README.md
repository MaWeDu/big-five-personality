# Ocean Personality – Streamlit App

Start the app with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model files

The project already contains both trained pipelines:

- `models/best_model.joblib` — 19-question short test
- `models/best_ocean_model.joblib` — 50-question OCEAN test

Both pipelines require the questionnaire values plus age, gender and handedness.
The app therefore requests these three values before the questionnaire. The
OCEAN pipeline also receives the five calculated dimension scores.

The pipelines were saved with scikit-learn 1.9.1, which is pinned in
`requirements.txt`. Install the requirements before starting the app.

## OCEAN test

The 50-question test calculates reverse-scored OCEAN dimension scores and also
uses the included 50-question classifier pipeline.

## Notes

- The short-test result is a model classification, not a diagnosis.
- The OCEAN results are answer scores, not percentile norms.
- The 19 short-question prompts are mapped to the item IDs in the notebooks.
  Validate the wording against the original source questionnaire before use.


## UI update
The profile inputs are displayed in a white card. Handedness uses a three-position slider (left / both / right) with localized labels.

## Single-window navigation

Language switching now uses Streamlit buttons and `st.session_state`.
No query-string links are used, so changing DE/EN/ES stays in the same browser tab and keeps the current app state.


## Fish progress indicator
During the questionnaire, a fish now moves along the progress bar as the user advances through the questions.
