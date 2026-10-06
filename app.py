import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Seitentitel
st.set_page_config(page_title="Big Five Personality Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Big Five Personality Predictor")
st.write("Gib die Werte deiner Fragen (1-5) und demografischen Daten ein, um deinen Persönlichkeitstyp vorherzusagen.")

# Modell laden
MODEL_PATH = Path("best_model.joblib")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.warning("⚠️ Es wurde noch kein trainiertes Modell (`best_model.joblib`) gefunden. Bitte führe zuerst das Start-Skript (`python run.py`) aus!")
else:
    with st.form("prediction_form"):
        st.subheader("Fragebogen (Items 1 bis 5)")
        
        # Eingaben für ein paar typische Features als Beispiel
        col1, col2 = st.columns(2)
        
        with col1:
            n1 = st.slider("N1 (Neuroticism): Get stressed out easily", 1, 5, 3)
            n3 = st.slider("N3 (Neuroticism): Worry about things", 1, 5, 4)
            e1 = st.slider("E1 (Extraversion): Life of the party", 1, 5, 3)
            e3 = st.slider("E3 (Extraversion): Comfortable around people", 1, 5, 4)
            
        with col2:
            c4 = st.slider("C4 (Conscientiousness): Make a mess of things", 1, 5, 2)
            a4 = st.slider("A4 (Agreeableness): Sympathize with others", 1, 5, 4)
            age = st.number_input("Alter", min_value=13, max_value=100, value=25)
            
        gender = st.selectbox("Geschlecht", ["Female", "Male", "Other"])
        hand = st.selectbox("Händigkeit", ["Right", "Left", "Both"])

        # Submit-Button
        submit = st.form_submit_button("Vorhersage generieren")

    if submit:
        # Ein DataFrame im exakten Format erstellen (Default-Werte für nicht im Formular genannte Fragen)
        input_data = pd.DataFrame([{
            'N1': n1, 'N2': 3, 'N3': n3, 'N4': 3, 'N5': 3, 'N6': 3, 'N7': 3, 'N8': 3, 'N9': 3, 'N10': 3,
            'E1': e1, 'E3': e3, 'E4': 3, 'E5': 3, 'E7': 3, 'E9': 3, 'E10': 3, 'C4': c4, 'A4': a4,
            'age': age, 'gender': gender, 'hand': hand
        }])

        # Vorhersage
        pred_class = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data).max()

        # Schicke Ausgabe in Streamlit
        st.success("Erfolgreich berechnet!")
        st.metric(label="Vorhergesagter Persönlichkeitstyp", value=pred_class)
        st.write(f"**Konfidenz:** {confidence:.2%}")