"""
Big Five (OCEAN) Personality Test: the Streamlit app for the Personality Type Predictor.

The app loads the trained pipeline that modeling.ipynb saved as
models/personality_pipeline.joblib and predicts one of four Big Five personality
types from 19 questionnaire answers plus age, gender and writing hand.

It does NOT train anything and does not read the dataset. It only loads the
saved pipeline, collects the answers and shows the prediction, which is what
the project brief asks for.

Run it with:    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# 1. Page setup and constants
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Big Five Personality Test",
    page_icon="🌊",
    layout="centered",
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "personality_pipeline.joblib"

# The 19 statements from the codebook, grouped the way the app shows them.
GROUPS = [
    ("Emotional style", [
        ("N1", "I get stressed out easily."),
        ("N2", "I am relaxed most of the time."),
        ("N3", "I worry about things."),
        ("N4", "I seldom feel blue."),
        ("N5", "I am easily disturbed."),
        ("N6", "I get upset easily."),
        ("N7", "I change my mood a lot."),
        ("N8", "I have frequent mood swings."),
        ("N9", "I get irritated easily."),
        ("N10", "I often feel blue."),
    ]),
    ("Social style", [
        ("E1", "I am the life of the party."),
        ("E3", "I feel comfortable around people."),
        ("E4", "I keep in the background."),
        ("E5", "I start conversations."),
        ("E7", "I talk to a lot of different people at parties."),
        ("E9", "I don't mind being the center of attention."),
        ("E10", "I am quiet around strangers."),
    ]),
    ("Everyday habits", [
        ("C4", "I make a mess of things."),
        ("A4", "I sympathize with others' feelings."),
    ]),
]
ALL_CODES = [code for _, items in GROUPS for code, _ in items]

# The answer scale from the codebook, with one colour per step:
# warm = disagree, light blue-grey = neutral, teal = agree.
SCALE = [
    (1, "Disagree", "#F08A6C"),
    (2, "Slightly disagree", "#F2B48C"),
    (3, "Neutral", "#BDD9E2"),
    (4, "Slightly agree", "#A5E6DA"),
    (5, "Agree", "#5FCBB5"),
]

GENDERS = ["Female", "Male", "Other"]
HANDS = ["Left", "Both", "Right"]

# One sea creature per personality type: colour, texts and the rule from the codebook.
PROFILES = {
    "Moderate": {
        "fish": "Clownfish", "color": "#E8871E", "deep": "#B35F0A", "share": "43%",
        "nutshell": "Balanced, no extreme traits",
        "why_fish": "at home in the reef, neither hiding nor showing off",
        "tagline": "Balanced — no trait pulls strongly in either direction.",
        "rule": "the default, when no other rule matches",
        "body": (
            "Your answers sat near the middle on most statements, in both the emotional and "
            "the social group. No trait crossed the threshold that defines one of the more "
            "distinctive types."
        ),
        "body2": (
            "This is the largest group in the dataset at roughly 43% of respondents, and also "
            "the one the model predicts most readily, so keep that in mind when you read the "
            "confidence number."
        ),
    },
    "Resilient": {
        "fish": "Blue Tang", "color": "#2E9FC9", "deep": "#14607F", "share": "31%",
        "nutshell": "Emotionally stable, calm under pressure",
        "why_fish": "glides through rough water without losing its course",
        "tagline": "Emotionally steady, and comfortable with other people.",
        "rule": "low Neuroticism",
        "body": (
            "Your answers were low on the statements about stress, worry and mood swings, "
            "while the social statements landed around or above the middle. In the Big Five "
            "that combination is low Neuroticism with ordinary to high Extraversion."
        ),
        "body2": (
            "People in this group tend to describe themselves as calm under pressure and slow "
            "to be rattled. It is the second most common type, covering about 31% of "
            "respondents."
        ),
    },
    "Overcontroller": {
        "fish": "Pufferfish", "color": "#9B6BB3", "deep": "#5F3B74", "share": "14%",
        "nutshell": "Anxious and introverted",
        "why_fish": "keeps to itself, and puffs up when the world gets too close",
        "tagline": "Feels things intensely, and prefers the quieter end of the room.",
        "rule": "low Extraversion and high Neuroticism",
        "body": (
            "Your answers were high on the statements about stress, worry and mood, and low on "
            "the ones about parties, conversations and attention. That pairing (high "
            "Neuroticism with low Extraversion) is what defines this type."
        ),
        "body2": (
            "About 14% of the dataset falls here. The label describes a pattern of answers, "
            "not a problem: many people in this group describe themselves as thoughtful and "
            "careful rather than anxious."
        ),
    },
    "Undercontroller": {
        "fish": "Reef Shark", "color": "#3FAE7A", "deep": "#1F6B49", "share": "12%",
        "nutshell": "Impulsive, less concerned with rules",
        "why_fish": "moves first, thinks later, and never asks permission",
        "tagline": "Spontaneous and direct, with less patience for rules.",
        "rule": "low Conscientiousness and low Agreeableness",
        "body": (
            "This type is defined by low Conscientiousness together with low Agreeableness: "
            "being less concerned with order, and more blunt than accommodating. Your answers "
            "to those two statements leaned that way."
        ),
        "body2": (
            "It is the smallest group at about 12% of the dataset, and the hardest for the "
            "model to recognise: only two of the 19 statements measure the traits that define "
            "it, so confidence here is usually lower."
        ),
    },
}

# The four traits behind the profile rules. Reverse-worded statements are flipped
# (6 - answer) so that a high score always means "more of this trait".
# The averages were calculated once from data/data_clean.csv (19,588 people),
# so the app itself never has to read the data.
TRAITS = [
    {"name": "Emotional reactivity", "trait": "Neuroticism",
     "codes": ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10"],
     "reversed": ["N2", "N4"], "average": 3.10,
     "hint": "Low points to Resilient, high to Overcontroller"},
    {"name": "Sociability", "trait": "Extraversion",
     "codes": ["E1", "E3", "E4", "E5", "E7", "E9", "E10"],
     "reversed": ["E4", "E10"], "average": 2.96,
     "hint": "Low, together with high reactivity, points to Overcontroller"},
    {"name": "Orderliness", "trait": "Conscientiousness",
     "codes": ["C4"], "reversed": ["C4"], "average": 3.35,
     "hint": "Low, together with low empathy, points to Undercontroller"},
    {"name": "Empathy", "trait": "Agreeableness",
     "codes": ["A4"], "reversed": [], "average": 4.03,
     "hint": "Low, together with low orderliness, points to Undercontroller"},
]


def html(markup):
    """Show a block of HTML. Lines are un-indented first, because Streamlit's
    markdown would otherwise turn indented lines into a code block."""
    lines = [line.strip() for line in markup.splitlines() if line.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def scroll_to_top():
    """Jump to the top of the page.

    Streamlit keeps the scroll position when the screen changes. Without this,
    the result page would open halfway down, where the button was clicked.
    A tiny script scrolls the page back up; it runs once per screen change.
    """
    st.session_state.page_turns += 1    # makes the script new every time, so it runs again
    script = f"""
    <script>
    // page turn {st.session_state.page_turns}
    (function () {{
      const page = window.parent.document;
      function toTop() {{
        window.parent.scrollTo(0, 0);
        // the part of the page that scrolls (its name differs between Streamlit versions)
        page.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main')
            .forEach(function (el) {{ el.scrollTo(0, 0); }});
      }}
      toTop();
      setTimeout(toTop, 150);   // once more, after the new screen has finished drawing
    }})();
    </script>
    """
    with st.container(key="scrolltop"):
        try:
            st.html(script, unsafe_allow_javascript=True)       # newer Streamlit versions
        except TypeError:
            import streamlit.components.v1 as components       # older Streamlit versions
            components.html(script, height=0)


# --------------------------------------------------------------------------
# 2. Ocean styling (colours, fonts and fish taken from the design mockup)
# --------------------------------------------------------------------------

html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* --- the sea: dark at the surface of the page, lighter blue further down --- */
.stApp, [data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #06202F 0%, #0B3A50 40%, #15657C 100%) fixed;
  color: #EAF4F7;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"], [data-testid="stBottomBlockContainer"] { background: transparent; }
.block-container { max-width: 940px; padding-top: 1.2rem; padding-bottom: 4rem; position: relative; z-index: 1; }

.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp li {
  font-family: "IBM Plex Sans", system-ui, sans-serif;
}
.stApp p, .stApp label, .stApp li { color: #EAF4F7; }
[data-testid="stWidgetLabel"] p { font-size: 14px; font-weight: 600; color: #EAF4F7; }

/* --- rising bubbles --- */
.reef-bubble {
  position: fixed; bottom: -60px; border-radius: 50%; pointer-events: none; z-index: 0;
  background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18);
  animation: rise linear infinite;
}
@keyframes rise {
  0% { transform: translateY(0); opacity: 0; }
  10% { opacity: 1; }
  100% { transform: translateY(-110vh) translateX(26px); opacity: 0; }
}

/* --- header, titles and text --- */
.reef-top { display: flex; align-items: center; justify-content: space-between; padding: 6px 0 4px; }
.reef-brand { display: flex; align-items: center; gap: 11px; font-weight: 600; font-size: 16px; }
.stApp .reef-tag { font-size: 13px; color: #8FB8C6; }
.stApp .reef-eyebrow {
  margin: 26px 0 10px; font-size: 13px; letter-spacing: 0.15em; text-transform: uppercase;
  color: #7FD4C4; font-weight: 600;
}
.reef-title {
  margin: 0 0 14px; font-family: Fraunces, Georgia, serif; font-size: 50px; line-height: 1.08;
  font-weight: 700; color: #EAF4F7;
}
.stApp .reef-lede { margin: 0 0 22px; font-size: 17.5px; line-height: 1.6; color: #BDD9E2; }
.reef-h2 { font-family: Fraunces, Georgia, serif; font-size: 27px; font-weight: 600; color: #EAF4F7; margin: 0; }
.stApp .reef-card-title { margin: 0 0 6px; font-size: 19px; font-weight: 600; color: #7FD4C4; }
.stApp .reef-muted { font-size: 13.5px; color: #8FB8C6; }
.stApp .reef-soft { font-size: 15px; color: #BDD9E2; }
.reef-group { display: flex; align-items: center; gap: 10px; margin: 18px 0 4px; }
.reef-group b { font-size: 16px; font-weight: 600; color: #7FD4C4; }
.reef-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #7FD4C4; }

/* --- glass cards (Streamlit containers with a key get the class st-key-<key>) --- */
.st-key-about, .st-key-actionbar, [class*="st-key-q_"] {
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16);
  border-radius: 18px; padding: 22px 26px;
}
.st-key-actionbar { background: rgba(10,40,55,0.75); border-color: rgba(127,212,196,0.35); }
[class*="st-key-q_"] {
  background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.13);
  border-radius: 13px; padding: 8px 18px;
}
[class*="st-key-group_"] { gap: 9px; }
.stApp .reef-statement { font-size: 15.5px; line-height: 1.4; color: #EAF4F7; }

/* --- the 1-5 answers (and gender) as square buttons instead of radio dots ---
   (written to work with both older and newer Streamlit versions) */
.stApp .stElementContainer:has([data-testid="stRadio"]),
.stApp [data-testid="stRadio"], .stApp [data-testid="stRadio"] > div { width: 100% !important; }
.stApp [role="radiogroup"] { display: flex; gap: 6px; flex-wrap: nowrap; justify-content: flex-end; width: 100%; }
.stApp [role="radiogroup"] label {
  width: 50px; height: 42px; margin: 0 !important; padding: 0 !important;
  display: flex; align-items: center; justify-content: center; cursor: pointer;
  border-radius: 11px; background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.25);
  transition: background 0.15s ease;
}
/* hide the round dot: every part of the option that is not the number text */
.stApp [role="radiogroup"] label div:not([data-testid="stMarkdownContainer"]):not(:has([data-testid="stMarkdownContainer"])) {
  display: none;
}
.stApp [role="radiogroup"] label > div { margin: 0; padding: 0; }
.stApp [role="radiogroup"] label p { margin: 0; font-size: 15px; font-weight: 700; color: #EAF4F7; }
.stApp [role="radiogroup"] label:hover { background: rgba(127,212,196,0.25); }
.stApp [role="radiogroup"] label:has(input:checked) { background: #7FD4C4; border-color: #7FD4C4; }
.stApp [role="radiogroup"] label:has(input:checked) p { color: #06202F !important; }
.st-key-gender [role="radiogroup"] { justify-content: stretch; gap: 8px; }
.st-key-gender [role="radiogroup"] > div, .st-key-gender [role="radiogroup"] > label { flex: 1 1 0; min-width: 0; }
.st-key-gender [role="radiogroup"] label { width: 100%; height: 44px; }
.st-key-gender [role="radiogroup"] label p { font-size: 14.5px; font-weight: 600; white-space: nowrap; }

/* --- the answer scale legend above the statements (stays visible while scrolling) --- */
.st-key-scale, [data-testid="stLayoutWrapper"]:has(> .st-key-scale) {
  position: sticky; top: 3.6rem; z-index: 5;    /* newer Streamlit wraps containers in a layout wrapper */
}
.st-key-scale {
  background: rgba(6,32,47,0.93); backdrop-filter: blur(8px);
  border: 1px solid rgba(127,212,196,0.35); border-radius: 13px; padding: 12px 18px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.stApp .reef-scale-hint { font-size: 14px; line-height: 1.5; color: #BDD9E2; }
.stApp .reef-scale-hint b { font-size: 15.5px; color: #EAF4F7; }
.reef-scale { display: flex; gap: 6px; justify-content: flex-end; }
.reef-scale-step { width: 50px; display: flex; flex-direction: column; align-items: center; gap: 5px; }
.reef-scale-step .num {
  width: 28px; height: 28px; border-radius: 8px; border: 1.5px solid; display: flex;
  align-items: center; justify-content: center; font-size: 14px; font-weight: 700;
}
.reef-scale-step .txt { font-size: 11.5px; line-height: 1.2; text-align: center; color: #DCEDF2; min-height: 28px; }
.reef-scale-bar {
  height: 5px; width: 274px; margin: 8px 0 0 auto; border-radius: 99px;
  background: linear-gradient(90deg, #F08A6C, #F2B48C, #BDD9E2, #A5E6DA, #5FCBB5);
}

/* --- age box --- */
.stApp [data-baseweb="input"] {
  background: rgba(255,255,255,0.10) !important; border: 1px solid rgba(255,255,255,0.25) !important;
  border-radius: 10px !important;
}
.stApp [data-baseweb="input"] input { color: #EAF4F7 !important; -webkit-text-fill-color: #EAF4F7; background: transparent !important; }
.stApp [data-testid="stNumberInput"] button { background: transparent !important; color: #EAF4F7 !important; }

/* --- writing hand: a slider with a hand on the thumb ---
   (the thumb is [role="slider"] in older Streamlit, the box around the range input in newer) */
.st-key-hand [role="slider"], .st-key-hand div:has(> div > input[type="range"]) {
  background: transparent !important; box-shadow: none !important; border: 0 !important;
  width: 34px !important; height: 34px !important;
}
.st-key-hand [role="slider"]::after, .st-key-hand div:has(> div > input[type="range"])::after {
  content: "🖐"; position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
  font-size: 27px; line-height: 1; pointer-events: none; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.4));
}
.st-key-hand [data-testid="stSliderThumbValue"], .st-key-hand [data-testid="stThumbValue"] {
  color: #7FD4C4 !important; font-weight: 700; top: -30px;
}
.st-key-hand [data-testid="stSliderTickBarMin"], .st-key-hand [data-testid="stSliderTickBarMax"],
.st-key-hand [data-testid="stTickBarMin"], .st-key-hand [data-testid="stTickBarMax"] { color: #8FB8C6 !important; }

/* --- age box: a "Press Enter to apply" hint while the box is selected --- */
.st-key-age [data-testid="stNumberInput"] { position: relative; }
.st-key-age [data-testid="stNumberInput"]:focus-within::after {
  content: "Press Enter to apply"; position: absolute; right: 2px; bottom: -21px;
  font-size: 12.5px; color: #8FB8C6;
}

/* --- buttons --- */
.stApp [data-testid="stFormSubmitButton"] button, .stApp [data-testid="stButton"] button {
  background: #7FD4C4; color: #06202F; border: 0; border-radius: 12px;
  padding: 0.85rem 1.8rem; width: 100%;
}
.stApp [data-testid="stFormSubmitButton"] button p, .stApp [data-testid="stButton"] button p {
  color: #06202F !important; font-size: 16px; font-weight: 700;
}
.stApp [data-testid="stFormSubmitButton"] button:hover, .stApp [data-testid="stButton"] button:hover {
  background: #A5E6DA; color: #06202F;
}
[data-testid="stForm"] { border: 0; padding: 0; }

/* the small back button at the top of the result page: outlined instead of filled */
.stApp .st-key-back_top [data-testid="stButton"] button {
  width: auto; background: transparent; border: 1px solid rgba(127,212,196,0.55);
  border-radius: 99px; padding: 0.4rem 1.1rem; min-height: 0;
}
.stApp .st-key-back_top [data-testid="stButton"] button p { color: #7FD4C4 !important; font-size: 14.5px; font-weight: 600; }
.stApp .st-key-back_top [data-testid="stButton"] button:hover { background: rgba(127,212,196,0.15); }

/* the scroll-to-top script takes no space on the page */
.st-key-scrolltop, [data-testid="stLayoutWrapper"]:has(> .st-key-scrolltop) {
  position: absolute; height: 0; overflow: hidden; margin: 0; padding: 0;
}

/* --- the fish --- */
.swim { animation: swim 3.2s ease-in-out infinite; }
@keyframes swim {
  0%, 100% { transform: translateX(-12px) translateY(0) rotate(-2deg); }
  50% { transform: translateX(12px) translateY(-10px) rotate(2deg); }
}
.tail { transform-origin: 78% 50%; animation: flick 0.6s ease-in-out infinite; }
.fin { transform-origin: 50% 25%; animation: flick 0.9s ease-in-out infinite; }
@keyframes flick { 0%, 100% { transform: rotate(-11deg); } 50% { transform: rotate(11deg); } }
.pop { animation: pop 0.5s cubic-bezier(.2, 1.4, .4, 1); }
@keyframes pop { 0% { transform: scale(0.4); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }

/* --- result cards --- */
.reef-hero {
  background: rgba(255,255,255,0.09); border: 2px solid #7FD4C4; border-radius: 22px;
  padding: 10px 32px 30px; text-align: center; margin-bottom: 20px;
}
.reef-hero .name { margin: 0 0 12px; font-family: Fraunces, Georgia, serif; font-size: 56px; line-height: 1; font-weight: 700; color: #EAF4F7; }
.reef-hero .tagline { margin: 0 auto 10px; font-size: 19px; line-height: 1.5; max-width: 38ch; color: #DCEDF2; }
.reef-chip {
  display: inline-block; margin-top: 14px; padding: 6px 14px; border-radius: 99px;
  background: rgba(127,212,196,0.16); border: 1px solid rgba(127,212,196,0.45);
  font-size: 14px; color: #A5E6DA;
}
.reef-panel {
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16);
  border-radius: 18px; padding: 24px 26px; height: 100%;
}
.reef-panel h3 { margin: 0 0 12px; font-family: Fraunces, Georgia, serif; font-size: 24px; font-weight: 600; color: #EAF4F7; }
.reef-panel p { margin: 0 0 12px; font-size: 16px; line-height: 1.65; color: #DCEDF2; }
.reef-bar-label { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; font-size: 14.5px; color: #EAF4F7; }
.reef-bar-track { height: 11px; background: rgba(255,255,255,0.14); border-radius: 99px; overflow: hidden; margin-bottom: 15px; }
.reef-bar-fill { height: 11px; border-radius: 99px; }
.reef-traits { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 6px 0 22px; }
.reef-trait { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16); border-radius: 16px; padding: 16px 16px 14px; }
.reef-trait .value { font-family: Fraunces, Georgia, serif; font-size: 32px; font-weight: 700; color: #EAF4F7; line-height: 1.1; }
/* --- the four types, previewed in a row on the first page --- */
.reef-types { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 10px 0 28px; }
.reef-type {
  background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14);
  border-radius: 16px; padding: 8px 14px 16px; text-align: center;
  display: flex; flex-direction: column;
}
.reef-type .fish { height: 84px; display: flex; align-items: center; justify-content: center; }
.reef-type .name { font-family: Fraunces, Georgia, serif; font-size: 20px; font-weight: 700; line-height: 1.2; }
.reef-type .nut { font-size: 13.5px; line-height: 1.4; color: #DCEDF2; margin-top: 4px; }
.reef-type .share { font-size: 12.5px; color: #8FB8C6; margin-top: auto; padding-top: 8px; }   /* always at the bottom */
.reef-creature {
  display: flex; align-items: center; gap: 24px; margin-bottom: 14px; padding: 16px 24px;
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16); border-radius: 20px;
}
.stApp .reef-fineprint { font-size: 13px; line-height: 1.6; color: #8FB8C6; margin-top: 18px; }

/* --- phones: stack the cards --- */
@media (max-width: 700px) {
  .reef-title { font-size: 36px; }
  .reef-hero .name { font-size: 42px; }
  .reef-traits { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .reef-types { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .reef-creature { flex-direction: column; text-align: center; }
}
</style>
<div class="reef-bubble" style="left:6%;  width:14px; height:14px; animation-duration:13s"></div>
<div class="reef-bubble" style="left:19%; width:8px;  height:8px;  animation-duration:17s; animation-delay:3s"></div>
<div class="reef-bubble" style="left:34%; width:18px; height:18px; animation-duration:15s; animation-delay:6s"></div>
<div class="reef-bubble" style="left:58%; width:10px; height:10px; animation-duration:19s; animation-delay:1s"></div>
<div class="reef-bubble" style="left:73%; width:16px; height:16px; animation-duration:14s; animation-delay:8s"></div>
<div class="reef-bubble" style="left:91%; width:9px;  height:9px;  animation-duration:16s; animation-delay:4s"></div>
""")


# Colour each answer button with its step of the scale (1 = warm ... 5 = teal).
# The selected button is filled with that colour. Two selectors per rule, because
# the button sits one level deeper in newer Streamlit versions.
scale_css = ""
for position, _, colour in SCALE:
    rows = f'[class*="st-key-q_"] [role="radiogroup"] > :nth-child({position})'
    scale_css += (
        f"{rows} p {{ color: {colour}; }}\n"
        f"{rows} label:has(input:checked), {rows}:is(label):has(input:checked) "
        f"{{ background: {colour}; border-color: {colour}; }}\n"
    )
html(f"<style>\n{scale_css}</style>")


def fish_svg(profile_name, width=260):
    """Return an animated SVG of the sea creature for one personality type."""
    info = PROFILES[profile_name]
    body, deep = info["color"], info["deep"]

    if profile_name == "Overcontroller":        # pufferfish: round, with spines
        shape = f"""
        <circle cx="95" cy="60" r="42" fill="{body}"/>
        <circle cx="95" cy="60" r="42" fill="none" stroke="{deep}" stroke-width="3"/>
        <g stroke="{deep}" stroke-width="3" stroke-linecap="round">
        <line x1="95" y1="12" x2="95" y2="2"/><line x1="131" y1="26" x2="139" y2="18"/>
        <line x1="143" y1="60" x2="153" y2="60"/><line x1="131" y1="94" x2="139" y2="102"/>
        <line x1="95" y1="108" x2="95" y2="118"/><line x1="59" y1="94" x2="51" y2="102"/>
        <line x1="47" y1="60" x2="37" y2="60"/><line x1="59" y1="26" x2="51" y2="18"/>
        </g>
        <polygon class="tail" points="137,60 172,40 172,80" fill="{deep}"/>"""
    elif profile_name == "Undercontroller":      # reef shark: sleek, with a dorsal fin
        shape = f"""
        <path d="M30 62 Q78 24 136 56 Q156 62 170 62 Q150 76 136 70 Q78 100 30 62 Z" fill="{body}"/>
        <polygon class="fin" points="86,34 100,8 112,38" fill="{deep}"/>
        <polygon points="92,78 104,98 116,76" fill="{deep}"/>
        <polygon class="tail" points="166,62 196,36 190,62 196,88" fill="{deep}"/>"""
    elif profile_name == "Moderate":             # clownfish: white stripes
        shape = f"""
        <ellipse cx="96" cy="62" rx="52" ry="34" fill="{body}"/>
        <path d="M72 32 Q68 62 72 92 L84 88 Q80 62 84 36 Z" fill="#FFFFFF"/>
        <path d="M112 34 Q108 62 112 90 L124 86 Q120 62 124 38 Z" fill="#FFFFFF"/>
        <polygon class="fin" points="90,30 104,10 114,32" fill="{deep}"/>
        <polygon class="tail" points="146,62 182,36 182,88" fill="{deep}"/>"""
    else:                                        # Resilient: blue tang with a yellow tail
        shape = f"""
        <ellipse cx="96" cy="62" rx="54" ry="36" fill="{body}"/>
        <path d="M70 34 Q110 50 138 44 Q110 70 74 86 Z" fill="{deep}" opacity="0.55"/>
        <polygon class="fin" points="88,28 104,8 116,30" fill="{deep}"/>
        <polygon class="tail" points="148,62 184,40 178,62 184,84" fill="#F2C94C"/>"""

    height = round(width * 124 / 210)
    return f"""
    <svg class="swim" width="{width}" height="{height}" viewBox="0 0 210 124" role="img" aria-label="{info['fish']}">
    {shape}
    <circle cx="68" cy="52" r="7" fill="#FFFFFF"/>
    <circle cx="70" cy="52" r="3.4" fill="#06202F"/>
    </svg>"""


def trait_score(answers, trait):
    """Average answer for one trait, with reverse-worded statements flipped."""
    values = [6 - answers[c] if c in trait["reversed"] else answers[c] for c in trait["codes"]]
    return sum(values) / len(values)


# --------------------------------------------------------------------------
# 3. Load the saved pipeline (trained and saved by modeling.ipynb)
# --------------------------------------------------------------------------


@st.cache_resource
def load_model(path):
    """Load the trained pipeline once and keep it in memory."""
    return joblib.load(path)


html("""
<div class="reef-top">
<div class="reef-brand">
<svg width="26" height="26" viewBox="0 0 26 26" fill="none" stroke="#7FD4C4" stroke-width="1.6" aria-hidden="true">
<path d="M2 17c3-4 5-4 8 0s5 4 8 0 5-4 6-2"/><path d="M2 10c3-4 5-4 8 0s5 4 8 0 5-4 6-2"/>
</svg>
<span>Big Five Personality Test</span>
</div>
<span class="reef-tag">OCEAN model · short form with 19 of the 50 statements</span>
</div>
""")

if not MODEL_PATH.exists():
    st.error(
        "No trained model found at `models/personality_pipeline.joblib`.\n\n"
        "Run the notebooks first (see the README): `eda.ipynb` creates `data/data_clean.csv`, "
        "and `modeling.ipynb` trains the pipeline and saves it."
    )
    st.stop()

model = load_model(MODEL_PATH)

# --------------------------------------------------------------------------
# 4. Remember the answers between the two screens
# --------------------------------------------------------------------------
# "form" = the questionnaire (step 1), "result" = the prediction (step 2).
# The answers are kept, so going back to the questionnaire shows them filled in.

if "screen" not in st.session_state:
    st.session_state.screen = "form"
    st.session_state.inputs = {"age": 29, "gender": "Female", "hand": "Right"}
    st.session_state.inputs.update({code: 3 for code in ALL_CODES})   # every answer starts at Neutral

if "go_to_top" not in st.session_state:
    st.session_state.go_to_top = False    # True right after the screen has changed
    st.session_state.page_turns = 0       # counts the screen changes (used by scroll_to_top)


def back_to_form():
    """Used by both back buttons on the result page."""
    st.session_state.screen = "form"
    st.session_state.go_to_top = True


# Right after a screen change, start at the top of the new screen
if st.session_state.go_to_top:
    scroll_to_top()
    st.session_state.go_to_top = False


# --------------------------------------------------------------------------
# 5. Step 1: the questionnaire
# --------------------------------------------------------------------------

if st.session_state.screen == "form":
    saved = st.session_state.inputs

    html("""
    <p class="reef-eyebrow">Step 1 of 2 · The questionnaire</p>
    <div class="reef-title">What kind of personality do you have?</div>
    <p class="reef-lede" style="margin-bottom:12px">This personality test will help you understand
    why you behave the way you do and how your personality is organized. It is based on the
    Big&nbsp;Five model, known as <b style="color:#7FD4C4">OCEAN</b>: Openness, Conscientiousness,
    Extraversion, Agreeableness and Neuroticism.</p>
    <p class="reef-lede">Answer 19 short statements, a short form of the original 50-item
    questionnaire, and tell us three things about yourself. A machine-learning model trained on
    about 19,500 completed questionnaires predicts which of four personality types fits you best,
    and shows how sure it is.</p>
    """)

    # --- Preview: the four personality types the model can predict ---
    type_cards = "".join(
        f'<div class="reef-type">'
        f'<div class="fish">{fish_svg(name, width=130)}</div>'
        f'<div class="name" style="color:{p["color"]}">{name}</div>'
        f'<div class="nut">{p["nutshell"]}</div>'
        f'<div class="share">{p["fish"]} · {p["share"]} of the data</div>'
        f'</div>'
        for name, p in PROFILES.items()
    )
    html(f"""
    <div style="margin:4px 0 2px"><span class="reef-h2" style="font-size:22px">Four possible results</span></div>
    <p class="reef-soft" style="margin:0">The model predicts one of these four personality types. Each one is
    shown with a sea creature as its symbol.</p>
    <div class="reef-types">{type_cards}</div>
    """)

    # enter_to_submit=False: pressing Enter in the age box only applies the age.
    # The form is sent only with the button at the bottom.
    with st.form("questionnaire", border=False, enter_to_submit=False):

        # --- About you ---
        with st.container(key="about"):
            html('<p class="reef-card-title">About you</p>')
            col_age, col_gender = st.columns(2)
            age = col_age.number_input("Age", min_value=13, max_value=100,
                                       value=int(saved["age"]), step=1, key="age")
            gender = col_gender.radio("Gender", GENDERS, horizontal=True, key="gender",
                                      index=GENDERS.index(saved["gender"]))
            # A slider rather than a dropdown: writing hand goes from left to right.
            hand = st.select_slider("Writing hand", options=HANDS, value=saved["hand"], key="hand")

        # --- The 19 statements ---
        html("""
        <div style="margin:30px 0 6px"><span class="reef-h2">The 19 statements</span></div>
        <p class="reef-soft" style="margin:0 0 6px">Read each statement and tap how much you agree
        with it. There is no right answer, and every statement starts at 3 (Neutral).</p>
        """)

        # The scale legend, lined up with the five buttons of every statement.
        # It stays at the top of the screen while you scroll through the statements.
        steps = "".join(
            f'<div class="reef-scale-step">'
            f'<span class="num" style="color:{colour}; border-color:{colour}">{value}</span>'
            f'<span class="txt">{label.replace(" ", "<br>")}</span></div>'
            for value, label, colour in SCALE
        )
        with st.container(key="scale"):
            col_hint, col_scale = st.columns([1.35, 1], vertical_alignment="center")
            col_hint.markdown(
                '<div class="reef-scale-hint"><b>How much do you agree?</b><br>'
                "The same scale for all 19 statements: from 1 (Disagree) on the left "
                "to 5 (Agree) on the right.</div>",
                unsafe_allow_html=True,
            )
            col_scale.markdown(
                f'<div style="padding-bottom:12px"><div class="reef-scale">{steps}</div>'
                f'<div class="reef-scale-bar"></div></div>',
                unsafe_allow_html=True,
            )

        answers = {}
        for group_name, items in GROUPS:
            html(f'<div class="reef-group"><span class="reef-dot"></span><b>{group_name}</b>'
                 f'<span class="reef-muted">{len(items)} statements</span></div>')
            with st.container(key=f"group_{group_name.split()[0].lower()}"):
                for code, text in items:
                    # One glass card per statement: the text on the left, five buttons on the right
                    with st.container(key=f"q_{code}"):
                        col_text, col_buttons = st.columns([1.35, 1], vertical_alignment="center")
                        col_text.markdown(f'<span class="reef-statement">{text}</span>',
                                          unsafe_allow_html=True)
                        answers[code] = col_buttons.radio(
                            text,                          # label, hidden but read by screen readers
                            options=[1, 2, 3, 4, 5],
                            index=saved[code] - 1,
                            horizontal=True,
                            label_visibility="collapsed",
                            key=code,
                        )

        # --- Action bar with the button ---
        st.write("")
        with st.container(key="actionbar"):
            col_info, col_button = st.columns([1.4, 1], vertical_alignment="center")
            col_info.markdown(
                '<p style="margin:0 0 4px; font-size:16.5px; font-weight:600">Ready for your result?</p>'
                '<p style="margin:0; font-size:14px; color:#8FB8C6">Every statement starts at 3 '
                '(Neutral). Change the ones that matter.</p>',
                unsafe_allow_html=True,
            )
            submitted = col_button.form_submit_button("Show my personality type")

    if submitted:
        st.session_state.inputs = {"age": int(age), "gender": gender, "hand": hand, **answers}
        st.session_state.screen = "result"
        st.session_state.go_to_top = True    # open the result page at the top
        st.rerun()

# --------------------------------------------------------------------------
# 6. Step 2: predict and show the result
# --------------------------------------------------------------------------

else:
    inputs = st.session_state.inputs

    # One row with exactly the columns the pipeline was trained on (names, order, types)
    X_new = pd.DataFrame([inputs])
    if hasattr(model, "feature_names_in_"):
        X_new = X_new.reindex(columns=model.feature_names_in_)

    # .ravel() flattens the result first: most models return ["Moderate"],
    # but CatBoost returns [["Moderate"]] (one extra layer), which would break the lookup below
    prediction = str(model.predict(X_new).ravel()[0])
    info = PROFILES[prediction]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_, model.predict_proba(X_new)[0]))

    confidence_chip = (
        f'<div class="reef-chip">Model confidence: <b>{probabilities[prediction]:.0%}</b></div>'
        if probabilities else ""
    )

    # --- Back button at the top (the same as "Change my answers" further down) ---
    st.button("← Back to the questionnaire", on_click=back_to_form, key="back_top")

    # --- The predicted personality type ---
    html(f"""
    <p class="reef-eyebrow" style="margin-top:14px">Step 2 of 2 · Your result</p>
    <div class="reef-hero pop" style="border-color:{info['color']}">
    <div style="display:flex; justify-content:center; align-items:center; height:170px">
    {fish_svg(prediction)}
    </div>
    <p style="margin:0 0 6px; font-size:15px; color:#BDD9E2">Your most likely personality type</p>
    <div class="name">{prediction}</div>
    <p class="tagline">{info['tagline']}</p>
    <p style="margin:0; font-size:15px; color:#8FB8C6">Ocean symbol:
    <b style="color:{info['color']}">{info['fish']}</b> — {info['why_fish']}</p>
    {confidence_chip}
    </div>
    """)

    # --- What it means + how the four compare ---
    col_meaning, col_bars = st.columns(2)
    with col_meaning:
        html(f"""
        <div class="reef-panel">
        <h3>What this type means</h3>
        <p>{info['body']}</p>
        <p style="margin:0">{info['body2']}</p>
        </div>
        """)
    with col_bars:
        if probabilities:
            bars = ""
            for name, value in sorted(probabilities.items(), key=lambda kv: -kv[1]):
                weight = "700" if name == prediction else "400"
                bars += (
                    f'<div class="reef-bar-label"><span style="font-weight:{weight}">{name} '
                    f'<span style="color:#8FB8C6">· {PROFILES[name]["fish"]}</span></span>'
                    f'<span style="font-size:13.5px; color:#BDD9E2">{value:.0%}</span></div>'
                    f'<div class="reef-bar-track"><div class="reef-bar-fill" '
                    f'style="width:{value * 100:.1f}%; background:{PROFILES[name]["color"]}"></div></div>'
                )
            html(f"""
            <div class="reef-panel">
            <h3 style="margin-bottom:4px">How the four compare</h3>
            <p style="font-size:14px; color:#8FB8C6; margin-bottom:18px">Probability the model assigns to each type</p>
            {bars}
            </div>
            """)

    # --- What pushed the answer this way: your four trait scores vs. the dataset average ---
    cards = ""
    for trait in TRAITS:
        mine = trait_score(inputs, trait)
        diff = mine - trait["average"]
        if abs(diff) < 0.25:
            compared = "about average"
        else:
            compared = f"{abs(diff):.1f} {'above' if diff > 0 else 'below'} average"
        cards += (
            f'<div class="reef-trait">'
            f'<div style="font-size:14px; font-weight:600; color:#7FD4C4">{trait["name"]}</div>'
            f'<div style="font-size:12.5px; color:#8FB8C6; margin-bottom:8px">{trait["trait"]}</div>'
            f'<div class="value">{mine:.1f}</div>'
            f'<div style="font-size:13.5px; color:#DCEDF2; margin:2px 0 8px">{compared} '
            f'<span style="color:#8FB8C6">(avg {trait["average"]:.1f})</span></div>'
            f'<div style="font-size:12.5px; line-height:1.45; color:#8FB8C6">{trait["hint"]}</div>'
            f'</div>'
        )
    html(f"""
    <div style="margin-top:26px"><span class="reef-h2" style="font-size:24px">What pushed the answer this way</span></div>
    <p class="reef-soft" style="margin:4px 0 10px">Your score per trait (1 to 5, reverse-worded statements flipped), next to the average of the 19,588 people in the data.</p>
    <div class="reef-traits">{cards}</div>
    """)

    st.button("Change my answers", on_click=back_to_form, key="back_bottom")

    # --- The four personality types ---
    creatures = ""
    for name, creature in PROFILES.items():
        border = creature["color"] if name == prediction else "rgba(255,255,255,0.16)"
        you = (' <span style="font-size:13px; color:#7FD4C4; font-weight:600">your result</span>'
               if name == prediction else "")
        creatures += f"""
        <div class="reef-creature" style="border-color:{border}">
        <div style="flex-shrink:0; width:170px; display:flex; justify-content:center">{fish_svg(name, width=160)}</div>
        <div>
        <div style="display:flex; align-items:baseline; gap:12px; flex-wrap:wrap">
        <span style="font-family:Fraunces, Georgia, serif; font-size:26px; font-weight:700; color:{creature['color']}">{name}</span>
        <span style="font-size:14.5px; color:#8FB8C6">{creature['fish']} · {creature['share']} of the data</span>{you}
        </div>
        <p style="margin:6px 0 6px; font-size:16px; line-height:1.5; color:#DCEDF2">{creature['tagline']}</p>
        <p style="margin:0; font-size:14px; color:#9FC4D0">Defined by: {creature['rule']}</p>
        </div>
        </div>"""
    html(f"""
    <div style="margin:34px 0 6px"><span class="reef-h2" style="font-size:24px">The four personality types</span></div>
    <p class="reef-soft" style="margin:0 0 14px">Each type is paired with a sea creature as its symbol, a small nod to the OCEAN model. The share shows how much of the data each type covers.</p>
    {creatures}
    """)

# --------------------------------------------------------------------------
# 7. Fine print
# --------------------------------------------------------------------------

model_name = type(model.steps[-1][1]).__name__ if hasattr(model, "steps") else type(model).__name__
html(f"""
<p class="reef-fineprint">Model: <code style="color:#BDD9E2">{model_name}</code> pipeline, loaded from
<code style="color:#BDD9E2">models/personality_pipeline.joblib</code>. A student project for a
machine-learning course: the model recognises patterns in questionnaire answers and is not a
psychological assessment.</p>
""")
