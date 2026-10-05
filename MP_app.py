"""Lokale Streamlit-App: gespeicherte Pipeline laden, Eingaben vorhersagen.

Start im Projektordner: python -m streamlit run app.py
Die trainierte Datei liegt daneben unter models/best_pipeline.joblib.
"""

from pathlib import Path
from urllib.parse import quote

import joblib
import pandas as pd
import streamlit as st


# Deutsche Übersetzungen der Originalaussagen aus codebook.txt.
# Die Antworten werden als Rohwerte 1–5 übergeben, nicht umgekehrt.
QUESTION_GROUPS = {
    "Umgang mit Gefühlen": {
        "N1": "Ich gerate leicht unter Stress.",
        "N2": "Ich bin die meiste Zeit entspannt.",
        "N3": "Ich mache mir Sorgen über Dinge.",
        "N4": "Ich fühle mich selten niedergeschlagen.",
        "N5": "Ich lasse mich leicht aus der Ruhe bringen.",
        "N6": "Ich rege mich leicht auf.",
        "N7": "Meine Stimmung ändert sich häufig.",
        "N8": "Ich habe häufig Stimmungsschwankungen.",
        "N9": "Ich bin schnell gereizt.",
        "N10": "Ich fühle mich oft niedergeschlagen.",
    },
    "Umgang mit anderen Menschen": {
        "E1": "Ich sorge auf Partys für Stimmung.",
        "E3": "Ich fühle mich in Gesellschaft anderer Menschen wohl.",
        "E4": "Ich halte mich im Hintergrund.",
        "E5": "Ich beginne Gespräche.",
        "E7": "Ich spreche auf Partys mit vielen verschiedenen Menschen.",
        "E9": "Es macht mir nichts aus, im Mittelpunkt der Aufmerksamkeit zu stehen.",
        "E10": "In Gegenwart fremder Menschen bin ich still.",
    },
    "Alltag und Mitgefühl": {
        "C4": "Ich bringe Dinge durcheinander.",
        "A4": "Ich fühle mit anderen mit.",
    },
}

GENDER_VALUES = {"Männlich": "Male", "Weiblich": "Female", "Andere": "Other"}
HAND_VALUES = {"Rechts": "Right", "Links": "Left"}

TYPE_LABELS = {
    "Moderate": "Moderat",
    "Resilient": "Resilient",
    "Overcontroller": "Überkontrolliert",
    "Undercontroller": "Unterkontrolliert",
}
TYPE_DESCRIPTIONS = {
    "Moderate": "Ein eher ausgeglichenes Profil ohne ausgeprägte Extreme.",
    "Resilient": "Ein Profil mit vergleichsweise geringer emotionaler Belastung.",
    "Overcontroller": "Ein eher zurückhaltendes Profil mit stärker ausgeprägter emotionaler Belastung.",
    "Undercontroller": "Ein Profil mit geringerer Ausprägung von Gewissenhaftigkeit und Verträglichkeit.",
}

FISH_NAMES = {
    "Moderate": "Clownfisch",
    "Resilient": "Pottwal",
    "Overcontroller": "Hai",
    "Undercontroller": "Einsiedlerkrebs",
}

TYPE_DETAILS = {
    "Moderate": "Im Projekt ist Moderat die Gruppe, für die keine der anderen Zuordnungsregeln greift. Das Modell erkennt diese Gruppe anhand entsprechender Antwortmuster. Die Zuordnung bedeutet nicht, dass alle Eigenschaften einer Person gleich stark ausgeprägt sind.",
    "Resilient": "Resilient bedeutet widerstandsfähig. Im Projekt steht dieser Typ für einen niedrigen Neurotizismus-Wert: Die Antworten deuten auf vergleichsweise weniger Sorgen, Gereiztheit und Stimmungsschwankungen hin. Antwortmuster mit diesen Merkmalen werden im Projekt dieser Gruppe zugeordnet.",
    "Overcontroller": "Im Projekt verbindet dieser Typ einen niedrigen Extraversion-Wert mit einem hohen Neurotizismus-Wert. Das beschreibt ein eher zurückhaltendes Antwortmuster, zusammen mit häufiger berichteten Sorgen oder emotionaler Anspannung. Überkontrolliert ist dabei der Name der berechneten Gruppe.",
    "Undercontroller": "Dieser Typ wurde im Projekt aus niedrigen Werten für Gewissenhaftigkeit und Verträglichkeit gebildet. Dabei spielen unter anderem die Antworten zu Ordnung und zum Mitgefühl mit anderen eine Rolle. Unterkontrolliert ist der Gruppenname und keine Bewertung deiner Person.",
}


def new_creature_svg(personality_type):
    """Hai, Pottwal und Einsiedlerkrebs im gleichen freundlichen Illustrationsstil."""
    if personality_type == "Overcontroller":
        colors = ("#d6f2ff", "#77b6d6", "#3b729b")
        body = '<path d="M72 100 Q38 66 20 51 L30 91 L16 133 Q49 128 72 107Z" fill="url(#creature-fin-overcontroller)"/><path d="M136 62 Q143 12 168 21 L180 67Z" fill="url(#creature-fin-overcontroller)"/><path d="M69 86 Q117 48 184 63 Q223 66 268 99 Q239 128 190 134 Q108 140 69 108Z" fill="url(#creature-overcontroller)"/><path d="M84 107 Q159 109 236 113 Q194 145 115 127Z" fill="#dff3f5"/><path d="M145 111 Q143 152 177 157 L186 122Z" fill="url(#creature-fin-overcontroller)"/><path d="M184 89 l-6 18 M175 87 l-6 18 M167 85 l-6 18" stroke="#477c99" stroke-width="3" stroke-linecap="round"/><ellipse cx="223" cy="88" rx="15" ry="18" fill="#fffdf5"/><ellipse cx="228" cy="90" rx="8" ry="11" fill="#17334e"/><circle cx="226" cy="84" r="4" fill="white"/><path d="M235 111 q12 6 22 -4" fill="none" stroke="#305d7f" stroke-width="3" stroke-linecap="round"/><path d="M207 67 q12 -6 21 -1" fill="none" stroke="#487ea0" stroke-width="3" stroke-linecap="round"/><ellipse cx="138" cy="75" rx="33" ry="7" fill="white" opacity=".25"/>'
    elif personality_type == "Resilient":
        colors = ("#b2d5f1", "#608dae", "#334d80")
        body = '<path d="M82 113 Q46 92 23 102 Q16 88 7 91 Q10 115 46 125 Q24 143 17 155 Q49 161 82 128Z" fill="url(#creature-fin-resilient)"/><path d="M91 86 Q126 66 164 62 Q216 49 251 67 Q270 80 265 123 Q261 146 228 147 Q147 151 91 123Z" fill="url(#creature-resilient)"/><path d="M106 124 Q169 135 252 127 Q247 146 207 143 Q145 142 106 124Z" fill="#a7ccd9"/><path d="M146 117 Q139 157 170 163 Q184 155 190 136Z" fill="url(#creature-fin-resilient)"/><path d="M112 81 Q122 57 140 69" fill="url(#creature-fin-resilient)"/><ellipse cx="224" cy="91" rx="14" ry="17" fill="#fffdf3"/><ellipse cx="228" cy="94" rx="7" ry="10" fill="#17334e"/><circle cx="225" cy="87" r="4" fill="white"/><path d="M239 123 q11 5 21 -3" fill="none" stroke="#263f65" stroke-width="3" stroke-linecap="round"/><path d="M184 59 l7 -17 M186 41 q-14 -15 -19 -3 M190 39 q8 -17 17 -8" fill="none" stroke="#a9e6ec" stroke-width="4" stroke-linecap="round"/><ellipse cx="178" cy="72" rx="32" ry="7" fill="white" opacity=".22"/>'
    else:
        colors = ("#ffe4c7", "#dcad80", "#a56859")
        body = '<path d="M161 129 Q225 154 255 117 Q282 71 244 41 Q210 11 173 43 Q137 70 151 106Z" fill="url(#creature-undercontroller)" stroke="#9c695d" stroke-width="2"/><path d="M168 111 Q146 60 191 42 Q238 25 253 68 Q266 109 226 113 Q190 117 186 87 Q182 59 216 62 Q241 66 230 87 Q219 102 208 88" fill="none" stroke="#9f665b" stroke-width="8" stroke-linecap="round"/><path d="M165 63 Q192 31 225 39" fill="none" stroke="#fff2d1" stroke-width="8" stroke-linecap="round" opacity=".65"/><ellipse cx="129" cy="121" rx="46" ry="30" fill="#f2936c"/><path d="M104 131 L71 154 L45 147 M128 140 L103 166 L78 165 M152 138 L173 160 L193 159" fill="none" stroke="#e6795d" stroke-width="10" stroke-linecap="round"/><path d="M99 115 Q65 95 42 117 M153 115 Q171 102 183 123" fill="none" stroke="#e6795d" stroke-width="10" stroke-linecap="round"/><path d="M48 117 Q18 122 19 91 Q34 84 41 103 Q37 81 53 77 Q70 98 48 117Z" fill="#ffb786" stroke="#e58b67" stroke-width="2"/><path d="M112 106 L104 77 M143 105 L147 73" stroke="#ed8d68" stroke-width="11" stroke-linecap="round"/><ellipse cx="103" cy="74" rx="13" ry="16" fill="#fffdf4"/><ellipse cx="148" cy="70" rx="13" ry="16" fill="#fffdf4"/><ellipse cx="106" cy="77" rx="7" ry="10" fill="#25334d"/><ellipse cx="148" cy="74" rx="7" ry="10" fill="#25334d"/><circle cx="103" cy="71" r="4" fill="white"/><circle cx="145" cy="68" r="4" fill="white"/><path d="M117 127 q11 10 23 -1" fill="none" stroke="#934f47" stroke-width="3" stroke-linecap="round"/>'
    name = personality_type.lower()
    light, middle, shadow = colors
    gradients = f'<defs><radialGradient id="creature-{name}" cx="35%" cy="20%" r="85%"><stop stop-color="{light}"/><stop offset=".5" stop-color="{middle}"/><stop offset="1" stop-color="{shadow}"/></radialGradient><linearGradient id="creature-fin-{name}" x2="0" y2="1"><stop stop-color="{middle}"/><stop offset="1" stop-color="{shadow}"/></linearGradient></defs>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 290 180" role="img" aria-label="{FISH_NAMES[personality_type]}">{gradients}{body}</svg>'


def fish_svg(personality_type):
    """Freundliche Animationsfilm-Fische mit weichen Formen und Lichtreflexen."""
    if personality_type != "Moderate":
        return new_creature_svg(personality_type)
    prefix = personality_type.lower()
    light, color, shadow = {
        "Moderate": ("#ffd789", "#ff922e", "#d95627"),
        "Resilient": ("#b4ffe5", "#43d7bc", "#14899b"),
        "Overcontroller": ("#c4eeb5", "#70b89a", "#337977"),
        "Undercontroller": ("#c0ccff", "#759fed", "#415cad"),
    }[personality_type]
    definitions = f'<defs><radialGradient id="{prefix}-body" cx="38%" cy="23%" r="80%"><stop stop-color="{light}"/><stop offset=".55" stop-color="{color}"/><stop offset="1" stop-color="{shadow}"/></radialGradient><linearGradient id="{prefix}-fin" x2="0" y2="1"><stop stop-color="{light}"/><stop offset="1" stop-color="{color}"/></linearGradient></defs>'
    body_fill = f'url(#{prefix}-body)'
    fin_fill = f'url(#{prefix}-fin)'
    if personality_type == "Overcontroller":
        body = f'<path d="M35 99 Q8 146 77 129 Q133 116 178 88 Q190 36 231 44 Q273 49 268 88 Q265 112 235 115 Q207 115 196 99 Q141 133 86 149 Q6 169 18 111Z" fill="{body_fill}" stroke="#30666a" stroke-width="2"/><path d="M31 117 Q100 116 176 76 Q196 33 225 39" fill="none" stroke="#c4eeb5" stroke-width="7" stroke-linecap="round" opacity=".7"/><g fill="#2a7774" opacity=".5"><ellipse cx="83" cy="131" rx="6" ry="4"/><ellipse cx="117" cy="119" rx="7" ry="4"/><ellipse cx="150" cy="104" rx="6" ry="4"/><ellipse cx="179" cy="95" rx="5" ry="4"/></g><ellipse cx="236" cy="75" rx="17" ry="20" fill="#fff9e9"/><ellipse cx="241" cy="78" rx="10" ry="13" fill="#25506d"/><ellipse cx="244" cy="79" rx="6" ry="9" fill="#152e47"/><circle cx="239" cy="71" r="5" fill="white"/><circle cx="246" cy="82" r="2" fill="white"/><path d="M218 52 q13 -11 27 -2" fill="none" stroke="#356166" stroke-width="4" stroke-linecap="round"/><path d="M230 100 q13 8 24 -3" fill="none" stroke="#28545e" stroke-width="3" stroke-linecap="round"/><ellipse cx="220" cy="95" rx="9" ry="5" fill="#ffd0b5" opacity=".6"/>'
    else:
        body = f'<path d="M90 95 Q46 44 25 59 Q44 95 25 130 Q50 148 90 95Z" fill="{fin_fill}" stroke="{shadow}" stroke-width="2"/><path d="M111 58 Q125 17 163 30 Q191 35 196 66Z" fill="{fin_fill}" stroke="{shadow}" stroke-width="2"/><path d="M117 122 Q133 158 169 143 L187 120Z" fill="{fin_fill}"/><path d="M53 90 Q62 50 130 46 Q215 36 248 82 Q267 111 230 133 Q192 161 119 141 Q60 127 53 90Z" fill="{body_fill}" stroke="{shadow}" stroke-width="2"/><path d="M35 73 L70 96 M35 114 L70 98 M128 43 L137 58 M151 37 L153 54 M174 43 L171 57" fill="none" stroke="{shadow}" stroke-width="2" opacity=".5"/>'
        if personality_type == "Moderate":
            body += '<path d="M97 51 Q83 91 102 134 M152 46 Q137 95 156 145 M198 52 Q183 88 200 140" fill="none" stroke="#784739" stroke-width="20"/><path d="M97 51 Q83 91 102 134 M152 46 Q137 95 156 145 M198 52 Q183 88 200 140" fill="none" stroke="#fff5de" stroke-width="13"/>'
        elif personality_type == "Resilient":
            body += '<path d="M76 77 Q127 50 188 65 M73 98 Q129 77 185 90 M93 119 Q137 100 190 114" fill="none" stroke="#168fab" stroke-width="7" stroke-linecap="round" opacity=".65"/><path d="M246 93 q25 -4 14 9 q12 13 -10 10" fill="#f8b5c3"/>'
        else:
            body += '<path d="M86 78 Q118 112 154 124 L183 55" fill="none" stroke="#344e94" stroke-width="12" stroke-linejoin="round"/><g fill="#ffe599"><circle cx="111" cy="76" r="5"/><circle cx="132" cy="96" r="5"/><circle cx="155" cy="68" r="5"/></g>'
        body += f'<path d="M137 93 Q106 75 113 106 Q118 122 138 110Z" fill="{fin_fill}" stroke="{shadow}" stroke-width="2"/><path d="M116 98 l16 5" stroke="{shadow}" stroke-width="2" opacity=".5"/><ellipse cx="124" cy="61" rx="27" ry="8" fill="white" opacity=".3" transform="rotate(-9 124 61)"/><ellipse cx="222" cy="80" rx="19" ry="23" fill="#fffaf0"/><ellipse cx="229" cy="84" rx="11" ry="15" fill="#284c72"/><ellipse cx="232" cy="85" rx="7" ry="10" fill="#14283f"/><circle cx="225" cy="75" r="6" fill="white"/><circle cx="236" cy="87" r="2.5" fill="white"/><path d="M206 54 q12 -10 25 -1" fill="none" stroke="{shadow}" stroke-width="4" stroke-linecap="round"/><path d="M229 114 q12 8 20 -2" fill="none" stroke="{shadow}" stroke-width="3" stroke-linecap="round"/><ellipse cx="211" cy="109" rx="11" ry="6" fill="#ffc3b7" opacity=".65"/>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 290 180" role="img" aria-label="{FISH_NAMES[personality_type]}">{definitions}{body}</svg>'


def net_svg():
    """Weich gezeichnetes Seilnetz als Rahmen für beide Typnamen."""
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 250" aria-hidden="true"><defs><linearGradient id="rope" x2="0" y2="1"><stop stop-color="#ffe8b6"/><stop offset=".5" stop-color="#d0aa72"/><stop offset="1" stop-color="#a3764f"/></linearGradient><pattern id="mesh" width="42" height="42" patternUnits="userSpaceOnUse"><path d="M0 0 L42 42 M42 0 L0 42" fill="none" stroke="#dab887" stroke-width="2" opacity=".55"/><circle cx="21" cy="21" r="3" fill="#e4cda3" opacity=".65"/></pattern><clipPath id="net-shape"><path d="M43 38 Q180 7 316 39 Q349 106 300 192 Q180 258 62 197 Q14 133 43 38Z"/></clipPath></defs><ellipse cx="181" cy="220" rx="119" ry="10" fill="#001e30" opacity=".28"/><path d="M43 38 Q180 7 316 39 Q349 106 300 192 Q180 258 62 197 Q14 133 43 38Z" fill="#143f53"/><rect x="20" y="20" width="320" height="215" fill="url(#mesh)" clip-path="url(#net-shape)"/><path d="M43 38 Q180 7 316 39 Q349 106 300 192 Q180 258 62 197 Q14 133 43 38Z" fill="none" stroke="#052333" stroke-width="15"/><path d="M43 38 Q180 7 316 39 Q349 106 300 192 Q180 258 62 197 Q14 133 43 38Z" fill="none" stroke="url(#rope)" stroke-width="10"/><path d="M43 38 Q180 7 316 39 Q349 106 300 192 Q180 258 62 197 Q14 133 43 38Z" fill="none" stroke="#fff2cd" stroke-width="2" stroke-dasharray="3 9" opacity=".65"/><path d="M42 37 Q28 15 19 32 Q16 48 42 37 M317 40 Q344 18 349 36 Q350 54 317 40" fill="none" stroke="url(#rope)" stroke-width="6" stroke-linecap="round"/></svg>'


def ship_svg():
    """Eine dezente Schiffssilhouette für die Animation hinter dem Titel."""
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 240" aria-hidden="true"><g fill="#afe1e9" stroke="#afe1e9" stroke-linejoin="round"><path d="M28 132 Q230 155 482 123 L441 183 Q251 221 78 174Z"/><rect x="142" y="82" width="220" height="59" rx="9"/><rect x="177" y="54" width="131" height="36" rx="4"/><rect x="225" y="18" width="34" height="47" rx="3"/><path d="M111 133 V60 M105 70 h23 M379 135 V76 M365 82 h29" fill="none" stroke-width="7"/><path d="M78 134 V113 H435 V128" fill="none" stroke-width="4"/></g><g fill="#17576d"><rect x="162" y="97" width="26" height="17" rx="4"/><rect x="204" y="97" width="26" height="17" rx="4"/><rect x="246" y="97" width="26" height="17" rx="4"/><rect x="288" y="97" width="26" height="17" rx="4"/><circle cx="158" cy="160" r="7"/><circle cx="220" cy="165" r="7"/><circle cx="282" cy="165" r="7"/><circle cx="344" cy="158" r="7"/></g></svg>'


def apply_ocean_design():
    """Unterwasserfarben, bewegte Dekoration und gut erkennbare Auswahlkarten."""
    man = '<circle cx="40" cy="17" r="11"/><path d="M26 33 h28 v29 h-8 v24 h-11 V62 h-9Z"/><path d="M26 36 l-9 26 M54 36 l9 26" fill="none" stroke="currentColor" stroke-width="9" stroke-linecap="round"/>'
    woman = '<circle cx="40" cy="17" r="11"/><path d="M30 33 h20 l14 35 H16Z"/><path d="M32 68 v18 M48 68 v18 M28 37 l-12 25 M52 37 l12 25" fill="none" stroke="currentColor" stroke-width="9" stroke-linecap="round"/>'
    question = '<text x="40" y="72" text-anchor="middle" font-family="sans-serif" font-size="78" font-weight="bold">?</text>'
    hand = '<path d="M28 85 Q16 72 10 57 Q8 50 14 48 Q18 47 24 59 V28 Q24 20 30 20 Q36 20 36 28 V17 Q36 9 42 9 Q48 9 48 17 V25 Q48 17 54 17 Q60 17 60 25 V33 Q60 26 66 26 Q72 26 72 34 V60 Q72 76 60 85Z"/>'
    icons = [man, woman, question, f'<g transform="translate(80 0) scale(-1 1)">{hand}</g>', hand]
    icon_css = ""
    for index, shapes in enumerate(icons):
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 96" fill="#90e5ee" color="#90e5ee">{shapes}</svg>'
        group = "gender_choice" if index < 3 else "hand_choice"
        position = index + 1 if index < 3 else index - 2
        icon_css += f'.st-key-{group} [role="radiogroup"] > div:nth-child({position}) label::before {{background-image:url("data:image/svg+xml,{quote(svg)}");}}'
    trident = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 110"><defs><linearGradient id="gold"><stop stop-color="#fff2bb"/><stop offset="1" stop-color="#da9b42"/></linearGradient></defs><g stroke="url(#gold)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" fill="none"><path d="M60 99 V15 M31 26 V47 Q31 62 60 62 Q89 62 89 47 V26"/></g><path d="M60 5 L49 25 H71Z M31 12 L20 32 H42Z M89 12 L78 32 H100Z" fill="#ffe5a0"/><path d="M47 78 H73" stroke="#b87537" stroke-width="6" stroke-linecap="round"/></svg>'
    chest = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 110"><defs><linearGradient id="wood" x2="0" y2="1"><stop stop-color="#cb8a57"/><stop offset="1" stop-color="#805049"/></linearGradient></defs><path d="M19 57 Q70 -5 122 46 L112 59Z" fill="url(#wood)" stroke="#ecc77b" stroke-width="5"/><path d="M24 65 Q70 45 119 63 L113 99 H28Z" fill="url(#wood)" stroke="#ecc77b" stroke-width="5"/><path d="M43 66 V97 M96 66 V98" stroke="#ecc77b" stroke-width="8"/><rect x="64" y="64" width="17" height="21" rx="4" fill="#ffe59b"/><circle cx="72" cy="74" r="3" fill="#82534b"/><g fill="#ffe99d"><ellipse cx="56" cy="58" rx="11" ry="5"/><ellipse cx="76" cy="55" rx="11" ry="5"/><ellipse cx="92" cy="59" rx="11" ry="5"/><path d="M70 19 l4 8 9 4 -9 4 -4 9 -4 -9 -9 -4 9 -4Z"/></g></svg>'
    for button_key, artwork in [("own_type", trident), ("other_types", chest)]:
        icon_css += f'.st-key-{button_key} button::before {{background-image:url("data:image/svg+xml,{quote(artwork)}");}}'
    css = '''
    <style>
    [data-testid="stAppViewContainer"] {background:linear-gradient(165deg,#136b86 0%,#07374e 42%,#031b30 100%);color:#edfaff;}
    [data-testid="stHeader"] {background:transparent;}
    [data-testid="stMainBlockContainer"] {max-width:920px;padding-top:2.6rem;position:relative;z-index:1;}
    [data-testid="stMainBlockContainer"] h1, [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3, [data-testid="stWidgetLabel"] p {color:#edfaff;}
    [data-testid="stCaptionContainer"] {color:#bedce8;}
    [data-testid="stForm"] {position:relative;z-index:1;background:rgba(4,30,49,.94);border:1px solid #548b9c;border-radius:24px;padding:26px;box-shadow:0 18px 70px #00162466;}
    [data-testid="stTextInput"] input {background:#eaf8fc;color:#123b50;}
    [data-testid="stTextInput"] [data-baseweb="input"] {background:#eaf8fc;border-radius:12px;}
    .ocean-hero {position:relative;isolation:isolate;z-index:1;padding:22px 0 25px;overflow:hidden;}
    .sea-ship {position:absolute;pointer-events:none;bottom:16px;right:8%;width:340px;max-width:60vw;opacity:.12;transform-origin:50% 65%;animation:sink 13s ease-in forwards;}
    .sea-ship svg {width:100%;}
    .seabed {position:absolute;bottom:0;left:0;width:100%;height:30px;border-radius:50% 35% 0 0;background:linear-gradient(#4b6c6d77,#46545599);}
    .ocean-kicker {color:#94e9e9;font-size:13px;letter-spacing:5px;font-weight:700;}
    .ocean-hero h1 {font-size:clamp(36px,6vw,60px);line-height:1.1;margin:12px 0 16px;letter-spacing:-2px;}
    .ocean-hero p {max-width:580px;color:#cfe7ee;font-size:17px;line-height:1.65;}
    .sea-scene {position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;}
    .sea-scene::before {content:"";position:absolute;inset:0;background:repeating-conic-gradient(from 160deg at 50% -20%,transparent 0deg 9deg,#93eff00b 10deg 15deg,transparent 16deg 30deg);}
    .sea-fish {position:absolute;width:155px;opacity:.38;animation:swim 19s ease-in-out infinite alternate;}
    .sea-fish:nth-child(1) {top:18%;left:1%;}
    .sea-fish:nth-child(2) {top:48%;right:1%;width:185px;animation-delay:-7s;}
    .sea-fish:nth-child(3) {top:78%;left:2%;animation-delay:-12s;}
    .sea-fish:nth-child(4) {top:8%;right:3%;animation-delay:-4s;}
    .sea-diver {position:absolute;width:180px;opacity:.25;top:62%;left:0%;transform:rotate(-18deg);animation:float 10s ease-in-out infinite alternate;}
    .sea-diver.second {top:30%;left:auto;right:0%;transform:scaleX(-1) rotate(-18deg);animation-delay:-5s;}
    .bubble {position:absolute;bottom:-60px;border:1px solid #b3f6f77a;border-radius:50%;background:#abecff0d;animation:rise 17s linear infinite;}
    @keyframes swim {to {translate:35px -20px;}}
    @keyframes float {to {translate:18px -35px;}}
    @keyframes rise {to {translate:30px -110vh;}}
    @keyframes sink {0%{transform:translateY(calc(-100vh + 240px)) rotate(75deg);} 87%{transform:translateY(-22px) rotate(75deg);} 96%{transform:translateY(3px) rotate(6deg);} 100%{transform:translateY(0) rotate(9deg);}}
    .type-overview {position:relative;z-index:1;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:0 0 18px;}
    .type-mini {border:1px solid #47829755;background:#082e4377;border-radius:18px;padding:12px 6px;text-align:center;}
    .type-mini svg {width:88px;height:55px;display:block;margin:0 auto 5px;}
    .type-mini strong {font-size:15px;color:#e4f9ff;display:block;}
    .type-mini small {font-size:12px;color:#aecfdc;display:block;margin-top:4px;}
    .stRadio [role="radiogroup"] {gap:12px;flex-wrap:wrap;}
    .st-key-gender_choice [role="radiogroup"] label,
    .st-key-hand_choice [role="radiogroup"] label {position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px;min-width:100px;min-height:130px;border:1px solid #557b90;border-radius:16px;background:#113b52;padding:12px;cursor:pointer;}
    .st-key-gender_choice [role="radiogroup"] label::before,
    .st-key-hand_choice [role="radiogroup"] label::before {content:"";display:block;width:48px;height:58px;background-size:contain;background-repeat:no-repeat;background-position:center;}
    .stRadio [role="radiogroup"] label:has(input:checked) {background:#175a70;border-color:#8cf0ed;box-shadow:0 0 0 2px #8cf0ed55;}
    .stRadio [role="radiogroup"] label:focus-within {outline:2px solid #b7ffff;outline-offset:3px;}
    .stRadio [role="radiogroup"] label > div > div:first-child {display:none;}
    .stRadio [role="radiogroup"] label p {color:#edfaff;}
    [data-testid="stFormSubmitButton"] button {background:#8fe8e4;color:#062f43;border:0;border-radius:12px;min-height:50px;font-weight:700;}
    [data-testid="stFormSubmitButton"] button:hover {background:#b7ffff;color:#062f43;}
    [data-testid="stSliderTickBar"] {visibility:hidden;}
    [data-testid="stSliderThumbValue"] p {color:#a6ffff;}
    [data-testid="stSlider"] [role="group"] > div > div:first-child {background:#538b9d;}
    [data-testid="stSlider"] [role="group"] > div > div[data-rac] {background:#a6ffff;}
    [data-testid="stSlider"] [role="group"] > div::after {content:"";position:absolute;inset:0;pointer-events:none;background:radial-gradient(circle at 1% 50%,#a6ffff 0 3px,transparent 4px),radial-gradient(circle at 25% 50%,#a6ffff 0 3px,transparent 4px),radial-gradient(circle at 50% 50%,#a6ffff 0 3px,transparent 4px),radial-gradient(circle at 75% 50%,#a6ffff 0 3px,transparent 4px),radial-gradient(circle at 99% 50%,#a6ffff 0 3px,transparent 4px);}
    .answer-scale {display:flex;justify-content:space-between;margin-top:-22px;margin-bottom:12px;color:#d4f5fb;font-size:14px;padding:0 2px;}
    .result-card {position:relative;z-index:1;background:#082c44;border:1px solid #548b9c;border-radius:24px;padding:30px;margin-top:24px;}
    .result-layout {display:flex;align-items:center;justify-content:space-around;gap:24px;flex-wrap:wrap;}
    .result-net {position:relative;display:flex;align-items:center;justify-content:center;height:235px;width:340px;max-width:100%;}
    .result-net > svg {position:absolute;inset:0;width:100%;height:100%;filter:drop-shadow(0 7px 10px #00162288);}
    .result-type {position:relative;padding:12px 14px;background:#10394eed;border-radius:15px;text-align:center;box-shadow:0 4px 18px #031e3077;}
    .result-type strong {display:block;color:#fff5d4;font-size:26px;font-weight:750;}
    .result-type small {display:block;color:#d9c7a7;font-size:15px;margin-top:6px;}
    .result-fish {width:250px;text-align:center;color:#c6e9ef;}
    .result-fish svg {width:100%;animation:float 5s ease-in-out infinite alternate;filter:drop-shadow(0 12px 12px #00162777);}
    .result-card .result-description {font-size:17px;line-height:1.7;color:#dceef4;margin-top:22px;}
    .result-note {font-size:13px;color:#adcdd9;}
    .st-key-own_type button, .st-key-other_types button {position:relative;z-index:1;display:flex;flex-direction:column;gap:12px;width:100%;min-height:190px;padding:20px;background:linear-gradient(145deg,#15435a,#082c44);color:#edfaff;border:1px solid #548b9c;border-radius:22px;white-space:normal;}
    .st-key-own_type button div, .st-key-own_type button span, .st-key-own_type button p,
    .st-key-other_types button div, .st-key-other_types button span, .st-key-other_types button p {white-space:normal!important;text-overflow:clip!important;overflow:visible!important;-webkit-line-clamp:unset!important;}
    .st-key-own_type button::before, .st-key-other_types button::before {content:"";width:85px;height:78px;background-size:contain;background-position:center;background-repeat:no-repeat;}
    .st-key-own_type button:hover, .st-key-other_types button:hover {border-color:#a6ffff;background:#175a70;color:#fff;}
    .st-key-own_type button:focus-visible, .st-key-other_types button:focus-visible {outline:2px solid #a6ffff;outline-offset:3px;}
    .type-info {position:relative;z-index:1;background:#082c44;border:1px solid #548b9c;border-radius:20px;padding:22px;margin:12px 0;color:#dceef4;line-height:1.7;}
    .type-info h3 {margin:0 0 5px;font-size:21px;}
    .type-info small {display:block;color:#aecfdc;font-size:14px;margin-bottom:15px;}
    .type-info svg {width:140px;float:right;margin:0 0 12px 18px;}
    .type-info::after {content:"";display:block;clear:both;}
    @media(max-width:700px) {.sea-fish,.sea-diver{opacity:.14;} .type-overview{grid-template-columns:repeat(2,minmax(0,1fr));} [data-testid="stForm"]{padding:16px;} .stRadio [role="radiogroup"] label{min-width:75px;} .result-card{padding:18px;} .result-type strong{font-size:23px;}}
    @media(prefers-reduced-motion:reduce) {.sea-fish,.sea-diver,.bubble,.result-fish svg,.sea-ship{animation:none;} .sea-ship{transform:rotate(9deg);}}
    ''' + icon_css + '</style>'
    diver = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 250 140"><g fill="#9adce9" stroke="#9adce9" stroke-width="8" stroke-linecap="round"><circle cx="190" cy="46" r="16"/><path d="M166 62 L113 86 L72 71 M118 84 L80 113 M153 69 L183 85 L207 71" fill="none"/><path d="M65 66 l-40 -9 23 22Z M74 108 l-28 18 40 -4Z"/><rect x="126" y="42" width="38" height="18" rx="8" transform="rotate(-25 145 51)"/><rect x="194" y="39" width="20" height="11" rx="4" fill="#07374e" stroke-width="3"/></g></svg>'
    fishes = ''.join(f'<div class="sea-fish">{fish_svg(name)}</div>' for name in FISH_NAMES)
    bubbles = ''.join(f'<i class="bubble" style="left:{position}%;width:{size}px;height:{size}px;animation-delay:-{index * 3}s;"></i>' for index, (position, size) in enumerate([(6,18),(18,9),(35,14),(65,10),(83,22),(94,12)]))
    st.markdown(css + f'<div class="sea-scene" aria-hidden="true">{fishes}<div class="sea-diver">{diver}</div><div class="sea-diver second">{diver}</div>{bubbles}<div class="seabed"></div><div class="sea-ship">{ship_svg()}</div></div>', unsafe_allow_html=True)


def show_ocean_result(prediction):
    st.markdown(
        f'<section class="result-card"><div class="ocean-kicker">DEIN ERGEBNIS</div>'
        f'<div class="result-layout"><div class="result-net">{net_svg()}<div class="result-type"><strong>{TYPE_LABELS[prediction]}</strong><small>({prediction})</small></div></div>'
        f'<div class="result-fish">{fish_svg(prediction)}<div>{FISH_NAMES[prediction]}</div></div></div>'
        f'<p class="result-description">{TYPE_DESCRIPTIONS[prediction]}</p>'
        '</section>',
        unsafe_allow_html=True,
    )


def show_type_information(personality_type):
    st.markdown(
        f'<section class="type-info">{fish_svg(personality_type)}'
        f'<h3>{TYPE_LABELS[personality_type]}</h3><small>({personality_type})</small>'
        f'<p>{TYPE_DETAILS[personality_type]}</p></section>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_pipeline(model_path):
    """Die bereits trainierte Pipeline einmal laden und wiederverwenden."""
    return joblib.load(model_path)


def make_input_frame(answers, age, gender, hand, feature_order):
    """Alle Rohwerte mit den Spaltennamen und der Reihenfolge des Trainings."""
    values = dict(answers)
    values["age"] = int(age)
    values["gender"] = GENDER_VALUES[gender]
    values["hand"] = HAND_VALUES[hand]

    missing = set(feature_order) - set(values)
    extra = set(values) - set(feature_order)
    if missing or extra:
        raise ValueError("Fragebogen und Eingabespalten der Pipeline stimmen nicht überein.")

    input_df = pd.DataFrame([values])
    return input_df.loc[:, feature_order]


def main():
    st.set_page_config(page_title="OCEAN · Persönlichkeitsprofil", page_icon="🌊", layout="centered")
    apply_ocean_design()
    st.markdown('<div class="ocean-hero"><div class="ocean-kicker">OCEAN · PERSÖNLICHKEIT ENTDECKEN</div><h1>Tauche in dein<br>Persönlichkeitsprofil ein.</h1><p>19 Aussagen. Deine Perspektive. Entdecke, welchem der vier Persönlichkeitstypen das Modell deine Antworten zuordnet.</p></div>', unsafe_allow_html=True)
    type_cards = ''.join(f'<div class="type-mini">{fish_svg(name)}<strong>{TYPE_LABELS[name]}</strong><small>({name})</small></div>' for name in TYPE_LABELS)
    st.markdown(f'<div class="type-overview" aria-label="Die vier Persönlichkeitstypen">{type_cards}</div>', unsafe_allow_html=True)

    # Portabler Pfad: relativ zur app.py, unabhängig vom Terminal-Arbeitsordner.
    model_path = Path(__file__).resolve().parent / "models" / "best_pipeline.joblib"
    if not model_path.is_file():
        st.error("Die trainierte Pipeline fehlt. Führe zuerst das Modelling-Notebook aus und lege die erzeugte best_pipeline.joblib im Ordner models neben app.py ab.")
        st.stop()

    try:
        pipeline = load_pipeline(str(model_path))
    except Exception:
        st.error("Die Pipeline konnte nicht geladen werden. Prüfe die Modelldatei und verwende dieselben Bibliotheksversionen wie beim Training.")
        st.stop()

    with st.form("personality_form"):
        st.subheader("Deine Angaben")
        age_column, _ = st.columns([1, 3])
        with age_column:
            age_text = st.text_input("Alter in Jahren", value="30", max_chars=3)
        demographic_columns = st.columns(2)
        with demographic_columns[0]:
            gender = st.radio("Geschlecht", list(GENDER_VALUES), horizontal=True, key="gender_choice")
        with demographic_columns[1]:
            hand = st.radio("Schreibhand", ["Links", "Rechts"], horizontal=True, key="hand_choice")

        st.subheader("Wie sehr stimmst du den Aussagen zu?")
        st.markdown("**1** = Stimme nicht zu · **2** = Stimme eher nicht zu · **3** = Neutral · **4** = Stimme eher zu · **5** = Stimme zu")
        answers = {}
        question_number = 1
        for group_name, questions in QUESTION_GROUPS.items():
            st.markdown(f"### {group_name}")
            for feature, statement in questions.items():
                answers[feature] = st.select_slider(
                    f"{question_number}. {statement}",
                    options=[1, 2, 3, 4, 5], value=3, key=feature,
                )
                st.markdown('<div class="answer-scale" aria-hidden="true"><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span></div>', unsafe_allow_html=True)
                question_number += 1

        submitted = st.form_submit_button("Persönlichkeitstyp vorhersagen", type="primary")

    if submitted:
        if not age_text.strip().isascii() or not age_text.strip().isdecimal() or not 1 <= int(age_text.strip()) <= 120:
            st.error("Bitte gib dein Alter als ganze Zahl zwischen 1 und 120 ein.")
            return
        age = int(age_text.strip())
        try:
            with st.spinner("Dein Profil wird berechnet …"):
                input_df = make_input_frame(
                    answers, age, gender, hand, pipeline.feature_names_in_.tolist()
                )
                prediction = pipeline.predict(input_df)[0]
            if prediction not in TYPE_LABELS:
                raise ValueError("Unbekannte Zielklasse.")
        except Exception:
            st.error("Die Vorhersage konnte nicht erstellt werden. Prüfe, ob die gespeicherte Pipeline zum verwendeten Fragebogen passt.")
        else:
            # Das Ergebnis bleibt erhalten, wenn ein Infofeld angeklickt wird.
            st.session_state["prediction"] = prediction
            st.session_state["show_own_type"] = False
            st.session_state["show_other_types"] = False

    prediction = st.session_state.get("prediction")
    if prediction is not None:
        show_ocean_result(prediction)
        own_column, other_column = st.columns(2)
        with own_column:
            if st.button("Möchtest du mehr über deinen Persönlichkeitstyp erfahren?", key="own_type", width="stretch"):
                st.session_state["show_own_type"] = not st.session_state.get("show_own_type", False)
        with other_column:
            if st.button("Andere Typen entdecken", key="other_types", width="stretch"):
                st.session_state["show_other_types"] = not st.session_state.get("show_other_types", False)
        if st.session_state.get("show_own_type"):
            show_type_information(prediction)
        if st.session_state.get("show_other_types"):
            for other_type in TYPE_LABELS:
                if other_type != prediction:
                    show_type_information(other_type)


if __name__ == "__main__":
    main()
