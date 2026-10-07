from __future__ import annotations

UI = {
    "de": {"title":"Dein Persönlichkeits-Ozean", "subtitle":"Entdecke Muster in deinen Antworten – ruhig, klar und ohne Schubladendenken.", "test_type":"Test wählen", "short":"Kurztest · 19 Fragen", "long":"OCEAN-Test · 50 Fragen", "questions":"Fragen", "answer":"Antwort", "scale_note":"1 = trifft gar nicht zu · 5 = trifft völlig zu", "evaluate":"Auswertung anzeigen", "your_type":"Dein Ergebnis", "ocean_profile":"Dein OCEAN-Profil", "ocean_intro":"Die fünf Dimensionen bilden ein Profil, keine Diagnose.", "probability_note":"Die Prozentwerte zeigen die Modell-Sicherheit zwischen den vier Klassen, nicht Anteile deiner Persönlichkeit.", "ocean_note":"Die Werte sind Antwort-Scores (0–100) und keine klinische Beurteilung.", "model_missing":"Das Kurztest-Modell ist noch nicht eingebunden.", "ocean_model_missing":"Das OCEAN-Modell ist noch nicht eingebunden.", "model_result":"Modellvorhersage", "profile_title":"Ein paar Angaben vorab", "profile_text":"Alter, Geschlecht und Händigkeit werden von deinem Modell als Eingaben verwendet.", "age":"Alter", "gender":"Geschlecht", "female":"Weiblich", "male":"Männlich", "other":"Andere Angabe", "hand":"Händigkeit", "right":"Rechts", "left":"Links", "both":"Beidhändig", "continue":"Weiter zum Test", "choose_test":"Welchen Test möchtest du machen?", "choose_test_text":"Der Kurztest zeigt eine der vier gelernten Profilklassen. Der OCEAN-Test beschreibt fünf Persönlichkeitsdimensionen.", "short_kicker":"ca. 3 Minuten", "long_kicker":"ca. 7–10 Minuten", "short_detail":"19 Fragen · vier Persönlichkeitsprofile", "long_detail":"50 Fragen · fünf OCEAN-Dimensionen", "start_short":"Kurztest starten", "start_long":"OCEAN-Test starten", "back_home":"Zur Startseite", "question":"Frage", "of":"von", "progress_text":"{done} von {total} Fragen beantwortet", "previous":"Zurück", "next":"Weiter", "show_result":"Ergebnis anzeigen"},
    "en": {"title":"Your Personality Ocean", "subtitle":"Explore patterns in your answers – calmly, clearly, and without putting you in a box.", "test_type":"Choose a test", "short":"Short test · 19 questions", "long":"OCEAN test · 50 questions", "questions":"questions", "answer":"Answer", "scale_note":"1 = strongly disagree · 5 = strongly agree", "evaluate":"Show results", "your_type":"Your result", "ocean_profile":"Your OCEAN profile", "ocean_intro":"The five dimensions form a profile, not a diagnosis.", "probability_note":"Percentages show the model's confidence across the four classes, not shares of your personality.", "ocean_note":"Scores reflect answers (0–100) and are not a clinical assessment.", "model_missing":"The short-test model has not been added yet. Put the exported model at models/personality_19.cbm.", "choose_test":"Which test would you like to take?", "choose_test_text":"The short test assigns one of four learned profile classes. The OCEAN test describes five personality dimensions.", "short_kicker":"about 3 minutes", "long_kicker":"about 7–10 minutes", "short_detail":"19 questions · four personality profiles", "long_detail":"50 questions · five OCEAN dimensions", "start_short":"Start short test", "start_long":"Start OCEAN test", "back_home":"Back to home", "question":"Question", "of":"of", "progress_text":"{done} of {total} questions answered", "previous":"Back", "next":"Next", "show_result":"Show results"},
    "es": {"title":"Tu océano de personalidad", "subtitle":"Explora patrones en tus respuestas, con calma y sin encasillarte.", "test_type":"Elegir prueba", "short":"Prueba corta · 19 preguntas", "long":"Prueba OCEAN · 50 preguntas", "questions":"preguntas", "answer":"Respuesta", "scale_note":"1 = no me describe · 5 = me describe totalmente", "evaluate":"Ver resultados", "your_type":"Tu resultado", "ocean_profile":"Tu perfil OCEAN", "ocean_intro":"Las cinco dimensiones forman un perfil, no un diagnóstico.", "probability_note":"Los porcentajes muestran la confianza del modelo entre las cuatro clases, no partes de tu personalidad.", "ocean_note":"Los valores son puntuaciones de respuesta (0–100), no una evaluación clínica.", "model_missing":"El modelo de la prueba corta aún no está integrado. Guarda el modelo exportado en models/personality_19.cbm.", "choose_test":"¿Qué prueba quieres realizar?", "choose_test_text":"La prueba corta asigna una de cuatro clases de perfil aprendidas. La prueba OCEAN describe cinco dimensiones de personalidad.", "short_kicker":"unos 3 minutos", "long_kicker":"unos 7–10 minutos", "short_detail":"19 preguntas · cuatro perfiles", "long_detail":"50 preguntas · cinco dimensiones OCEAN", "start_short":"Iniciar prueba corta", "start_long":"Iniciar prueba OCEAN", "back_home":"Volver al inicio", "question":"Pregunta", "of":"de", "progress_text":"{done} de {total} preguntas respondidas", "previous":"Atrás", "next":"Siguiente", "show_result":"Ver resultados"},
}

UI["en"].update({"ocean_model_missing":"The OCEAN model has not been added yet.","model_result":"Model prediction","profile_title":"A few details first","profile_text":"Your model uses age, gender and handedness as inputs.","age":"Age","gender":"Gender","female":"Female","male":"Male","other":"Other","hand":"Handedness","right":"Right","left":"Left","both":"Both","continue":"Continue to test"})
UI["es"].update({"ocean_model_missing":"El modelo OCEAN aún no está integrado.","model_result":"Predicción del modelo","profile_title":"Algunos datos primero","profile_text":"El modelo usa la edad, el género y la lateralidad como entradas.","age":"Edad","gender":"Género","female":"Mujer","male":"Hombre","other":"Otra opción","hand":"Lateralidad","right":"Derecha","left":"Izquierda","both":"Ambidiestro/a","continue":"Continuar a la prueba"})

UI["de"].update({"profile_form_title":"Deine Angaben","left_hand":"Linke Hand","both_hands":"Beide","right_hand":"Rechte Hand"})
UI["en"].update({"profile_form_title":"Your details","left_hand":"Left hand","both_hands":"Both","right_hand":"Right hand"})
UI["es"].update({"profile_form_title":"Tus datos","left_hand":"Mano izquierda","both_hands":"Ambas","right_hand":"Mano derecha"})

LIKERT = {
    "de": {1:"Trifft gar nicht zu", 2:"Trifft eher nicht zu", 3:"Teils / teils", 4:"Trifft eher zu", 5:"Trifft völlig zu"},
    "en": {1:"Strongly disagree", 2:"Disagree", 3:"Neither", 4:"Agree", 5:"Strongly agree"},
    "es": {1:"Nada de acuerdo", 2:"Poco de acuerdo", 3:"Ni de acuerdo ni en desacuerdo", 4:"De acuerdo", 5:"Totalmente de acuerdo"},
}

ITEM_TEXT = {
"N1":("Ich gerate leicht unter Stress.","I get stressed out easily.","Me estreso con facilidad."),"N2":("Ich bin meistens entspannt.","I am relaxed most of the time.","Estoy relajado la mayor parte del tiempo."),"N3":("Ich mache mir über viele Dinge Sorgen.","I worry about things.","Me preocupo por las cosas."),"N4":("Ich fühle mich selten niedergeschlagen.","I seldom feel blue.","Rara vez me siento triste."),"N5":("Ich bin leicht aus der Ruhe zu bringen.","I am easily disturbed.","Me perturbo con facilidad."),"N6":("Ich rege mich schnell auf.","I get upset easily.","Me pongo molesto con facilidad."),"N7":("Meine Stimmung wechselt häufig.","I change my mood a lot.","Mi estado de ánimo cambia mucho."),"N8":("Ich habe häufige Stimmungsschwankungen.","I have frequent mood swings.","Tengo cambios de humor frecuentes."),"N9":("Ich bin schnell gereizt.","I get irritated easily.","Me irrita fácilmente."),"N10":("Ich fühle mich oft niedergeschlagen.","I often feel blue.","A menudo me siento triste."),
"E1":("Ich bin der Mittelpunkt auf Partys.","I am the life of the party.","Soy el alma de la fiesta."),"E2":("Ich rede nicht viel.","I don't talk a lot.","No hablo mucho."),"E3":("Ich fühle mich unter Menschen wohl.","I feel comfortable around people.","Me siento cómodo entre la gente."),"E4":("Ich halte mich im Hintergrund.","I keep in the background.","Me mantengo en un segundo plano."),"E5":("Ich fange Gespräche an.","I start conversations.","Inicio conversaciones."),"E6":("Ich habe wenig zu sagen.","I have little to say.","Tengo poco que decir."),"E7":("Ich spreche auf Partys mit vielen verschiedenen Leuten.","I talk to a lot of different people at parties.","Hablo con mucha gente diferente en las fiestas."),"E8":("Ich ziehe ungern die Aufmerksamkeit auf mich.","I don't like to draw attention to myself.","No me gusta llamar la atención."),"E9":("Es macht mir nichts aus, im Mittelpunkt zu stehen.","I don't mind being the center of attention.","No me importa ser el centro de atención."),"E10":("Ich bin in Gegenwart von Fremden ruhig.","I am quiet around strangers.","Soy callado con los desconocidos."),
"A1":("Ich kümmere mich wenig um andere.","I feel little concern for others.","Me importan poco los demás."),"A2":("Ich interessiere mich für Menschen.","I am interested in people.","Me interesan las personas."),"A3":("Ich beleidige Leute.","I insult people.","Insulto a la gente."),"A4":("Ich habe Mitgefühl für die Gefühle anderer.","I sympathize with others' feelings.","Sympatizo con los sentimientos de los demás."),"A5":("Die Probleme anderer interessieren mich nicht.","I am not interested in other people's problems.","No me interesan los problemas de los demás."),"A6":("Ich habe ein weiches Herz.","I have a soft heart.","Tengo buen corazón."),"A7":("Ich interessiere mich nicht wirklich für andere.","I am not really interested in others.","No me interesan realmente los demás."),"A8":("Ich nehme mir Zeit für andere.","I take time out for others.","Dedico tiempo a los demás."),"A9":("Ich nachempfinde die Emotionen anderer.","I feel others' emotions.","Siento las emociones de los demás."),"A10":("Ich sorge dafür, dass sich andere wohlfühlen.","I make people feel at ease.","Hago que la gente se sienta cómoda."),
"C1":("Ich bin immer vorbereitet.","I am always prepared.","Siempre estoy preparado."),"C2":("Ich lasse meine Sachen herumliegen.","I leave my belongings around.","Dejo mis pertenencias por ahí."),"C3":("Ich achte auf Details.","I pay attention to details.","Presto atención a los detalles."),"C4":("Ich mache ein Chaos aus Dingen.","I make a mess of things.","Hago un lío de las cosas."),"C5":("Ich erledige Aufgaben sofort.","I get chores done right away.","Hago las tareas de inmediato."),"C6":("Ich vergesse oft, Dinge an ihren Platz zurückzustellen.","I often forget to put things back in their proper place.","A menudo olvido guardar las cosas en su lugar."),"C7":("Ich mag Ordnung.","I like order.","Me gusta el orden."),"C8":("Ich drücke mich vor meinen Pflichten.","I shirk my duties.","Eludo mis deberes."),"C9":("Ich halte mich an einen Zeitplan.","I follow a schedule.","Sigo un horario."),"C10":("Ich arbeite sehr exakt.","I am exacting in my work.","Soy exigente en mi trabajo."),
"O1":("Ich habe einen großen Wortschatz.","I have a rich vocabulary.","Tengo un vocabulario rico."),"O2":("Es fällt mir schwer, abstrakte Ideen zu verstehen.","I have difficulty understanding abstract ideas.","Tengo dificultad para entender ideas abstractas."),"O3":("Ich habe eine blühende Fantasie.","I have a vivid imagination.","Tengo una imaginación vívida."),"O4":("Ich interessiere mich nicht für abstrakte Ideen.","I am not interested in abstract ideas.","No me interesan las ideas abstractas."),"O5":("Ich habe hervorragende Ideen.","I have excellent ideas.","Tengo excelentes ideas."),"O6":("Ich habe keine gute Fantasie.","I do not have a good imagination.","No tengo una buena imaginación."),"O7":("Ich verstehe Dinge schnell.","I am quick to understand things.","Entiendo las cosas rápidamente."),"O8":("Ich benutze schwierige Wörter.","I use difficult words.","Uso palabras difíciles."),"O9":("Ich verbringe Zeit damit, über Dinge nachzudenken.","I spend time reflecting on things.","Paso tiempo reflexionando sobre las cosas."),"O10":("Ich bin voller Ideen.","I am full of ideas.","Estoy lleno de ideas.")}

SHORT_IDS = ["N1","N2","N3","N4","N5","N6","N7","N8","N9","N10","E1","E3","E4","E5","E7","E9","E10","C4","A4"]
SHORT_QUESTIONS = [{"id":item, "text":dict(zip(("de","en","es"), ITEM_TEXT[item]))} for item in SHORT_IDS]

REVERSE_ITEMS = {"N2", "N4", "E2", "E4", "E6", "E8", "E10", "A1", "A3", "A5", "A7", "C2", "C4", "C6", "C8", "O2", "O4", "O6"}
LONG_QUESTIONS = [
    {"id": item, "trait": item[0], "reverse": item in REVERSE_ITEMS,
     "text": dict(zip(("de", "en", "es"), ITEM_TEXT[item]))}
    for trait in ("O", "C", "E", "A", "N")
    for item in [f"{trait}{number}" for number in range(1, 11)]
]

TRAIT_META = {
 "O":{"de":{"name":"Offenheit", "description":"Neugier, Ideen und neue Perspektiven."},"en":{"name":"Openness", "description":"Curiosity, ideas and new perspectives."},"es":{"name":"Apertura", "description":"Curiosidad, ideas y nuevas perspectivas."}},
 "C":{"de":{"name":"Gewissenhaftigkeit", "description":"Planung, Verlässlichkeit und Selbstorganisation."},"en":{"name":"Conscientiousness", "description":"Planning, reliability and self-organisation."},"es":{"name":"Responsabilidad", "description":"Planificación, fiabilidad y organización."}},
 "E":{"de":{"name":"Extraversion", "description":"Kontaktfreude und Aktivität."},"en":{"name":"Extraversion", "description":"Sociability and activity."},"es":{"name":"Extraversión", "description":"Sociabilidad y actividad."}},
 "A":{"de":{"name":"Verträglichkeit", "description":"Mitgefühl, Vertrauen und Rücksicht."},"en":{"name":"Agreeableness", "description":"Compassion, trust and consideration."},"es":{"name":"Amabilidad", "description":"Compasión, confianza y consideración."}},
 "N":{"de":{"name":"Neurotizismus", "description":"Emotionale Reaktivität und Anspannung."},"en":{"name":"Neuroticism", "description":"Emotional reactivity and tension."},"es":{"name":"Neuroticismo", "description":"Reactividad emocional y tensión."}},
}

def ocean_scores(answers, questions):
    groups={trait:[] for trait in TRAIT_META}
    for question in questions:
        value=answers[question['id']]
        groups[question['trait']].append(6-value if question['reverse'] else value)
    return {trait: (sum(values)/len(values)-1)*25 for trait,values in groups.items()}

def class_copy(label, lang):
    text = {
        "Moderate": {
            "de": {
                "tagline": "Ausgewogenes Profil",
                "description": "Deine Antworten liegen zwischen den vier gelernten Profilen überwiegend im mittleren Bereich.",
                "tendencies": ["ausgewogen", "anpassungsfähig", "situationsabhängig", "weniger extrem ausgeprägt"],
                "explanation": "Dieses Profil deutet darauf hin, dass keine der stärker ausgeprägten Tendenzen eindeutig dominiert. Je nach Situation kannst du eher ruhig oder aktiv, kontrolliert oder spontan reagieren. Das kann mit Flexibilität einhergehen, bedeutet aber nicht, dass du in jeder Eigenschaft genau im Durchschnitt liegst.",
            },
            "en": {
                "tagline": "Balanced profile",
                "description": "Your answers mostly fall in the middle range of the four learned profiles.",
                "tendencies": ["balanced", "adaptable", "situation-dependent", "less extreme"],
                "explanation": "This profile suggests that none of the more pronounced tendencies clearly dominates. Depending on the situation, you may respond more calmly or actively, more controlled or spontaneously. This can go along with flexibility, but it does not mean that every trait is exactly average.",
            },
            "es": {
                "tagline": "Perfil equilibrado",
                "description": "Tus respuestas se sitúan principalmente en el rango medio de los cuatro perfiles aprendidos.",
                "tendencies": ["equilibrado", "adaptable", "dependiente de la situación", "menos extremo"],
                "explanation": "Este perfil sugiere que ninguna de las tendencias más marcadas domina claramente. Según la situación, puedes reaccionar de forma más tranquila o activa, más controlada o espontánea. Esto puede asociarse con flexibilidad, pero no significa que todos tus rasgos sean exactamente promedio.",
            },
        },
        "Resilient": {
            "de": {
                "tagline": "Stabiles Profil",
                "description": "Deine Antworten zeigen in diesem Modell eine eher stabile, belastbare Tendenz.",
                "tendencies": ["emotional stabil", "belastbar", "ruhig", "zuversichtlich"],
                "explanation": "Dieses Profil ist mit einer vergleichsweise stabilen Reaktion auf Belastungen verbunden. Herausforderungen werden tendenziell ruhiger verarbeitet und negative Emotionen können weniger stark oder weniger dauerhaft auftreten. Auch resiliente Personen erleben natürlich Stress und schwierige Phasen.",
            },
            "en": {
                "tagline": "Stable profile",
                "description": "In this model, your answers indicate a relatively stable and resilient tendency.",
                "tendencies": ["emotionally stable", "resilient", "calm", "confident"],
                "explanation": "This profile is associated with a comparatively stable response to stress. Challenges may tend to be processed more calmly, while negative emotions may be less intense or less persistent. Resilient people still experience stress and difficult periods, of course.",
            },
            "es": {
                "tagline": "Perfil estable",
                "description": "En este modelo, tus respuestas indican una tendencia relativamente estable y resiliente.",
                "tendencies": ["emocionalmente estable", "resiliente", "tranquilo", "confiado"],
                "explanation": "Este perfil se asocia con una respuesta comparativamente estable ante el estrés. Los retos tienden a procesarse con más calma y las emociones negativas pueden ser menos intensas o menos persistentes. Las personas resilientes también experimentan estrés y etapas difíciles.",
            },
        },
        "Overcontroller": {
            "de": {
                "tagline": "Stark kontrolliertes Profil",
                "description": "Deine Antworten ähneln in diesem Modell eher einem kontrollierten, zurückhaltenden Muster.",
                "tendencies": ["selbstkontrolliert", "vorsichtig", "zurückhaltend", "strukturiert"],
                "explanation": "Dieses Profil deutet auf eine stärkere Kontrolle von Verhalten und Emotionen hin. Entscheidungen werden möglicherweise eher überlegt getroffen und Impulse stärker zurückgehalten. Das kann Struktur und Verlässlichkeit unterstützen, kann in manchen Situationen aber auch mit größerer Vorsicht oder Zurückhaltung verbunden sein.",
            },
            "en": {
                "tagline": "Highly controlled profile",
                "description": "Your answers are more similar to a controlled, restrained pattern in this model.",
                "tendencies": ["self-controlled", "cautious", "reserved", "structured"],
                "explanation": "This profile suggests stronger control over behaviour and emotions. Decisions may be made more deliberately and impulses may be held back more often. This can support structure and reliability, while in some situations it may also be associated with greater caution or restraint.",
            },
            "es": {
                "tagline": "Perfil muy controlado",
                "description": "Tus respuestas se parecen más a un patrón controlado y reservado en este modelo.",
                "tendencies": ["autocontrolado", "prudente", "reservado", "estructurado"],
                "explanation": "Este perfil sugiere un mayor control del comportamiento y de las emociones. Las decisiones pueden tomarse de forma más reflexiva y los impulsos pueden contenerse con mayor frecuencia. Esto puede favorecer la estructura y la fiabilidad, aunque en algunas situaciones también puede relacionarse con mayor prudencia o reserva.",
            },
        },
        "Undercontroller": {
            "de": {
                "tagline": "Weniger kontrolliertes Profil",
                "description": "Deine Antworten ähneln in diesem Modell eher einem spontaneren Muster.",
                "tendencies": ["spontan", "emotional reaktiv", "flexibel", "ausdrucksstark"],
                "explanation": "Dieses Profil deutet auf spontanere Reaktionen und eine geringere Zurückhaltung von Impulsen hin. Gefühle und Reaktionen können unmittelbarer zum Ausdruck kommen. Das kann mit Flexibilität und Ausdrucksstärke verbunden sein, in manchen Situationen aber auch mit stärkeren oder schnelleren emotionalen Reaktionen.",
            },
            "en": {
                "tagline": "Less controlled profile",
                "description": "Your answers are more similar to a more spontaneous pattern in this model.",
                "tendencies": ["spontaneous", "emotionally reactive", "flexible", "expressive"],
                "explanation": "This profile suggests more spontaneous reactions and less restraint of impulses. Feelings and reactions may be expressed more immediately. This can be associated with flexibility and expressiveness, while in some situations emotions may also arise more strongly or quickly.",
            },
            "es": {
                "tagline": "Perfil menos controlado",
                "description": "Tus respuestas se parecen más a un patrón más espontáneo en este modelo.",
                "tendencies": ["espontáneo", "emocionalmente reactivo", "flexible", "expresivo"],
                "explanation": "Este perfil sugiere reacciones más espontáneas y una menor contención de los impulsos. Los sentimientos y las reacciones pueden expresarse de forma más inmediata. Esto puede asociarse con flexibilidad y expresividad, aunque en algunas situaciones las emociones también pueden aparecer con mayor intensidad o rapidez.",
            },
        },
    }
    return text.get(label, text["Moderate"])[lang]

