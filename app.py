from pathlib import Path
import base64
import streamlit as st
import joblib
import pandas as pd
from content import LIKERT, LONG_QUESTIONS, SHORT_QUESTIONS, TRAIT_META, UI, class_copy, ocean_scores

ROOT=Path(__file__).parent; HERO_IMAGE=ROOT/'assets'/'coast-hero.png'
MODEL_PATHS={'short':ROOT/'models'/'best_model.joblib','long':ROOT/'models'/'best_ocean_model.joblib'}

@st.cache_resource
def model(test):
    path=MODEL_PATHS[test]
    return joblib.load(path) if path.exists() else None

def css():
    image = base64.b64encode(HERO_IMAGE.read_bytes()).decode()
    st.markdown(f'''<style>
    [data-testid="stHeader"]{{height:0;background:transparent}}[data-testid="stToolbar"], [data-testid="stDecoration"]{{display:none}}.stApp{{background:linear-gradient(90deg,#ffffffb8,#ffffff14),url("data:image/png;base64,{image}") center/cover fixed;color:#073b4c;min-height:100vh}}.block-container{{max-width:1390px;padding:3.1rem 4.5rem 7rem}}[data-testid="stSidebar"]{{display:none}}h1,h2,h3{{font-family:Georgia,serif;color:#073b4c}}
    .brand{{font-family:Georgia,serif;font-size:1.6rem;letter-spacing:.13em;color:#063b4c;margin:0}}.brand small{{display:block;font-family:Arial;font-size:.67rem;letter-spacing:.28em;margin-top:.25rem}}.nav-note{{text-align:right;font-size:1.05rem;color:#063b4c;padding-top:.6rem}}.language-switcher{{display:flex;justify-content:flex-end;align-items:center;gap:.55rem;margin-top:-.4rem}}.hero{{padding:3rem 1.2rem 1.3rem;color:#073b4c;margin-bottom:1.25rem}}.hero h1{{font-size:4rem;line-height:1.02;max-width:540px;margin:0}}.hero p{{font-size:1.25rem;line-height:1.5;max-width:520px;margin:1.4rem 0}}.card,.question{{background:#fffffff0;border:1px solid #dbe8e7;border-radius:18px;padding:1.6rem;margin:1rem 0;box-shadow:0 14px 35px #0a4d5e1f}}.test-card{{min-height:178px}}.question{{padding:2.2rem}}.question h2{{font-size:1.7rem;line-height:1.35;margin:0}}.eyebrow{{color:#087f91;font-weight:700;letter-spacing:.06em;text-transform:uppercase;font-size:.78rem}}.muted{{color:#315d70}}.scale{{display:flex;justify-content:space-between;color:#4e7379;font-size:.8rem;gap:1rem}}.trait{{background:#fffdf8e8;border-left:6px solid #10a8a9;border-radius:14px;padding:1rem 1.15rem;margin:.8rem 0}}.type-analysis{{background:#fffffff0;border:1px solid #dbe8e7;border-radius:18px;padding:1.5rem 1.6rem;margin:1.2rem 0;box-shadow:0 14px 35px #0a4d5e1f}}.type-analysis h3{{margin-top:0}}.probability-panel{{background:rgba(255,255,255,.95);border:1px solid #dbe8e7;border-radius:20px;padding:1.15rem 1.35rem;margin:1rem 0;box-shadow:0 14px 35px #0a4d5e1f}}.prob-row{{margin:.65rem 0 1rem}}.prob-head{{display:flex;justify-content:space-between;align-items:center;color:#073b4c;font-weight:750;margin-bottom:.38rem}}.prob-value{{font-size:1.05rem;color:#007d98}}.prob-track{{height:14px;background:#d9e4e7;border-radius:999px;overflow:hidden;box-shadow:inset 0 1px 2px #073b4c24}}.prob-fill{{height:100%;border-radius:999px;background:linear-gradient(90deg,#159bc0,#65d3e5)}}.prob-note{{color:#587178;font-size:.88rem;margin-top:.25rem}}.fish-progress{{position:relative;height:42px;margin:.15rem 0 .65rem}}.fish-track{{position:absolute;left:0;right:0;top:18px;height:8px;border-radius:999px;background:#252832;box-shadow:inset 0 1px 2px #0005}}.fish-fill{{position:absolute;left:0;top:18px;height:8px;border-radius:999px;background:linear-gradient(90deg,#35a8e0,#75d6e8)}}.fish-marker{{position:absolute;top:-1px;transform:translateX(-50%) scaleX(-1);font-size:28px;line-height:1;filter:drop-shadow(0 2px 2px #073b4c55)}}.profile-form-title{{text-align:center;margin:.15rem 0 .35rem;font-size:2rem}}.hand-icon{{text-align:center;font-size:2.7rem;line-height:1;margin:.2rem 0 .15rem;filter:grayscale(1)}}.hand-scale-labels{{display:flex;justify-content:space-between;gap:.5rem;color:#315d70;font-size:.82rem;font-weight:700;margin:-.15rem .15rem .15rem}}div[data-testid="stForm"]{{background:rgba(255,255,255,.94);border:1px solid #dbe8e7;border-radius:20px;padding:1.5rem 1.6rem 1.3rem;box-shadow:0 14px 35px #0a4d5e1f}}div[data-testid="stForm"] label{{color:#073b4c!important;font-weight:700}}div[data-testid="stForm"] [data-baseweb="input"],div[data-testid="stForm"] [data-baseweb="select"]>div{{background:#fff!important}}.tendency{{display:inline-block;background:#daf4f7;border:1px solid #a9e0e7;border-radius:999px;padding:.35rem .7rem;margin:.2rem .25rem .2rem 0;font-size:.9rem;font-weight:600}}.stRadio>div{{display:grid;gap:.45rem}}.stRadio label{{background:#fff;border-radius:12px;color:#073b4c!important;font-weight:600;border:1px solid #e0e7e8;padding:.7rem 1rem}}.stRadio label:has(input:checked){{background:#daf4f7;color:#073b4c!important;border-color:#a9e0e7}}div.stButton>button{{border-radius:999px;padding:.7rem 1.6rem;border:0;background:#00658e;color:#fff;font-weight:700}}div.stButton>button:hover{{background:#004f72;color:#fff}}@media(max-width:800px){{.block-container{{padding:2rem 1.2rem}}.hero h1{{font-size:2.8rem}}}}
.block-container {{
    padding-top: 0.75rem !important;
}}

    /* Readable text and inputs */
    [data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"]{{
        background:rgba(255,255,255,.94);border-radius:18px;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]{{
        background:rgba(255,255,255,.94)!important;
        border:1px solid #d7e5e9!important;border-radius:20px!important;
        box-shadow:0 12px 30px #063b4c20;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] label,
    [data-testid="stVerticalBlockBorderWrapper"] h3{{color:#073b4c!important}}
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"]>div{{
        background:#fff!important;color:#073b4c!important;
        border:1px solid #a9bec5!important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{{
        background:transparent!important;border:0!important;
        box-shadow:0 1px 5px #003b4b70;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"]>div>div{{
        color:#007da3!important;
    }}
    [data-testid="stButton"] button[kind="secondary"]{{
        background:#00658e!important;color:white!important;
        border:1px solid #005678!important;
    }}
    [data-testid="stButton"] button[kind="secondary"] p{{color:white!important}}
    
    /* Profile card */
    div[data-testid="stForm"]{{background:rgba(255,255,255,.96)!important;border:1px solid #dce8ec!important;border-radius:20px!important;padding:1.5rem!important;box-shadow:0 12px 30px #063b4c26!important}}
    div[data-testid="stForm"] label,div[data-testid="stForm"] p,div[data-testid="stForm"] h3{{color:#073b4c!important}}
    /* Hand slider */
    .st-key-hand [data-testid="stThumbValue"],
    .st-key-hand [data-testid="stSliderThumbValue"],
    .st-key-hand [data-testid="stTickBar"],
    .st-key-hand [data-testid="stSliderTickBar"],
    .st-key-hand [role="slider"] > div,
    .st-key-hand [role="slider"] > span {{
        display:none!important;
    }}
    .st-key-hand [role="slider"],
    .st-key-hand [data-baseweb="slider"] [role="slider"] {{
        position:relative!important;
        width:38px!important;height:38px!important;
        background:transparent!important;border:0!important;
        box-shadow:none!important;overflow:visible!important;
    }}
    .st-key-hand [role="slider"]::after,
    .st-key-hand div:has(> div > input[type="range"])::after {{
        content:"🖐"!important;position:absolute!important;
        left:50%;top:50%;transform:translate(-50%,-50%);
        font-size:29px;line-height:1;pointer-events:none;
        filter:drop-shadow(0 2px 4px rgba(0,0,0,.4));
    }}
    .st-key-hand [role="slider"]:focus-visible {{
        outline:2px solid #00658e!important;outline-offset:3px;
    }}
    /* Remove default number bubbles and tick labels */
    .st-key-hand [data-testid="stSlider"] [data-testid*="ThumbValue"],
    .st-key-hand [data-testid="stSlider"] [data-testid*="TickBar"] {{
        display:none!important;
    }}
    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button{{background:#00658e!important;color:white!important}}
    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button p{{color:white!important}}
</style>''',unsafe_allow_html=True)
def t(k): return UI[st.session_state.lang][k]
def go(test): st.session_state.update(test=test,step=0,answers={},page='profile')
def set_language(lang):
    st.session_state.lang=lang

def header():
    a,b=st.columns([3,2])
    with a:
        st.markdown("<p class='brand'>〰 OCEAN INSIGHTS<small>DISCOVER · UNDERSTAND · GROW</small></p>",unsafe_allow_html=True)
    with b:
        c1,c2,c3=st.columns(3)
        with c1:
            st.button("🇩🇪",key=f"lang_de_{st.session_state.page}",on_click=set_language,args=("de",),use_container_width=True)
        with c2:
            st.button("🇬🇧",key=f"lang_en_{st.session_state.page}",on_click=set_language,args=("en",),use_container_width=True)
        with c3:
            st.button("🇪🇸",key=f"lang_es_{st.session_state.page}",on_click=set_language,args=("es",),use_container_width=True)

def welcome():
    header(); left,right=st.columns([1.05,0.9],gap='large')
    with left:
        st.markdown(f"<section class='hero'><h1>{t('title').replace('Dein Persönlichkeits-Ozean','Entdecke<br>deinen<br>Persönlichkeitstyp')}</h1><p>{t('subtitle')}</p></section>",unsafe_allow_html=True)
        a,b=st.columns(2)
        with a:
            st.markdown(f"<div class='card test-card'><p class='eyebrow'>{t('short_kicker')}</p><h3>〰 {t('short')}</h3><p class='muted'>{t('short_detail')}</p></div>",unsafe_allow_html=True)
            st.button(t('start_short')+'  →',on_click=lambda:go('short'),use_container_width=True)
        with b:
            st.markdown(f"<div class='card test-card'><p class='eyebrow'>{t('long_kicker')}</p><h3>◌ {t('long')}</h3><p class='muted'>{t('long_detail')}</p></div>",unsafe_allow_html=True)
            st.button(t('start_long')+'  →',on_click=lambda:go('long'),use_container_width=True)
    with right:
        st.markdown(f"<div class='card'><p class='eyebrow'>{t('choose_test')}</p><h2>{t('choose_test_text')}</h2><p class='muted'>〰 〰 〰</p></div>",unsafe_allow_html=True)

def profile():
    header(); left,right=st.columns([1.05,0.9],gap='large')
    with left:
        st.markdown(f"<section class='hero'><h1>{t('profile_title')}</h1><p>{t('profile_text')}</p></section>",unsafe_allow_html=True)
        st.button('← '+t('back_home'),on_click=lambda:st.session_state.update(page='welcome'))
    with right:
        with st.form('profile_form',enter_to_submit=False):
            st.markdown(f"<h3 class='profile-form-title'>{t('profile_form_title')}</h3>",unsafe_allow_html=True)
            age=st.number_input(t('age'),min_value=1,max_value=100,value=25,step=1,key='profile_age')
            gender_options=['Female','Male','Other']
            gender_label=st.selectbox(t('gender'),gender_options,format_func=lambda x: {'Female':t('female'),'Male':t('male'),'Other':t('other')}[x],key='profile_gender')

            st.markdown(f"<div style='font-weight:700;color:#073b4c;margin-top:.45rem'>{t('hand')}</div><div class='hand-scale-labels'><span>{t('left_hand')}</span><span>{t('both_hands')}</span><span>{t('right_hand')}</span></div>",unsafe_allow_html=True)
            hand_value=st.slider(
                t('hand'),
                min_value=0,
                max_value=2,
                value=2,
                step=1,
                label_visibility='collapsed',
                key='hand'
            )
            hand_map={0:'Left',1:'Both',2:'Right'}

            if st.form_submit_button(t('continue'),use_container_width=True):
                st.session_state.demographics={'age':age,'gender':gender_label,'hand':hand_map[hand_value]}
                st.session_state.page='test'; st.rerun()

def feature_frame(test, answers):
    row={**answers, **st.session_state.demographics}
    if test=='long':
        scores=ocean_scores(answers,LONG_QUESTIONS)
        row.update({f'{trait}_score':value for trait,value in scores.items()})
    return pd.DataFrame([row])

def test():
    qs=SHORT_QUESTIONS if st.session_state.test=='short' else LONG_QUESTIONS; i=st.session_state.step; q=qs[i]
    header(); left,right=st.columns([1.05,0.9],gap='large')
    with left:
        st.markdown(f"<section class='hero'><h1>{t('title').replace('Dein Persönlichkeits-Ozean','Entdecke<br>deinen<br>Persönlichkeitstyp')}</h1><p>{t('subtitle')}</p></section>",unsafe_allow_html=True)
        st.button('← '+t('back_home'),on_click=lambda:st.session_state.update(page='welcome'))
    with right:
        progress_pct=((i+1)/len(qs))*100
        fish_pos=max(3,min(97,progress_pct))
        st.markdown(f"<div class='card'><p class='muted'>{t('question')} {i+1} {t('of')} {len(qs)} · {(i+1)/len(qs):.0%}</p>",unsafe_allow_html=True)
        st.markdown(
            f"<div class='fish-progress'><div class='fish-track'></div>"
            f"<div class='fish-fill' style='width:{progress_pct:.2f}%'></div>"
            f"<div class='fish-marker' style='left:{fish_pos:.2f}%'>🐟</div></div>",
            unsafe_allow_html=True
        )
        st.markdown(f"<section class='question'><h2>{q['text'][st.session_state.lang]}</h2></section>",unsafe_allow_html=True)
        old=st.session_state.answers.get(q['id'])
        opts=[1,2,3,4,5]
        answer=st.radio(t('answer'),opts,index=opts.index(old) if old is not None else None,format_func=lambda v:f"{v}   {LIKERT[st.session_state.lang][v]}",label_visibility='collapsed',key=f"answer_{st.session_state.test}_{q['id']}_{st.session_state.lang}")
        a,b=st.columns(2)
        with a:
            if i and st.button('← '+t('previous'),use_container_width=True): st.session_state.step-=1;st.rerun()
        with b:
            if st.button(t('show_result') if i==len(qs)-1 else t('next')+' →',disabled=answer is None,use_container_width=True):
                st.session_state.answers[q['id']]=answer
                if i==len(qs)-1: st.session_state.page='result'
                else: st.session_state.step+=1
                st.rerun()

def result():
    st.button('← '+t('back_home'),on_click=lambda:st.session_state.update(page='welcome'))
    if st.session_state.test=='short':
        m=model('short')
        if not m: st.warning(t('model_missing'));return
        ranked=sorted(zip(m.classes_,m.predict_proba(feature_frame('short',st.session_state.answers))[0]),key=lambda x:x[1],reverse=True); win=ranked[0][0]; cp=class_copy(win,st.session_state.lang)
        st.markdown(f"<section class='hero'><p class='eyebrow'>{t('your_type')}</p><h1>{win}</h1><p>{cp['tagline']}</p></section>",unsafe_allow_html=True)
        analysis_title={'de':'Tendenzen & Einordnung','en':'Tendencies & explanation','es':'Tendencias y explicación'}[st.session_state.lang]
        tendency_title={'de':'Typische Tendenzen','en':'Typical tendencies','es':'Tendencias típicas'}[st.session_state.lang]
        note={'de':'Diese Einordnung beschreibt eine statistische Tendenz auf Basis deiner Antworten und ist keine psychologische Diagnose.','en':'This interpretation describes a statistical tendency based on your answers and is not a psychological diagnosis.','es':'Esta interpretación describe una tendencia estadística basada en tus respuestas y no es un diagnóstico psicológico.'}[st.session_state.lang]
        chips=''.join(f"<span class='tendency'>{x}</span>" for x in cp['tendencies'])
        st.markdown(f"<div class='type-analysis'><p style='margin-top:0;margin-bottom:1.35rem'>{cp['description']}</p><h3>{analysis_title}</h3><p><b>{tendency_title}</b></p><div>{chips}</div><p style='margin-top:1rem'>{cp['explanation']}</p><p class='muted' style='font-size:.88rem;margin-bottom:0'>{note}</p></div>",unsafe_allow_html=True)
        prob_rows=''.join(
            f"<div class='prob-row'><div class='prob-head'><span>{n}</span><span class='prob-value'>{p:.0%}</span></div>"
            f"<div class='prob-track'><div class='prob-fill' style='width:{float(p)*100:.2f}%'></div></div></div>"
            for n,p in ranked
        )
        st.markdown(f"<div class='probability-panel'>{prob_rows}<div class='prob-note'>{t('probability_note')}</div></div>",unsafe_allow_html=True)
    else:
        m=model('long')
        if not m: st.warning(t('ocean_model_missing'));return
        ranked=sorted(zip(m.classes_,m.predict_proba(feature_frame('long',st.session_state.answers))[0]),key=lambda x:x[1],reverse=True)
        st.markdown(f"<section class='hero'><p class='eyebrow'>{t('your_type')}</p><h1>{t('ocean_profile')}</h1><p>{t('ocean_intro')}</p></section>",unsafe_allow_html=True)
        for tr,score in ocean_scores(st.session_state.answers,LONG_QUESTIONS).items():
            meta=TRAIT_META[tr][st.session_state.lang];st.markdown(f"<div class='trait'><b>{meta['name']} · {score:.0f}/100</b><br><span class='muted'>{meta['description']}</span></div>",unsafe_allow_html=True);st.progress(score/100)
        st.markdown(f"<h3>{t('model_result')}</h3>",unsafe_allow_html=True)
        prob_rows=''.join(
            f"<div class='prob-row'><div class='prob-head'><span>{name}</span><span class='prob-value'>{probability:.0%}</span></div>"
            f"<div class='prob-track'><div class='prob-fill' style='width:{float(probability)*100:.2f}%'></div></div></div>"
            for name,probability in ranked
        )
        st.markdown(f"<div class='probability-panel'>{prob_rows}</div>",unsafe_allow_html=True)
        if ranked:
            cp=class_copy(ranked[0][0],st.session_state.lang)
            analysis_title={'de':'Tendenzen des Modell-Typs','en':'Model-type tendencies','es':'Tendencias del tipo del modelo'}[st.session_state.lang]
            tendency_title={'de':'Typische Tendenzen','en':'Typical tendencies','es':'Tendencias típicas'}[st.session_state.lang]
            note={'de':'Diese Einordnung beschreibt eine statistische Tendenz auf Basis deiner Antworten und ist keine psychologische Diagnose.','en':'This interpretation describes a statistical tendency based on your answers and is not a psychological diagnosis.','es':'Esta interpretación describe una tendencia estadística basada en tus respuestas y no es un diagnóstico psicológico.'}[st.session_state.lang]
            chips=''.join(f"<span class='tendency'>{x}</span>" for x in cp['tendencies'])
            st.markdown(f"<div class='type-analysis'><h3>{analysis_title}: {ranked[0][0]}</h3><p><b>{cp['tagline']}</b></p><p><b>{tendency_title}</b></p><div>{chips}</div><p style='margin-top:1rem'>{cp['explanation']}</p><p class='muted' style='font-size:.88rem;margin-bottom:0'>{note}</p></div>",unsafe_allow_html=True)
        st.caption(t('ocean_note'))

st.set_page_config(page_title='Ocean Personality',page_icon='🌊',layout='centered')
for k,v in {'lang':'de','page':'welcome','test':'short','step':0,'answers':{},'demographics':{}}.items():st.session_state.setdefault(k,v)
css();{'welcome':welcome,'profile':profile,'test':test,'result':result}[st.session_state.page]()
