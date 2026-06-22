
import streamlit as st
import pandas as pd
import docx
import fitz
import pytesseract
from PIL import Image
import nltk
import string
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from skills_data import SKILL_LIBRARY, ROLE_SKILL_MAPPING, SKILL_RESOURCES
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title='PulseMatch AI', layout='wide', initial_sidebar_state='expanded')

# Custom CSS
st.markdown("""
<style>
    .skill-tag { display: inline-block; padding: 5px 12px; margin: 4px; border-radius: 20px; font-weight: 500; font-size: 14px; }
    .match-tag { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .miss-tag { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nltk():
    for pkg in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        nltk.download(pkg)
load_nltk()

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return ' '.join([lemmatizer.lemmatize(t.translate(str.maketrans('', '', string.punctuation))) for t in tokens if t.isalpha() and t not in stop_words])

def extract_skills(text, skills_list):
    text_low = text.lower()
    text_tokens = set(word_tokenize(text_low))
    return list(set([s for s in skills_list if (s in text_tokens if ' ' not in s else s in text_low)]))

def predict_qualification(found_skills):
    if not os.path.exists('trained_model.pkl'): return 'Unknown'
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Create input DF ensuring column order matches model
    input_df = pd.DataFrame([{f: (1 if f in found_skills else 0) for f in model.feature_names_in_}])
    prediction = model.predict(input_df)[0]
    return 'Qualified ✅' if prediction == 1 else 'Needs More Skills ❌'

# Sidebar
st.sidebar.title("⚙️ Settings")
selected_role = st.sidebar.selectbox('Target Role', list(ROLE_SKILL_MAPPING.keys()))
uploaded_file = st.sidebar.file_uploader('Upload Resume', type=['docx', 'pdf', 'png', 'jpg'])
analyze_btn = st.sidebar.button('🚀 Start Analysis', use_container_width=True)

# Main
st.title('🚀 PulseMatch AI')

if uploaded_file and analyze_btn:
    def get_text(file):
        ext = file.name.split('.')[-1].lower()
        if ext == 'docx': return '\n'.join([p.text for p in docx.Document(file).paragraphs])
        if ext == 'pdf':
            with fitz.open(stream=file.read(), filetype='pdf') as doc: return ''.join([p.get_text() for p in doc])
        return pytesseract.image_to_string(Image.open(file))

    raw_text = get_text(uploaded_file)
    clean_text = preprocess_text(raw_text)
    found = set(extract_skills(clean_text, SKILL_LIBRARY))
    target = set(ROLE_SKILL_MAPPING[selected_role])
    matched, missing = target.intersection(found), target - found
    score = (len(matched) / len(target) * 100) if target else 0
    verdict = predict_qualification(found)

    st.subheader("📊 Analysis Dashboard")
    m1, m2, m3 = st.columns(3)
    m1.metric("Match Score", f"{score:.1f}%")
    m2.metric("AI Verdict", verdict)
    m3.metric("Detected Skills", len(found))

    c1, c2 = st.columns([1.2, 2])
    with c1:
        fig = go.Figure(go.Indicator(mode='gauge+number', value=score, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2e7d32"}}))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"### Skill Breakdown for **{selected_role}**")
        if matched:
            st.markdown("".join([f'<span class="skill-tag match-tag">✅ {s.title()}</span>' for s in sorted(matched)]), unsafe_allow_html=True)
        if missing:
            st.markdown("".join([f'<span class="skill-tag miss-tag">❌ {s.title()}</span>' for s in sorted(missing)]), unsafe_allow_html=True)

    st.divider()
    st.subheader("🎯 Personalized Learning Roadmap")
    if missing:
        for s in sorted(missing):
            with st.expander(f"🖳 Master {s.title()}"):
                res = SKILL_RESOURCES.get(s, {})
                st.markdown(f"📺 [YouTube]({res.get('youtube', '#')}) | 🎓 [Coursera]({res.get('coursera', '#')}) | 💻 [Udemy]({res.get('udemy', '#')})")
    else: st.success("Perfect match!")
else: st.info("Please upload a resume and click Start Analysis.")
