
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

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    .skill-tag { display: inline-block; padding: 5px 12px; margin: 4px; border-radius: 20px; font-weight: 500; font-size: 14px; }
    .match-tag { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .miss-tag { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .roadmap-card { 
        background-color: #f1f3f5; 
        border-radius: 10px; 
        padding: 20px; 
        margin-bottom: 20px; 
        border-left: 5px solid #d32f2f;
    }
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
    return [s for s in skills_list if (s.lower() in text_low)]

def predict_qualification(found_skills):
    if not found_skills: return 'Needs More Skills'
    if not os.path.exists('trained_model.pkl'): return 'Unknown'
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    input_df = pd.DataFrame([{f: (1 if f in found_skills else 0) for f in model.feature_names_in_}])
    prediction = model.predict(input_df)[0]
    return 'Qualified' if prediction == 1 else 'Needs More Skills'

# Sidebar
st.sidebar.title("Settings")
selected_role = st.sidebar.selectbox('Target Role', list(ROLE_SKILL_MAPPING.keys()))
uploaded_file = st.sidebar.file_uploader('Upload Resume', type=['docx', 'pdf', 'png', 'jpg'])
analyze_btn = st.sidebar.button('Start Analysis', use_container_width=True)

st.title('PulseMatch AI: Resume Intelligence')

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

    st.subheader("Analysis Dashboard")
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
            st.markdown("".join([f'<span class="skill-tag match-tag">MATCH: {s.title()}</span>' for s in sorted(matched)]), unsafe_allow_html=True)
        if missing:
            st.markdown("".join([f'<span class="skill-tag miss-tag">MISSING: {s.title()}</span>' for s in sorted(missing)]), unsafe_allow_html=True)

    st.divider()
    st.subheader("🎯 Personalized Learning Roadmap")
    if missing:
        st.write(f"To reach 100% alignment for the **{selected_role}** role, prioritize learning these {len(missing)} skills:")
        for s in sorted(missing):
            res = SKILL_RESOURCES.get(s, {})
            with st.expander(f"📖 Learning Path for {s.upper()}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.link_button("📺 YouTube", res.get('youtube', '#'), use_container_width=True)
                col2.link_button("🎓 Coursera", res.get('coursera', '#'), use_container_width=True)
                col3.link_button("💻 Udemy", res.get('udemy', '#'), use_container_width=True)
                col4.link_button("📝 Documentation", res.get('documentation', '#'), use_container_width=True)
    else: 
        st.success("Your profile is a perfect match for this role! No skill gaps detected.")
else: 
    st.info("Please upload a resume in the sidebar and click 'Start Analysis'.")
