
import streamlit as st
import pandas as pd
import docx
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from skills_data import SKILL_LIBRARY, ROLE_SKILL_MAPPING
import plotly.graph_objects as go
from ml_logic import predict_qualification

# Setup NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed = []
    for t in tokens:
        t = t.translate(str.maketrans('', '', string.punctuation))
        if t.isalpha() and t not in stop_words:
            processed.append(lemmatizer.lemmatize(t))
    return ' '.join(processed)

def extract_skills(text, skills_list):
    return [s for s in skills_list if s in text.lower()]

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("🚀 AI Resume Analyzer & Skill Gap Detector")

with st.sidebar:
    st.header("Target Settings")
    selected_role = st.selectbox("Choose Target Role", list(ROLE_SKILL_MAPPING.keys()))
    target_skills = ROLE_SKILL_MAPPING[selected_role]

uploaded_file = st.file_uploader("Upload Resume (DOCX)", type="docx")

if uploaded_file:
    doc = docx.Document(uploaded_file)
    resume_text = "\n".join([p.text for p in doc.paragraphs])
    clean_resume = preprocess_text(resume_text)
    found_skills = set(extract_skills(clean_resume, SKILL_LIBRARY))
    required_skills = set(target_skills)
    
    matched = required_skills.intersection(found_skills)
    missing = required_skills - found_skills
    score = (len(matched) / len(required_skills) * 100) if required_skills else 0

    # Visuals
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': "Role Match Score"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "green" if score > 70 else "orange"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Analysis for {selected_role}")
        st.write("**Matched Skills:**")
        st.write(", ".join([f"✅ {s}" for s in matched]) if matched else "No specific matches found.")
        
        st.write("**Missing Skills:**")
        st.error(", ".join(missing) if missing else "You have all required skills!")

    st.divider()
    st.subheader("Personalized Learning Roadmap")
    for skill in missing:
        st.sidebar.markdown(f'**ML Prediction:** {predict_qualification(found_skills, SKILL_LIBRARY)}')
st.info(f"📖 Recommendation: Take a course on **{skill.title()}** to improve your suitability for this role.")
