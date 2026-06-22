
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
from nltk.tokenize import word_tokenize
from skills_data import SKILL_LIBRARY, ROLE_SKILL_MAPPING, LEARNING_ROADMAPS
import plotly.graph_objects as go

@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
load_nltk()

def extract_text_from_any(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    text = ""
    if file_ext == 'docx':
        doc = docx.Document(uploaded_file)
        text = '\n'.join([p.text for p in doc.paragraphs])
    elif file_ext == 'pdf':
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc: text += page.get_text()
    elif file_ext in ['jpg', 'jpeg', 'png']:
        text = pytesseract.image_to_string(Image.open(uploaded_file))
    return text

def extract_skills(text, skills_list):
    text_low = text.lower()
    return [s for s in skills_list if s.lower() in text_low]

st.set_page_config(page_title="ResumePulse AI", layout="wide")
st.title("🚀 ResumePulse AI: Advanced Resume Intelligence")

selected_role = st.selectbox("Target Specialization:", list(ROLE_SKILL_MAPPING.keys()))
target_skills = ROLE_SKILL_MAPPING[selected_role]
uploaded_file = st.file_uploader("Upload Resume", type=["docx", "pdf", "jpg", "png"])

if uploaded_file and st.button("🔍 Analyze Resume"):
    resume_text = extract_text_from_any(uploaded_file)
    if resume_text.strip():
        found_skills = set(extract_skills(resume_text, SKILL_LIBRARY))
        matched = set(target_skills).intersection(found_skills)
        missing = set(target_skills) - found_skills
        score = (len(matched) / len(target_skills) * 100) if target_skills else 0

        st.divider()
        st.subheader("📊 Analysis Results")
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(mode='gauge+number', value=score, title={'text': 'Match Score'}))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write(f"**Matched:** {', '.join(matched) if matched else 'None'}")
            st.error(f"**Missing:** {', '.join(missing) if missing else 'None'}")

        st.subheader("🎯 Detailed Professional Roadmap")
        if missing:
            for s in missing:
                role_roadmaps = LEARNING_ROADMAPS.get(selected_role, {})
                detail = role_roadmaps.get(s, f"Master {s.title()} by building hands-on projects and studying advanced implementation patterns.")
                st.info(f"**{s.upper()}**: {detail}")
        else:
            st.success("You are a subject matter expert for this role!")
