
import streamlit as st
import pandas as pd
import docx
import nltk
import string
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from skills_data import SKILL_LIBRARY, ROLE_SKILL_MAPPING
import plotly.graph_objects as go

# Setup NLTK
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

load_nltk()

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
    text_low = text.lower()
    return [s for s in skills_list if s.lower() in text_low]

def predict_qualification(skills_found, all_features):
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl') if '__file__' in locals() else 'trained_model.pkl'
    if not os.path.exists(model_path):
        return "Model file missing (Please upload trained_model.pkl)"
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        input_dict = {feature: (1 if feature in skills_found else 0) for feature in all_features}
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)
        return "Qualified" if prediction[0] == 1 else "Needs More Skills"
    except Exception as e:
        return f"ML Error: {str(e)}"

# UI Layout
st.set_page_config(page_title="PulseMatch AI", layout="wide")
st.title("🚀 PulseMatch AI: Resume Analyzer & Skill Gap Detector")

# Role Selection
st.subheader("1. Select Your Target Specialization")
selected_role = st.selectbox("Choose role:", list(ROLE_SKILL_MAPPING.keys()))
target_skills = ROLE_SKILL_MAPPING[selected_role]

# File Upload
st.subheader("2. Upload Resume")
uploaded_file = st.file_uploader("Select DOCX file", type="docx")

if uploaded_file:
    try:
        doc = docx.Document(uploaded_file)
        resume_text = '\n'.join([p.text for p in doc.paragraphs])

        if resume_text.strip():
            clean_resume = preprocess_text(resume_text)
            found_skills = set(extract_skills(clean_resume, SKILL_LIBRARY))

            req_skills_set = set(target_skills)
            matched = req_skills_set.intersection(found_skills)
            missing = req_skills_set - found_skills
            score = (len(matched) / len(req_skills_set) * 100) if req_skills_set else 0

            # --- ML Results (Moved from Sidebar to Main Area) ---
            verdict = predict_qualification(found_skills, SKILL_LIBRARY)
            st.divider()
            st.subheader("📊 Analysis Results")
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.success(f"**Machine Learning Verdict:** {verdict}")
            with v_col2:
                st.info(f"**Skills Detected in Resume:** {len(found_skills)}")

            # Main Display
            c1, c2 = st.columns([1, 2])
            with c1:
                fig = go.Figure(go.Indicator(
                    mode='gauge+number', value=score,
                    title={'text': f'{selected_role} Match %'},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': 'green' if score > 70 else 'orange'}}
                ))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.write("**Matched Skills:** " + (", ".join(matched) if matched else "None"))
                st.write("**Missing Skills for this Role:**")
                st.error(", ".join(missing) if missing else "None!")

            st.divider()
            st.subheader("Personalized Learning Recommendations")
            if missing:
                for s in missing:
                    st.info(f"💡 Study **{s.title()}** to improve your standing for this role.")
            else:
                st.success("Perfect fit! You possess all the core skills defined for this specialization.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
