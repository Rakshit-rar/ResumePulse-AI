
import streamlit as st
import pandas as pd
import docx
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from skills_data import SKILL_LIBRARY, ROLE_SKILL_MAPPING
import plotly.graph_objects as go

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

def predict_qualification(skills_found, all_features):
    try:
        with open('trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        input_data = pd.DataFrame([{feature: (1 if feature in skills_found else 0) for feature in all_features}])
        prediction = model.predict(input_data)
        return "Qualified" if prediction[0] == 1 else "Needs More Skills"
    except:
        return "Model not found"

# Streamlit UI
st.set_page_config(page_title="PulseMatch AI", layout="wide")
st.title("🚀 PulseMatch AI: Resume Analyzer & Skill Gap Detector")

# Move Role Selection to main part
st.subheader("1. Select Your Target CS Specialization")
selected_role = st.selectbox("Choose a role to analyze your resume against:", list(ROLE_SKILL_MAPPING.keys()))
target_skills = ROLE_SKILL_MAPPING[selected_role]

st.subheader("2. Upload Your Resume")
uploaded_file = st.file_uploader("Upload Resume (DOCX format)", type="docx")

if uploaded_file:
    doc = docx.Document(uploaded_file)
    resume_text = '\n'.join([p.text for p in doc.paragraphs])
    clean_resume = preprocess_text(resume_text)
    found_skills = set(extract_skills(clean_resume, SKILL_LIBRARY))
    required_skills = set(target_skills)

    matched = required_skills.intersection(found_skills)
    missing = required_skills - found_skills
    score = (len(matched) / len(required_skills) * 100) if required_skills else 0
    
    # Display ML Verdict in sidebar
    ml_status = predict_qualification(found_skills, SKILL_LIBRARY)
    st.sidebar.header("ML Analysis Results")
    st.sidebar.success(f"**Verdict:** {ml_status}")
    st.sidebar.info("The model predicts suitability based on technical skill patterns identified in the dataset.")

    col1, col2 = st.columns([1, 2])

    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': f"{selected_role} Match"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "green" if score > 70 else "orange"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Skill Breakdown")
        st.write("**Matched Skills:**")
        st.write(", ".join([f"✅ {s}" for s in matched]) if matched else "No core matches for this specific role.")

        st.write("**Missing Skills for this Role:**")
        st.error(", ".join(missing) if missing else "You match all core requirements for this specialization!")

    st.divider()
    st.subheader("Personalized Learning Roadmap")
    if missing:
        for skill in missing:
            st.info(f"📖 Recommendation: Consider a certificate or project involving **{skill.title()}** to improve your standing for {selected_role} roles.")
    else:
        st.success(f"Excellent! Your resume shows strong alignment with the {selected_role} skill set.")
