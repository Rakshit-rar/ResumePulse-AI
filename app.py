
import streamlit as st
import pandas as pd
import docx
import fitz # PyMuPDF
import pytesseract
from PIL import Image
import nltk
import string
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from skills_data import SKILL_LIBRARY, ROLE_SKILL_MAPPING, SKILL_RESOURCES # Import SKILL_RESOURCES
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
    # Ensure both are lowered for matching
    text_low = text.lower()
    # Using set for efficiency when checking for presence
    text_tokens = set(word_tokenize(text_low))
    found = []
    for skill in skills_list:
        # For single-word skills, check if token is present
        if ' ' not in skill:
            if skill in text_tokens:
                found.append(skill)
        # For multi-word skills, check if substring is present in original low-cased text
        else:
            if skill in text_low:
                found.append(skill)
    return list(set(found))

def predict_qualification(skills_found, all_features):
    model_path = 'trained_model.pkl'
    if not os.path.exists(model_path):
        return 'Model file missing'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Ensure features are in the same order as model training
        # Use feature_names_in_ if available, otherwise assume all_features in order
        model_feature_names = getattr(model, 'feature_names_in_', all_features)

        input_dict = {feature: (1 if feature in skills_found else 0) for feature in model_feature_names}
        input_data = pd.DataFrame([input_dict])

        prediction = model.predict(input_data)
        return 'Qualified' if prediction[0] == 1 else 'Needs More Skills'
    except Exception as e:
        return f'ML Analysis Error: {str(e)}'

def get_learning_resources(skill_name):
    return SKILL_RESOURCES.get(skill_name.lower(), {
        'youtube': f'https://www.youtube.com/results?search_query={skill_name.replace(" ", "+")}+tutorial',
        'udemy': f'https://www.udemy.com/courses/search/?q={skill_name.replace(" ", "+")}',
        'coursera': f'https://www.coursera.org/search?query={skill_name.replace(" ", "+")}',
        'documentation': f'https://www.google.com/search?q={skill_name.replace(" ", "+")}+documentation'
    })

# Streamlit UI
st.set_page_config(page_title="PulseMatch AI", layout="wide")
st.title("🚀 PulseMatch AI: Resume Analyzer & Skill Gap Detector")

# Move Role Selection to main part
st.subheader("1. Configure Analysis")
col_a, col_b = st.columns(2)
with col_a:
    selected_role = st.selectbox("Target Specialization:", list(ROLE_SKILL_MAPPING.keys()))
with col_b:
    uploaded_file = st.file_uploader("Upload Resume (DOCX, PDF, JPG, PNG)", type=["docx", "pdf", "jpg", "png"])

analyze_clicked = st.button("🔍 Analyze Resume")

if uploaded_file and analyze_clicked:
    with st.spinner('Processing your profile...'):
        # Universal text extraction
        def extract_text_from_any(uploaded_file):
            file_ext = uploaded_file.name.split('.')[-1].lower()
            text = ''
            try:
                if file_ext == 'docx':
                    doc = docx.Document(uploaded_file)
                    text = '
'.join([p.text for p in doc.paragraphs])
                elif file_ext == 'pdf':
                    # Need to reset file pointer for PyMuPDF if it was read by another process (like Streamlit internal handling)
                    uploaded_file.seek(0)
                    with fitz.open(stream=uploaded_file.read(), filetype='pdf') as doc:
                        for page in doc: text += page.get_text()
                elif file_ext in ['jpg', 'jpeg', 'png']:
                    uploaded_file.seek(0)
                    img = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(img)
            except Exception as e:
                return f'Error reading file: {str(e)}'
            return text

        resume_text = extract_text_from_any(uploaded_file)

        if resume_text and not resume_text.startswith('Error'):
            clean_resume = preprocess_text(resume_text)
            found_skills = set(extract_skills(clean_resume, SKILL_LIBRARY))
            target_skills = set(ROLE_SKILL_MAPPING[selected_role])

            matched = target_skills.intersection(found_skills)
            missing = target_skills - found_skills
            score = (len(matched) / len(target_skills) * 100) if target_skills else 0

            # Display ML Verdict in sidebar
            ml_status = predict_qualification(found_skills, SKILL_LIBRARY)

            st.divider()
            st.subheader("📊 Analysis Results")

            m1, m2, m3 = st.columns(3)
            m1.metric('Match Score', f'{score:.1f}%')
            m2.metric('ML Verdict', ml_status)
            m3.metric('Skills Detected', len(found_skills))

            col1, col2 = st.columns([1, 2])

            with col1:
                fig = go.Figure(go.Indicator(
                    mode = 'gauge+number',
                    value = score,
                    title = {'text': f'{selected_role} Match'},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': 'green' if score > 70 else 'orange'}}
                ))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown(f'**Target Role:** {selected_role}')
                st.write(f'**Matched Skills:** {', '.join([f'✅ {s.title()}' for s in sorted(list(matched))]) if matched else 'None matching the requirement identified')
                st.error(f'**Missing Skills:** {', '.join([f'❌ {s.title()}' for s in sorted(list(missing))]) if missing else 'You match all core requirements!')

            st.divider()
            st.subheader("🎯 Professional Learning Roadmap")
            if missing:
                for skill in sorted(list(missing)):
                    st.markdown(f"##### 📚 Skill: {skill.upper()}")
                    resources = get_learning_resources(skill) # Call your function here
                    if 'youtube' in resources: st.markdown(f"- **YouTube:** [Tutorials for {skill.title()}]({resources['youtube']})")
                    if 'udemy' in resources: st.markdown(f"- **Udemy:** [Courses on {skill.title()}]({resources['udemy']})")
                    if 'coursera' in resources: st.markdown(f"- **Coursera:** [Specializations for {skill.title()}]({resources['coursera']})")
                    if 'documentation' in resources: st.markdown(f"- **Documentation:** [Official Docs for {skill.title()}]({resources['documentation']})")
                    st.markdown("---") # Separator for readability
            else:
                st.success(f"Excellent! Your resume shows strong alignment with the {selected_role} skill set.")
        else:
            st.warning(f'Could not process file: {resume_text}')
