
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
    text = ''
    try:
        if file_ext == 'docx':
            doc = docx.Document(uploaded_file)
            text = '\n'.join([p.text for p in doc.paragraphs])
        elif file_ext == 'pdf':
            with fitz.open(stream=uploaded_file.read(), filetype='pdf') as doc:
                for page in doc: text += page.get_text()
        elif file_ext in ['jpg', 'jpeg', 'png']:
            text = pytesseract.image_to_string(Image.open(uploaded_file))
    except Exception as e:
        return f'Error reading file: {str(e)}'
    return text

def extract_skills(text, skills_list):
    text_low = text.lower()
    return [s for s in skills_list if s.lower() in text_low]

def predict_qualification(found_skills, all_features):
    model_path = 'trained_model.pkl'
    if not os.path.exists(model_path):
        return 'Model file missing'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else all_features
        input_dict = {feat: (1 if feat in found_skills else 0) for feat in features}
        input_df = pd.DataFrame([input_dict])
        
        if hasattr(model, 'feature_names_in_'):
            input_df = input_df[list(model.feature_names_in_)]
        
        prediction = model.predict(input_df)
        return 'Qualified' if prediction[0] == 1 else 'Needs More Skills'
    except Exception as e:
        return f'Analysis Error: {str(e)}'

st.set_page_config(page_title='ResumePulse AI', layout='wide')
st.title('🚀 ResumePulse AI: Advanced Resume Intelligence')

st.markdown('### 1. Configure Analysis')
col_a, col_b = st.columns(2)
with col_a:
    selected_role = st.selectbox('Target Specialization:', list(ROLE_SKILL_MAPPING.keys()))
with col_b:
    uploaded_file = st.file_uploader('Upload Resume (PDF, DOCX, JPG, PNG)', type=['docx', 'pdf', 'jpg', 'png'])

analyze_clicked = st.button('🔍 Analyze Resume')

if uploaded_file and analyze_clicked:
    with st.spinner('Processing your profile...'):
        resume_text = extract_text_from_any(uploaded_file)
        if resume_text and not resume_text.startswith('Error'):
            target_skills = ROLE_SKILL_MAPPING[selected_role]
            found_skills = set(extract_skills(resume_text, SKILL_LIBRARY))
            matched = set(target_skills).intersection(found_skills)
            missing = set(target_skills) - found_skills
            score = (len(matched) / len(target_skills) * 100) if target_skills else 0
            ml_verdict = predict_qualification(found_skills, SKILL_LIBRARY)

            st.divider()
            st.subheader('📊 Analysis Results')

            m1, m2, m3 = st.columns(3)
            m1.metric('Match Score', f'{score:.1f}%')
            m2.metric('ML Verdict', ml_verdict)
            m3.metric('Skills Detected', len(found_skills))

            c1, c2 = st.columns([1, 2])
            with c1:
                fig = go.Figure(go.Indicator(mode='gauge+number', value=score, title={'text': 'Visual Match'}))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown(f'**Target Role:** {selected_role}')
                st.write(f'**Matched Skills:** {list(matched)}')
                st.error(f'**Missing Skills:** {list(missing)}')

            st.subheader('🎯 Professional Learning Roadmap')
            if missing:
                for s in missing:
                    role_roadmaps = LEARNING_ROADMAPS.get(selected_role, {})
                    detail = role_roadmaps.get(s, f'Improve your proficiency in {s} via specialized projects.')
                    st.info(f'**{s.upper()}**: {detail}')
            else:
                st.success('Your profile perfectly aligns with this role requirements!')
        else:
            st.warning(f'Could not process file: {resume_text}')
