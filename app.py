
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
from skills_data import SKILL_LIBRARY, ROLE_SKILL_MAPPING, SKILL_RESOURCES
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
    text_tokens = set(word_tokenize(text_low))
    found = []
    for skill in skills_list:
        if ' ' not in skill:
            if skill in text_tokens: found.append(skill)
        else:
            if skill in text_low: found.append(skill)
    return list(set(found))

def predict_qualification(found_skills, all_features):
    model_path = 'trained_model.pkl'
    if not os.path.exists(model_path): return 'Model file missing'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_feature_names = getattr(model, 'feature_names_in_', all_features)
        input_dict = {feature: (1 if feature in found_skills else 0) for feature in model_feature_names}
        input_data = pd.DataFrame([input_dict])
        prediction = model.predict(input_data)
        return 'Qualified' if prediction[0] == 1 else 'Needs More Skills'
    except Exception as e: return f'ML Error: {str(e)}'

def get_learning_resources(skill_name):
    return SKILL_RESOURCES.get(skill_name, {
        'youtube': f'https://www.youtube.com/results?search_query={skill_name.replace(" ", "+")}',
        'udemy': f'https://www.udemy.com/courses/search/?q={skill_name.replace(" ", "+")}',
        'coursera': f'https://www.coursera.org/search?query={skill_name.replace(" ", "+")}'
    })

st.set_page_config(page_title='PulseMatch AI', layout='wide')
st.title('🚀 PulseMatch AI: Resume Analyzer')

selected_role = st.selectbox('Target Specialization:', list(ROLE_SKILL_MAPPING.keys()))
uploaded_file = st.file_uploader('Upload Resume', type=['docx', 'pdf', 'jpg', 'png'])

if uploaded_file and st.button('🔍 Analyze'):
    with st.spinner('Analyzing...'):
        def extract_text(file_obj):
            ext = file_obj.name.split('.')[-1].lower()
            try:
                if ext == 'docx':
                    return '\n'.join([p.text for p in docx.Document(file_obj).paragraphs])
                elif ext == 'pdf':
                    file_obj.seek(0)
                    with fitz.open(stream=file_obj.read(), filetype='pdf') as doc:
                        return ''.join([p.get_text() for p in doc])
                else:
                    file_obj.seek(0)
                    return pytesseract.image_to_string(Image.open(file_obj))
            except Exception as e:
                return f'Error: {str(e)}'

        resume_text = extract_text(uploaded_file)
        if resume_text and not resume_text.startswith('Error'):
            clean_resume = preprocess_text(resume_text)
            found_skills = set(extract_skills(clean_resume, SKILL_LIBRARY))
            target_skills = set(ROLE_SKILL_MAPPING[selected_role])
            matched = target_skills.intersection(found_skills)
            missing = target_skills - found_skills
            score = (len(matched) / len(target_skills) * 100) if target_skills else 0
            ml_status = predict_qualification(found_skills, SKILL_LIBRARY)

            st.divider()
            st.subheader('📊 Results')
            m1, m2, m3 = st.columns(3)
            m1.metric('Match Score', f'{score:.1f}%')
            m2.metric('ML Verdict', ml_status)
            m3.metric('Skills Detected', len(found_skills))

            c1, c2 = st.columns([1, 2])
            with c1:
                st.plotly_chart(go.Figure(go.Indicator(mode='gauge+number', value=score, title={'text': 'Match'})), use_container_width=True)
            with c2:
                st.write(f'**Role:** {selected_role}')
                matched_str = ", ".join([f"✅ {s.title()}" for s in sorted(list(matched))]) if matched else "None"
                st.write(f"**Matched:** {matched_str}")
                missing_str = ", ".join([f"❌ {s.title()}" for s in sorted(list(missing))]) if missing else "None"
                st.write(f"**Missing:** {missing_str}")

            st.divider()
            st.subheader('🎯 Roadmap')
            if missing:
                for skill in sorted(list(missing)):
                    res = get_learning_resources(skill)
                    st.markdown(f"##### 📚 {skill.upper()}")
                    st.markdown(f"[YouTube]({res['youtube']}) | [Udemy]({res['udemy']}) | [Coursera]({res['coursera']})")
            else: st.success('Perfect match!')
