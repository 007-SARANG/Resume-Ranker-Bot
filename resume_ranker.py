# Resume Ranker and Feedback Bot

import os
import docx2txt
import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
from sentence_transformers import SentenceTransformer, util

# Initialize tools
language_tool = language_tool_python.LanguageTool('en-US')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- File Reading Utilities ---
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return " ".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)

def extract_text(file):
    if file.name.endswith('.pdf'):
        with open(file.name, 'wb') as f:
            f.write(file.read())
        return extract_text_from_pdf(file.name)
    elif file.name.endswith('.docx'):
        with open(file.name, 'wb') as f:
            f.write(file.read())
        return extract_text_from_docx(file.name)
    else:
        return file.read().decode('utf-8')

# --- Skill Matching and Feedback ---
def get_similarity_score(resume_text, jd_text):
    embeddings = bert_model.encode([resume_text, jd_text])
    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score)

def get_language_feedback(text):
    matches = language_tool.check(text)
    return [match.message for match in matches[:5]]

def get_missing_keywords(resume_text, jd_text):
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    return list(jd_words - resume_words)[:10]  # Top 10 missing

# --- Streamlit UI ---
st.title("AI Resume Ranker & Feedback Bot")

resume_file = st.file_uploader("Upload Your Resume (PDF or DOCX)", type=['pdf', 'docx', 'txt'])
jd_text = st.text_area("Paste the Job Description Here")

if st.button("Evaluate Resume") and resume_file and jd_text:
    resume_text = extract_text(resume_file)
    score = get_similarity_score(resume_text, jd_text)
    feedback = get_language_feedback(resume_text)
    missing = get_missing_keywords(resume_text, jd_text)

    st.subheader("📊 Match Score")
    st.write(f"{score * 100:.2f}% match with job description")

    st.subheader("❌ Missing Keywords")
    st.write(", ".join(missing))

    st.subheader("📝 Language Suggestions")
    for f in feedback:
        st.write("-", f)

    st.success("Analysis Complete!")
