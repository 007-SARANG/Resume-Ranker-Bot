import streamlit as st
import docx2txt
import PyPDF2
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Ranker", layout="wide")
st.title("📄 AI Resume Ranker Bot")

uploaded_files = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)
job_description = st.text_area("Paste Job Description Here 👇", height=250)

if st.button("🔍 Rank Resumes") and uploaded_files and job_description:

    resumes = []
    names = []

    for file in uploaded_files:
        text = ""
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file.read())
                text = docx2txt.process(tmp.name)
                os.unlink(tmp.name)

        resumes.append(text)
        names.append(file.name)

    # TF-IDF and Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_description] + resumes)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Ranking
    st.subheader("📊 Ranking Results")
    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(ranked, 1):
        st.markdown(f"### {i}. {name}")
        st.write(f"**Relevance Score:** `{round(score * 100, 2)}%`")
        st.divider()

else:
    st.info("👆 Upload resumes and paste a job description to get rankings.")
