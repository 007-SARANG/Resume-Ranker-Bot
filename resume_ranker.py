import streamlit as st
import docx2txt
import PyPDF2
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gingerit.gingerit import GingerIt

# Title
st.title("📄 Resume Ranker & Feedback Bot")

# Upload Resume(s)
uploaded_files = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)

# Job Description
job_description = st.text_area("Paste Job Description Here")

# Button
if st.button("🔍 Rank Resumes") and uploaded_files and job_description:

    resumes = []
    names = []

    for file in uploaded_files:
        file_text = ""
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                file_text += page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file.read())
                file_text = docx2txt.process(tmp.name)
                os.unlink(tmp.name)

        resumes.append(file_text)
        names.append(file.name)

    # TF-IDF Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_description] + resumes)
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Grammar Feedback using GingerIt
    parser = GingerIt()

    st.subheader("📊 Ranking Results")
    ranked_data = sorted(zip(names, resumes, similarity_scores), key=lambda x: x[2], reverse=True)

    for i, (name, text, score) in enumerate(ranked_data, start=1):
        st.markdown(f"### {i}. {name}")
        st.write(f"**Relevance Score**: {round(score * 100, 2)}%")

        # Grammar Feedback
        try:
            result = parser.parse(text[:1200])  # Limit size for API
            corrections = result.get('corrections', [])
            st.write(f"**Grammar Suggestions**: {len(corrections)}")
            if corrections:
                st.expander("🔧 See suggestions").write(corrections)
        except Exception as e:
            st.warning("Grammar check skipped due to error or rate limit.")

        st.divider()

else:
    st.info("Please upload resumes and enter a job description to begin.")
