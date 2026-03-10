import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import io

st.title("📄 Upload & Ingest")
st.write("Upload your contract document for analysis")

uploaded_file = st.file_uploader("Choose a contract file", type=["pdf", "docx"])

if uploaded_file:
    # Extract text based on file type
    text = ""
    
    if uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
    
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(uploaded_file.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
    
    st.session_state.contract_text = text
    
    st.success(f"✅ File uploaded successfully! Extracted {len(text)} characters.")
    
    with st.expander("Preview extracted text"):
        st.text_area("Contract Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)
    
    if st.button("🔍 Run Analysis", type="primary"):
        st.session_state.analysis_done = True
        st.switch_page("pages/report.py")
else:
    st.info("Please upload a PDF or DOCX file to begin.")
