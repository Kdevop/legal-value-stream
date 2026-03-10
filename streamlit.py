import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import io

# Initialize session state
if "contract_text" not in st.session_state:
    st.session_state.contract_text = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "flags" not in st.session_state:
    st.session_state.flags = {}

# Page definitions
upload_page = st.Page("pages/upload.py", title="Upload & Ingest", icon="📄")
report_page = st.Page("pages/report.py", title="Risk Report", icon="⚠️")

pg = st.navigation([upload_page, report_page])
pg.run()
