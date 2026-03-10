# Legal Contract Risk Assessor

A modular tool designed to automatically parse, analyze, and assess legal risks in contracts using Retrieval-Augmented Generation (RAG) and Large Language Models.

## Core Features
* **Document Parsing**: Supports PDF and DOCX file processing.
* **Risk Detection**: Categorizes clauses (e.g., Liability, Data Training, Audit Rights) based on 2026 UK regulatory standards.
* **Precedent Enrichment**: Automatically queries a vector database of UK employment case law to provide legal context for high-risk clauses.
* **Structured Output**: Generates clear, actionable risk reports in JSON format.

## Technology Stack
* **Frameworks**: Flask, Streamlit 
* **AI/LLM**: OpenAI API (via OpenRouter), Gemini, Sentence-Transformers 
* **Data Handling**: ChromaDB (Vector Storage), Pydantic (Validation), PyMuPDF/python-docx 

## Getting Started
1. **Environment**: Create a `.env` file and add your `OPENROUTER_API_KEY`.
2. **Installation**: Install dependencies using: `pip install -r requirements.txt`.
3. **Data Ingestion**: Run `fetch.py` to collect and store relevant UK case law precedent.
4. **Analysis**: Use `contract_processor.py` to audit a document:
   `python contract_processor.py path/to/contract.pdf`
