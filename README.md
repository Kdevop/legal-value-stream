# Legal Contract Risk Assessor

A modular tool designed to automatically parse, analyze, and assess legal risks in contracts using Retrieval-Augmented Generation (RAG) and Large Language Models.

## Core Features
* [cite_start]**Document Parsing**: Supports PDF and DOCX file processing[cite: 1].
* [cite_start]**Risk Detection**: Categorizes clauses (e.g., Liability, Data Training, Audit Rights) based on 2026 UK regulatory standards[cite: 1].
* [cite_start]**Precedent Enrichment**: Automatically queries a vector database of UK employment case law to provide legal context for high-risk clauses[cite: 1].
* [cite_start]**Structured Output**: Generates clear, actionable risk reports in JSON format[cite: 1].

## Technology Stack
* [cite_start]**Frameworks**: Flask, Streamlit [cite: 1]
* [cite_start]**AI/LLM**: OpenAI API (via OpenRouter), Gemini, Sentence-Transformers [cite: 1]
* [cite_start]**Data Handling**: ChromaDB (Vector Storage), Pydantic (Validation), PyMuPDF/python-docx [cite: 1]

## Getting Started
1. [cite_start]**Environment**: Create a `.env` file and add your `OPENROUTER_API_KEY`[cite: 1].
2. [cite_start]**Installation**: Install dependencies using: `pip install -r requirements.txt`[cite: 1].
3. [cite_start]**Data Ingestion**: Run `fetch.py` to collect and store relevant UK case law precedents[cite: 1].
4. [cite_start]**Analysis**: Use `contract_processor.py` to audit a document[cite: 1]:
   `python contract_processor.py path/to/contract.pdf`
