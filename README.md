Legal Contract Risk Assessor
============================

A modular, AI-driven framework for automated contract analysis, designed to identify legal risks in accordance with 2026 UK regulatory standards using Retrieval-Augmented Generation (RAG).

System Architecture & Data Flow
-------------------------------

The system operates as a multi-stage pipeline:

1.  **Data Ingestion (fetch.py)**: Connects to the National Archives Find Case Law API to fetch and structure UK employment case law, creating a local knowledge base.
    
2.  **Vector Storage (chunk.py)**: Embeds legal cases into a ChromaDB vector store using all-MiniLM-L6-v2 for efficient semantic retrieval.
    
3.  **Document Parsing (contract\_processor.py)**: Uses pdfplumber and python-docx to extract text from legal contracts, preserving page-level indexing.
    
4.  **Risk Analysis & Enrichment (risk\_analyser.py)**:
    
    *   **Clause Extraction**: Analyzes document segments against a 2026 risk rubric (Liability, Data Training, Audit Rights, etc.) using structured JSON output.
        
    *   **Legal Precedent Enrichment**: Automatically cross-references high-risk clauses against the vector database to provide relevant case law context.
        

Technology Stack
----------------

*   **AI/LLM**: Model support via OpenRouter GPT-4o-mini).
    
*   **Vector Search**: ChromaDB combined with sentence-transformers for semantic context retrieval.
    
*   **Data & Validation**: Pydantic for schema-validated JSON outputs.
    
*   **Backend**: Flask for API orchestration.
    

Getting Started
---------------

### 1\. Prerequisites

Ensure you have an .env file in the project root containing your API credentials:

```Plaintext
OPENROUTER_API_KEY=your_key_here   
```

### 2\. Installation

Install the necessary dependencies:

``` Bash  
pip install -r requirements.txt
```

### 3\. Data Ingestion

Populate the vector store with relevant UK case law precedents:

``` Bash 
python fetch.py
```

### 4\. Running an Audit

Analyze a contract file to generate a structured risk report:

``` Bash  
python contract_processor.py path/to/contract.pdf --output results.json
 ```
