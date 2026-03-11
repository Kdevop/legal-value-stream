import os, json
from flask import Flask, request, jsonify, render_template
import chunk as rag

app = Flask(__name__)

# Load cases into ChromaDB on startup
store = rag.VectorStoreManager()
store.load_cases("employment_cases.json")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/assess", methods=["POST"])
def assess():
    # Handle file upload
    if "file" in request.files:
        f = request.files["file"]
        filename = f.filename.lower()

        if filename.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(f)
                text = " ".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return jsonify({"error": "pypdf not installed. Run: pip install pypdf"}), 500

        elif filename.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(f)
                text = " ".join(p.text for p in doc.paragraphs)
            except ImportError:
                return jsonify({"error": "python-docx not installed. Run: pip install python-docx"}), 500

        else:
            # .txt or any plain text file
            text = f.read().decode("utf-8", errors="ignore")

    else:
        text = (request.json or {}).get("text", "")

    if not text or not text.strip():
        return jsonify({"error": "No text could be extracted from the submission"}), 400

    return jsonify(store.assess_risk(text))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
