import os, json
from flask import Flask, request, jsonify, render_template
import chunk as rag

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB limit for uploads

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
    # only accept file uploads; text entry removed
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

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
        # unsupported extensions
        return jsonify({"error": "Unsupported file type. Only PDF and DOCX are allowed."}), 400

    if not text or not text.strip():
        return jsonify({"error": "No text could be extracted from the submission"}), 400

    return jsonify(rag.assess_risk(text))

if __name__ == "__main__":
    app.run(debug=False, port=5000)

