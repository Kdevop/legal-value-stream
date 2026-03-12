import os, json, tempfile
from flask import Flask, request, jsonify, render_template
import chunk as rag
import chunk as rag
from contract_processor import DocParser, ClauseExtractor   
from risk_analyser import RiskAnalyser

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB limit for uploads

# Load cases into ChromaDB on startup
rag.load_cases("employment_cases.json")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/assess", methods=["POST"])
def assess():
    api_key = os.getenv("OPENROUTER_API_KEY")
    store = rag._get_store()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    suffix = os.path.splitext(f.filename.lower())[1]

    if suffix not in [".pdf", ".docx"]:
        return jsonify({"error": "Only PDF and DOCX files are supported"}), 400

    # Save to temp file so DocParser handles page splitting correctly
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        pages = DocParser().parse_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not pages:
        return jsonify({"error": "No text could be extracted from the submission"}), 400

    extractor = ClauseExtractor(api_key=api_key)
    clauses = extractor.extract_clauses(pages).get("clauses", [])

    if not clauses:
        return jsonify({"total_clauses": 0, "high_risk_count": 0, "clauses": []})

    analyser = RiskAnalyser(
        collection=store.collection,
        embedder=store.embedder,
        api_key=api_key
    )
    return jsonify(analyser.analyse_all_clauses(clauses))

if __name__ == "__main__":
    app.run(debug=False, port=5000)
