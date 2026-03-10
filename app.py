from flask import Flask, request, jsonify, render_template_string
import chunk as rag

app = Flask(__name__)

# Load cases into ChromaDB on startup
rag.load_cases("employment_cases.json")

@app.route("/")
def index():
    return render_template_string(open("templates/index.html").read())

@app.route("/assess", methods=["POST"])
def assess():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(rag.assess_risk(text))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
