import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import worker

# ----------------------------
# Flask setup
# ----------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)


# ----------------------------
# Routes
# ----------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process-message", methods=["POST"])
def process_message():
    data = request.get_json()
    user_message = data.get("userMessage", "")

    bot_response = worker.process_prompt(user_message)

    return jsonify({
        "botResponse": bot_response
    }), 200


@app.route("/process-document", methods=["POST"])
def process_document():
    if "file" not in request.files:
        return jsonify({
            "botResponse": "No file uploaded. Please upload a PDF."
        }), 400

    file = request.files["file"]
    file_path = file.filename
    file.save(file_path)

    worker.process_document(file_path)

    return jsonify({
        "botResponse": (
            "Your document has been processed successfully. "
            "You can now ask questions about it."
        )
    }), 200


# ----------------------------
# Run server
# ----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

