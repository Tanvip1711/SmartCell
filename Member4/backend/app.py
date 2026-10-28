from api import api
from pathlib import Path

from flask import Flask, jsonify, render_template

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static"
)

app.register_blueprint(api, url_prefix="/api")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/result")
def result_page():
    return render_template("result.html")


@app.route("/health")
def health():
    return jsonify({
        "project": "SmartCELL",
        "status": "OK"
    })


if __name__ == "__main__":
    app.run(debug=True)

