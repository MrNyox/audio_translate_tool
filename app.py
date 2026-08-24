from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

import config
from routes import bp as api_bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    config.JOB_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    app.register_blueprint(api_bp)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.errorhandler(RequestEntityTooLarge)
    def handle_upload_too_large(_exc):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"Upload is too large. Maximum size is {config.MAX_UPLOAD_MB} MB.",
                }
            ),
            413,
        )

    @app.errorhandler(404)
    def handle_not_found(_exc):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Not found."}), 404
        return _exc

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )
