"""Flask application factory."""

from __future__ import annotations

import logging

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import config


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    # Long uploads and SSE streams must not be buffered or truncated.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from app.api.routes import bp as api_bp

    app.register_blueprint(api_bp)

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc):
        return jsonify({
            "error": f"Upload exceeds the {config.MAX_UPLOAD_MB} MB limit. "
                     f"Raise MAX_UPLOAD_MB in .env to allow bigger files."
        }), 413

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        return jsonify({"error": exc.description, "status": exc.code}), exc.code or 500

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        app.logger.exception("unhandled error")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    from app.core.pipeline import restore_all

    restored = restore_all()
    if restored:
        app.logger.info("restored %d completed job(s) from disk", restored)

    return app
