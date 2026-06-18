"""
SkinNova AI – Flask Backend Entry Point
"""

from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from api.auth_routes import auth_bp
from api.analysis_routes import analysis_bp
from api.product_routes import product_bp
from api.routine_routes import routine_bp
from api.environment_routes import env_bp
from api.full_analysis_routes import full_analysis_bp
from api.auth_routes import init_db
import os

def create_app():
    app = Flask(__name__)
    frontend_dir = os.environ.get(
        "SKINNOVA_FRONTEND_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")),
    )

    # ── Config ──────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "skinnova-secret-2024")
    app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "skinnova-jwt-2024")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
    app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "uploads")
    app.config["MODEL_DIR"] = os.path.join(os.path.dirname(__file__), "models")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # ── Extensions ──────────────────────────────────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    JWTManager(app)
    init_db()

    # ── Blueprints ──────────────────────────────────────────
    app.register_blueprint(auth_bp,     url_prefix="/api/auth")
    app.register_blueprint(analysis_bp, url_prefix="/api/analyze")
    app.register_blueprint(product_bp,  url_prefix="/api/products")
    app.register_blueprint(routine_bp,  url_prefix="/api/routine")
    app.register_blueprint(env_bp,      url_prefix="/api/environment")
    app.register_blueprint(full_analysis_bp, url_prefix="/api/analyze")

    @app.route("/api/health")
    def health():
        return {"status": "ok", "version": "1.0.0"}

    @app.route("/")
    def serve_frontend():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/<path:path>")
    def serve_frontend_assets(path):
        requested = os.path.join(frontend_dir, path)
        if os.path.isfile(requested):
            return send_from_directory(frontend_dir, path)
        return send_from_directory(frontend_dir, "index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
