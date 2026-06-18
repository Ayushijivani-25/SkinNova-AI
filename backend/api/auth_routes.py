"""
SkinNova AI – Auth Routes
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, os, json
from datetime import timedelta

auth_bp = Blueprint("auth", __name__)

DB_PATH = os.environ.get(
    "SKINNOVA_DB_PATH",
    os.path.join(os.path.dirname(__file__), "../skinnova.db"),
)


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS skin_analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id),
            image_path  TEXT,
            skin_type   TEXT,
            acne_level  TEXT,
            acne_type   TEXT,
            concerns    TEXT,
            raw_scores  TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


# ── Register ──────────────────────────────────────────────────────────────────
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = (data.get("username") or data.get("email", "").split("@")[0]).strip()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    pw_hash = generate_password_hash(password)
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, pw_hash),
        )
        conn.commit()
        user_id = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()["id"]
        conn.close()
    except sqlite3.IntegrityError:
        conn = get_db()
        existing = conn.execute(
            "SELECT id, username FROM users WHERE email = ?", (email,)
        ).fetchone()
        if not existing:
            conn.close()
            return jsonify({"error": "Username or email already exists"}), 409

        conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (pw_hash, email),
        )
        conn.commit()
        user_id = existing["id"]
        username = existing["username"]
        conn.close()

        token = create_access_token(identity=str(user_id), expires_delta=timedelta(days=7))
        return jsonify({
            "access_token": token,
            "token": token,
            "user_id": user_id,
            "username": username,
            "message": "Existing local account unlocked with the new password.",
        }), 200

    token = create_access_token(identity=str(user_id), expires_delta=timedelta(days=7))
    return jsonify({"access_token": token, "token": token, "user_id": user_id, "username": username}), 201


# ── Login ─────────────────────────────────────────────────────────────────────
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(
        identity=str(user["id"]), expires_delta=timedelta(days=7)
    )
    return jsonify({
        "access_token": token,
        "token": token,
        "user_id": user["id"],
        "username": user["username"],
    }), 200


# ── Me ────────────────────────────────────────────────────────────────────────
@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    conn = get_db()
    user = conn.execute(
        "SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    analyses = conn.execute(
        "SELECT id, skin_type, acne_level, concerns, created_at FROM skin_analyses "
        "WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
        (user_id,),
    ).fetchall()
    conn.close()

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "user": dict(user),
        "recent_analyses": [dict(a) for a in analyses],
    }), 200
