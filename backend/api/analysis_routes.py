"""
SkinNova AI – Analysis Routes
POST /api/analyze/skin        → Full skin image analysis
GET  /api/analyze/history     → Past analyses for user
"""

import os, json, uuid
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from models.predictor import full_skin_analysis
from api.auth_routes import get_db
import logging

logger = logging.getLogger(__name__)
analysis_bp = Blueprint("analysis", __name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Skin Analysis ──────────────────────────────────────────────────────────────
@analysis_bp.route("/skin", methods=["POST"])
@jwt_required()
def analyze_skin():
    user_id = get_jwt_identity()

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use multipart/form-data with key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        return jsonify({"error": "Invalid file. Supported formats: jpg, jpeg, png, webp"}), 400

    # Save upload
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)

    # Run all three models
    try:
        result = full_skin_analysis(filepath)
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        os.remove(filepath)
        return jsonify({"error": "Skin analysis failed. Please try a clearer face image."}), 500

    # Generate human-readable insights
    insights = _build_insights(result)

    # Save to DB
    conn = get_db()
    conn.execute(
        """INSERT INTO skin_analyses
           (user_id, image_path, skin_type, acne_level, acne_type, concerns, raw_scores)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, filepath,
            result["skin_type"], result["acne_level"], result["acne_type"],
            json.dumps(result["concerns"]), json.dumps(result),
        ),
    )
    conn.commit()
    analysis_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    return jsonify({
        "analysis_id":     analysis_id,
        "skin_type":       result["skin_type"],
        "acne_level":      result["acne_level"],
        "acne_type":       result["acne_type"],
        "concerns":        result["concerns"],
        "risk_flags":      result["risk_flags"],
        "combination_risk": result["combination_risk"],
        "insights":        insights,
        "image_path":      filename,
    }), 200


# ── History ───────────────────────────────────────────────────────────────────
@analysis_bp.route("/history", methods=["GET"])
@jwt_required()
def get_history():
    user_id = get_jwt_identity()
    limit = min(int(request.args.get("limit", 10)), 50)

    conn = get_db()
    rows = conn.execute(
        """SELECT id, skin_type, acne_level, acne_type, concerns, created_at
           FROM skin_analyses WHERE user_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    conn.close()

    analyses = []
    for row in rows:
        d = dict(row)
        d["concerns"] = json.loads(d["concerns"]) if d["concerns"] else []
        analyses.append(d)

    return jsonify({"analyses": analyses, "count": len(analyses)}), 200


# ── Insight Builder ───────────────────────────────────────────────────────────
def _build_insights(result: dict) -> list:
    insights = []
    skin  = result["skin_type"]
    acne  = result["acne_level"]
    flags = result["risk_flags"]

    # Skin type insight
    descriptions = {
        "oily":        "Your skin produces excess sebum, making pores appear larger and skin look shiny.",
        "dry":         "Your skin has reduced sebum production, causing tightness and potential flakiness.",
        "normal":      "Your skin has balanced oil production – maintain it with consistent hydration.",
        "combination": "Your T-zone is oily while cheeks may be normal or dry – use zone-specific products.",
        "sensitive":   "Your skin reacts easily – prioritize gentle, fragrance-free formulations.",
    }
    insights.append({
        "type": "skin_type",
        "icon": "💧",
        "title": f"Skin Type: {skin.title()}",
        "detail": descriptions.get(skin, ""),
    })

    # Acne insight
    if acne != "none":
        acne_msgs = {
            "mild":     "Mild acne detected (blackheads/whiteheads). Salicylic acid cleansers can help.",
            "moderate": "Moderate acne detected (papules/pustules). Consider a BHA routine and niacinamide.",
            "severe":   "Severe cystic acne detected. We recommend consulting a dermatologist alongside your skincare routine.",
        }
        insights.append({
            "type": "acne",
            "icon": "⚠️",
            "title": f"Acne: {acne.title()} ({result['acne_type']})",
            "detail": acne_msgs.get(acne, ""),
        })

    # Risk flag insights
    if flags.get("early_acne_risk", 0) > 0.6:
        insights.append({
            "type": "risk",
            "icon": "🔬",
            "title": "Early Acne Formation Risk Detected",
            "detail": "Pre-inflammatory signs detected in skin texture. A preventive BHA toner can intercept breakouts.",
        })
    if flags.get("pore_congestion", 0) > 0.6:
        insights.append({
            "type": "risk",
            "icon": "🔬",
            "title": "Pore Congestion Likely in T-Zone",
            "detail": "Elevated pore blockage risk detected. Use a clay mask 1–2x per week.",
        })
    if flags.get("texture_imbalance", 0) > 0.55:
        insights.append({
            "type": "risk",
            "icon": "🔬",
            "title": "Skin Texture Imbalance Detected",
            "detail": "Dehydration and uneven texture patterns suggest your skin barrier needs support. Ceramide + HA is key.",
        })
    if flags.get("micro_inflammation", 0) > 0.55:
        insights.append({
            "type": "risk",
            "icon": "🔬",
            "title": "Micro-Inflammation Patterns Detected",
            "detail": "Subclinical inflammation may be present. Reduce active ingredient frequency and add centella asiatica.",
        })

    # Concerns
    for concern in result.get("concerns", []):
        concern_tips = {
            "dark_spots": ("Dark Spots Detected", "Vitamin C + SPF 50 is your best strategy. Alpha Arbutin can also help fade existing spots."),
            "wrinkles":   ("Early Wrinkle Signs", "Retinol (start 0.025%) + Peptides at night, SPF every morning."),
            "pores":      ("Enlarged Pores",       "Niacinamide 10% reduces pore appearance over 4–8 weeks."),
            "blackheads": ("Blackheads Present",   "BHA (salicylic acid) 2% exfoliates inside pores. Use 3x/week."),
            "acne":       ("Active Acne Concern",  "Benzoyl peroxide 2.5% or tea tree for spot treatment."),
        }
        if concern in concern_tips:
            title, detail = concern_tips[concern]
            insights.append({"type": "concern", "icon": "🎯", "title": title, "detail": detail})

    return insights
