"""
SkinNova AI – Routine Routes
POST /api/routine/check     → Check routine for conflicts
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from engines.conflict_detector import analyze_routine

routine_bp = Blueprint("routine", __name__)


@routine_bp.route("/check", methods=["POST"])
@jwt_required()
def check_routine():
    """
    Body (JSON):
    {
      "ingredients": "Vitamin C, Retinol, Niacinamide, Hyaluronic Acid, SPF 50",
      "skin_type": "oily",
      "frequency_map": {
        "retinol": 3,
        "aha": 4,
        "vitamin c": 7
      }
    }

    OR pass a list of product ingredient blobs:
    {
      "products": [
        {"name": "CeraVe AM", "ingredients": "Niacinamide, Ceramides, Hyaluronic Acid"},
        {"name": "Paula's BHA", "ingredients": "Salicylic Acid 2%, Green Tea"}
      ],
      "skin_type": "oily"
    }
    """
    data = request.get_json() or {}

    skin_type    = data.get("skin_type", "normal")
    frequency    = data.get("frequency_map", {})

    # Support both flat ingredient string and product list
    if "ingredients" in data:
        ingredients_text = data["ingredients"]
    elif "products" in data:
        # Merge all product ingredients
        parts = []
        for p in data["products"]:
            parts.append(p.get("ingredients", ""))
        ingredients_text = ", ".join(parts)
    else:
        return jsonify({"error": "Provide 'ingredients' string or 'products' list"}), 400

    if not ingredients_text.strip():
        return jsonify({"error": "Empty ingredient list"}), 400

    report = analyze_routine(
        ingredients_text=ingredients_text,
        skin_type=skin_type,
        frequency_map=frequency,
    )

    return jsonify(report.to_dict()), 200
