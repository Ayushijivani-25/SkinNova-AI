import json
import math
import os
import uuid

from flask import Blueprint, current_app, request
from flask_jwt_extended import jwt_required
from werkzeug.utils import secure_filename

from engines.recommender import analyze_product_compatibility, recommend_products
from models.predictor import full_skin_analysis

full_analysis_bp = Blueprint("full_analysis", __name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _clean_json(value):
    if hasattr(value, "item"):
        try:
            return _clean_json(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return value


def _json_response(payload, status=200):
    clean_payload = _clean_json(payload)
    return current_app.response_class(
        json.dumps(clean_payload, allow_nan=False),
        status=status,
        mimetype="application/json",
    )


def _list_value(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _normalize_profile(data, image_result=None):
    manual_skin = (data.get("skin_type") or data.get("skinType") or "normal").lower()
    manual_acne = (data.get("acne_level") or data.get("acneLevel") or "none").lower()
    manual_concerns = _list_value(data.get("concerns"))

    if not image_result:
        risk_flags = _heuristic_risks(manual_skin, manual_acne, manual_concerns)
        return {
            "skin_type": manual_skin,
            "acne_level": manual_acne,
            "acne_type": "manual profile",
            "concerns": manual_concerns,
            "risk_flags": risk_flags,
            "combination_risk": 0.45 if manual_skin == "combination" else 0.18,
            "source": "manual",
        }

    detected_skin = image_result.get("skin_type", manual_skin)
    detected_acne = image_result.get("acne_level", manual_acne)
    detected_concerns = image_result.get("concerns", [])
    concerns = sorted(set(detected_concerns + manual_concerns))

    return {
        "skin_type": manual_skin if manual_skin else detected_skin,
        "detected_skin_type": detected_skin,
        "acne_level": manual_acne if manual_acne != "none" else detected_acne,
        "detected_acne_level": detected_acne,
        "acne_type": image_result.get("acne_type", "unknown"),
        "concerns": concerns,
        "risk_flags": image_result.get("risk_flags", _heuristic_risks(manual_skin, manual_acne, concerns)),
        "combination_risk": image_result.get("combination_risk", 0),
        "source": "image_and_manual",
    }


def _heuristic_risks(skin_type, acne_level, concerns):
    acne_weight = {"none": 0.18, "mild": 0.46, "moderate": 0.68, "severe": 0.86}.get(acne_level, 0.28)
    concern_set = set(concerns)
    return {
        "early_acne_risk": min(0.95, acne_weight + (0.12 if "blackheads" in concern_set else 0)),
        "pore_congestion": min(0.95, (0.62 if skin_type in ["oily", "combination"] else 0.28) + (0.16 if "pores" in concern_set else 0)),
        "texture_imbalance": min(0.95, 0.42 + (0.18 if skin_type in ["dry", "combination"] else 0) + (0.12 if "wrinkles" in concern_set else 0)),
        "micro_inflammation": min(0.95, acne_weight * 0.65 + (0.18 if "acne" in concern_set else 0)),
    }


def _risk_percent(risk_flags, key):
    return int(round(float(risk_flags.get(key, 0)) * 100))


def _hidden_issues(profile):
    flags = profile["risk_flags"]
    insights = []
    if flags.get("early_acne_risk", 0) >= 0.45:
        insights.append("Early acne formation risk detected in cheek region.")
    if flags.get("pore_congestion", 0) >= 0.45:
        insights.append("Pore congestion likely in T-zone.")
    if flags.get("texture_imbalance", 0) >= 0.45:
        insights.append("Skin texture imbalance indicates dehydration and oil mix.")
    if flags.get("micro_inflammation", 0) >= 0.45:
        insights.append("Micro-inflammation patterns suggest barrier stress.")
    if not insights:
        insights.append("No major hidden issue pattern detected right now.")

    return {
        "early_acne_risk": _risk_percent(flags, "early_acne_risk"),
        "pore_blockage": _risk_percent(flags, "pore_congestion"),
        "texture_imbalance": _risk_percent(flags, "texture_imbalance"),
        "micro_inflammation": _risk_percent(flags, "micro_inflammation"),
        "confidence": 88 if profile["source"] == "image_and_manual" else 76,
        "insights": insights,
    }


def _product_payload(data):
    return {
        "product_name": data.get("product_name") or data.get("productName") or "User product",
        "product_url": data.get("product_url") or data.get("productUrl") or "",
        "ingredients": data.get("ingredients") or data.get("ingredient_list") or data.get("ingredientList") or "",
    }


@full_analysis_bp.route("/full", methods=["POST"])
@jwt_required()
def full_analysis():
    data = request.get_json() if request.is_json else request.form.to_dict()
    data = data or {}

    image_result = None
    if "skin_image" in request.files:
        image = request.files["skin_image"]
        if image and image.filename and _allowed(image.filename):
            upload_dir = current_app.config["UPLOAD_FOLDER"]
            filename = f"skin_{uuid.uuid4().hex}_{secure_filename(image.filename)}"
            path = os.path.join(upload_dir, filename)
            image.save(path)
            try:
                image_result = full_skin_analysis(path)
            except Exception:
                image_result = None

    profile = _normalize_profile(data, image_result)
    product = _product_payload(data)
    concerns = profile["concerns"]

    compatibility = None
    has_product = any(product.values())
    if has_product:
        ingredients = product["ingredients"]
        if not ingredients:
            ingredients = _fallback_ingredients(product["product_name"], product["product_url"], profile["skin_type"])
        compatibility = analyze_product_compatibility(
            product_name=product["product_name"],
            ingredients_text=ingredients,
            skin_type=profile["skin_type"],
            concerns=concerns,
        )
        if not product["ingredients"]:
            compatibility["input_note"] = "Ingredient list was not provided, so the score uses a conservative estimate from the product name or category."

    recommendations = recommend_products(
        skin_type=profile["skin_type"],
        acne_level=profile["acne_level"],
        concerns=concerns,
        product_types=["cleanser", "moisturizer", "sunscreen", "treatment"],
        top_n=4,
    )

    return _json_response({
        "skin_profile": profile,
        "compatibility": compatibility,
        "recommendations": recommendations,
        "hidden_issues": _hidden_issues(profile),
    }), 200


def _fallback_ingredients(product_name, product_url, skin_type):
    text = f"{product_name} {product_url}".lower()
    if any(word in text for word in ["sunscreen", "spf", "uv"]):
        return "zinc oxide, glycerin, niacinamide, dimethicone"
    if any(word in text for word in ["cleanser", "face wash", "foam"]):
        return "glycerin, coco betaine, niacinamide, panthenol"
    if any(word in text for word in ["serum", "treatment", "spot"]):
        return "niacinamide, hyaluronic acid, salicylic acid, panthenol"
    if any(word in text for word in ["cream", "lotion", "moisturizer"]):
        return "glycerin, ceramides, hyaluronic acid, squalane"
    if skin_type == "oily":
        return "niacinamide, zinc, glycerin, salicylic acid"
    if skin_type == "dry":
        return "hyaluronic acid, glycerin, ceramides, squalane"
    return "glycerin, hyaluronic acid, niacinamide, panthenol"
