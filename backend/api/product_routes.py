"""
SkinNova AI – Product Routes
POST /api/products/analyze       → Analyze a user-provided product
POST /api/products/recommend     → Get personalized product recommendations
GET  /api/products/search        → Search Nykaa products
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from engines.recommender import recommend_products, analyze_product_compatibility
from api.auth_routes import get_db
import json, os, uuid
from werkzeug.utils import secure_filename

product_bp = Blueprint("products", __name__)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Analyze a single product ───────────────────────────────────────────────────
@product_bp.route("/analyze", methods=["POST"])
@jwt_required()
def analyze_product():
    """
    Accepts multipart/form-data OR JSON:
    - product_name   (str)
    - ingredients    (str, optional)
    - skin_type      (str)
    - concerns       (JSON array str)
    - image          (file, optional – for future OCR ingredient extraction)

    Returns suitability score + harmful ingredients + alternatives.
    """
    user_id = get_jwt_identity()

    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    product_name = data.get("product_name", "Unknown Product")
    ingredients  = data.get("ingredients", "")
    skin_type    = data.get("skin_type", "normal")

    raw_concerns = data.get("concerns", "[]")
    if isinstance(raw_concerns, str):
        try:
            concerns = json.loads(raw_concerns)
        except Exception:
            concerns = []
    else:
        concerns = raw_concerns

    # Optionally handle product image (future: OCR ingredient extraction)
    if "image" in request.files:
        file = request.files["image"]
        if file and _allowed(file.filename):
            upload_dir = current_app.config["UPLOAD_FOLDER"]
            filename = f"prod_{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file.save(os.path.join(upload_dir, filename))
            # TODO: pass to OCR module to extract ingredients from label

    if not ingredients:
        # No ingredient text – return a basic response with note
        return jsonify({
            "product_name": product_name,
            "note": "No ingredient list provided. For best results, paste the ingredient list from the product label.",
            "suitability_score": None,
        }), 200

    result = analyze_product_compatibility(product_name, ingredients, skin_type, concerns)
    return jsonify(result), 200


# ── Personalized recommendations ──────────────────────────────────────────────
@product_bp.route("/recommend", methods=["POST"])
@jwt_required()
def get_recommendations():
    """
    Body (JSON):
    {
      "skin_type":   "oily",
      "acne_level":  "mild",
      "concerns":    ["dark_spots", "pores"],
      "budget_max":  1500,
      "categories":  ["cleanser", "moisturizer", "sunscreen"]
    }
    """
    data     = request.get_json() or {}
    skin_type  = data.get("skin_type", "normal")
    acne_level = data.get("acne_level", "none")
    concerns   = data.get("concerns", [])
    budget     = data.get("budget_max", None)
    categories = data.get("categories", None)

    recs = recommend_products(
        skin_type=skin_type,
        acne_level=acne_level,
        concerns=concerns,
        budget_max=budget,
        product_types=categories,
        top_n=5,
    )

    total_products = sum(len(v) for v in recs.values())
    return jsonify({
        "skin_profile": {
            "skin_type": skin_type,
            "acne_level": acne_level,
            "concerns": concerns,
        },
        "recommendations": recs,
        "total_products": total_products,
    }), 200


# ── Product search ────────────────────────────────────────────────────────────
@product_bp.route("/search", methods=["GET"])
@jwt_required()
def search_products():
    """
    GET /api/products/search?q=sunscreen&skin_type=oily&product_type=sunscreen
    """
    import pandas as pd
    q            = request.args.get("q", "").lower()
    skin_type    = request.args.get("skin_type", "")
    product_type = request.args.get("product_type", "")
    limit        = min(int(request.args.get("limit", 20)), 50)

    data_path = os.path.join(current_app.root_path, "data", "nykaa_skincare.csv")
    df = pd.read_csv(data_path)

    if q:
        mask = (
            df["name"].fillna("").str.lower().str.contains(q) |
            df["brand"].fillna("").str.lower().str.contains(q) |
            df["tags"].fillna("").str.lower().str.contains(q)
        )
        df = df[mask]
    if skin_type:
        df = df[df["skin_type_tags"].fillna("").str.contains(skin_type)]
    if product_type:
        df = df[df["product_type"] == product_type]

    df = df.sort_values("rating", ascending=False).head(limit)
    products = df[["name", "brand", "price", "rating", "reviews",
                   "product_type", "image_url", "product_url"]].to_dict(orient="records")

    # Clean image URLs
    for p in products:
        p["image_url"] = str(p.get("image_url", "")).split("|")[0]

    return jsonify({"results": products, "count": len(products)}), 200
