"""
SkinNova AI – Product Recommendation & Compatibility Engine
Uses the cleaned Nykaa dataset to recommend and score products.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from engines.conflict_detector import check_single_product, _normalize, _tokenize_ingredients
import logging
import math

logger = logging.getLogger(__name__)

_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/nykaa_skincare.csv")
_df: Optional[pd.DataFrame] = None


def _load_data() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_DATA_PATH)
        _df["ingredients"] = _df["ingredients"].fillna("")
        _df["skin_type_tags"] = _df["skin_type_tags"].fillna("all")
        logger.info(f"Loaded {len(_df)} skincare products from Nykaa dataset")
    return _df


def _safe_float(value, default=0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _safe_int(value, default=0) -> int:
    try:
        number = _safe_float(value, default)
        return int(number)
    except Exception:
        return default


def _safe_text(value, default="") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


# ── Harmful ingredients with risk levels ─────────────────────────────────────
HARMFUL_INGREDIENTS = {
    "parabens":            {"risk": "medium", "detail": "Potential hormone disruptor; controversial."},
    "sodium lauryl sulfate": {"risk": "medium", "detail": "Harsh surfactant; disrupts skin barrier."},
    "fragrance":           {"risk": "medium", "detail": "Common allergen; irritant for sensitive skin."},
    "mineral oil":         {"risk": "low",    "detail": "Occlusive; can clog pores in oily skin."},
    "alcohol denat":       {"risk": "medium", "detail": "Drying; disrupts skin barrier."},
    "sd alcohol":          {"risk": "medium", "detail": "Drying alcohol; barrier disruption."},
    "formaldehyde":        {"risk": "high",   "detail": "Known carcinogen; banned in EU cosmetics."},
    "phthalates":          {"risk": "high",   "detail": "Endocrine disruptors; avoid."},
    "hydroquinone":        {"risk": "high",   "detail": "Skin lightener; potentially toxic with long-term use."},
    "mercury":             {"risk": "high",   "detail": "Toxic heavy metal; illegal in cosmetics."},
    "lead":                {"risk": "high",   "detail": "Toxic heavy metal."},
    "triclosan":           {"risk": "medium", "detail": "Antibiotic resistance concerns; hormone disruption."},
    "coal tar":            {"risk": "high",   "detail": "Potential carcinogen."},
    "oxybenzone":          {"risk": "medium", "detail": "UV filter linked to hormone disruption; reef-damaging."},
    "toluene":             {"risk": "high",   "detail": "Toxic solvent found in nail products."},
    "synthetic dyes":      {"risk": "low",    "detail": "Potential allergen."},
    "isopropyl myristate": {"risk": "medium", "detail": "Highly comedogenic."},
    "coconut oil":         {"risk": "low",    "detail": "Comedogenic for acne-prone skin."},
}


def _score_product_compatibility(
    ingredients: str,
    skin_type: str,
    concerns: List[str],
) -> Dict:
    """
    Compute a 0–100 suitability score for a product given user skin profile.
    """
    score = 70  # base score
    flags = []
    positives = []
    negatives = []

    ingr_list = _tokenize_ingredients(ingredients)

    # ── Check harmful ingredients ─────────────────────────────────────────
    for harm, meta in HARMFUL_INGREDIENTS.items():
        if _normalize(harm) in ingr_list:
            penalty = {"high": 20, "medium": 10, "low": 5}[meta["risk"]]
            score -= penalty
            negatives.append({"ingredient": harm, "risk": meta["risk"], "detail": meta["detail"]})

    # ── Skin type bonus/penalty ───────────────────────────────────────────
    SKIN_GOOD = {
        "oily":        ["salicylic acid", "bha", "niacinamide", "zinc", "tea tree", "kaolin"],
        "dry":         ["hyaluronic acid", "ceramides", "glycerin", "shea butter", "squalane"],
        "combination": ["niacinamide", "hyaluronic acid", "glycerin"],
        "sensitive":   ["ceramides", "centella asiatica", "aloe vera", "oat", "allantoin"],
        "normal":      ["hyaluronic acid", "vitamin c", "niacinamide"],
    }
    SKIN_BAD = {
        "oily":    ["coconut oil", "mineral oil", "petrolatum", "isopropyl myristate"],
        "dry":     ["alcohol denat", "sd alcohol", "salicylic acid"],
        "sensitive": ["fragrance", "alcohol denat", "essential oils", "menthol"],
        "acne":    ["coconut oil", "lanolin", "isopropyl myristate"],
    }

    for good in SKIN_GOOD.get(skin_type, []):
        if good in ingr_list:
            score += 5
            positives.append(f"{good} is beneficial for {skin_type} skin")

    for bad in SKIN_BAD.get(skin_type, []):
        if bad in ingr_list:
            score -= 8
            negatives.append({"ingredient": bad, "risk": "medium",
                               "detail": f"Not ideal for {skin_type} skin."})

    # ── Concern bonus ────────────────────────────────────────────────────
    CONCERN_INGREDIENTS = {
        "acne":       ["salicylic acid", "bha", "benzoyl peroxide", "tea tree", "niacinamide", "zinc"],
        "dark_spots": ["vitamin c", "niacinamide", "kojic acid", "alpha arbutin", "tranexamic acid"],
        "wrinkles":   ["retinol", "peptides", "vitamin c", "hyaluronic acid", "collagen"],
        "pores":      ["niacinamide", "salicylic acid", "bha", "retinol", "kaolin"],
        "blackheads": ["salicylic acid", "bha", "glycolic acid", "aha", "charcoal"],
    }
    for concern in concerns:
        for helpful_ing in CONCERN_INGREDIENTS.get(concern, []):
            if helpful_ing in ingr_list:
                score += 4
                positives.append(f"{helpful_ing} targets your {concern} concern")

    score = max(0, min(100, score))

    return {
        "suitability_score": round(score),
        "harmful_ingredients": negatives,
        "beneficial_ingredients": positives[:5],
        "reasoning": _build_reasoning(score, negatives, positives, skin_type),
    }


def _build_reasoning(score, negatives, positives, skin_type):
    if score >= 80:
        base = f"Highly suitable for {skin_type} skin."
    elif score >= 60:
        base = f"Moderately suitable for {skin_type} skin."
    elif score >= 40:
        base = f"Use with caution for {skin_type} skin."
    else:
        base = f"Not recommended for {skin_type} skin."

    if negatives:
        base += f" Contains {len(negatives)} potentially problematic ingredient(s)."
    if positives:
        base += f" Has {len(positives)} beneficial active(s) for your skin profile."
    return base


# ── Public API ────────────────────────────────────────────────────────────────

def recommend_products(
    skin_type: str,
    acne_level: str,
    concerns: List[str],
    budget_max: float = None,
    product_types: List[str] = None,
    top_n: int = 5,
) -> Dict[str, List[Dict]]:
    """
    Returns category-wise product recommendations.
    
    Returns:
        {
          "cleanser": [...],
          "moisturizer": [...],
          "treatment": [...],
          "sunscreen": [...],
        }
    """
    df = _load_data()
    categories = product_types or ["cleanser", "moisturizer", "sunscreen", "treatment", "toner"]
    result = {}

    for cat in categories:
        cat_df = df[df["product_type"] == cat].copy()

        if budget_max:
            cat_df = cat_df[cat_df["price"] <= budget_max]

        # Filter by skin type tag
        def matches_skin(row):
            tags = str(row["skin_type_tags"])
            if "all" in tags:
                return True
            return skin_type in tags or (
                acne_level in ["moderate", "severe"] and "acne" in tags
            )

        cat_df = cat_df[cat_df.apply(matches_skin, axis=1)]

        # Score and sort
        if len(cat_df) == 0:
            result[cat] = []
            continue

        scored = []
        for _, row in cat_df.iterrows():
            compat = _score_product_compatibility(
                row["ingredients"], skin_type, concerns
            )
            # Weighted final score
            rating = _safe_float(row.get("rating"), 0.0)
            reviews = _safe_int(row.get("reviews"), 0)
            price = _safe_float(row.get("price"), 0.0)
            rating_score = min(max(rating, 0.0) / 5 * 100, 100)
            review_score = min(max(reviews, 0) / 500 * 100, 100)
            final = (
                compat["suitability_score"] * 0.5
                + rating_score * 0.3
                + review_score * 0.2
                if reviews > 0
                else compat["suitability_score"] * 0.6 + rating_score * 0.4
            )
            scored.append({
                "name":               _safe_text(row.get("name"), "Skincare Product"),
                "brand":              _safe_text(row.get("brand"), "Unknown Brand"),
                "price":              round(price, 2),
                "rating":             round(rating, 1),
                "reviews":            reviews,
                "image_url":          _safe_text(row.get("image_url")).split("|")[0],
                "product_url":        _safe_text(row.get("product_url"), "#"),
                "suitability_score":  compat["suitability_score"],
                "final_score":        round(final, 1),
                "harmful_ingredients": compat["harmful_ingredients"],
                "beneficial_ingredients": compat["beneficial_ingredients"],
                "reasoning":          compat["reasoning"],
            })

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        result[cat] = scored[:top_n]

    return result


def analyze_product_compatibility(
    product_name: str,
    ingredients_text: str,
    skin_type: str,
    concerns: List[str],
) -> Dict:
    """
    Analyze a user-provided product against their skin profile.
    Used for the Skin–Product Compatibility Engine (Module 1).
    """
    compat = _score_product_compatibility(ingredients_text, skin_type, concerns)
    conflict = check_single_product(ingredients_text, skin_type)

    # Find similar/better alternatives
    df = _load_data()
    alt_type = _guess_product_type(product_name, ingredients_text)
    alternatives = []
    if alt_type:
        alts = recommend_products(skin_type, "none", concerns, product_types=[alt_type], top_n=3)
        alternatives = alts.get(alt_type, [])

    return {
        "product_name":       product_name,
        "suitability_score":  compat["suitability_score"],
        "reasoning":          compat["reasoning"],
        "harmful_ingredients": compat["harmful_ingredients"],
        "beneficial_ingredients": compat["beneficial_ingredients"],
        "conflicts":          conflict["warnings"],
        "alternatives":       alternatives,
    }


def _guess_product_type(name: str, ingredients: str) -> Optional[str]:
    text = (name + ingredients).lower()
    if any(w in text for w in ["cleanser", "face wash", "foaming"]): return "cleanser"
    if any(w in text for w in ["moisturizer", "cream", "lotion"]): return "moisturizer"
    if any(w in text for w in ["spf", "sunscreen", "sun protect"]): return "sunscreen"
    if any(w in text for w in ["serum", "treatment", "spot"]): return "treatment"
    if any(w in text for w in ["toner", "essence"]): return "toner"
    return None
