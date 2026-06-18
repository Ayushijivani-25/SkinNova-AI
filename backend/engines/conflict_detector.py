"""
SkinNova AI – Routine Conflict Detector Engine
Checks skincare ingredient combinations for safety, conflicts, and synergies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import re

# ── Conflict Knowledge Base ──────────────────────────────────────────────────
# Format: ((ingredient_a, ingredient_b), severity, reason)
CONFLICTS: List[Tuple[Tuple[str, str], str, str]] = [
    # HIGH severity
    (("vitamin c", "retinol"),        "high",
     "Vitamin C (low pH) destabilizes Retinol and both together cause irritation/redness."),
    (("vitamin c", "niacinamide"),    "medium",
     "Historically considered a conflict – can form nicotinic acid causing flushing. "
     "Modern research suggests it's mild, but avoid in high concentrations."),
    (("aha", "retinol"),              "high",
     "AHA exfoliants (glycolic/lactic acid) combined with Retinol over-exfoliate and damage skin barrier."),
    (("bha", "retinol"),              "high",
     "BHA (salicylic acid) + Retinol causes severe dryness and barrier disruption."),
    (("benzoyl peroxide", "retinol"), "high",
     "Benzoyl Peroxide oxidizes and inactivates Retinol, and the combination is extremely drying."),
    (("benzoyl peroxide", "vitamin c"), "high",
     "Benzoyl Peroxide oxidizes Vitamin C, rendering it ineffective."),
    (("aha", "vitamin c"),            "medium",
     "Both are acidic – layering can cause over-exfoliation and irritation for sensitive skin."),
    (("retinol", "alpha arbutin"),    "medium",
     "No direct conflict, but both are actives – use on alternate nights to avoid irritation."),
    (("exfoliant", "retinol"),        "high",
     "Any physical or chemical exfoliant used with Retinol risks skin barrier damage."),
    (("tretinoin", "glycolic acid"),  "high",
     "Tretinoin (prescription retinoid) + AHAs severely increases irritation and peeling."),

    # MEDIUM severity
    (("niacinamide", "aha"),          "low",
     "Niacinamide can slightly raise pH and reduce AHA effectiveness – use with a gap."),
    (("copper peptides", "vitamin c"), "medium",
     "Copper peptides can oxidize Vitamin C; best used at different times of day."),
    (("copper peptides", "retinol"),  "medium",
     "Can interfere with each other's mechanisms; alternate days recommended."),
    (("salicylic acid", "glycolic acid"), "medium",
     "Double exfoliation – over-exfoliation risk, especially for sensitive/dry skin."),
]

# ── Safe & Beneficial Combinations ──────────────────────────────────────────
SYNERGIES: List[Tuple[Tuple[str, str], str]] = [
    (("niacinamide", "retinol"),
     "Niacinamide buffers Retinol irritation and boosts its anti-aging effects – great combo."),
    (("hyaluronic acid", "retinol"),
     "Hyaluronic Acid hydrates and counteracts Retinol dryness – highly recommended."),
    (("niacinamide", "zinc"),
     "Niacinamide + Zinc is ideal for oily/acne-prone skin – reduces sebum and inflammation."),
    (("vitamin c", "vitamin e"),
     "Synergistic antioxidant pair – Vitamin E stabilizes and extends Vitamin C efficacy."),
    (("vitamin c", "ferulic acid"),
     "Ferulic Acid dramatically boosts Vitamin C stability and effectiveness."),
    (("aha", "hyaluronic acid"),
     "HA applied after AHA helps replenish moisture lost during exfoliation."),
    (("peptides", "hyaluronic acid"),
     "Peptides + HA is a gentle, effective anti-aging and hydration combo."),
    (("ceramides", "niacinamide"),
     "Both support skin barrier integrity – excellent for dry or sensitive skin."),
    (("spf", "vitamin c"),
     "Vitamin C in AM routine boosts SPF photoprotection effectiveness."),
    (("retinol", "ceramides"),
     "Ceramides repair the barrier disruption that Retinol can cause."),
]

# ── Overuse Thresholds ───────────────────────────────────────────────────────
OVERUSE_RULES: List[Tuple[str, int, str]] = [
    ("retinol",         2,  "Retinol should be used max 2–3x per week when starting out."),
    ("aha",             3,  "AHA exfoliants more than 3x/week cause micro-tears and barrier damage."),
    ("bha",             3,  "Salicylic Acid daily use can over-dry skin; limit to 3x/week for sensitive skin."),
    ("benzoyl peroxide", 1, "Benzoyl Peroxide daily use is very drying; use sparingly."),
    ("vitamin c",       7,  "Vitamin C can be used daily (AM) – no overuse concern at normal concentrations."),
]

# ── Harmful Ingredients for Skin Types ──────────────────────────────────────
HARMFUL_BY_SKIN_TYPE: Dict[str, List[Tuple[str, str]]] = {
    "oily": [
        ("coconut oil",   "Highly comedogenic (clogs pores) – avoid for oily/acne-prone skin."),
        ("mineral oil",   "Occlusive; can worsen pore congestion on oily skin."),
        ("petrolatum",    "Heavy occlusive; traps sebum – not ideal for very oily skin."),
        ("isopropyl myristate", "Highly comedogenic; commonly triggers breakouts."),
    ],
    "dry": [
        ("alcohol denat", "Denatured alcohol strips skin of moisture – harmful for dry skin."),
        ("sd alcohol",    "Drying alcohol; disrupts skin barrier."),
        ("salicylic acid", "Can over-dry already dry skin; use with caution."),
        ("benzoyl peroxide", "Highly drying; not recommended for dry skin without heavy moisturizer."),
    ],
    "sensitive": [
        ("fragrance",     "Fragrance (parfum) is the #1 cause of contact dermatitis in sensitive skin."),
        ("essential oils", "Many essential oils are irritants/allergens for sensitive skin."),
        ("alcohol denat", "Causes stinging and barrier disruption on sensitive skin."),
        ("synthetic dyes", "FD&C / D&C dyes can cause allergic reactions."),
        ("menthol",       "Cooling sensation misleads – actually an irritant for sensitive skin."),
    ],
    "acne": [
        ("coconut oil",   "Highly comedogenic; triggers breakouts."),
        ("isopropyl myristate", "Blocks pores; major acne trigger."),
        ("lanolin",       "Can be comedogenic for acne-prone skin."),
        ("sodium lauryl sulfate", "Disrupts skin barrier and worsens acne."),
    ],
}

# ── Normalize ingredient names ────────────────────────────────────────────────
_ALIASES = {
    "l-ascorbic acid": "vitamin c",
    "ascorbic acid": "vitamin c",
    "ascorbyl glucoside": "vitamin c",
    "retinyl palmitate": "retinol",
    "tretinoin": "retinol",
    "retin-a": "retinol",
    "glycolic acid": "aha",
    "lactic acid": "aha",
    "mandelic acid": "aha",
    "tartaric acid": "aha",
    "salicylic acid": "bha",
    "beta hydroxy acid": "bha",
    "beta-hydroxy acid": "bha",
    "bp": "benzoyl peroxide",
    "parfum": "fragrance",
}


def _normalize(name: str) -> str:
    n = name.strip().lower()
    return _ALIASES.get(n, n)


def _tokenize_ingredients(text: str) -> List[str]:
    """Split ingredient list by comma or newline, normalize each entry."""
    parts = re.split(r"[,\n;/]", text)
    return [_normalize(p) for p in parts if p.strip()]


# ── Public API ────────────────────────────────────────────────────────────────
@dataclass
class ConflictReport:
    conflicts:  List[Dict] = field(default_factory=list)
    synergies:  List[Dict] = field(default_factory=list)
    overuse:    List[Dict] = field(default_factory=list)
    harmful:    List[Dict] = field(default_factory=list)
    safe:       bool = True
    summary:    str  = ""

    def to_dict(self):
        return {
            "safe": self.safe,
            "summary": self.summary,
            "conflicts": self.conflicts,
            "synergies": self.synergies,
            "overuse_warnings": self.overuse,
            "harmful_ingredients": self.harmful,
        }


def analyze_routine(
    ingredients_text: str,
    skin_type: str = "normal",
    frequency_map: Dict[str, int] = None,  # {ingredient: uses_per_week}
) -> ConflictReport:
    """
    Main entry point.

    Args:
        ingredients_text: Raw ingredient list string (comma/newline separated)
        skin_type:        User's detected skin type
        frequency_map:    Optional dict of how often each ingredient is used/week

    Returns:
        ConflictReport dataclass
    """
    ingredients = _tokenize_ingredients(ingredients_text)
    freq = frequency_map or {}
    report = ConflictReport()

    # 1. Check conflicts
    for (a, b), severity, reason in CONFLICTS:
        if a in ingredients and b in ingredients:
            report.conflicts.append({
                "ingredient_a": a, "ingredient_b": b,
                "severity": severity, "reason": reason,
            })
            if severity == "high":
                report.safe = False

    # 2. Check synergies
    for (a, b), note in SYNERGIES:
        if a in ingredients and b in ingredients:
            report.synergies.append({"pair": [a, b], "note": note})

    # 3. Check overuse
    for ingredient, max_days, warning in OVERUSE_RULES:
        days_used = freq.get(ingredient, 0)
        if days_used > max_days:
            report.overuse.append({
                "ingredient": ingredient,
                "days_per_week": days_used,
                "recommended_max": max_days,
                "warning": warning,
            })

    # 4. Check harmful by skin type
    skin_harms = HARMFUL_BY_SKIN_TYPE.get(skin_type, [])
    for ingredient, reason in skin_harms:
        norm = _normalize(ingredient)
        if norm in ingredients:
            report.harmful.append({
                "ingredient": ingredient,
                "skin_type": skin_type,
                "reason": reason,
            })

    # 5. Summary
    n_conf = len(report.conflicts)
    n_harm = len(report.harmful)
    n_over = len(report.overuse)
    n_syn  = len(report.synergies)

    if n_conf == 0 and n_harm == 0 and n_over == 0:
        report.summary = (
            f"✅ Your routine looks safe! "
            + (f"Found {n_syn} beneficial ingredient synergies." if n_syn else "")
        )
    else:
        parts = []
        if n_conf: parts.append(f"{n_conf} ingredient conflict(s) detected")
        if n_harm: parts.append(f"{n_harm} ingredient(s) potentially harmful for {skin_type} skin")
        if n_over: parts.append(f"{n_over} overuse warning(s)")
        report.summary = "⚠️ " + " | ".join(parts) + "."

    return report


def check_single_product(ingredients_text: str, skin_type: str = "normal") -> Dict:
    """Simplified check for a single product's ingredients."""
    ingredients = _tokenize_ingredients(ingredients_text)
    warnings = []
    harmful  = []

    for (a, b), severity, reason in CONFLICTS:
        if a in ingredients and b in ingredients:
            warnings.append({"type": "conflict", "severity": severity, "detail": reason})

    for ingredient, reason in HARMFUL_BY_SKIN_TYPE.get(skin_type, []):
        if _normalize(ingredient) in ingredients:
            harmful.append({"ingredient": ingredient, "detail": reason})

    return {
        "ingredients_found": ingredients,
        "warnings": warnings,
        "harmful_for_skin_type": harmful,
        "overall_safe": len([w for w in warnings if w["severity"] == "high"]) == 0 and len(harmful) == 0,
    }
