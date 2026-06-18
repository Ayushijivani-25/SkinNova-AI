"""
SkinNova AI – Environment Impact Analyzer
Analyzes how weather/pollution conditions affect the user's specific skin type.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class EnvironmentAlert:
    category:    str   # humidity / temperature / pollution / uv
    severity:    str   # low / medium / high
    message:     str
    tips:        List[str] = field(default_factory=list)


# ── Rule Engine ────────────────────────────────────────────────────────────────
def analyze_environment(
    humidity: float,        # 0–100 %
    temperature: float,     # °C
    pm25: float,            # μg/m³
    uv_index: float,        # 0–11+
    skin_type: str,         # oily / dry / normal / combination / sensitive
    concerns: List[str],    # e.g. ['acne', 'dark_spots']
) -> Dict:
    """
    Returns a full environment impact report for the user's skin.
    """
    alerts: List[EnvironmentAlert] = []

    # ── Humidity rules ────────────────────────────────────────────────────
    if humidity > 75:
        if skin_type in ["oily", "combination", "acne"]:
            alerts.append(EnvironmentAlert(
                category="humidity", severity="high",
                message="High humidity detected → Increased sebum production and acne risk, especially in T-zone.",
                tips=[
                    "Switch to a lightweight, oil-free moisturizer.",
                    "Use a BHA toner (salicylic acid) to manage pore congestion.",
                    "Blotting papers are your best friend today.",
                    "Avoid heavy, cream-based sunscreens – go for gel SPF.",
                ],
            ))
        elif skin_type == "dry":
            alerts.append(EnvironmentAlert(
                category="humidity", severity="low",
                message="High humidity is beneficial for dry skin – helps retain moisture.",
                tips=["You can use a lighter moisturizer today."],
            ))
        else:
            alerts.append(EnvironmentAlert(
                category="humidity", severity="medium",
                message="High humidity may increase sweat and product pilling.",
                tips=["Use minimal product layers today.", "Opt for water-based formulas."],
            ))

    elif humidity < 35:
        severity = "high" if skin_type in ["dry", "sensitive"] else "medium"
        alerts.append(EnvironmentAlert(
            category="humidity", severity=severity,
            message="Low humidity → Transepidermal water loss (TEWL) is elevated. Skin dehydration risk.",
            tips=[
                "Apply humectants (hyaluronic acid, glycerin) immediately after cleansing.",
                "Layer a barrier occlusive (squalane or light oil) on top to lock in moisture.",
                "Avoid alcohol-based toners and mists today.",
                "Consider a humidifier indoors.",
            ],
        ))

    # ── Temperature rules ─────────────────────────────────────────────────
    if temperature > 35:
        alerts.append(EnvironmentAlert(
            category="temperature", severity="high",
            message=f"Extreme heat ({temperature}°C) → Increased sweating, pore dilation, and product breakdown.",
            tips=[
                "Reapply SPF every 2 hours if outdoors.",
                "Use a thermal water spray to cool skin.",
                "Avoid heavy makeup – opt for tinted SPF only.",
                "Keep your toner and moisturizer in the fridge for a cooling effect.",
            ],
        ))
    elif temperature < 10:
        if skin_type in ["dry", "sensitive"]:
            alerts.append(EnvironmentAlert(
                category="temperature", severity="high",
                message=f"Cold weather ({temperature}°C) → Drastically accelerates moisture loss and chapping.",
                tips=[
                    "Switch to a richer, cream-based moisturizer with ceramides.",
                    "Don't skip SPF – UV rays are still present and snow reflects them.",
                    "Use a facial oil as the final layer at night.",
                    "Avoid foaming cleansers – use cream or oil-based cleansers.",
                ],
            ))
        else:
            alerts.append(EnvironmentAlert(
                category="temperature", severity="medium",
                message=f"Cold weather ({temperature}°C) may dry out skin. Hydration is key.",
                tips=[
                    "Add a hydrating serum (hyaluronic acid) to your routine.",
                    "Don't skip moisturizer even if skin feels normal.",
                ],
            ))

    # ── Pollution (PM2.5) rules ───────────────────────────────────────────
    if pm25 > 150:
        alerts.append(EnvironmentAlert(
            category="pollution", severity="high",
            message=f"Hazardous pollution (PM2.5: {pm25} μg/m³) → Particles penetrate pores and generate free radicals.",
            tips=[
                "Double cleanse in the evening: oil cleanser first, then foam cleanser.",
                "Apply Vitamin C serum in AM – antioxidant shield against free radical damage.",
                "Use Niacinamide to strengthen skin barrier integrity.",
                "Minimize outdoor time during peak pollution hours (morning rush / evening).",
                "Probiotic or centella serums help with pollution-induced micro-inflammation.",
            ],
        ))
    elif pm25 > 55:
        alerts.append(EnvironmentAlert(
            category="pollution", severity="medium",
            message=f"Elevated pollution (PM2.5: {pm25} μg/m³) → Increased pore blockage and dullness risk.",
            tips=[
                "Apply antioxidant serum (Vitamin C or Niacinamide) before SPF.",
                "Double cleanse at night to remove particulate matter.",
                "AHA toner 2x this week to prevent buildup.",
            ],
        ))
    elif pm25 > 12:
        alerts.append(EnvironmentAlert(
            category="pollution", severity="low",
            message=f"Mild pollution (PM2.5: {pm25} μg/m³). Standard sun + pollution protection sufficient.",
            tips=["SPF + antioxidant serum in AM is your baseline pollution defense."],
        ))

    # ── UV Index rules ────────────────────────────────────────────────────
    if uv_index >= 11:
        alerts.append(EnvironmentAlert(
            category="uv", severity="high",
            message=f"Extreme UV Index ({uv_index}) → Direct skin damage, accelerated aging, and hyperpigmentation.",
            tips=[
                "SPF 50+ PA++++ is mandatory – non-negotiable.",
                "Reapply every 90 minutes if outdoors.",
                "Seek shade between 11AM–3PM.",
                "UV-induced dark spots risk is elevated – Vitamin C + SPF is your armor.",
                "Wear a hat / protective clothing.",
            ],
        ))
    elif uv_index >= 6:
        alerts.append(EnvironmentAlert(
            category="uv", severity="medium",
            message=f"High UV Index ({uv_index}) → Sunburn and premature aging risk.",
            tips=[
                "SPF 30–50 required.",
                "Reapply after sweating or 2 hours outdoors.",
            ],
        ))
    elif uv_index >= 3:
        alerts.append(EnvironmentAlert(
            category="uv", severity="low",
            message=f"Moderate UV Index ({uv_index}). SPF 30 is sufficient for daily protection.",
            tips=["SPF 30 applied once in AM should cover indoor/brief outdoor activity."],
        ))

    # ── Concern-specific overlays ─────────────────────────────────────────
    concern_flags = []
    if "acne" in concerns and humidity > 65:
        concern_flags.append("⚠️ High humidity + acne-prone skin: Risk of bacterial proliferation in clogged pores. Use a non-comedogenic, antibacterial cleanser twice daily.")
    if "dark_spots" in concerns and uv_index >= 6:
        concern_flags.append("⚠️ High UV + dark spot concern: UV is the primary trigger for hyperpigmentation. Vitamin C + SPF 50 is critical today.")
    if "pores" in concerns and pm25 > 55:
        concern_flags.append("⚠️ Elevated pollution + pore concern: Particulate matter accumulates in pores. Double cleanse tonight and use a BHA exfoliant.")
    if "wrinkles" in concerns and humidity < 35:
        concern_flags.append("⚠️ Low humidity + wrinkle concern: Dehydration exaggerates fine lines. Layer humectants + occlusive tonight.")

    # ── Overall risk level ────────────────────────────────────────────────
    high_alerts = [a for a in alerts if a.severity == "high"]
    overall_risk = "high" if len(high_alerts) >= 2 else ("medium" if alerts else "low")

    return {
        "overall_risk": overall_risk,
        "alerts": [
            {
                "category": a.category,
                "severity": a.severity,
                "message":  a.message,
                "tips":     a.tips,
            }
            for a in alerts
        ],
        "concern_flags":   concern_flags,
        "conditions": {
            "humidity":    humidity,
            "temperature": temperature,
            "pm25":        pm25,
            "uv_index":    uv_index,
        },
        "summary": _build_summary(alerts, skin_type, overall_risk),
    }


def _build_summary(alerts, skin_type, overall_risk):
    if overall_risk == "high":
        return (
            f"Today's environment poses HIGH risk for {skin_type} skin. "
            f"{len(alerts)} environmental factor(s) detected. "
            "Follow the protective tips carefully."
        )
    elif overall_risk == "medium":
        return (
            f"Moderate environmental impact on {skin_type} skin today. "
            "Some adjustments to your routine are recommended."
        )
    else:
        return f"Environment conditions are favorable for {skin_type} skin today. Maintain your usual routine."
