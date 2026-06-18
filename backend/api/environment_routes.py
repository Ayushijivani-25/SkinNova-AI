"""
SkinNova AI – Environment Routes
POST /api/environment/analyze   → Analyze environment impact on skin
GET  /api/environment/live      → Fetch live weather + AQI for user location
"""

import os
import requests
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from engines.environment import analyze_environment

env_bp = Blueprint("environment", __name__)

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
AQICN_API_KEY       = os.environ.get("AQICN_API_KEY", "")


# ── Manual input analysis ──────────────────────────────────────────────────────
@env_bp.route("/analyze", methods=["POST"])
@jwt_required()
def env_analyze():
    """
    Body (JSON):
    {
      "humidity":    70,
      "temperature": 34,
      "pm25":        80,
      "uv_index":    7,
      "skin_type":   "oily",
      "concerns":    ["acne", "dark_spots"]
    }
    """
    data = request.get_json() or {}

    required = ["humidity", "temperature", "pm25", "uv_index", "skin_type"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        result = analyze_environment(
            humidity    = float(data["humidity"]),
            temperature = float(data["temperature"]),
            pm25        = float(data["pm25"]),
            uv_index    = float(data["uv_index"]),
            skin_type   = str(data["skin_type"]),
            concerns    = data.get("concerns", []),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result), 200


# ── Live data from APIs ────────────────────────────────────────────────────────
@env_bp.route("/live", methods=["GET"])
@jwt_required()
def env_live():
    """
    GET /api/environment/live?lat=23.02&lon=72.57&skin_type=oily&concerns=acne,pores

    Fetches:
    - Weather (OpenWeatherMap) → humidity, temperature, UV
    - AQI (AQICN) → PM2.5
    Then calls analyze_environment automatically.
    """
    lat      = request.args.get("lat")
    lon      = request.args.get("lon")
    city     = request.args.get("city", "")
    skin_type = request.args.get("skin_type", "normal")
    concerns  = [c.strip() for c in request.args.get("concerns", "").split(",") if c.strip()]

    if not lat or not lon:
        return jsonify({"error": "Provide lat and lon query parameters"}), 400

    weather_data = _fetch_weather(lat, lon)
    aqi_data     = _fetch_aqi(lat, lon)

    if not weather_data:
        return jsonify({"error": "Could not fetch weather data. Check OPENWEATHER_API_KEY."}), 502

    humidity    = weather_data.get("humidity", 60)
    temperature = weather_data.get("temperature", 25)
    uv_index    = weather_data.get("uv_index", 5)
    pm25        = aqi_data.get("pm25", 35)

    result = analyze_environment(
        humidity=humidity, temperature=temperature,
        pm25=pm25, uv_index=uv_index,
        skin_type=skin_type, concerns=concerns,
    )

    result["fetched_data"] = {
        "humidity": humidity, "temperature": temperature,
        "uv_index": uv_index, "pm25": pm25,
        "city": weather_data.get("city", city),
    }

    return jsonify(result), 200


# ── Internal helpers ──────────────────────────────────────────────────────────
def _fetch_weather(lat, lon) -> dict:
    if not OPENWEATHER_API_KEY:
        # Return mock data if no API key configured
        return {
            "humidity": 65, "temperature": 30,
            "uv_index": 6, "city": "Unknown (no API key)"
        }
    try:
        # Current weather
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=5)
        w = r.json()

        # UV index (separate endpoint)
        uv_url = (
            f"https://api.openweathermap.org/data/2.5/uvi"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        )
        uv_r = requests.get(uv_url, timeout=5).json()

        return {
            "humidity":    w["main"]["humidity"],
            "temperature": w["main"]["temp"],
            "uv_index":    uv_r.get("value", 5),
            "city":        w.get("name", ""),
        }
    except Exception:
        return {}


def _fetch_aqi(lat, lon) -> dict:
    if not AQICN_API_KEY:
        return {"pm25": 40}  # mock
    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQICN_API_KEY}"
        r = requests.get(url, timeout=5).json()
        pm25 = r.get("data", {}).get("iaqi", {}).get("pm25", {}).get("v", 35)
        return {"pm25": pm25}
    except Exception:
        return {"pm25": 35}
