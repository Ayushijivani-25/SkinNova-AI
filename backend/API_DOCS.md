# SkinNova AI – API Documentation

## Base URL
```
http://localhost:5000/api
```

---

## 🔐 Authentication

### Register
```
POST /api/auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "securepassword"
}
```
**Response:**
```json
{
  "access_token": "<JWT>",
  "user_id": 1,
  "username": "john_doe"
}
```

### Login
```
POST /api/auth/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "securepassword"
}
```

### Get Profile
```
GET /api/auth/me
Authorization: Bearer <JWT>
```

---

## 📸 Skin Analysis (Module 5 + 2)

### Analyze Skin Image
```
POST /api/analyze/skin
Authorization: Bearer <JWT>
Content-Type: multipart/form-data

image: <file>
```
**Response:**
```json
{
  "analysis_id": 12,
  "skin_type": "oily",
  "acne_level": "moderate",
  "acne_type": "Pustules",
  "concerns": ["dark_spots", "pores"],
  "risk_flags": {
    "early_acne_risk": 0.71,
    "pore_congestion": 0.65,
    "texture_imbalance": 0.42,
    "micro_inflammation": 0.58
  },
  "insights": [
    {
      "type": "skin_type",
      "icon": "💧",
      "title": "Skin Type: Oily",
      "detail": "Your skin produces excess sebum..."
    }
  ]
}
```

### Analysis History
```
GET /api/analyze/history?limit=10
Authorization: Bearer <JWT>
```

---

## 🧴 Product Engine (Module 1 + 2)

### Analyze Product Compatibility
```
POST /api/products/analyze
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "product_name": "The Ordinary Niacinamide 10%",
  "ingredients": "Niacinamide, Zinc PCA, Glycerin, Aqua",
  "skin_type": "oily",
  "concerns": ["pores", "dark_spots"]
}
```
**Response:**
```json
{
  "product_name": "The Ordinary Niacinamide 10%",
  "suitability_score": 88,
  "reasoning": "Highly suitable for oily skin. Has 3 beneficial actives.",
  "harmful_ingredients": [],
  "beneficial_ingredients": [
    "niacinamide is beneficial for oily skin",
    "niacinamide targets your pores concern"
  ],
  "alternatives": [...]
}
```

### Get Personalized Recommendations
```
POST /api/products/recommend
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "skin_type": "oily",
  "acne_level": "mild",
  "concerns": ["pores", "dark_spots"],
  "budget_max": 1500,
  "categories": ["cleanser", "moisturizer", "sunscreen"]
}
```
**Response:**
```json
{
  "skin_profile": { "skin_type": "oily", "acne_level": "mild", "concerns": [...] },
  "recommendations": {
    "cleanser": [
      {
        "name": "Neutrogena Oil-Free Acne Wash",
        "brand": "Neutrogena",
        "price": 599,
        "rating": 4.3,
        "suitability_score": 85,
        "image_url": "...",
        "product_url": "..."
      }
    ],
    "moisturizer": [...],
    "sunscreen": [...]
  }
}
```

### Search Products
```
GET /api/products/search?q=sunscreen&skin_type=oily&product_type=sunscreen&limit=10
Authorization: Bearer <JWT>
```

---

## 🌦️ Environment Analyzer (Module 3)

### Manual Input
```
POST /api/environment/analyze
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "humidity": 78,
  "temperature": 35,
  "pm25": 95,
  "uv_index": 8,
  "skin_type": "oily",
  "concerns": ["acne", "pores"]
}
```
**Response:**
```json
{
  "overall_risk": "high",
  "alerts": [
    {
      "category": "humidity",
      "severity": "high",
      "message": "High humidity detected → Increased sebum production...",
      "tips": ["Switch to lightweight moisturizer", "Use BHA toner"]
    }
  ],
  "concern_flags": [
    "⚠️ High humidity + acne-prone skin: Risk of bacterial proliferation..."
  ],
  "summary": "Today's environment poses HIGH risk for oily skin."
}
```

### Live Data (requires API keys)
```
GET /api/environment/live?lat=23.02&lon=72.57&skin_type=oily&concerns=acne,pores
Authorization: Bearer <JWT>
```

---

## 🧪 Routine Conflict Checker (Module 4)

### Check Ingredient Conflicts
```
POST /api/routine/check
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "ingredients": "Vitamin C, Retinol, Niacinamide, Hyaluronic Acid",
  "skin_type": "oily",
  "frequency_map": {
    "retinol": 4,
    "aha": 5
  }
}
```
**Response:**
```json
{
  "safe": false,
  "summary": "⚠️ 1 ingredient conflict(s) detected | 1 overuse warning(s).",
  "conflicts": [
    {
      "ingredient_a": "vitamin c",
      "ingredient_b": "retinol",
      "severity": "high",
      "reason": "Vitamin C (low pH) destabilizes Retinol..."
    }
  ],
  "synergies": [
    {
      "pair": ["niacinamide", "retinol"],
      "note": "Niacinamide buffers Retinol irritation – great combo."
    }
  ],
  "overuse_warnings": [
    {
      "ingredient": "retinol",
      "days_per_week": 4,
      "recommended_max": 2,
      "warning": "Retinol should be used max 2-3x per week when starting out."
    }
  ],
  "harmful_ingredients": []
}
```

---

## 🏃 Running the Server

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment file
cp .env.example .env

# 3. Run training scripts (one-time, needs GPU for speed)
python models/training/train_skin_type.py
python models/training/train_acne_type.py
python models/training/train_skin_concern.py

# 4. Start Flask server
python app.py

# Production
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
```

---

## 📁 Project Structure

```
skinnova/
├── app.py                          ← Flask entry point
├── requirements.txt
├── .env.example
├── skinnova.db                     ← SQLite DB (auto-created)
│
├── api/
│   ├── auth_routes.py             ← Register / Login / Profile
│   ├── analysis_routes.py         ← Skin image analysis
│   ├── product_routes.py          ← Product compatibility & recs
│   ├── routine_routes.py          ← Routine conflict checker
│   └── environment_routes.py      ← Environment analyzer
│
├── engines/
│   ├── conflict_detector.py       ← Ingredient conflict rules
│   ├── recommender.py             ← Product recommendation engine
│   └── environment.py             ← Weather/AQI rule engine
│
├── models/
│   ├── predictor.py               ← Model loader & predict functions
│   ├── skin_type_model/           ← Trained after running script
│   │   ├── model.h5
│   │   └── class_indices.json
│   ├── acne_type_model/
│   │   └── model.h5
│   ├── skin_concern_model/
│   │   └── model.h5
│   └── training/
│       ├── train_skin_type.py     ← Dataset 3 (Oily-Dry)
│       ├── train_acne_type.py     ← Dataset 2 (AcneDataset)
│       └── train_skin_concern.py  ← Dataset 1 (Skin v2)
│
├── data/
│   ├── nykaa_skincare.csv         ← Cleaned product database
│   ├── Skin v2/                   ← Dataset 1 images
│   ├── AcneDataset/               ← Dataset 2 images
│   └── Oily-Dry-Skin-Types/       ← Dataset 3 images
│
└── static/
    └── uploads/                   ← User uploaded images
```

---

## 🎨 Figma Frontend Integration Notes

For each screen in your Figma design, here are the API calls to wire up:

| Screen               | API Call                          |
|----------------------|-----------------------------------|
| Login                | POST /api/auth/login              |
| Register             | POST /api/auth/register           |
| Home / Scan          | POST /api/analyze/skin            |
| Product Check        | POST /api/products/analyze        |
| Recommendations      | POST /api/products/recommend      |
| Routine Checker      | POST /api/routine/check           |
| Environment Widget   | GET  /api/environment/live        |
| Profile / History    | GET  /api/auth/me + /analyze/history |

All protected routes require `Authorization: Bearer <token>` header.
