# SkinNova AI - Working Project

This folder contains the fixed working version of SkinNova AI.

## Website Link

Open this local link while the servers are running:

http://127.0.0.1:5173

For deployment, the backend can also serve the website directly. Use `backend/wsgi.py`
with a command like `gunicorn wsgi:app`, and the deployed website will call
`/api` on the same public domain.

## Start The Project

Run `start_skinnova.ps1` from this folder. It starts:

- Backend API: http://127.0.0.1:5004
- Website: http://127.0.0.1:5173

## What Works

- Login and register
- Skin image upload and manual skin type selection
- Acne level and skin concern selection
- Product image, product name, product URL, and ingredient list input
- Skin-product compatibility score
- Harmful or risky ingredient detection
- Compatibility reasoning
- Alternative product suggestions
- Category-wise recommendations for cleanser, moisturizer, sunscreen, and treatment products
- Hidden skin issue detection for early acne, pore blockage, texture imbalance, and micro-inflammation

## Notes

The backend uses fast local AI-style analysis by default so the project starts quickly. To enable TensorFlow model loading later, set `SKINNOVA_USE_TENSORFLOW=1` before starting the backend and place trained models in the expected backend model folders.

## Deploy Notes

This project includes:

- `render.yaml` for Render deployment
- `backend/wsgi.py` for production servers
- `backend/requirements-deploy.txt` with only the packages needed for the working app

If you deploy on Render from a GitHub repo, create a new Blueprint from this folder.
