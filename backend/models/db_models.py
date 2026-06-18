"""
SkinNova AI – Database Models
Uses SQLite (no external DB needed for local dev)
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id           = db.Column(db.Integer, primary_key=True)
    username     = db.Column(db.String(80), unique=True, nullable=False)
    email        = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    analyses = db.relationship("SkinAnalysis", backref="user", lazy=True)

    def to_dict(self):
        return {"id": self.id, "username": self.username, "email": self.email}


class SkinAnalysis(db.Model):
    __tablename__ = "skin_analyses"

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    image_path     = db.Column(db.String(512))
    skin_type      = db.Column(db.String(50))          # oily / dry / normal / combination
    acne_level     = db.Column(db.String(50))          # none / mild / moderate / severe
    acne_type      = db.Column(db.String(100))         # Blackheads, Cysts, etc.
    concerns       = db.Column(db.Text)                # JSON list: ['pores','dark_spots',...]
    raw_scores     = db.Column(db.Text)                # JSON dict with all model scores
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "skin_type": self.skin_type,
            "acne_level": self.acne_level,
            "acne_type": self.acne_type,
            "concerns": json.loads(self.concerns) if self.concerns else [],
            "raw_scores": json.loads(self.raw_scores) if self.raw_scores else {},
            "created_at": self.created_at.isoformat(),
        }
