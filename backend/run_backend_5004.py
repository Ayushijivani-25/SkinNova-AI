import os

os.environ["SKINNOVA_DB_PATH"] = os.path.join(os.path.dirname(__file__), "skinnova_live.db")

from app import create_app

create_app().run(debug=False, host="127.0.0.1", port=5004)
