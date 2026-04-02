from flask import Flask

from app.routes import api
from utils.logger import configure_logging


def create_app() -> Flask:
    configure_logging()
    app = Flask(__name__)
    app.register_blueprint(api)
    return app
