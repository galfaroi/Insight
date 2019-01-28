from flask import Flask, make_response
from flask.json import jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api
from flask_jwt_extended import JWTManager
from flask_cors import CORS
import logging


# Define the WSGI application object
app = Flask(__name__)
app.config.from_object('config')


@app.route("/")
def hello():
    return make_response(jsonify({"hello": "world"}), 200)


# Json Web Token Manager instance
# define jwt
jwt = JWTManager(app)

# flask cors develop mode only
CORS(app)


# Define Blueprint
# api_bp = Blueprint('api', __name__)
api = Api(app)


# Define the database object which is imported
# by modules and controllers
db = SQLAlchemy(app)
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)


@app.before_first_request
def create_tables():
    db.create_all()


# Sample HTTP error handling
@app.errorhandler(404)
def not_found(error):
    return make_response('not_found', 404)


from app.routes.api_routes import api
# uploads.register_blueprint(api_bp)


# Register blueprint(s)
# uploads.register_blueprint(xyz_module)
# ..
