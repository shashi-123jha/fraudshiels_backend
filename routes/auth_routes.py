from flask import Blueprint, request, jsonify
from auth import signup_user, login_user

auth_routes = Blueprint("auth_routes", __name__)


@auth_routes.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()

    username = data.get("username")
    password = data.get("password")
    role = data.get("role", "user")

    response, status = signup_user(username, password, role)
    return jsonify(response), status


@auth_routes.route("/login", methods=["POST"])
def login():
    data = request.get_json()

    username = data.get("username")
    password = data.get("password")

    response, status = login_user(username, password)
    return jsonify(response), status
