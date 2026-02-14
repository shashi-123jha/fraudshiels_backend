import jwt
import datetime
from functools import wraps
from flask import request, jsonify
from db import users_collection

SECRET_KEY = "fraudshield_secret_key"


# =========================
# TOKEN REQUIRED DECORATOR
# =========================
def token_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):

            token = None

            if "Authorization" in request.headers:
                try:
                    token = request.headers["Authorization"].split(" ")[1]
                except:
                    return jsonify({"error": "Invalid token format"}), 401

            if not token:
                return jsonify({"error": "Token missing"}), 401

            try:
                data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

                if role and data.get("role") != role:
                    return jsonify({"error": "Unauthorized access"}), 403

            except jwt.ExpiredSignatureError:
                return jsonify({"error": "Token expired"}), 401
            except:
                return jsonify({"error": "Invalid token"}), 401

            return f(*args, **kwargs)

        return decorated
    return decorator


# =========================
# SIGNUP
# =========================
def signup_user(username, password, role="user"):

    if users_collection.find_one({"username": username}):
        return {"error": "User already exists"}, 400

    users_collection.insert_one({
        "username": username,
        "password": password,
        "role": role
    })

    return {"message": "User registered successfully"}, 201


# =========================
# LOGIN
# =========================
def login_user(username, password):

    user = users_collection.find_one({"username": username})

    if not user or user["password"] != password:
        return {"error": "Invalid credentials"}, 401

    token = jwt.encode({
        "username": username,
        "role": user["role"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=3)
    }, SECRET_KEY, algorithm="HS256")

    return {"token": token}, 200
