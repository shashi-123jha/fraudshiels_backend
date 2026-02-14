import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import joblib
from functools import wraps
from dotenv import load_dotenv

# =====================================
# LOAD ENV VARIABLES
# =====================================
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
MONGO_URI = os.getenv("MONGO_URI")

# =====================================
# APP SETUP
# =====================================
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# =====================================
# DATABASE CONNECTION
# =====================================
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # Test connection
    db = client["fraudshield"]
    users_collection = db["users"]
    transactions_collection = db["transactions"]
except Exception as e:
    print(f"MongoDB Connection Error: {e}")
    users_collection = None
    transactions_collection = None

# =====================================
# LOAD ML MODEL
# =====================================
try:
    model = joblib.load("fraud_model.pkl")
except Exception as e:
    print(f"Model Loading Error: {e}")
    model = None

# =====================================
# HOME ROUTE
# =====================================
@app.route("/")
def home():
    return "FraudShield Backend Running 🚀"

# =====================================
# TOKEN REQUIRED DECORATOR
# =====================================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]

        if not token:
            return jsonify({"error": "Token missing"}), 401

        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = users_collection.find_one({"email": data["email"]})
            if not current_user:
                return jsonify({"error": "User not found"}), 401
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401

        return f(current_user, *args, **kwargs)

    return decorated

# =====================================
# SIGNUP ROUTE
# =====================================
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    if users_collection is None:
        return jsonify({"error": "Database not available"}), 503
    
    data = request.json

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "user")

    if not name or not email or not password:
        return jsonify({"error": "All fields required"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already exists"}), 400

    hashed_password = generate_password_hash(password)

    users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password,
        "role": role,
        "created_at": datetime.datetime.utcnow()
    })

    return jsonify({"message": "User created successfully"}), 201

# =====================================
# LOGIN ROUTE
# =====================================
@app.route("/api/auth/login", methods=["POST"])
def login():
    if users_collection is None:
        return jsonify({"error": "Database not available"}), 503
    
    data = request.json

    email = data.get("email")
    password = data.get("password")

    user = users_collection.find_one({"email": email})

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode({
        "email": user["email"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({
        "message": "Login successful",
        "token": token
    }), 200

# =====================================
# FRAUD PREDICTION ROUTE
# =====================================
@app.route("/predict", methods=["POST"])
@token_required
def predict(current_user):
    if model is None:
        return jsonify({"error": "ML model not available"}), 503
    if transactions_collection is None:
        return jsonify({"error": "Database not available"}), 503
    
    data = request.json

    try:
        amount = float(data.get("amount", 0))
        transaction_type = int(data.get("transaction_type", 0))
        location = int(data.get("location", 0))
        hour = int(data.get("hour", 12))
        account_age_days = int(data.get("account_age_days", 365))
        
        # Feature engineering (same as training)
        high_risk_hour = 1 if hour < 6 else 0
        is_large_txn = 1 if amount > 25000 else 0

        features = [[
            amount,
            transaction_type,
            location,
            hour,
            account_age_days,
            high_risk_hour,
            is_large_txn
        ]]

        prediction = model.predict(features)[0]
        risk_score = float(model.predict_proba(features)[0][1]) * 100

        final_result = "Fraud Detected 🚨" if prediction == 1 else "Transaction Safe ✅"

        # Save transaction
        transactions_collection.insert_one({
            "user_email": current_user["email"],
            "amount": amount,
            "prediction": int(prediction),
            "risk_score": risk_score,
            "timestamp": datetime.datetime.utcnow()
        })

        return jsonify({
            "ml_prediction": int(prediction),
            "risk_score": round(risk_score, 2),
            "final_result": final_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
