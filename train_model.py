import pandas as pd
import numpy as np
import random
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight

# ============================================
# 1. Generate Advanced Synthetic Dataset
# ============================================

np.random.seed(42)

data = []

for _ in range(10000):

    amount = np.random.randint(100, 60000)
    transaction_type = np.random.randint(0, 5)
    location = np.random.randint(0, 3)
    hour = np.random.randint(0, 24)
    account_age_days = np.random.randint(1, 2000)

    # Advanced fraud logic
    fraud = 0

    if amount > 30000:
        fraud = 1

    if transaction_type == 4 and amount > 15000:
        fraud = 1

    if location == 2 and hour < 6:
        fraud = 1

    if account_age_days < 30 and amount > 10000:
        fraud = 1

    # Add randomness (realistic noise)
    if np.random.rand() < 0.02:
        fraud = 1

    data.append([
        amount,
        transaction_type,
        location,
        hour,
        account_age_days,
        fraud
    ])

df = pd.DataFrame(data, columns=[
    "amount",
    "transaction_type",
    "location",
    "hour",
    "account_age_days",
    "fraud"
])

# ============================================
# 2. Feature Engineering
# ============================================

df["high_risk_hour"] = df["hour"].apply(lambda x: 1 if x < 6 else 0)
df["is_large_txn"] = df["amount"].apply(lambda x: 1 if x > 25000 else 0)

# ============================================
# 3. Split Data
# ============================================

X = df.drop("fraud", axis=1)
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 4. Handle Class Imbalance
# ============================================

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}

# ============================================
# 5. Create ML Pipeline
# ============================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight=class_weights
    ))
])

# ============================================
# 6. Hyperparameter Tuning
# ============================================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ============================================
# 7. Evaluation
# ============================================

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n🔎 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🔥 ROC-AUC Score:")
print(roc_auc_score(y_test, y_proba))

# ============================================
# 8. Feature Importance
# ============================================

feature_importance = pd.Series(
    best_model.named_steps["model"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n📌 Feature Importance:")
print(feature_importance)

# ============================================
# 9. Save Model
# ============================================

joblib.dump(best_model, "fraud_model_v2.pkl")

print("\n✅ Advanced Model Trained and Saved Successfully!")
