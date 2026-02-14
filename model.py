import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample transaction dataset
data = {
    "amount": [100, 2000, 150, 5000, 60, 7000, 120, 9000],
    "location": [0, 1, 0, 1, 0, 1, 0, 1],  # 0 = Local, 1 = International
    "type": [0, 1, 0, 1, 0, 1, 0, 1],      # 0 = Normal, 1 = Suspicious
    "fraud": [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["amount", "location", "type"]]
y = df["fraud"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Logistic Regression model trained & saved")
