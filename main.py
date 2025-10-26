import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Train-test split (just to compare with CV later)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

print("🔹 Train-Test Accuracy:", model.score(X_test, y_test))

# Use StratifiedKFold to maintain class balance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=skf)

print("🔹 Cross Validation Scores:", cv_scores)
print("✅ Mean CV Accuracy:", cv_scores.mean())
