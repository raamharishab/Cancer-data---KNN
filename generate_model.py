"""
generate_model.py
-----------------
Run this ONCE to train the KNN model and save the pickled artefacts:
    python3 generate_model.py

Requires cancer_data.csv in the same directory.
Outputs:
    knn_model.pkl   – trained KNeighborsClassifier
    scaler.pkl      – fitted StandardScaler

NOTE: All 8 features are used (age, gender, bmi, smoking, genetic_risk,
      physical_activity, alcohol_intake, cancer_history).
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("cancer_data.csv")

# ── 2. Features & target (keep ALL columns including age & gender) ─────────────
X = df.drop("diagnosis", axis=1)   # age, gender, bmi, smoking, genetic_risk,
                                    # physical_activity, alcohol_intake, cancer_history
y = df["diagnosis"]

# ── 3. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. Train ──────────────────────────────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ── 7. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(knn,    "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅  Saved knn_model.pkl and scaler.pkl  (features: age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history)")
