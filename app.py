
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# =====================================
# 1. WATER QUALITY PARAMETERS
# =====================================

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

# =====================================
# 2. LOAD DATASET
# =====================================

try:
    df = pd.read_csv("water_potability.csv")
except FileNotFoundError:
    raise FileNotFoundError(
        "water_potability.csv not found. "
        "Place it in the same folder as app.py."
    )

required_columns = FEATURES + ["Potability"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

df = df[required_columns].copy()

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Potability"])
df = df[df["Potability"].isin([0, 1])]

if len(df) < 20:
    raise ValueError("Dataset contains too few valid samples.")

X = df[FEATURES]
y = df["Potability"].astype(int)

# =====================================
# 3. TRAIN AND TEST DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# 4. MACHINE LEARNING MODEL
# =====================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])

model.fit(X_train, y_train)

# =====================================
# 5. MODEL EVALUATION
# =====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Not Potable", "Potable"],
    zero_division=0,
    output_dict=True
)

potability_counts = y.value_counts().to_dict()

# =====================================
# 6. HOME PAGE
# =====================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    prediction = None
    confidence = None
    error = None
    inputs = {}

    if request.method == "POST":

        try:
            values = []

            for feature in FEATURES:

                value = float(request.form[feature])

                if not np.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Invalid value for {feature}"
                    )

                inputs[feature] = value
                values.append(value)

            input_data = pd.DataFrame(
                [values],
                columns=FEATURES
            )

            # Generate prediction
            prediction = int(model.predict(input_data)[0])

            probabilities = model.predict_proba(input_data)[0]

            classes = model.named_steps[
                "classifier"
            ].classes_

            confidence = round(
                float(
                    probabilities[
                        list(classes).index(prediction)
                    ]
                ) * 100,
                2
            )

            if prediction == 1:
                result = "POTABLE"
            else:
                result = "NOT POTABLE"

        except (ValueError, KeyError) as e:
            error = str(e)

    return render_template(
        "index.html",
        features=FEATURES,
        result=result,
        prediction=prediction,
        confidence=confidence,
        error=error,
        inputs=inputs,
        accuracy=round(accuracy, 2),
        counts=potability_counts,
        report=report
    )

# =====================================
# 7. START APPLICATION
# =====================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
