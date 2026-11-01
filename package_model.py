"""Model packaging and serialization script.

Reads the optimal configuration selected in Part C, retrains the final pipeline
on the training partition, exports model artefacts, and verifies reload consistency:
- models/delay_model.pkl
- models/delay_threshold.csv
- models/feature_order.csv
- models/impute_values.csv
- outputs/part_d_reload_check.csv
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from src.shipment_prep import FEATURES, clean_shipments


def run_package_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load Part C final configuration
    config_df = pd.read_csv("outputs/part_c_final_config.csv")
    config = config_df.iloc[0]

    k_features = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth_val = config["max_depth"]
    max_depth = None if str(max_depth_val).lower() == "unlimited" else int(max_depth_val)
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # 2. Rebuild Part C stratified training / validation / test splits
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 3. Fit the final pipeline on training rows
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=k_features)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            ),
        ),
    ])
    pipeline.fit(X_train, y_train)

    # 4. Save model artefacts
    # 4a. Model pickle
    joblib.dump(pipeline, "models/delay_model.pkl")

    # 4b. Alert threshold CSV (one column threshold)
    threshold_df = pd.DataFrame({"threshold": [threshold]})
    threshold_df.to_csv("models/delay_threshold.csv", index=False)

    # 4c. Feature order CSV (one column feature, 17 names in FEATURES order)
    feature_order_df = pd.DataFrame({"feature": FEATURES})
    feature_order_df.to_csv("models/feature_order.csv", index=False)

    # 4d. Impute values CSV (columns column, median — 3 medians from clean_shipments)
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)
    impute_df = pd.DataFrame([
        {"column": "distance_km", "median": round(float(medians["distance_km"]), 4)},
        {"column": "parcel_weight_kg", "median": round(float(medians["parcel_weight_kg"]), 4)},
        {"column": "declared_value_inr", "median": round(float(medians["declared_value_inr"]), 4)},
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)

    # 5. Reload verification
    reloaded_pipeline = joblib.load("models/delay_model.pkl")

    probs_in_memory = pipeline.predict_proba(X_test)[:, 1]
    probs_reloaded = reloaded_pipeline.predict_proba(X_test)[:, 1]

    max_prob_diff = float(np.max(np.abs(probs_in_memory - probs_reloaded)))
    test_rows_scored = len(X_test)

    reload_checks_df = pd.DataFrame([
        ("test_rows_scored", str(test_rows_scored)),
        ("reload_max_abs_probability_diff", str(round(max_prob_diff, 9))),
        ("threshold", str(round(threshold, 4))),
    ], columns=["metric", "value"])
    reload_checks_df.to_csv("outputs/part_d_reload_check.csv", index=False)

    print(
        f"Packaging complete: {test_rows_scored} test rows scored. "
        f"Max abs prob diff: {max_prob_diff:.1e}. Threshold: {threshold}"
    )


if __name__ == "__main__":
    run_package_model()
