"""Batch scoring and operational action assignment script.

Applies the packaged delay intelligence model to newly booked shipments:
- Ingests data/new_shipments.csv and data/hubs.csv
- Imputes missing values using verified training medians from models/impute_values.csv
- Preprocesses and engineers features via shared src.shipment_prep functions
- Enforces strict feature alignment using models/feature_order.csv
- Evaluates delay probability using serialized models/delay_model.pkl
- Applies decision threshold from models/delay_threshold.csv
- Joins hub operating segments from outputs/hub_segments.csv
- Assigns actionable logistics interventions:
    * Escalate to hub manager (flagged shipment at Escalate hub)
    * Proactive customer call (flagged shipment at Standard or Watch hub)
    * No action (unflagged shipment)

Outputs generated:
- outputs/new_shipment_actions.csv
"""

import os
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features


def predict_delays():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load inputs and artifacts
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    model = joblib.load("models/delay_model.pkl")
    threshold_val = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute_df = pd.read_csv("models/impute_values.csv")
    stored_medians = dict(zip(impute_df["column"], impute_df["median"].astype(float)))

    # 2. Clean new shipments using stored training medians
    # Note: clean_shipments preserves file order (new_shipments.csv has 0 dupes)
    clean_new, _ = clean_shipments(new_raw, medians=stored_medians)

    # 3. Build features using shared preparation logic
    feat_new = build_features(clean_new, hubs)

    # 4. Enforce exact feature ordering
    X_score = feat_new[feature_order]

    # 5. Predict probabilities (positive class, rounded to 4 decimals)
    probs = model.predict_proba(X_score)[:, 1].round(4)
    flagged = (probs >= threshold_val).astype(int)

    # 6. Merge hub segments while preserving initial file order
    # Add temporary row index to guarantee file order preservation
    feat_new["row_order"] = range(len(feat_new))
    feat_new["delay_probability"] = probs
    feat_new["flagged"] = flagged

    merged = feat_new.merge(
        hub_segments[["hub_id", "segment"]], on="hub_id", how="left"
    ).sort_values("row_order")

    # 7. Operational action assignment
    actions = []
    for _, row in merged.iterrows():
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                actions.append("Escalate to hub manager")
            else:
                actions.append("Proactive customer call")
        else:
            actions.append("No action")

    merged["action"] = actions

    output_cols = [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
    actions_df = merged[output_cols]
    actions_df.to_csv("outputs/new_shipment_actions.csv", index=False)

    total = len(actions_df)
    n_flagged = int(actions_df["flagged"].sum())
    n_escalate = int((actions_df["action"] == "Escalate to hub manager").sum())
    n_call = int((actions_df["action"] == "Proactive customer call").sum())
    n_none = int((actions_df["action"] == "No action").sum())

    print(
        f"Scoring complete for {total} new shipments. "
        f"Flagged: {n_flagged} (Escalate: {n_escalate}, Call: {n_call}, None: {n_none})"
    )


if __name__ == "__main__":
    predict_delays()
