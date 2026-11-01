"""Part B — Delivery-Hours Regression.

Evaluates regression models for parcel delivery-hours estimation:
- Data splitting without data leakage (840 / 180 / 180)
- Scaler leakage diagnostics
- Baseline mean model
- Decision tree depth diagnostic
- Model candidate evaluation (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
- Gradient boosting learning rate sensitivity
- 5-fold cross-validation comparison on train + validation rows
- Final model evaluation on unseen test split

Outputs generated:
- outputs/part_b_checks.csv
- outputs/part_b_tree_depth.csv
- outputs/part_b_metrics.csv
- outputs/part_b_gb_learning_rate.csv
- outputs/part_b_cv.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES


def run_part_b():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data and split
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # Pinned two-step split: 70% train, 15% val, 15% test (840 / 180 / 180)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    train_rows = len(X_train)
    val_rows = len(X_val)
    test_rows = len(X_test)

    # 2. Scaling check without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
    test_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))

    checks_df = pd.DataFrame([
        ("train_rows", str(train_rows)),
        ("val_rows", str(val_rows)),
        ("test_rows", str(test_rows)),
        ("train_scaled_max_abs_mean", str(round(train_scaled_max_abs_mean, 4))),
        ("test_scaled_max_abs_mean", str(round(test_scaled_max_abs_mean, 4))),
    ], columns=["metric", "value"])
    checks_df.to_csv("outputs/part_b_checks.csv", index=False)

    # 3 & 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    best_depth = None
    min_val_rmse = float("inf")

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)

        tr_pred = dt.predict(X_train)
        va_pred = dt.predict(X_val)

        tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
        va_rmse = np.sqrt(mean_squared_error(y_val, va_pred))

        d_label = "unlimited" if d is None else str(d)
        tree_records.append({
            "max_depth": d_label,
            "train_rmse": round(float(tr_rmse), 4),
            "val_rmse": round(float(va_rmse), 4),
        })

        if va_rmse < min_val_rmse:
            min_val_rmse = va_rmse
            best_depth = d

    tree_depth_df = pd.DataFrame(tree_records)
    tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

    # 5. Candidate models evaluation on validation split
    candidates = {
        "mean_baseline": None,
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_records = []
    val_predictions = {}

    for name, model in candidates.items():
        if name == "mean_baseline":
            y_pred_val = np.full(len(y_val), float(y_train.mean()))
        else:
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)

        val_predictions[name] = y_pred_val
        mae = mean_absolute_error(y_val, y_pred_val)
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred_val)

        metrics_records.append({
            "model": name,
            "split": "val",
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4),
        })

    # 6. Learning-rate experiment
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        va_pred = gb.predict(X_val)
        va_rmse = np.sqrt(mean_squared_error(y_val, va_pred))
        va_r2 = r2_score(y_val, va_pred)

        lr_records.append({
            "learning_rate": lr,
            "val_rmse": round(float(va_rmse), 4),
            "val_r2": round(float(va_r2), 4),
        })

    lr_df = pd.DataFrame(lr_records)
    lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    # 7. Cross-validated comparison and final model
    val_ranked = sorted(
        [r for r in metrics_records if r["model"] != "mean_baseline"],
        key=lambda x: x["RMSE"]
    )
    top_two_names = [val_ranked[0]["model"], val_ranked[1]["model"]]
    final_model_name = val_ranked[0]["model"]

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []

    for name in top_two_names:
        m = candidates[name]
        scores = cross_val_score(m, X_train_val, y_train_val, cv=kf)
        cv_records.append({
            "model": name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv("outputs/part_b_cv.csv", index=False)

    # Score final winning model on test rows (already fitted on training rows)
    winning_model = candidates[final_model_name]
    y_pred_test = winning_model.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred_test)

    metrics_records.append({
        "model": final_model_name,
        "split": "test",
        "MAE": round(float(test_mae), 4),
        "MSE": round(float(test_mse), 4),
        "RMSE": round(float(test_rmse), 4),
        "R2": round(float(test_r2), 4),
    })

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

    print(f"Part B complete. Best regression model: {final_model_name}")


if __name__ == "__main__":
    run_part_b()
