"""Part C — Delay Classifier and Alert Threshold.

Develops classification models to identify at-risk shipments before delay:
- Stratified 3-way split (840 / 180 / 180)
- Baseline & initial classifier benchmarking (Blind, Logistic Regression, Decision Tree, Random Forest)
- Feature screening (SelectKBest ANOVA F-score vs. Random Forest Gini importance)
- Noise column rejection verification
- Feature subset sweep (k in 5, 7, 10, 12, 17)
- Hyperparameter tuning (GridSearchCV vs. RandomizedSearchCV vs. Default)
- Probability threshold optimization based on operations penalty rule (recall >= 0.80)
- Single final test set evaluation

Outputs generated:
- outputs/part_c_models.csv
- outputs/part_c_feature_scores.csv
- outputs/part_c_selectk.csv
- outputs/part_c_tuning.csv
- outputs/part_c_thresholds.csv
- outputs/part_c_final_test.csv
- outputs/part_c_final_config.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


def run_part_c():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # 1. Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 2. Blind baseline and three classifiers on validation set with default 0.5 decision rule
    # Find best decision tree depth from [2, 3, 4, 5, 6, 8, 10]
    best_dt_depth = None
    best_dt_f1 = -1.0
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt_cand = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt_cand.fit(X_train, y_train)
        pred_cand = dt_cand.predict(X_val)
        score_f1 = f1_score(y_val, pred_cand, zero_division=0)
        if score_f1 > best_dt_f1:
            best_dt_f1 = score_f1
            best_dt_depth = d

    initial_models = {
        "blind_all_on_time": None,
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42))
        ]),
        f"decision_tree_depth_{best_dt_depth}": DecisionTreeClassifier(
            criterion="gini", max_depth=best_dt_depth, random_state=42
        ),
        "random_forest_default": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    models_records = []
    for name, model in initial_models.items():
        if name == "blind_all_on_time":
            y_pred = np.zeros(len(y_val), dtype=int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            acc = accuracy_score(y_val, y_pred)
            prec = 0.0
            rec = 0.0
            f1 = 0.0
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

        models_records.append({
            "model": name,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
        })

    models_df = pd.DataFrame(models_records)
    models_df.to_csv("outputs/part_c_models.csv", index=False)

    # 3. Feature screening: Random Forest Importances & SelectKBest F-scores
    rf_screen = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_screen.fit(X_train, y_train)

    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": [round(float(s), 4) for s in selector_all.scores_],
        "rf_importance": [round(float(imp), 4) for imp in rf_screen.feature_importances_],
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    })
    feature_scores_df["f_rank"] = (
        feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    )
    feature_scores_df = feature_scores_df[
        ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
    ].sort_values("f_rank")
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

    # Feature count sweep for k in [5, 7, 10, 12, 17]
    selectk_records = []
    for k in [5, 7, 10, 12, 17]:
        pipe_k = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_k.fit(X_train, y_train)
        pred_k = pipe_k.predict(X_val)
        val_f1_k = f1_score(y_val, pred_k, zero_division=0)
        selectk_records.append({
            "k": k,
            "val_f1": round(float(val_f1_k), 4),
        })

    selectk_df = pd.DataFrame(selectk_records)
    selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)

    # 4. Tuning on selected top k=7 features
    selector_7 = SelectKBest(f_classif, k=7)
    X_train_7 = selector_7.fit_transform(X_train, y_train)
    X_val_7 = selector_7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    rf_tune_base = RandomForestClassifier(random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(rf_tune_base, param_grid, cv=5)
    grid_search.fit(X_train_7, y_train)
    gs_pred = grid_search.predict(X_val_7)
    gs_val_f1 = f1_score(y_val, gs_pred, zero_division=0)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        rf_tune_base, param_grid, n_iter=9, cv=5, random_state=42
    )
    random_search.fit(X_train_7, y_train)
    rs_pred = random_search.predict(X_val_7)
    rs_val_f1 = f1_score(y_val, rs_pred, zero_division=0)

    # Default untuned forest on same 7 features
    rf_default_7 = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42
    )
    rf_default_7.fit(X_train_7, y_train)
    def_pred = rf_default_7.predict(X_val_7)
    def_val_f1 = f1_score(y_val, def_pred, zero_division=0)

    tuning_records = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": round(float(grid_search.best_score_), 4),
            "val_f1": round(float(gs_val_f1), 4),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(random_search.best_params_),
            "best_cv_accuracy": round(float(random_search.best_score_), 4),
            "val_f1": round(float(rs_val_f1), 4),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": round(float(def_val_f1), 4),
        },
    ]

    tuning_df = pd.DataFrame(tuning_records)
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

    # 5. Final model configuration selection
    # Highest val_f1 (ties -> default_forest_same_7_features, then grid_search, then randomized_search)
    priority_order = {
        "default_forest_same_7_features": 3,
        "grid_search": 2,
        "randomized_search": 1,
    }
    tuning_df["priority"] = tuning_df["search"].map(priority_order)
    winner_row = tuning_df.sort_values(
        by=["val_f1", "priority"], ascending=[False, False]
    ).iloc[0]
    chosen_from = winner_row["search"]

    if chosen_from == "default_forest_same_7_features":
        chosen_n_estimators = 100
        chosen_max_depth = None
        chosen_min_samples_leaf = 1
        model_kwargs = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
            "random_state": 42,
        }
    elif chosen_from == "grid_search":
        bp = grid_search.best_params_
        chosen_n_estimators = bp["n_estimators"]
        chosen_max_depth = bp["max_depth"]
        chosen_min_samples_leaf = bp["min_samples_leaf"]
        model_kwargs = {**bp, "random_state": 42}
    else:
        bp = random_search.best_params_
        chosen_n_estimators = bp["n_estimators"]
        chosen_max_depth = bp["max_depth"]
        chosen_min_samples_leaf = bp["min_samples_leaf"]
        model_kwargs = {**bp, "random_state": 42}

    # Final pipeline fitted on all 17 features of training set
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(**model_kwargs)),
    ])
    final_pipeline.fit(X_train, y_train)

    # 6. Alert threshold evaluation on validation probabilities
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    threshold_records = []
    for t in threshold_list:
        p = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, p).ravel()
        acc = accuracy_score(y_val, p)
        prec = precision_score(y_val, p, zero_division=0)
        rec = recall_score(y_val, p, zero_division=0)
        f1 = f1_score(y_val, p, zero_division=0)

        threshold_records.append({
            "threshold": t,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
        })

    thresholds_df = pd.DataFrame(threshold_records)
    thresholds_df.to_csv("outputs/part_c_thresholds.csv", index=False)

    # Operations rule: highest threshold whose validation recall is >= 0.80 (fallback 0.1)
    eligible = thresholds_df[thresholds_df["recall"] >= 0.80]
    if len(eligible) > 0:
        chosen_threshold = float(eligible["threshold"].max())
    else:
        chosen_threshold = 0.1

    # 7. One test evaluation at chosen threshold
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    p_test = (test_probs >= chosen_threshold).astype(int)
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, p_test).ravel()
    acc_te = accuracy_score(y_test, p_test)
    prec_te = precision_score(y_test, p_test, zero_division=0)
    rec_te = recall_score(y_test, p_test, zero_division=0)
    f1_te = f1_score(y_test, p_test, zero_division=0)

    final_test_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        "TN": int(tn_te),
        "FP": int(fp_te),
        "FN": int(fn_te),
        "TP": int(tp_te),
        "accuracy": round(float(acc_te), 4),
        "precision": round(float(prec_te), 4),
        "recall": round(float(rec_te), 4),
        "f1": round(float(f1_te), 4),
    }])
    final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)

    # Final config file
    max_depth_str = "unlimited" if chosen_max_depth is None else str(chosen_max_depth)
    final_config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": chosen_n_estimators,
        "max_depth": max_depth_str,
        "min_samples_leaf": chosen_min_samples_leaf,
        "threshold": chosen_threshold,
        "chosen_from": chosen_from,
    }])
    final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)

    print(
        f"Part C complete. Chosen configuration: {chosen_from}, "
        f"Alert threshold: {chosen_threshold}, Test F1: {f1_te:.4f}"
    )


if __name__ == "__main__":
    run_part_c()
