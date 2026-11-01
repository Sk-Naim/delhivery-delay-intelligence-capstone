"""Preprocessing module for shipment and hub data.

Contains shared constants, data cleaning functions, and feature engineering logic
used across all pipeline stages.
"""

import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw: pd.DataFrame, medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """Clean raw shipments export following prescribed sequential operations.

    Steps applied in order:
    i.   drop exact duplicate rows
    ii.  service_type: strip whitespace and convert to title case
    iii. payment_mode: replace missing values with 'Unknown'
    iv.  declared_value: parse rupee currency strings to numeric float declared_value_inr, drop original
    v.   dispatch_time: convert to datetime
    vi.  parcel_weight_kg: values > 100 kg flagged as scanner-fault sentinels and treated as missing
    vii. impute missing distance_km, parcel_weight_kg, declared_value_inr with medians

    Args:
        raw: Raw export shipments DataFrame
        medians: Optional dictionary of medians to use for imputation.
                 If None, computes medians from non-missing post-step-6 values.

    Returns:
        tuple of (clean_df, medians_dict)
    """
    # i. Drop exact duplicates
    clean_df = raw.drop_duplicates().copy()

    # ii. service_type: strip and title case
    clean_df["service_type"] = (
        clean_df["service_type"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # iii. payment_mode: fillna with "Unknown"
    clean_df["payment_mode"] = (
        clean_df["payment_mode"]
        .fillna("Unknown")
    )

    # iv. declared_value: strip currency symbol and commas, convert to numeric float
    clean_df["declared_value_inr"] = (
        clean_df["declared_value"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .replace("", np.nan)
        .astype(float)
    )
    clean_df = clean_df.drop(columns=["declared_value"])

    # v. dispatch_time: convert to datetime
    clean_df["dispatch_time"] = pd.to_datetime(
        clean_df["dispatch_time"]
    )

    # vi. parcel_weight_kg: values > 100 treated as missing sentinel
    clean_df.loc[
        clean_df["parcel_weight_kg"] > 100,
        "parcel_weight_kg",
    ] = np.nan

    # vii. compute or apply medians
    if medians is None:
        computed_medians = {
            "distance_km": round(float(clean_df["distance_km"].median()), 2),
            "parcel_weight_kg": round(float(clean_df["parcel_weight_kg"].median()), 2),
            "declared_value_inr": round(float(clean_df["declared_value_inr"].median()), 2),
        }
    else:
        computed_medians = {
            "distance_km": round(float(medians["distance_km"]), 2),
            "parcel_weight_kg": round(float(medians["parcel_weight_kg"]), 2),
            "declared_value_inr": round(float(medians["declared_value_inr"]), 2),
        }

    # replace respective column's missing values with their medians
    for column, median_value in computed_medians.items():
        clean_df[column] = (
            clean_df[column]
            .fillna(median_value)
        )

    return clean_df, computed_medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub metadata and engineer all operational and temporal features.

    Does not reference or require delivery_hours, ensuring compatibility with
    new, un-delivered shipments.

    Args:
        clean: Cleaned shipments DataFrame from clean_shipments()
        hubs: Master hub reference DataFrame

    Returns:
        DataFrame containing attached hub attributes and engineered features.
    """
    hub_cols = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    feat_df = clean.merge(
        hubs[hub_cols],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    feat_df["promised_hours"] = (
        feat_df["service_type"]
        .map(PROMISED_HOURS)
        .astype(int)
    )

    feat_df["is_cod"] = (
        feat_df["payment_mode"] == "COD"
    ).astype(int)

    feat_df["is_weekend"] = (
        feat_df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    feat_df["dispatch_hour"] = (
        feat_df["dispatch_time"].dt.hour
    ).astype(int)

    feat_df["log_declared_value"] = np.log(feat_df["declared_value_inr"])

    return feat_df
