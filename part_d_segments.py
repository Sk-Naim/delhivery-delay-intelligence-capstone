"""Part D — Hub Segments and Operational Clustering.

Clusters the 24 regional delivery hubs into operating segments:
- Builds hub profile table (shipments, mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops)
- Standardizes hub profile metrics
- K-Means cluster separation diagnostic (K in 2, 3, 4, 5) measuring across-to-within cluster distance ratio
- Segments hubs into Standard, Watch, Escalate based on empirical delay rate ordering
- Detects tier-versus-segment operational mismatches

Outputs generated:
- outputs/hub_profile.csv
- outputs/part_d_separation.csv
- outputs/hub_segments.csv
- outputs/segment_profile.csv
- outputs/tier_vs_segment_mismatch.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def run_part_d_segments():
    os.makedirs("outputs", exist_ok=True)

    modelling_path = "data/modelling_table.csv"
    hubs_path = "data/hubs.csv"

    df = pd.read_csv(modelling_path)
    hubs = pd.read_csv(hubs_path)

    # 1. Hub profile: aggregated at hub_id level, sorted by ascending hub_id
    hub_profile = (
        df.groupby("hub_id")
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
        .sort_values("hub_id")
    )

    # Save hub profile with 4 decimal places
    hub_profile_out = hub_profile.copy()
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        hub_profile_out[col] = hub_profile_out[col].round(4)
    hub_profile_out.to_csv("outputs/hub_profile.csv", index=False)

    # 2. K-Means clustering setup
    feature_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hub_profile[feature_cols])

    dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
    n_hubs = len(X_scaled)

    # 3. Separation evidence for K in [2, 3, 4, 5]
    separation_records = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # Cluster sizes in descending order joined by hyphens
        size_counts = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
        cluster_sizes_str = "-".join(map(str, size_counts))

        # Mean Euclidean distance across vs within
        across_dists = []
        within_dists = []
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = dist_matrix[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        ratio = np.mean(across_dists) / np.mean(within_dists)

        separation_records.append({
            "K": k,
            "across_over_within": round(float(ratio), 4),
            "cluster_sizes": cluster_sizes_str,
        })

    separation_df = pd.DataFrame(separation_records)
    separation_df.to_csv("outputs/part_d_separation.csv", index=False)

    # 4. K = 3 Segmentation, naming, and tier checking
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster"] = km3.fit_predict(X_scaled)

    # Order clusters by mean delay_rate: Standard (lowest), Watch (middle), Escalate (highest)
    cluster_means = (
        hub_profile.groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
    )
    cluster_to_segment = {
        cluster_means.index[0]: "Standard",
        cluster_means.index[1]: "Watch",
        cluster_means.index[2]: "Escalate",
    }
    hub_profile["segment"] = hub_profile["cluster"].map(cluster_to_segment)

    # Join with hubs metadata (hub_city, hub_tier)
    hub_segments = (
        hub_profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
        [["hub_id", "segment", "hub_city", "hub_tier"]]
        .sort_values("hub_id")
    )
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)

    # Segment profile: mean of the hub-level values
    segment_profile = (
        hub_profile.groupby("segment")
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .reindex(["Standard", "Watch", "Escalate"])
        .reset_index()
    )
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        segment_profile[col] = segment_profile[col].round(4)
    segment_profile.to_csv("outputs/segment_profile.csv", index=False)

    # Tier versus segment mismatch
    # Expected: Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_segments["expected_segment"] = hub_segments["hub_tier"].map(tier_map)
    mismatches = (
        hub_segments[hub_segments["segment"] != hub_segments["expected_segment"]]
        [["hub_id", "segment", "hub_city", "hub_tier"]]
        .sort_values("hub_id")
    )
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print("Part D segmentation complete. Hub segments and separation evidence saved.")


if __name__ == "__main__":
    run_part_d_segments()
