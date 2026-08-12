"""
Turns numeric KMeans cluster IDs into human-readable business segment
names by ranking clusters on spend/frequency/recency, so the labels
stay correct even if you change k or regenerate the data.

Run after clustering.py:  python scripts/label_segments.py
Output: outputs/segmented_customers_labeled.csv
"""

import pandas as pd

df = pd.read_csv("outputs/segmented_customers.csv")
profile = df.groupby("Segment")[["MonetaryValue", "Frequency", "Recency", "TenureYears"]].mean()

# Simple scoring heuristic: high spend+frequency+tenure and low recency = best customers
score = (
    profile["MonetaryValue"].rank(pct=True)
    + profile["Frequency"].rank(pct=True)
    + profile["TenureYears"].rank(pct=True)
    - profile["Recency"].rank(pct=True)
)
ordered = score.sort_values(ascending=False).index.tolist()

names = ["VIP / Premium Loyalists", "Steady Regulars", "Occasional Big Spenders",
         "Budget Shoppers", "At-Risk / Disengaged", "Other"]
label_map = {seg: names[i] if i < len(names) else f"Segment {seg}"
             for i, seg in enumerate(ordered)}

df["SegmentLabel"] = df["Segment"].map(label_map)
df.to_csv("outputs/segmented_customers_labeled.csv", index=False)

print("Segment label mapping:")
for seg, name in label_map.items():
    print(f"  Cluster {seg} -> {name}")
