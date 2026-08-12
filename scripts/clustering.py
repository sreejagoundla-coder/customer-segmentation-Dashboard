"""
Customer Segmentation - Clustering Pipeline

Loads data/customers.csv, engineers RFM + demographic features,
finds the optimal number of clusters (elbow + silhouette), fits
KMeans, and saves:
  - outputs/elbow_silhouette.png
  - outputs/pca_clusters.png
  - outputs/cluster_profiles.png
  - outputs/segmented_customers.csv

Run: python scripts/clustering.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------- load data
df = pd.read_csv("data/customers.csv")

# ---------------------------------------------------------- feature building
numeric_features = [
    "Age", "AnnualIncome", "TenureYears",
    "Recency", "Frequency", "MonetaryValue", "AvgBasketSize"
]
categorical_features = ["Gender", "PreferredChannel", "PreferredCategory", "LoyaltyMember"]

X_num = df[numeric_features].copy()
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

encoder = OneHotEncoder(sparse_output=False, drop="first")
X_cat = encoder.fit_transform(df[categorical_features])

X = np.hstack([X_num_scaled, X_cat])

# ------------------------------------------------ optimal k: elbow + silhouette
k_range = range(2, 11)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(list(k_range), sil_scores, marker="o", color="#DD8452")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
plt.tight_layout()
plt.savefig("outputs/elbow_silhouette.png", bbox_inches="tight")
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"Silhouette-optimal k = {best_k} (score={max(sil_scores):.3f})")

# ------------------------------------------------------------- final model
FINAL_K = best_k
kmeans = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
df["Segment"] = kmeans.fit_predict(X)

# ------------------------------------------------------------ PCA visualization
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
df["PCA1"], df["PCA2"] = coords[:, 0], coords[:, 1]

plt.figure(figsize=(7, 6))
palette = sns.color_palette("Set2", FINAL_K)
sns.scatterplot(
    data=df, x="PCA1", y="PCA2", hue="Segment",
    palette=palette, s=45, alpha=0.8, edgecolor="white", linewidth=0.3
)
plt.title(f"Customer Segments (PCA projection, k={FINAL_K})")
plt.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("outputs/pca_clusters.png", bbox_inches="tight")
plt.close()

# ------------------------------------------------------------ cluster profiles
profile = df.groupby("Segment")[numeric_features].mean().round(1)
profile["Count"] = df["Segment"].value_counts().sort_index()
profile["SharePct"] = (profile["Count"] / len(df) * 100).round(1)
profile.to_csv("outputs/segment_profile.csv")
print("\nSegment profile (mean values):")
print(profile)

# Bar chart of key metrics per segment
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
metrics_to_plot = ["Age", "AnnualIncome", "Recency", "Frequency", "MonetaryValue", "TenureYears"]
for ax, metric in zip(axes.flat, metrics_to_plot):
    sns.barplot(x=profile.index, y=profile[metric], hue=profile.index,
                palette=palette, legend=False, ax=ax)
    ax.set_title(metric)
    ax.set_xlabel("Segment")
plt.suptitle("Segment Characteristics", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("outputs/cluster_profiles.png", bbox_inches="tight")
plt.close()

# ------------------------------------------------------------ save results
df.drop(columns=["PCA1", "PCA2"]).to_csv("outputs/segmented_customers.csv", index=False)
print("\nSaved:")
print(" - outputs/elbow_silhouette.png")
print(" - outputs/pca_clusters.png")
print(" - outputs/cluster_profiles.png")
print(" - outputs/segment_profile.csv")
print(" - outputs/segmented_customers.csv")
