"""
Customer Segmentation - Interactive Streamlit App

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud (share.streamlit.io) by
pointing it at this file in your GitHub repo.
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="🧩")
sns.set_style("whitegrid")

NUMERIC_FEATURES = ["Age", "AnnualIncome", "TenureYears", "Recency", "Frequency",
                     "MonetaryValue", "AvgBasketSize"]
CATEGORICAL_FEATURES = ["Gender", "PreferredChannel", "PreferredCategory", "LoyaltyMember"]


@st.cache_data
def load_default_data():
    base = Path(__file__).resolve().parent
    data_path = base / "data" / "customers.csv"
    return pd.read_csv(data_path)


def run_clustering(df, k):
    X_num = StandardScaler().fit_transform(df[NUMERIC_FEATURES])
    X_cat = OneHotEncoder(sparse_output=False, drop="first").fit_transform(df[CATEGORICAL_FEATURES])
    X = np.hstack([X_num, X_cat])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    out = df.copy()
    out["Segment"] = labels
    out["PCA1"], out["PCA2"] = coords[:, 0], coords[:, 1]
    return out, sil


st.title("🧩 Customer Segmentation Dashboard")
st.caption("Cluster customers by demographics + purchase behavior with K-Means, "
           "then explore each segment's characteristics.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload your own customer CSV", type="csv")
    st.caption(
        "Expected columns: " + ", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    )
    st.divider()
    st.header("Clustering settings")
    k = st.slider("Number of segments (k)", min_value=2, max_value=10, value=5)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(df.columns)
    if missing:
        st.error(f"Uploaded file is missing required columns: {sorted(missing)}")
        st.stop()
else:
    df = load_default_data()
    st.info("Using bundled sample dataset (1,000 synthetic customers). Upload your own CSV in the sidebar to use real data.")

result, sil_score = run_clustering(df, k)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", len(result))
col2.metric("Segments", k)
col3.metric("Silhouette score", f"{sil_score:.3f}")
col4.metric("Avg spend", f"${result['MonetaryValue'].mean():,.0f}")

tab1, tab2, tab3 = st.tabs(["📊 Segment Map", "📈 Segment Profiles", "📋 Data"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set2", k)
    sns.scatterplot(data=result, x="PCA1", y="PCA2", hue="Segment",
                     palette=palette, s=45, alpha=0.85, ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_title("Customer Segments (PCA projection)")
    st.pyplot(fig)

with tab2:
    profile = result.groupby("Segment")[NUMERIC_FEATURES].mean().round(1)
    profile["Count"] = result["Segment"].value_counts().sort_index()
    profile["Share %"] = (profile["Count"] / len(result) * 100).round(1)
    st.dataframe(profile, width='stretch')

    metrics = ["Age", "AnnualIncome", "Recency", "Frequency", "MonetaryValue", "TenureYears"]
    fig2, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, m in zip(axes.flat, metrics):
        sns.barplot(x=profile.index, y=profile[m], hue=profile.index,
                    palette=palette, legend=False, ax=ax)
        ax.set_title(m)
        ax.set_xlabel("Segment")
    plt.tight_layout()
    st.pyplot(fig2)

with tab3:
    st.dataframe(result.drop(columns=["PCA1", "PCA2"]), width='stretch')
    csv = result.drop(columns=["PCA1", "PCA2"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download segmented customers as CSV", data=csv,
                        file_name="segmented_customers.csv", mime="text/csv")
