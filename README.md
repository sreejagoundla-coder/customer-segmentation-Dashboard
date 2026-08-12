Customer Segmentation Dashboard

An interactive Customer Segmentation project that combines demographic and behavioral/RFM data to group customers into meaningful business segments using K-Means clustering.

The project includes a Streamlit dashboard where users can upload customer data, select the number of segments, explore cluster visualizations, and download the segmented results.

## 🚀 Live Demo

👉 [Customer Segmentation Dashboard](https://customer-segmentation-dashboard-gs.streamlit.app/)



📌 Project Overview

Customer segmentation helps businesses understand different groups of customers based on their characteristics and purchasing behavior.

This project uses:

Demographics: Age, Gender, Income
RFM Data: Recency, Frequency, Monetary Value
Behavioral Data: Basket Size, Channel, Category Preference
Customer Information: Tenure and Loyalty Status

K-Means clustering is used to identify groups of customers with similar characteristics.

🛠️ Tech Stack
Python
Pandas — Data manipulation
NumPy — Numerical operations
Scikit-learn — Scaling, K-Means, PCA, evaluation
Matplotlib — Visualizations
Seaborn — Statistical visualizations
Streamlit — Interactive dashboard


📂 Project Structure
customer-segmentation/
│
├── app.py
│
├── data/
│   └── customers.csv
│
├── scripts/
│   ├── generate_data.py
│   ├── clustering.py
│   └── label_segments.py
│
├── outputs/
│   ├── plots/
│   └── segmented_data/
│
├── requirements.txt
└── README.md


🔍 Methodology
1. Feature Engineering

Numerical features such as:

Age
Income
Tenure
Recency
Frequency
Monetary Value
Average Basket Size

are standardized using StandardScaler.

Categorical features such as:

Gender
Channel
Category Preference
Loyalty Status

are converted into numerical form using One-Hot Encoding.

2. Choosing the Number of Clusters

K-Means clustering is evaluated for different values of K (2–10).

Two methods are used:

Elbow Method — analyzes cluster inertia
Silhouette Score — evaluates cluster separation

These help determine a suitable number of customer segments.

3. Customer Segmentation

After selecting the appropriate value of K, the K-Means algorithm assigns each customer to a cluster.

4. PCA Visualization

Principal Component Analysis (PCA) reduces the high-dimensional feature space to two dimensions so that customer clusters can be visualized using a scatter plot.

5. Business-Friendly Segment Labels

Numeric clusters are converted into meaningful customer segment names based on metrics such as:

Spending
Purchase frequency
Recency
Customer tenure

Example segments include:

VIP / Premium Loyalists
High-Value Customers
Regular Customers
At-Risk / Disengaged Customers

👩‍💻 Author

sreeja Goundla

Built as a Data Science / Machine Learning project using Python, Scikit-learn, and Streamlit.
