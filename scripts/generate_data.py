"""
Generates a synthetic customer dataset that mixes demographics
(age, gender, income) with behavioral / purchase data (RFM-style:
recency, frequency, monetary + category preference, channel, tenure).

Run: python scripts/generate_data.py
Output: data/customers.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

# --- Build 4 latent "true" customer archetypes to make clusters realistic ---
# 0: Budget shoppers (young, low income, frequent small purchases)
# 1: Premium loyalists (older, high income, high spend, frequent)
# 2: Occasional big spenders (mid-age, high income, infrequent but big baskets)
# 3: Disengaged / churn-risk (any age, low frequency, high recency, low spend)
archetype_weights = [0.30, 0.20, 0.25, 0.25]
archetypes = np.random.choice(4, size=N, p=archetype_weights)

age = np.zeros(N)
income = np.zeros(N)
recency = np.zeros(N)      # days since last purchase (lower = more recent)
frequency = np.zeros(N)    # purchases in last 12 months
monetary = np.zeros(N)     # total annual spend
tenure_years = np.zeros(N) # years as a customer

for i, a in enumerate(archetypes):
    if a == 0:  # Budget shoppers
        age[i] = np.random.normal(27, 5)
        income[i] = np.random.normal(35000, 8000)
        recency[i] = np.random.normal(15, 8)
        frequency[i] = np.random.normal(18, 5)
        monetary[i] = np.random.normal(600, 150)
        tenure_years[i] = np.random.exponential(1.5)
    elif a == 1:  # Premium loyalists
        age[i] = np.random.normal(45, 8)
        income[i] = np.random.normal(110000, 20000)
        recency[i] = np.random.normal(8, 4)
        frequency[i] = np.random.normal(24, 6)
        monetary[i] = np.random.normal(4200, 800)
        tenure_years[i] = np.random.exponential(4)
    elif a == 2:  # Occasional big spenders
        age[i] = np.random.normal(38, 9)
        income[i] = np.random.normal(95000, 18000)
        recency[i] = np.random.normal(45, 15)
        frequency[i] = np.random.normal(6, 2)
        monetary[i] = np.random.normal(2800, 700)
        tenure_years[i] = np.random.exponential(2.5)
    else:  # Disengaged / churn-risk
        age[i] = np.random.normal(35, 12)
        income[i] = np.random.normal(50000, 15000)
        recency[i] = np.random.normal(120, 30)
        frequency[i] = np.random.normal(3, 1.5)
        monetary[i] = np.random.normal(300, 120)
        tenure_years[i] = np.random.exponential(3)

# Clip to realistic bounds
age = np.clip(age, 18, 75).round().astype(int)
income = np.clip(income, 15000, 220000).round(2)
recency = np.clip(recency, 1, 365).round().astype(int)
frequency = np.clip(frequency, 1, 60).round().astype(int)
monetary = np.clip(monetary, 50, 15000).round(2)
tenure_years = np.clip(tenure_years, 0.1, 15).round(2)

gender = np.random.choice(["Male", "Female", "Other"], size=N, p=[0.47, 0.49, 0.04])
channel = np.random.choice(["Online", "In-Store", "Mobile App"], size=N, p=[0.5, 0.3, 0.2])
category_pref = np.random.choice(
    ["Electronics", "Fashion", "Home & Garden", "Groceries", "Beauty", "Sports"],
    size=N
)
avg_basket_size = (monetary / np.maximum(frequency, 1)).round(2)
loyalty_member = np.random.binomial(1, np.clip(0.3 + monetary / 20000, 0, 0.9), size=N)

df = pd.DataFrame({
    "CustomerID": [f"CUST{i+1:05d}" for i in range(N)],
    "Age": age,
    "Gender": gender,
    "AnnualIncome": income,
    "TenureYears": tenure_years,
    "PreferredChannel": channel,
    "PreferredCategory": category_pref,
    "Recency": recency,
    "Frequency": frequency,
    "MonetaryValue": monetary,
    "AvgBasketSize": avg_basket_size,
    "LoyaltyMember": loyalty_member,
})

df.to_csv("data/customers.csv", index=False)
print(f"Generated {len(df)} customers -> data/customers.csv")
print(df.head())
