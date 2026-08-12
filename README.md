# Customer Segmentation Project

Segment customers by combining **demographics** (age, gender, income) with
**behavioral / RFM data** (recency, frequency, monetary value, basket size,
channel, category preference) using K-Means clustering — then explore the
resulting segments through static reports and an interactive Streamlit app.

## What's inside

```
customer-segmentation/
├── app.py                      # Interactive Streamlit dashboard (deployable)
├── data/
│   └── customers.csv           # Sample dataset (1,000 synthetic customers)
├── scripts/
│   ├── generate_data.py        # Creates the sample dataset
│   ├── clustering.py           # Full pipeline: scaling, elbow/silhouette, KMeans, PCA, plots
│   └── label_segments.py       # Maps numeric clusters -> business-friendly names
├── outputs/                    # Generated plots + segmented CSVs
├── requirements.txt
└── README.md
```

## Method

1. **Feature engineering** — numeric features (age, income, tenure, recency,
   frequency, monetary value, avg basket size) are standardized with
   `StandardScaler`; categorical features (gender, channel, category
   preference, loyalty status) are one-hot encoded.
2. **Choosing k** — K-Means is fit for k = 2..10; the **elbow method**
   (inertia) and **silhouette score** are plotted to pick the best k.
3. **Segmentation** — final K-Means model assigns each customer to a segment.
4. **Visualization** — PCA reduces features to 2D for a segment scatter plot;
   bar charts show each segment's average age, income, recency, frequency,
   spend, and tenure.
5. **Labeling** — segments are ranked by spend/frequency/tenure/recency and
   mapped to readable names (e.g. "VIP / Premium Loyalists", "At-Risk /
   Disengaged").

## Run it locally

```bash
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt

python scripts/generate_data.py      # (optional) regenerate sample data
python scripts/clustering.py         # run the full clustering pipeline -> outputs/
python scripts/label_segments.py     # add human-readable segment names

streamlit run app.py                 # launch the interactive dashboard
```

The dashboard lets you upload your own customer CSV (same column schema as
`data/customers.csv`), pick the number of segments with a slider, and
download the labeled results.

## Push to GitHub

From inside the `customer-segmentation` folder:

```bash
git init
git add .
git commit -m "Initial commit: customer segmentation project"
git branch -M main
git remote add origin https://github.com/<your-username>/customer-segmentation.git
git push -u origin main
```
## live Demo
customer-segmentation-dashboard-gs
https://customer-segmentation-dashboard-gs.streamlit.app/
