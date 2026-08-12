from pathlib import Path
import pandas as pd
p = Path(__file__).resolve().parent / 'customer-segmentation' / 'customer-segmentation' / 'data' / 'customers.csv'
df = pd.read_csv(p)
print('path:', p)
print('shape:', df.shape)
print('columns:', list(df.columns))
print('\n--- head (10) ---')
print(df.head(10).to_string(index=False))
