import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset (download from Kaggle first)
df = pd.read_csv('fake_reviews_dataset.csv')

# Basic cleaning
df = df[['text_', 'label']].dropna()
df.columns = ['text', 'label']

# Convert labels: 'CG' (fake) = 1, 'OR' (real) = 0
df['label'] = df['label'].map({'CG': 1, 'OR': 0})

# Split data
train_df, test_df = train_test_split(df, te
st_size=0.2, random_state=42)

# Save
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Fake reviews: {df['label'].sum()}, Real reviews: {len(df) - df['label'].sum()}")