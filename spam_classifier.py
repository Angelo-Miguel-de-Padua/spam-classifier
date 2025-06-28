import pandas as pd

# Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())
print(df['label'].value_counts())



