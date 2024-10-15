import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('hyundi.csv')

# Display the first few rows and data info
print(df.head())
df.info()

# Check for and count duplicate rows
num_duplicate_rows = df.duplicated().sum()
print("Number of duplicate rows:", num_duplicate_rows)

# Drop duplicates
df_cleaned = df.drop_duplicates()

# Print the shape before and after
print("Shape before dropping duplicates:", df.shape)
print("Shape after dropping duplicates:", df_cleaned.shape)

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('processed_hyundi.csv', index=False)


