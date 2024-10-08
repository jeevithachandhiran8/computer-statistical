import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'method': ['Radial Velocity', 'Transit', 'Imaging', 'Microlensing', 'Astrometry'],
    'orbital_period': [365.25, 433.5, np.nan, 1234.56, 678.9],
    'mass': [1.0, 0.5, np.nan, 2.3, 1.1],
    'distance': [10.5, 20.3, 15.2, np.nan, 30.1],
    'year': [2001, 2005, 2010, 2015, 2020]
}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Calculate basic statistics (excluding non-numeric columns)
numeric_cols = ['orbital_period', 'mass', 'distance', 'year']
mean_values = df[numeric_cols].mean()
min_values = df[numeric_cols].min()
max_values = df[numeric_cols].max()
sum_values = df[numeric_cols].sum()
count_values = df[numeric_cols].count()

# Print the statistics
print("Mean Values:\n", mean_values)
print("Minimum Values:\n", min_values)
print("Maximum Values:\n", max_values)
print("Sum Values:\n", sum_values)
print("Count Values:\n", count_values)

# Aggregation of 1D data
sum_1d = df['mass'].sum()
mean_1d = df['mass'].mean()
count_1d = df['mass'].count()
min_1d = df['mass'].min()
max_1d = df['mass'].max()

print(f"Sum: {sum_1d}, Mean: {mean_1d}, Count: {count_1d}, Min: {min_1d}, Max: {max_1d}")

# Aggregation of 2D data
sum_2d = df[['orbital_period', 'distance']].sum()
mean_2d = df[['orbital_period', 'distance']].mean()
count_2d = df[['orbital_period', 'distance']].count()
min_2d = df[['orbital_period', 'distance']].min()
max_2d = df[['orbital_period', 'distance']].max()

print("Sum (Columnwise):\n", sum_2d)
print("Mean (Columnwise):\n", mean_2d)
print("Count (Columnwise):\n", count_2d)
print("Min (Columnwise):\n", min_2d)
print("Max (Columnwise):\n", max_2d)

# Aggregation of n-D data
grouped = df.groupby(['method', 'orbital_period']).agg({
    'mass': ['mean', 'std', 'min', 'max'],
    'distance': ['mean', 'std', 'min', 'max'],
    'year': ['mean', 'std', 'min', 'max']
})

print(grouped)
