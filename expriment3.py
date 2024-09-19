import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Load the Iris dataset directly from seaborn
data = sns.load_dataset('iris')

# Count the total number of records
print(f"Total Number of Records: {data.shape[0]}")

# List of numerical attributes
attributes = data.select_dtypes(include=[np.number]).columns.tolist()

# Print Mean and Standard Deviation of each attribute
print("\nMean and Standard Deviation of Numerical Attributes:")
summary_stats = []
for attribute in attributes:
    mean = data[attribute].mean()
    std_dev = data[attribute].std()
    summary_stats.append([attribute, mean, std_dev])
    print(f"{attribute}: Mean = {mean}, Standard Deviation = {std_dev}")

# Convert summary statistics to a DataFrame and display as a table
summary_df = pd.DataFrame(summary_stats, columns=['Attribute', 'Mean', 'Standard Deviation'])
print("\nSummary Statistics Table:")
print(summary_df)

# Plotting Normal Distribution and Histogram together
for attribute in attributes:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[attribute], kde=False, stat="density", bins=20, edgecolor='black')
    
    # Plot the normal distribution curve
    mean = data[attribute].mean()
    std_dev = data[attribute].std()
    x = np.linspace(data[attribute].min(), data[attribute].max(), 100)
    p = norm.pdf(x, mean, std_dev)
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.title(f"Normal Distribution and Histogram of {attribute}")
    plt.xlabel(attribute)
    plt.ylabel('Density')
    plt.show()
