import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset directly from seaborn
data = sns.load_dataset('iris')

# Statistical description of the dataset
description = data.describe()
print("Statistical Description of Iris Dataset:")
print(description)

# Select one attribute for further analysis
attribute = 'petal_width'

# Box plot for the selected attribute
plt.figure(figsize=(8, 6))
sns.boxplot(y=data[attribute])
plt.title(f"Box Plot of {attribute}")
plt.ylabel(attribute)
plt.show()

# Compare the statistical description with the box plot
print(f"\nStatistical Description of {attribute}:")
print(description[attribute])

# Dependency curve of the selected attribute
plt.figure(figsize=(8, 6))
sns.kdeplot(data[attribute], shade=False)
plt.title(f"Dependency Curve of {attribute}")
plt.xlabel(attribute)
plt.ylabel('Density')
plt.show()
