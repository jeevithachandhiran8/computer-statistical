# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, probplot
# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Select an attribute (for example, 'sepal length (cm)')
attribute = 'sepal length (cm)'
data = iris_df[attribute]
# i. Print the first five records
print("First five records of the selected attribute:")
print(iris_df.head())

# ii. Plot the combined Histogram and Kernel Density Estimation (KDE)
plt.figure(figsize=(8, 6))
sns.histplot(data, kde=True, bins=10, color='blue')
plt.title(f'Histogram and KDE of {attribute}')
plt.xlabel(attribute)
plt.ylabel('Count')
plt.show()
# iii. Print the Probability Plot
plt.figure(figsize=(6, 4))
probplot(data, dist="norm", plot=plt)
plt.title(f'Probability Plot of {attribute}')
plt.show()
# iv. Calculate skewness
data_skewness = skew(data)
print(f"Skewness of {attribute}: {data_skewness}")
# v. Calculate kurtosis
data_kurtosis = kurtosis(data)
print(f"Kurtosis of {attribute}: {data_kurtosis}")
