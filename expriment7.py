import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# **Load the Wine dataset from sklearn**
wine = load_wine()
X = pd.DataFrame(data=wine.data, columns=wine.feature_names)
y = wine.target

# **Standardize the data**
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# **Apply LDA**
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# **Print the number of classes and features**
print("Number of classes and Features:")
print(f"Number of classes: {len(wine.target_names)}")
print(f"Number of features: {len(wine.feature_names)}")

# **Print the variance ratio**
print("\nVariance Ratio:")
print(lda.explained_variance_ratio_)

# **Create the LDA scatter plot**
plt.figure(figsize=(10, 5))
for i in range(len(wine.target_names)):
    plt.scatter(
        X_lda[y == i, 0], X_lda[y == i, 1], label=wine.target_names[i], s=10
    )
plt.xlabel("LDA_1")
plt.ylabel("LDA_2")
plt.legend(title="target", loc="upper left", bbox_to_anchor=(1.05, 1))
plt.title("LDA Scatter Plot")
plt.show()
