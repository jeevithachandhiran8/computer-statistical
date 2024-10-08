import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Step 2: Load the Iris Dataset
iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target'])
# Step 3: Standardize the Features
features = iris['feature_names']
x = data.loc[:, features].values
x = StandardScaler().fit_transform(x)

# Step 4: Perform PCA with n=2
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])

# Step 5: Plot the Data with the New Principal Components
plt.figure(figsize=(8, 6))
plt.scatter(principalDf['Principal Component 1'], principalDf['Principal Component 2'], c=data['target'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Component PCA')
plt.colorbar()
plt.show()

# Step 6: Display the Variance Among the 2 Components
print('Explained variance ratio:', pca.explained_variance_ratio_)
