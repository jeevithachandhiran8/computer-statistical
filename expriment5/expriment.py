import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS

# Generate 100 random values between 10 to 30
x = np.linspace(-10, 30, 100)

# Generate 100 dependent values for the variable 'y' using the formula y = 10 + 4*x + e
e = np.random.normal(0, 1, 100)  # error term
y = 10 + 4*x + e

# Create a dataframe with X and Y values
df = pd.DataFrame({'X': x, 'Y': y})

# Plot the generated values using regplot() function
plt.figure(figsize=(8, 6))
plt.scatter(df['X'], df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X and Y')
plt.show()

# Perform OLS regression
ols_model = OLS(df['Y'], df['X']).fit()
print("OLS Regression Results:")
print(ols_model.summary())

# Define the likelihood function for MLE
def likelihood(params):
    beta0, beta1, sigma = params
    y_pred = beta0 + beta1*x
    residuals = y - y_pred
    likelihood = np.sum(np.log(np.sqrt(2*np.pi*sigma**2)) + (residuals**2)/(2*sigma**2))
    return likelihood

# Define the initial guess for MLE parameters
params0 = [2, 4, 1]  # initial guess for beta0, beta1, and sigma

# Perform MLE using L-BFGS-B algorithm
res = minimize(likelihood, params0, method="L-BFGS-B")

# Print MLE results
print("\nMLE Results:")
print(res)

# Calculate the standard deviation of residuals
residuals = y - (res.x[0] + res.x[1]*x)
sd_residuals = np.std(residuals)
print("\nStandard Deviation of Residuals:", sd_residuals)

# Compare MLE parameters with OLS parameters
print("\nComparison of MLE and OLS Parameters:")
print("MLE Parameters:", res.x)
print("OLS Parameters:", ols_model.params)
