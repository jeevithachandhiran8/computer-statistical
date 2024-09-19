import plotly.express as px

# Load the Iris dataset from plotly.express and name it as 'px'
df = px.data.iris()

# 1. Print the first 10 instances.
print("First 10 instances:")
print(df.head(10))

# 2. Print the Number of Rows and Columns of the dataset.
rows, columns = df.shape
print(f"\nTotal number of rows: {rows}")
print(f"Total number of columns: {columns}")

# 3. Print the Column names [Attribute Names] of the dataset.
print("\nColumn names [Attribute Names] of the dataset:")
print(df.columns.tolist())

# 4. Mean of all the Numerical Attributes
print("\nMean of all the Numerical Attributes:")
print(f"Sepal length - Mean = {df['sepal_length'].mean()}")
print(f"Sepal width - Mean = {df['sepal_width'].mean()}")
print(f"Petal length - Mean = {df['petal_length'].mean()}")
print(f"Petal width - Mean = {df['petal_width'].mean()}")
