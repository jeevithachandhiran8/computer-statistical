import matplotlib.pyplot as plt

# Sample data
product = ['computer', 'monitor', 'laptop', 'printer', 'tablet']
quantity = [320, 450, 300, 120, 280]

# Create a figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Bar Plot
ax1.bar(product, quantity, color='orange')
ax1.set_title('Bar Plot\nstore inventory')
ax1.set_xlabel('product')
ax1.set_ylabel('quantity')

# Horizontal Bar Plot (H Plot)
ax2.barh(product, quantity, color='orange')
ax2.set_title('H Plot\nstore inventory')
ax2.set_xlabel('quantity')
ax2.set_ylabel('product')

# Display the plots
plt.tight_layout()
plt.show()
