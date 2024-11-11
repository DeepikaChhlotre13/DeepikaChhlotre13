#Q1.Create a scatter plot to visualize the relationship between two variables, by generating a synthetic dataset.

'''import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)  # Set seed for reproducibility
x = np.random.rand(100) * 10  # 100 random values for x between 0 and 10
y = 2 * x + np.random.randn(100) * 2  # y is approximately 2x with some noise

# Create the scatter plot
plt.scatter(x, y, color='blue', alpha=0.6, label='Data Points')

# Add title and labels
plt.title('Scatter Plot of Synthetic Data')
plt.xlabel('X Values')
plt.ylabel('Y Values')

# Show a legend
plt.legend()

# Display the plot
plt.show()'''

#Q2. Generate a dataset of random numbers. Visualize the distribution of a numerical variable.


'''import numpy as np
import matplotlib.pyplot as plt

# Generate random data (normal distribution)
np.random.seed(0)  # Set seed for reproducibility
data = np.random.randn(1000)  # 1000 random values from a standard normal distribution (mean=0, std=1)

# Create a histogram to visualize the distribution
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# Add title and labels
plt.title('Distribution of Randomly Generated Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Display the plot
plt.show()'''

#Q3. Create the dataset representing categories and their corresponding values. Compare different
#    categories based on numerical values.

'''import matplotlib.pyplot as plt

# Dataset representing categories and their corresponding numeric values
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
values = [23, 45, 18, 60, 35]

# Create a bar chart to compare categories based on their values
plt.bar(categories, values, color='lightblue', edgecolor='black')

# Add title and labels
plt.title('Comparison of Categories Based on Numeric Values')
plt.xlabel('Categories')
plt.ylabel('Values')

# Display the plot
plt.show()'''

#Q4.Generate a dataset with categories and numerical values. Visualize the distribution of a numerical variable 
#   across different categories.

'''import numpy as np
import matplotlib.pyplot as plt

# Generate random data for categories
np.random.seed(42)  # Set seed for reproducibility

# Categories
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']

# Generate numerical values for each category
data = {
    'Category A': np.random.normal(10, 2, 50),  # Mean=10, Std=2, 50 samples
    'Category B': np.random.normal(15, 3, 50),  # Mean=15, Std=3, 50 samples
    'Category C': np.random.normal(20, 4, 50),  # Mean=20, Std=4, 50 samples
    'Category D': np.random.normal(25, 5, 50),  # Mean=25, Std=5, 50 samples
    'Category E': np.random.normal(30, 6, 50),  # Mean=30, Std=6, 50 samples
}

# Prepare data for plotting: Values from all categories
values = [data[category] for category in categories]

# Create a box plot to visualize the distribution of numerical values
plt.boxplot(values, labels=categories, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='red'))

# Add title and labels
plt.title('Distribution of Numerical Values Across Categories')
plt.xlabel('Categories')
plt.ylabel('Numerical Value')

# Display the plot
plt.show()'''


#Q5. Generate a synthetic dataset with correlated features. Visualize the correlation matrix of a
#    dataset using a heatmap.
'''import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with correlated features
n_samples = 1000

# Feature 1: Normally distributed random data
x1 = np.random.normal(0, 1, n_samples)

# Feature 2: Correlated with Feature 1 (e.g., x2 = x1 * 0.7 + noise)
x2 = 0.7 * x1 + np.random.normal(0, 0.5, n_samples)

# Feature 3: Correlated with both x1 and x2 (e.g., x3 = x1 + x2 + noise)
x3 = x1 + x2 + np.random.normal(0, 0.2, n_samples)

# Create a DataFrame to hold the features
data = pd.DataFrame({'Feature 1': x1, 'Feature 2': x2, 'Feature 3': x3})

# Calculate the correlation matrix
corr_matrix = data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', cbar=True)
plt.title('Correlation Matrix Heatmap')
plt.show()'''
