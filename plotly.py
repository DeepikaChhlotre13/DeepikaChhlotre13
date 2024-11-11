#Q1. Using the given dataset,to generate a 3D scatter plot to visualize the distribution of data points 
#    in a three-dimentional space.
'''np.random.seed(30)
     data = {
        'X': np.random.uniform(-10, 10, 300)
        'V': np.random.uniform(-10, 10, 300)
        'Z': np.random.uniform(-10, 10, 300)
     }
df = pd.DataFrame(data)'''

'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(30)

# Create the dataset
data = {
    'X': np.random.uniform(-10, 10, 300),
    'V': np.random.uniform(-10, 10, 300),
    'Z': np.random.uniform(-10, 10, 300)
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the data
ax.scatter(df['X'], df['V'], df['Z'], c='blue', marker='o', alpha=0.6)

# Label axes
ax.set_xlabel('X Axis')
ax.set_ylabel('V Axis')
ax.set_zlabel('Z Axis')

# Title
ax.set_title('3D Scatter Plot of Data Distribution')

# Show plot
plt.show()'''

#Q2. Using the Student Grades, create a violin plot to display the distribution of scores across different
#     grade categories.

'''np.random.seed(15)
data = {
     'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'],200),
     ''Score: np.random.randint(50, 100, 200)
     }
df = pd.DataFrame(data)

1. Using the sales data, generates a heatmap to visualize the variation in sales across different months and days.
np.random.seed(20)
data = {
    'Month': np.random.choice(['jan', 'Feb', 'Mar', 'Apr', 'May'],100)
    'Day': np.random.choice(range(1, 31),100),
    'Sales': np.random.randint(1000, 5000, 100)
    }
df = pd.DataFrame(data)'''

'''import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the data for grades and scores
np.random.seed(15)
data = {
    'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200),
    'Score': np.random.randint(50, 100, 200)
}
df_grades = pd.DataFrame(data)

# Plotting a Violin Plot
plt.figure(figsize=(10, 6))
sns.Violinplot(x='Grade', y='Score', data=df_grades, palette="muted")
plt.title('Distribution of Scores Across Different Grade Categories')
plt.xlabel('Grade')
plt.ylabel('Score')
plt.show()'''


# Generate the sales data
'''np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df_sales = pd.DataFrame(data)

# Pivot the DataFrame to get a matrix of Sales by Month and Day
sales_pivot = df_sales.pivot_table(index='Day', columns='Month', values='Sales', aggfunc='mean')

# Plotting the Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(sales_pivot, cmap='coolwarm', annot=True, fmt='.0f', cbar_kws={'label': 'Sales Amount'}, linewidths=0.5)
plt.title('Sales Variation Across Different Months and Days')
plt.xlabel('Month')
plt.ylabel('Day')
plt.show()'''

#Q3. Using the sales data, generate a heatmap to visualize the variation in sales across different months and Days.

'''np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)'''

'''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data generation
np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)

# Aggregating sales by month and day
pivot_table = df.pivot_table(values='Sales', index='Month', columns='Day', aggfunc='sum', fill_value=0)

# Plotting heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt="d", linewidths=.5)
plt.title("Sales Heatmap by Month and Day", fontsize=16)
plt.xlabel('Day of the Month', fontsize=12)
plt.ylabel('Month', fontsize=12)
plt.show()'''

    
#Q4. Using the given x and y data, generate a 3D surface plot to visualize the function z = sin(√x^2+y^2) 


'''x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

data = {
       'X': x.flatten(),
       'Y': y.flatten(),
       'Z': z.flatten()
}
df = pd.DataFrame(data)'''

'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the x and y grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Compute the function z = sin(sqrt(x^2 + y^2))
z = np.sin(np.sqrt(x**2 + y**2))

# Flatten the arrays to create a DataFrame
data = {
    'X': x.flatten(),
    'Y': y.flatten(),
    'Z': z.flatten()
}
df = pd.DataFrame(data)

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Set axis labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Title for the plot
ax.set_title(r'$z = \sin(\sqrt{x^2 + y^2})$')

# Show the plot
plt.show()'''

#Q5. Using the given dataset, create a bubble chart to represent each country's population (y-axis), GDP (xaxis), 
# and bubble size proportional to the population.

'''np.random.seed(25)
data = {
        'Country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
        'Population':
        np.random.randint(100, 1000, 5),
        'GDP': np.random.randint(500, 2000, 5)
}
df = pd.DataFrame(data)'''

'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given data
np.random.seed(25)
data = {
    'Country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
    'Population': np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000, 5)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the bubble chart
plt.figure(figsize=(10, 6))

# Scatter plot with bubble sizes proportional to population
plt.scatter(
    df['GDP'],              # x-axis (GDP)
    df['Population'],       # y-axis (Population)
    s=df['Population']*10,  # Bubble size proportional to population (scaled)
    alpha=0.5,              # Transparency
    c=df['Population'],     # Color by population
    cmap='viridis',         # Color map
    edgecolors="w",         # White border around bubbles
    linewidth=1
)

# Adding labels and title
for i, country in enumerate(df['Country']):
    plt.text(
        df['GDP'][i], df['Population'][i], country, 
        fontsize=12, ha='center', va='center'
    )

plt.title('Bubble Chart: Population vs GDP by Country', fontsize=16)
plt.xlabel('GDP (in billions)', fontsize=14)
plt.ylabel('Population (in millions)', fontsize=14)
plt.colorbar(label='Population')  # Colorbar to indicate population scale
plt.grid(True)

# Show the plot
plt.show()'''



 











