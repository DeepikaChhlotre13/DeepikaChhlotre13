#Q1.Create a scatter plot using matplotlib to visualize the relationship between two arrays ,x and y for the 
#   given data.      
#   x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   
#   y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]                                                      

'''import matplotlib.pyplot as plt

# Given data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]

# Create a scatter plot
plt.scatter(x, y, color='blue', label='Data Points')

# Add title and labels
plt.title('Scatter Plot of x vs y')
plt.xlabel('x values')
plt.ylabel('y values')

# Show a legend
plt.legend()

# Display the plot
plt.show()'''

#Q2.Generate a line plot to visualize the trend of values for the given data.

'''import matplotlib.pyplot as plt

# Given data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]

# Create a line plot
plt.plot(x, y, color='green', marker='o', label='Trend Line')

# Add title and labels
plt.title('Line Plot of x vs y')
plt.xlabel('x values')
plt.ylabel('y values')

# Show a legend
plt.legend()

# Display the plot
plt.show()'''


#Q3.Display a bar chart to represent the frequency of each item in the given array categories.
#    categories = ['A', 'B', 'C', 'D', 'E',]  
#    values = [25, 40, 30, 35, 20]


'''import matplotlib.pyplot as plt

# Given data
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 35, 20]

# Create the bar chart
plt.bar(categories, values, color='lightcoral')

# Add title and labels
plt.title('Frequency of Each Category')
plt.xlabel('Categories')
plt.ylabel('Values')

# Display the plot
plt.show()'''

#Q4.Create a histogram to visualize the distribution of values in the array data.
#    data = np.random.normal(0, 1, 1000)

'''import matplotlib.pyplot as plt
import numpy as np

# Generate data: 1000 values from a normal distribution with mean=0, std=1
data = np.random.normal(0, 1, 1000)

# Create the histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Add title and labels
plt.title('Histogram of Normally Distributed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Display the plot
plt.show()'''

#Q5.Show a pie chart to represent the percentage distribution of different sections in the array 'section'.
#   sections = ['section A', 'section B', 'section C', 'section D',]
#   sizes = [25, 30, 15, 40]

import matplotlib.pyplot as plt

# Given data
sections = ['section A', 'section B', 'section C', 'section D']
sizes = [25, 30, 15, 40]

# Create the pie chart
plt.pie(sizes, labels=sections, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral', 'gold'])

# Add title
plt.title('Percentage Distribution of Sections')

# Display the plot
plt.show()
