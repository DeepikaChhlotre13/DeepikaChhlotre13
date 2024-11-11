#Q1.Create a Bokeh plot displaying a sine wave. Set x-values from 0 to 10 and y-values as the sine of x.

'''import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file

# Generate data
x = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = np.sin(x)                 # y-values as sine of x

# Output to an HTML file
output_file("sine_wave.html")

# Create a figure object
p = figure(title="Sine Wave", x_axis_label='x', y_axis_label='sin(x)', width=800, height=400)

# Add a line to the plot
p.line(x, y, legend_label="sin(x)", line_width=2, color="blue")

# Show the plot
show(p)'''

#Q2.Create a Bokeh scatter plot using randomly generated x and y values. Use different sizes and colors for the
#markers based on the 'sizes' and 'colors' columns.

'''import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource

# Generate random data
n = 100  # Number of points
x = np.random.random(n) * 10  # Random x values between 0 and 10
y = np.random.random(n) * 10  # Random y values between 0 and 10
sizes = np.random.randint(5, 20, size=n)  # Random sizes for the markers
colors = np.random.choice(['red', 'green', 'blue', 'orange', 'purple'], size=n)  # Random colors

# Create a pandas DataFrame
df = pd.DataFrame({
    'x': x,
    'y': y,
    'sizes': sizes,
    'colors': colors
})

# Create a ColumnDataSource
source = ColumnDataSource(df)

# Set up the output file
output_file("scatter_plot.html")

# Create the scatter plot
p = figure(title="Random Scatter Plot with Varying Sizes and Colors", x_axis_label='X', y_axis_label='Y')

# Add scatter renderer (circle markers)
p.scatter(x='x', y='y', source=source, size='sizes', color='colors', legend_field='colors', fill_alpha=0.6)

# Customize the plot
p.legend.title = 'Color'
p.legend.location = 'top_left'

# Show the plot
show(p)'''

#Q3.Generate a Bokeh bar chart representing the counts of different fruits using the following dataset.
    
#fruits = ['Apples', 'Oranges', 'Bananas', 'Pears']
#counts = [20, 25, 30, 35]

'''from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource

# Data
fruits = ['Apples', 'Oranges', 'Bananas', 'Pears']
counts = [20, 25, 30, 35]

# Create a pandas DataFrame or directly use ColumnDataSource
data = {'fruits': fruits, 'counts': counts}
source = ColumnDataSource(data)

# Set up the output file
output_file("fruit_counts_bar_chart.html")

# Create the bar chart figure
p = figure(x_range=fruits, title="Fruit Counts", toolbar_location=None, tools="")

# Add bars
p.vbar(x='fruits', top='counts', width=0.5, source=source, legend_field="fruits", color="skyblue", alpha=0.7)

# Customize the plot
p.xaxis.axis_label = "Fruits"
p.yaxis.axis_label = "Count"
p.legend.title = "Fruits"
p.legend.location = "top_left"
p.grid.grid_line_color = "white"

# Show the plot
show(p)'''


#Q4. Create a Bokeh histogram to visualize the distribution of the given data.
#data_hist = np.random.randn(1000)
#hist, edges = np.histogram(data_hist, bins=30)

'''import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource

# Generate random data
data_hist = np.random.randn(1000)  # 1000 random numbers from a standard normal distribution

# Create the histogram data
hist, edges = np.histogram(data_hist, bins=30)

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(top=hist, left=edges[:-1], right=edges[1:], bottom=np.zeros_like(hist)))

# Set up the output file
output_file("histogram.html")

# Create the figure for the histogram
p = figure(title="Histogram of Random Data", 
           x_axis_label='Value', 
           y_axis_label='Frequency',
           tools="pan, box_zoom, reset")

# Add the histogram bars (quad glyphs)
p.quad(top='top', bottom='bottom', left='left', right='right', 
       source=source, color="skyblue", alpha=0.7)

# Customize the plot
p.grid.grid_line_color = "white"
p.legend.location = "top_left"

# Show the plot
show(p)'''

#Q5.Create a Bokeh heatmap using the provided dataset.
'''data_heatmap = np.random.rand(10, 10)
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
xx, yy = np.meshgrid(x, y)'''



'''import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.layouts import column
from bokeh.io import push_notebook

# Data generation
data_heatmap = np.random.rand(10, 10)  # 10x10 array of random values
x = np.linspace(0, 1, 10)  # X values (evenly spaced between 0 and 1)
y = np.linspace(0, 1, 10)  # Y values (evenly spaced between 0 and 1)

# Create a meshgrid for the X and Y values
xx, yy = np.meshgrid(x, y)

# Flatten the meshgrid and the data
x_flat = xx.flatten()
y_flat = yy.flatten()
z_flat = data_heatmap.flatten()

# Create a ColumnDataSource
from bokeh.models import ColumnDataSource
source = ColumnDataSource(data=dict(x=x_flat, y=y_flat, z=z_flat))

# Set up the output file
output_file("heatmap.html")

# Create a colormap (using Viridis palette) for the heatmap
mapper = linear_cmap(field_name='z', palette=Viridis256, low=min(z_flat), high=max(z_flat))

# Create the figure
p = figure(title="Heatmap", x_axis_label='X', y_axis_label='Y', 
           tools="hover,pan,box_zoom,reset", tooltips=[('Value', '@z')])

# Create the heatmap using the 'rect' glyph
p.rect(x='x', y='y', width=1/10, height=1/10, source=source, 
       color=mapper, line_color=None)

# Add a color bar to indicate the color scale
color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
p.add_layout(color_bar, 'right')

# Show the plot
show(p, notebook_handle=True)'''
