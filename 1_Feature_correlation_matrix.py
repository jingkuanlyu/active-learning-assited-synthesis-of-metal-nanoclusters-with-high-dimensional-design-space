import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Load the data
data_df = pd.read_csv(r'C:\Users\3_category.csv')

# Create a custom palette with 6 colors
custom_palette = sns.color_palette("rocket_r")
columns_to_convert = ["water volume ratio", "Au concentration", "SR:Au ratio", "NaOH", "NaBH4"]
data_df[columns_to_convert] = data_df[columns_to_convert].astype('float64')

# Plot the data using the new classification
g = sns.pairplot(data_df, palette=custom_palette, x_vars=["water volume ratio", "Au concentration", "SR:Au ratio", "NaOH", "NaBH4"],
                 y_vars=["water volume ratio", "Au concentration", "SR:Au ratio", "NaOH", "NaBH4"], height=2, hue='Objective', diag_kind='kde')

# Adjust the size of each individual plot
fig = g.fig
fig.set_size_inches(16, 15)  # Set the overall figure size

# Define the limits for each variable
limits = {
    "water volume ratio": (0.25, 1.05),
    "Au concentration": (0, 5.1),
    "SR:Au ratio": (0, 5.1),
    "NaOH": (-0.2, 2.1),
    "NaBH4": (0, 3.1)
}

# Custom labels
fontsize = 25
custom_labels = {
    "water volume ratio": r"$VR_{water}$",
    "Au concentration": r"$[Au]$",
    "SR:Au ratio": r"$SR:Au$",
    "NaOH": r"$V_{NaOH}$",
    "NaBH4": r"$V_{NaBH_4}$"
}

# Function to format numbers to 2 significant figures
def format_to_2_sig_figs(x, pos):
    if x == 0:
        return '0'
    if abs(x) < 0.01:
        return f'{x:.1e}'  # Use scientific notation for very small numbers
    return f'{x:.2g}'

# Create formatter
formatter = FuncFormatter(format_to_2_sig_figs)

# Define manual tick positions for each variable
manual_ticks = {
    "water volume ratio": [0.4, 0.6, 0.8, 1],
    "Au concentration": [0, 1, 2, 3, 4, 5],
    "SR:Au ratio": [0, 1, 2, 3, 4, 5],
    "NaOH": [0, 0.5, 1, 1.5, 2],
    "NaBH4": [0, 1, 2, 3]
}

# Apply all customizations to axes in one loop
for ax in g.axes.flatten():
    current_xlabel = ax.get_xlabel()
    current_ylabel = ax.get_ylabel()
    
    # Set custom labels and limits
    if current_xlabel in custom_labels:
        ax.set_xlabel(custom_labels[current_xlabel], fontsize=fontsize)
        ax.set_xlim(limits[current_xlabel])
        # Set manual x-axis ticks
        ax.set_xticks(manual_ticks[current_xlabel])
    if current_ylabel in custom_labels:
        ax.set_ylabel(custom_labels[current_ylabel], fontsize=fontsize)
        ax.set_ylim(limits[current_ylabel])
        # Set manual y-axis ticks
        ax.set_yticks(manual_ticks[current_ylabel])
    
    # Set y-label position
    ax.yaxis.set_label_coords(-0.2, 0.5)
    
    # Set tick formatting
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

# Adjust the font size of the legend
if g._legend:
    g._legend.set_title(g._legend.get_title().get_text(), prop={'size': fontsize-5})
    for text in g._legend.get_texts():
        text.set_fontsize(fontsize-5)
    g._legend.set_bbox_to_anchor((1, 0.5))  # Move the legend to the top-right outside the plot

plt.savefig(r'C:\Users\3_feature correlation matrixb.png', dpi=1000)
# plt.show()