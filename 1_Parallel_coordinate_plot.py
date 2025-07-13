import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Sample data (replace with your actual data)
data = {
    'Batch': [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'Phase': [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 'Seed'],
    '<i>VR</i><sub>water</sub>': [0.62, 0.65, 0.6, 0.62, 0.6, 0.6, 0.62, 0.6, 0.55, 0.52, 0.61, 0.7, 0.7, 0.39, 0.77, 0.48, 1, 0.97],
    '[Au]': [3.85, 3.54, 3.97, 4.02, 3.5, 4.53, 3.97, 4.63, 3.64, 3.99, 4.7, 4.71, 3.86, 5, 0.6, 0.63, 0.5, 1.12],
    'SR:Au': [2.92, 2.93, 2.92, 2.94, 2.95, 2.87, 2.88, 2.81, 2.93, 2.43, 2.66, 2.88, 2.91, 0.5, 2.98, 3.9, 0.5, 3.09],
    '<i>V</i><sub>NaOH</sub>': [1.18, 0.79, 1.21, 1.01, 1.04, 1.13, 1.25, 1.3, 2, 1.71, 0.87, 0.15, 0.56, 1.65, 0.58, -0.1, 2, 0.38],
    '<i>V</i><sub>NaBH<sub>4</sub></sub>': [2.86, 2.63, 3, 3, 3, 3, 2.83, 2.87, 2.15, 2.89, 2.99, 2.72, 2.37, 0.28, 0.63, 0.72, 0.1, 0.62],
    '<i>OBj. value</i>': [2.25, 2.26, 2.41, 2.43, 2.46, 2.46, 2.32, 2.29, 1.37, 1.13, 1.95, 1.76, 1.92, -0.08, 0.05, 0.01, 0.01, 0.81]
}

df = pd.DataFrame(data)

Batch_colors = {
        1: '#FFBA08',   2: '#FCAF07',   3: '#FAA407',   4: '#F79906',
        5: '#F48E06',   6: '#EF7B05',   7: '#EA6504',   8: '#E44F03',
        9: '#DE3902',   10: '#D92301',  11: '#D30D00',  12: '#C70001',
        13: '#AF0105',  14: '#970208',  15: '#7F030C',  16: '#67040F',
        17: '#4F0513',  18: '#370617'
    }

# Create color mapping
def get_color_and_dash(row):

    if row['Phase'] == 'Seed':
        return 'gray', 'dash'
    elif row['Batch'] == 13:
        return Batch_colors.get(13), 'solid'
    else:
        # Different shades of teal for different Batchs
        # Interpolated colors from batch 1 ('#FFBA08') to batch 18 ('#370617')
        return Batch_colors.get(row['Batch'], '#0891b2'), 'dash'

# Set font for the plot
font_family = "Arial"
axis_font_size = 30
title_font_size = 20

# Create subplot figure
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    subplot_titles=('', ''),
    horizontal_spacing=0,
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

# Parameters for parallel coordinates (excluding <i>OBj. value</i>)
parallel_params = ['<i>VR</i><sub>water</sub>', '[Au]', 'SR:Au', '<i>V</i><sub>NaOH</sub>', '<i>V</i><sub>NaBH<sub>4</sub></sub>', 'Batch']

# Define parameter ranges for individual scaling
param_ranges = {
    '<i>VR</i><sub>water</sub>': (0.3, 1.0),
    '[Au]': (0.5, 5.0),
    'SR:Au': (0.5, 5.0),
    '<i>V</i><sub>NaOH</sub>': (-0.1, 2.0),
    '<i>V</i><sub>NaBH<sub>4</sub></sub>': (0.1, 3.0),
    'Batch' : (0.8, 18.2)  
}

# Create parallel coordinates plot with individual scaling
for i, row in df.iterrows():
    color, dash_style = get_color_and_dash(row)
    
    # Normalize each parameter to 0-1 scale for plotting, then scale to common y-axis range
    normalized_values = []
    for param in parallel_params:
        min_val, max_val = param_ranges[param]
        # Normalize to 0-1, then scale to common range (0-100 for better visualization)
        normalized_val = ((row[param] - min_val) / (max_val - min_val)) * 100 + 2
        normalized_values.append(normalized_val)
    
    # Add line trace for parallel coordinates
    fig.add_trace(
        go.Scatter(
            x=list(range(len(parallel_params))),
            y=normalized_values,
            mode='lines+markers',
            line=dict(color=color, width=2, dash=dash_style),
            marker=dict(size=6, color=color),
            # name=f"Batch {row['Batch']}, Phase {row['Phase']}",
            showlegend=False,
            # hovertemplate='<b>Batch %{customdata[0]}, Phase %{customdata[1]}</b><br>' +
            #              '<i>VR</i><sub>water</sub>: ' + str(row['<i>VR</i><sub>water</sub>']) + '<br>' +
            #              '[Au]: ' + str(row['[Au]']) + '<br>' +
            #              'SR:Au: ' + str(row['SR:Au']) + '<br>' +
            #              '<i>V</i><sub>NaOH</sub>: ' + str(row['<i>V</i><sub>NaOH</sub>']) + '<br>' +
            #              '<i>V</i><sub>NaBH<sub>4</sub></sub>: ' + str(row['<i>V</i><sub>NaBH<sub>4</sub></sub>']) + '<br>' +
            #              'GHSV: ' + str(row['GHSV']) + '<extra></extra>',
            # customdata=[[row['Batch'], row['Phase']]] * len(parallel_params)
        ),
        row=1, col=1
    )

# Customize parallel coordinates subplot with individual parameter scales
fig.update_xaxes(
    tickmode='array',
    tickvals=list(range(len(parallel_params))),
    ticktext=['<br><i>VR</i><sub>water</sub>', '<br>[Au]', '<br>SR:Au', '<br><i>V</i><sub>NaOH</sub>', '<br><i>V</i><sub>NaBH<sub>4</sub></sub>', ''],  # Custom tick labels for parameters
    title_text="",
    gridwidth=2,
    gridcolor="gray",
    tickfont=dict(size=axis_font_size, family=font_family, color="black"),
    zeroline=False,  # Remove the zero line which might interfere
    row=1, col=1
)

# Set y-axis range for normalized values
fig.update_yaxes(
    range=[0, 105],
    showticklabels=False,  # Hide the normalized tick labels
    title_text="",
    showgrid=False,  # Remove horizontal grid lines
    zeroline=False,  # Remove the zero line which might interfere
    row=1, col=1
)

# Add custom annotations for parameter scales
annotations = []
for i, param in enumerate(parallel_params):
    min_val, max_val = param_ranges[param]
    
    # Add max value at top with more space from parameter name
    annotations.append(
        dict(
            x=i/(len(parallel_params)-0.41)+0.052,
            y=1.08,
            xref="x domain", 
            yref="y domain",
            text=f"{max_val}",
            showarrow=False,
            xanchor="center",
            font=dict(size=axis_font_size, family=font_family, color="black")
        )
    )
    
    # Add min value at bottom with more space
    annotations.append(
        dict(
            x=i/(len(parallel_params)-0.41)+0.052,
            y=-0.08,
            xref="x domain",
            yref="y domain", 
            text=f"{min_val}",
            showarrow=False,
            xanchor="center",
            font=dict(size=axis_font_size, family=font_family, color="black")
        )
    )

# Create horizontal bar chart
# Sort data by <i>OBj. value</i> for better visualization
df_sorted = df.sort_values('Batch', ascending=True)

for i, row in df_sorted.iterrows():
    color, _ = get_color_and_dash(row)
    
    fig.add_trace(
        go.Bar(
            x=[row['<i>OBj. value</i>']],
            y=[f"Batch {row['Batch']}"],
            orientation='h',
            marker_color=color,
            # name=f"Batch {row['Batch']}, Phase {row['Phase']}",
            showlegend=False,
            # hovertemplate='<b>Batch %{text}, Phase %{customdata}</b><br>' +
            #              'STY<sub>HA</sub>: %{x:.2f}<extra></extra>',
            text=[row['Phase']],
            textfont=dict(size=axis_font_size, family=font_family, color="black"),
            textposition='outside'
            # customdata=[row['Phase']]
        ),
        row=1, col=2
    )

# Customize bar chart subplot
fig.update_xaxes(
    title_text="Objective Value",
    title_font_size=axis_font_size,
    title_font_family=font_family,
    title_font_color="black",
    range=[-0.3, 2.7],
    showgrid=True,
    gridwidth=1,
    gridcolor="lightgray",
    showticklabels=True,
    tickfont=dict(size=axis_font_size, family=font_family, color="black"),
    tickformat=".1f",
    row=1, col=2
)

fig.update_yaxes(
    title_text="",
    showgrid=False,  # Remove horizontal grid lines
    showticklabels=False,
    row=1, col=2
)

# Update overall layout
fig.update_layout(
    height=600,
    width=1600,
    template='plotly_white',
    margin=dict(t=50, b=10, l=8, r=8),
    annotations=annotations[0:10]  # Add the custom annotations
)

# Add custom legend
# Create dummy traces for legend
legend_data = [
    {'name': 'Batch 18', 'color': Batch_colors.get(18), 'dash': 'dash'},
    {'name': 'Batch 17', 'color': Batch_colors.get(17), 'dash': 'dash'},
    {'name': 'Batch 16', 'color': Batch_colors.get(16), 'dash': 'dash'},
    {'name': 'Batch 15', 'color': Batch_colors.get(15), 'dash': 'dash'},
    {'name': 'Batch 14', 'color': Batch_colors.get(14), 'dash': 'dash'},
    {'name': 'Batch 13', 'color': Batch_colors.get(13), 'dash': 'solid'},
    {'name': 'Batch 12', 'color': Batch_colors.get(12), 'dash': 'dash'},
    {'name': 'Batch 11', 'color': Batch_colors.get(11), 'dash': 'dash'},
    {'name': 'Batch 10', 'color': Batch_colors.get(10), 'dash': 'dash'},
    {'name': 'Batch 9', 'color': Batch_colors.get(9), 'dash': 'dash'},
    {'name': 'Batch 8', 'color': Batch_colors.get(8), 'dash': 'dash'},
    {'name': 'Batch 7', 'color': Batch_colors.get(7), 'dash': 'dash'},
    {'name': 'Batch 6', 'color': Batch_colors.get(6), 'dash': 'dash'},
    {'name': 'Batch 5', 'color': Batch_colors.get(5), 'dash': 'dash'},
    {'name': 'Batch 4', 'color': Batch_colors.get(4), 'dash': 'dash'},
    {'name': 'Batch 3', 'color': Batch_colors.get(3), 'dash': 'dash'},
    {'name': 'Batch 2', 'color': Batch_colors.get(2), 'dash': 'dash'},
    {'name': 'Seed', 'color': 'gray', 'dash': 'dash'},
]


# Show the plot
fig.write_html("hybrid_parallel_bar_plot.html", auto_open=True)
