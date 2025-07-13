from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_gaussian_process, plot_convergence
from skopt.utils import create_result, expected_minimum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
from sklearn.inspection import partial_dependence


class BayesianOptimizer:
    def __init__(self, dimensions, n_initial_points=9, acq_func='gp_hedge', objective='max'):
        self.optimizer = Optimizer(dimensions, base_estimator='GP', acq_func=acq_func, acq_func_kwargs={'xi': 2, 'kappa': 2},
                                   n_initial_points=n_initial_points, random_state=50)
        self.objective = objective

    def get_next_parameters(self):
        return [self.optimizer.ask(n_points=9)]

    def update(self, x_list, y_list):
        if not isinstance(x_list[0], list):
            x_list = [x_list]
            y_list = [y_list]
        if self.objective == 'max':
            y_list = [y for y in y_list]
        self.optimizer.tell(x_list, y_list)

    def get_result(self):
        result = create_result(Xi=self.optimizer.Xi, yi=self.optimizer.yi, space=self.optimizer.space, 
                               rng=self.optimizer.rng, models=self.optimizer.models)
        return result


# Define the parameters
water_vol_ratio = Real(0.3, 1.0, name='water_vol_ratio')
Au_Conc = Real(0.5, 5, name='Au_Conc')
SR_Au_ratio = Real(0.5, 5.0, name='SR_Au_ratio')
NaOH = Real(-0.1, 2.0, name='NaOH') #1M NaOH 0-200 ul/ 1M HCl 0-10uL
# NaOH_tot = Real(-0.0625, 5.75, name='NaOH_tot')
NaBH4 = Real(0.1, 3.0, name='NaBH4') #5 mM NaBH4 10-300 ul


# Combine the parameters into a list to form the parameter space
dimensions = [water_vol_ratio, Au_Conc, SR_Au_ratio, NaOH, NaBH4]

# Initialize the optimizer
optimizer = BayesianOptimizer(dimensions, n_initial_points=9, objective='max')

# Read initial parameters and function results from CSV
data_df = pd.read_csv(r'C:\Users\1_result_update.csv')

# Separate the parameters and results
initial_params = data_df[[dim.name for dim in dimensions]].values.tolist()
function_results = data_df['result'].values.tolist()
# print(initial_params, function_results)

# Update the optimizer with the function results
optimizer.update(initial_params, function_results)

# Extract the trained Gaussian Process model
trained_model = optimizer.optimizer.models[-1]

# Transform the initial parameters to the model's input space
transformed_params = optimizer.get_result().space.transform(initial_params)

# Create a SHAP explainer for the Gaussian Process model
explainer = shap.Explainer(trained_model.predict, transformed_params)

# Compute SHAP values for the transformed parameters
shap_values = explainer(transformed_params)

custom_feature_names = [r'$VR_{water}$', r'$[Au]$', r'$SR:Au$', r'$V_{NaOH}$', r'$V_{NaBH_4}$']
custom_feature_names_units = [r'$VR_{water}$',
            r'$[Au]$ (mM)',
            r'$SR:Au$',
            r'$V_{NaOH}$ ($\times 10^2$ μL)',
            r'$V_{NaBH_4}$ ($\times 10^2$ μL)']

shap_values.feature_names = custom_feature_names

# Convert SHAP values to DataFrame and save to Excel
shap_df = pd.DataFrame(
    shap_values.values, 
    columns=custom_feature_names
)
shap_df.to_excel(r'C:\Users\4_shap_values.xlsx', index=False)
print("SHAP values saved to shap_values.xlsx")
print(shap_values)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, transformed_params, feature_names=custom_feature_names, show=False)

plt.xlabel('SHAP Value', fontsize=14)
plt.xticks(fontsize=12)

plt.savefig(r'C:\Users\shap_summary_plot_tot.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# Draw the SHAP waterfall plot for the selected instance
shap.waterfall_plot(shap_values[109], max_display=5, show=True)
shap.plots.scatter(shap_values[:, custom_feature_names[0]])

# Function to create spider plot with normalized average SHAP values
def create_spider_plot(shap_values, feature_names, figsize=(9, 9)):
    """
    Create a spider/radar plot using normalized average absolute SHAP values
    
    Parameters:
    shap_values: SHAP values object
    feature_names: List of feature names
    figsize: Figure size tuple
    """
    # Calculate average absolute SHAP values for each feature
    avg_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Normalize the values to 0-1 scale
    normalized_values = (avg_abs_shap - 0) / (avg_abs_shap.max() - 0)

    # Number of features
    N = len(feature_names)
    
    # Calculate angles for each feature on the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Add the first value at the end to close the polygon
    values = np.concatenate((normalized_values, [normalized_values[0]]))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Plot the polygon
    ax.plot(angles, values, 'o-', linewidth=2, label='Normalized Avg |SHAP|', color='#4a148c', markersize=6)
    ax.fill(angles, values, alpha=0.5, color='#b39ddb')
    
    # Add feature names
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=30)
    ax.tick_params(axis='x', which='major', pad=70)
    ax.set_rlabel_position(32)
    
    # Set y-axis properties
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.0'], fontsize=20)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=1, alpha=0.7)
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=1, alpha=0.7, dashes=(15, 9))
    
    # Add value labels on each point
    for angle, value, name in zip(angles[:-1], normalized_values, feature_names):
        ax.annotate(f'{value:.2f}', 
                   xy=(angle, value), 
                   xytext=(-5, -30), 
                   textcoords='offset points',
                   fontsize=18,
                   ha='center')
    
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    return fig, ax

# Create and display the spider plot
print("Creating spider plot with normalized average SHAP values...")
fig, ax = create_spider_plot(shap_values, custom_feature_names)
plt.savefig(r'C:\Users\spider_plot.png', dpi=600, bbox_inches='tight')
plt.show()



# Function to create PDP plot between any two parameters
def create_pdp_plot(feature_idx1, feature_idx2, colormap='RdYlBu_r', levels=50, 
                   vmin=-1.5, vmax=1.5, figsize=(8, 6), show_contour_lines=True, 
                   label_fontsize=20, colorbar_tick_interval=0.5):
   
    # Parameter ranges for denormalization
    param_ranges = {
        0: [0.3, 1.0],    # water_vol_ratio
        1: [0.5, 5.0],    # Au_Conc
        2: [0.5, 5.0],    # SR_Au_ratio
        3: [-0.1, 2.0],   # NaOH
        # 3: [-0.0625, 5.75], # NaOH_tot
        4: [0.1, 3.0]     # NaBH4
    }
    
    feature_indices = (feature_idx1, feature_idx2)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute partial dependence
    pdp = partial_dependence(
        trained_model, transformed_params, features=[feature_indices], 
        kind="average", grid_resolution=100
    )
    
    # Create meshgrid for smooth contour plot
    grid_values = pdp["grid_values"]
    XX, YY = np.meshgrid(grid_values[0], grid_values[1])
    Z = pdp.average[(0,)].T
    
    # Denormalize grid values back to original parameter ranges
    range1 = param_ranges[feature_idx1]
    range2 = param_ranges[feature_idx2]
    
    XX_denorm = XX * (range1[1] - range1[0]) + range1[0]
    YY_denorm = YY * (range2[1] - range2[0]) + range2[0]
        
    # Create smooth filled contour plot with denormalized values
    contour = ax.contourf(XX_denorm, YY_denorm, Z, levels=levels, cmap=colormap, 
                          norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    
    # Optional: Add contour lines for better visualization
    if show_contour_lines:
        ax.contour(XX_denorm, YY_denorm, Z, levels=10, colors='white', 
                   alpha=0.3, linewidths=0.5)
    
    # Set labels and tick sizes
    ax.set_xlabel(custom_feature_names_units[feature_idx1], fontsize=label_fontsize)
    ax.set_ylabel(custom_feature_names_units[feature_idx2], fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=label_fontsize)

    # Format x and y axis tick labels to 2 decimal places
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # # Apply tight layout first
    # plt.tight_layout()

    # Add colorbar with better positioning
    clb = plt.colorbar(contour, ax=ax, pad=0.04, shrink=1, aspect=15)
    clb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    clb.ax.tick_params(labelsize=label_fontsize-2)
    
    # Set specific tick intervals for colorbar
    tick_interval = colorbar_tick_interval
    ticks = np.arange(np.ceil(vmin / tick_interval) * tick_interval, 
                      np.floor(vmax / tick_interval) * tick_interval + tick_interval, 
                      tick_interval)
    clb.set_ticks(ticks)

    # Set colorbar title on the right side with vertical orientation
    clb.ax.text(3.4, 0.5, "Partial dependence", fontsize=label_fontsize-2, 
                rotation=90, va='center', ha='left', transform=clb.ax.transAxes)
    
    # plt.show()
    plt.subplots_adjust(
    left=0.14,    # Left margin
    bottom=0.15,  # Bottom margin (reduce to minimize whitespace)
    right=0.92,   # Right margin 
    top=0.92      # Top margin
)
    return fig, ax

# Individual plots (keeping original functionality)
fig, ax = create_pdp_plot(0, 1)
plt.savefig(r'C:\Users\pdp_water_vol_ratio_vs_Au_Conc.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(2, 1)
plt.savefig(r'C:\Users\pdp_SR_Au_vs_Au_Conc.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(4, 1)
plt.savefig(r'C:\Users\pdp_NaBH4_vs_Au_Conc.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(0, 2)
plt.savefig(r'C:\Users\pdp_water_vol_ratio_vs_SR_Au.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(0, 3)
plt.savefig(r'C:\Users\pdp_water_vol_ratio_vs_NaOH.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(0, 4)
plt.savefig(r'C:\Users\pdp_water_vol_ratio_vs_NaBH4.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(2, 3)
plt.savefig(r'C:\Users\pdp_SR_Au_vs_NaOH.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(2, 4)
plt.savefig(r'C:\Users\pdp_SR_Au_vs_NaBH4.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(4, 3)
plt.savefig(r'C:\Users\pdp_NaBH4_vs_NaOH.png', dpi=600)
plt.close()

fig, ax = create_pdp_plot(1, 3)
plt.savefig(r'C:\Users\pdp_Au_Conc_vs_NaOH.png', dpi=600)
plt.close()