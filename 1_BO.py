from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_gaussian_process, plot_convergence
from skopt.utils import create_result, expected_minimum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BayesianOptimizer:
    def __init__(self, dimensions, n_initial_points=9, acq_func='gp_hedge', objective='max'):
        self.optimizer = Optimizer(dimensions, base_estimator='GP', acq_func=acq_func, acq_func_kwargs={'xi': 2, 'kappa': 2},  #'xi': 0.01, 'kappa': 0.01 during exploitation phase
                                   n_initial_points=n_initial_points, random_state=50)
        self.objective = objective

    def get_next_parameters(self):
        return [self.optimizer.ask(n_points=9)]

    def update(self, x_list, y_list):
        if not isinstance(x_list[0], list):
            x_list = [x_list]
            y_list = [y_list]
        if self.objective == 'max':
            y_list = [-1 * y for y in y_list]
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

# Update the optimizer with the function results
optimizer.update(initial_params, function_results)

# Get new parameters
new_params = optimizer.get_next_parameters()[0]

#Normalize the new parameters
normalized_params = np.zeros_like(new_params)

for i, params in enumerate(new_params):
    for j, value in enumerate(params):
        min_val, max_val = optimizer.get_result().space.bounds[j]
        normalized_params[i, j] = (value - min_val) / (max_val - min_val)

# Predict the result for the new parameters
predicted_means, predicted_stds = optimizer.optimizer.models[-1].predict(normalized_params.tolist(), return_std=True)
predicted_means = [-mean for mean in predicted_means]

# Combine the new parameters, predicted means, and predicted standard deviations
output_data = [param_list + [mean, std] for param_list, mean, std in zip(new_params, predicted_means, predicted_stds)]

# Convert to DataFrame and write to CSV
output_df = pd.DataFrame(output_data, columns=[dim.name for dim in dimensions] + ['predicted_mean', 'predicted_std'])
output_df.to_csv(r'C:\Users\1_new_params.csv', index=False)

print(expected_minimum(optimizer.get_result(), random_state=20))
plot_convergence(optimizer.get_result())
plt.show()
