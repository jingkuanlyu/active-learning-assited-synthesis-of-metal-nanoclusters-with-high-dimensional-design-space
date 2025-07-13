from skopt.space import Real
from skopt.sampler import Lhs
import pandas as pd

class BayesianOptimizer:
    def __init__(self, dimensions, n_initial_points=9):
        self.dimensions = dimensions
        self.n_initial_points = n_initial_points
        self.lhs = Lhs(criterion="maximin", iterations=10000)
        self.initial_samples = self.lhs.generate(dimensions, n_initial_points, random_state=0)

    def get_initial_samples(self):
        return [self.initial_samples]

# Define the parameters
water_vol_ratio = Real(0.3, 1.0, name='water_vol_ratio')
Au_Conc = Real(0.5, 5, name='Au_Conc')
SR_Au_ratio = Real(0.5, 5.0, name='SR_Au_ratio')
NaOH = Real(-0.1, 2.0, name='NaOH') #1M NaOH 0-200 ul/ 1M HCl 0-10uL
NaBH4 = Real(0.1, 3.0, name='NaBH4') #5 mM NaBH4 10-300 ul

# Combine the parameters into a list to form the parameter space
dimensions = [water_vol_ratio, Au_Conc, SR_Au_ratio, NaOH, NaBH4]

# Get initial parameters using Latin Hypercube Sampling
initial_params = BayesianOptimizer(dimensions, n_initial_points=9).get_initial_samples()[0]

# Convert initial parameters to DataFrame and write to CSV
initial_params_df = pd.DataFrame(initial_params, columns=[dim.name for dim in dimensions])
initial_params_df.to_csv(r'C:\Users\1_initial_params.csv', index=False)
