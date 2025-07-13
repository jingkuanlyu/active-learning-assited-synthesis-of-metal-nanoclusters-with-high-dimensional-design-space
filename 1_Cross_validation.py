from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_gaussian_process, plot_convergence
from skopt.utils import create_result, expected_minimum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer

# Configuration Constants - Modify these for your use case
N_INITIAL_POINTS = 9  # Number of initial points for Bayesian optimization
N_CV_SPLITS = 5  # Number of cross-validation folds
RANDOM_STATE = 50  # Random state for reproducibility
GOOD_PERFORMANCE_THRESHOLD = 0.7  # R² threshold for considering a fold "good"

class BayesianOptimizer:
    def __init__(self, dimensions, n_initial_points=N_INITIAL_POINTS, acq_func='gp_hedge', objective='max'):
        self.optimizer = Optimizer(dimensions, base_estimator='GP', acq_func=acq_func, acq_func_kwargs={'xi': 2, 'kappa': 2},
                                   n_initial_points=n_initial_points, random_state=RANDOM_STATE)
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
optimizer = BayesianOptimizer(dimensions, n_initial_points=N_INITIAL_POINTS, objective='max')

# Read initial parameters and function results from CSV
data_df = pd.read_csv(r'C:\Users\1_result_update.csv')

# Separate the parameters and results
initial_params = data_df[[dim.name for dim in dimensions]].values.tolist()
function_results = data_df['result'].values.tolist()
# print(initial_params, function_results)

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
# print(optimizer.get_result().x_iters)
print(optimizer.get_result().models[-1])

# Conduct Stratified Cross-Validation with n=5
def analyze_optimal_bin_size(y, n_splits=5):
    """
    Analyze optimal bin size for stratified cross-validation with continuous targets
    """
    print("\n" + "="*60)
    print("BIN SIZE ANALYSIS FOR STRATIFIED CROSS-VALIDATION")
    print("="*60)
    
    # Key considerations for bin size
    print("KEY CONSIDERATIONS FOR BIN SIZE SELECTION:")
    print("-" * 45)
    print("1. Statistical Power: More bins = better stratification but fewer samples per bin")
    print("2. Sample Distribution: Bins should have roughly equal sample counts")
    print("3. Cross-validation Constraints: Each bin needs ≥ n_splits samples")
    print("4. Target Variability: Bins should capture meaningful differences")
    print("5. Practical Limits: Too many bins = overstratification")
    
    n_samples = len(y)
    print(f"\nDATASET CHARACTERISTICS:")
    print(f"  Total samples: {n_samples}")
    print(f"  Target range: [{np.min(y):.4f}, {np.max(y):.4f}]")
    print(f"  Target std: {np.std(y):.4f}")
    print(f"  CV folds: {n_splits}")
    
    # Test different bin sizes
    possible_bins = range(2, min(n_samples + 1, 11))  # Test 2 to 10 bins or n_samples
    bin_analysis = []
    
    print(f"\nBIN SIZE ANALYSIS:")
    print(f"{'Bins':<5} {'Min/Bin':<8} {'Feasible':<9} {'Balance':<8} {'Strat_Eff':<10} {'Recommendation':<15}")
    print("-" * 70)
    
    for n_bins in possible_bins:
        # Check if stratification is feasible
        min_samples_per_bin = n_samples // n_bins
        feasible = min_samples_per_bin >= n_splits
        
        # Calculate bin balance using KBinsDiscretizer
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        try:
            y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
            
            # Check bin balance
            bin_counts = np.bincount(y_binned.astype(int))
            balance_score = np.min(bin_counts) / np.max(bin_counts) if np.max(bin_counts) > 0 else 0
            
            # Calculate stratification effectiveness (within-bin variance / total variance)
            total_var = np.var(y)
            within_bin_var = 0
            for bin_id in range(n_bins):
                bin_mask = y_binned == bin_id
                if np.sum(bin_mask) > 1:
                    within_bin_var += np.var(y[bin_mask]) * np.sum(bin_mask)
            
            stratification_eff = 1 - (within_bin_var / (n_samples * total_var)) if total_var > 0 else 0
            
            # Recommendation logic
            if not feasible:
                recommendation = "❌ Too Few"
            elif balance_score < 0.5:
                recommendation = "⚠️ Unbalanced"
            elif stratification_eff > 0.8 and balance_score > 0.7:
                recommendation = "✅ Excellent"
            elif stratification_eff > 0.6 and balance_score > 0.6:
                recommendation = "✅ Good"
            elif stratification_eff > 0.4:
                recommendation = "⚪ Fair"
            else:
                recommendation = "❌ Poor"
            
        except:
            balance_score = 0
            stratification_eff = 0
            recommendation = "❌ Failed"
        
        bin_analysis.append({
            'n_bins': n_bins,
            'min_per_bin': min_samples_per_bin,
            'feasible': feasible,
            'balance_score': balance_score,
            'stratification_eff': stratification_eff,
            'recommendation': recommendation
        })
        
        print(f"{n_bins:<5} {min_samples_per_bin:<8} {str(feasible):<9} {balance_score:<8.3f} {stratification_eff:<10.3f} {recommendation:<15}")
    
    # Find optimal bin size
    feasible_bins = [b for b in bin_analysis if b['feasible']]
    if feasible_bins:
        # Score based on balance and stratification effectiveness
        for b in feasible_bins:
            b['overall_score'] = (b['balance_score'] * 0.4 + b['stratification_eff'] * 0.6)
        
        optimal_bin = max(feasible_bins, key=lambda x: x['overall_score'])
        
        print(f"\nRECOMMENDED BIN SIZE: {optimal_bin['n_bins']}")
        print(f"  Balance Score: {optimal_bin['balance_score']:.3f}")
        print(f"  Stratification Effectiveness: {optimal_bin['stratification_eff']:.3f}")
        print(f"  Overall Score: {optimal_bin['overall_score']:.3f}")
        
        # Additional guidance
        print(f"\nADDITIONAL GUIDANCE:")
        if optimal_bin['n_bins'] == 2:
            print("  • Using binary stratification (high/low values)")
            print("  • Consider if this captures sufficient target variability")
        elif optimal_bin['n_bins'] <= 3:
            print("  • Using few bins - good for small datasets")
            print("  • May miss subtle patterns in target distribution")
        elif optimal_bin['n_bins'] >= 8:
            print("  • Using many bins - captures fine-grained patterns")
            print("  • Risk of overstratification with small samples per bin")
        else:
            print("  • Balanced choice between stratification and sample size")
        
        return optimal_bin['n_bins'], bin_analysis
    else:
        print(f"\nWARNING: No feasible bin size found!")
        print(f"Consider using regular k-fold cross-validation instead.")
        return min(n_splits, n_samples), bin_analysis

def visualize_binning_strategies(y, bin_sizes=[2, 3, 5]):
    """
    Visualize different binning strategies
    """
    fig, axes = plt.subplots(1, len(bin_sizes), figsize=(5*len(bin_sizes), 4))
    if len(bin_sizes) == 1:
        axes = [axes]
    
    for i, n_bins in enumerate(bin_sizes):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        try:
            y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
            
            # Create histogram
            axes[i].hist(y, bins=20, alpha=0.7, color='lightblue', edgecolor='black', label='Original')
            
            # Show bin boundaries
            bin_edges = discretizer.bin_edges_[0]
            for edge in bin_edges[1:-1]:  # Exclude min and max
                axes[i].axvline(x=edge, color='red', linestyle='--', linewidth=2)
            
            # Color code the bins
            colors = plt.cm.Set3(np.linspace(0, 1, n_bins))
            for bin_id in range(n_bins):
                bin_mask = y_binned == bin_id
                y_bin = y[bin_mask]
                if len(y_bin) > 0:
                    axes[i].hist(y_bin, bins=20, alpha=0.6, color=colors[bin_id], 
                               label=f'Bin {bin_id} (n={len(y_bin)})')
            
            axes[i].set_title(f'{n_bins} Bins Strategy')
            axes[i].set_xlabel('Target Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Binning failed:\n{str(e)}', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{n_bins} Bins Strategy (Failed)')
    
    plt.tight_layout()
    plt.show()

def compare_cv_strategies(optimizer, bin_sizes=[2, 3, 5], n_splits=5):
    """
    Compare cross-validation performance with different bin sizes
    """
    print(f"\nCOMPARING CV STRATEGIES WITH DIFFERENT BIN SIZES:")
    print("=" * 60)
    
    X = np.array(optimizer.optimizer.Xi)
    y = np.array(optimizer.optimizer.yi)
    gp_model = optimizer.optimizer.models[-1]
    
    results_comparison = []
    
    # Regular K-Fold as baseline
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores_kfold = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        from copy import deepcopy
        fold_gp = deepcopy(gp_model)
        fold_gp.fit(X_train, y_train)
        y_pred, _ = fold_gp.predict(X_val, return_std=True)
        r2 = r2_score(y_val, y_pred)
        cv_scores_kfold.append(r2)
    
    results_comparison.append({
        'strategy': 'Regular K-Fold',
        'bins': 'N/A',
        'mean_r2': np.mean(cv_scores_kfold),
        'std_r2': np.std(cv_scores_kfold),
        'min_r2': np.min(cv_scores_kfold),
        'max_r2': np.max(cv_scores_kfold)
    })
    
    # Stratified with different bin sizes
    for n_bins in bin_sizes:
        try:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
            
            # Check if stratification is possible
            bin_counts = np.bincount(y_binned.astype(int))
            if np.min(bin_counts) < n_splits:
                print(f"  Skipping {n_bins} bins: insufficient samples per bin")
                continue
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores_stratified = []
            
            for train_idx, val_idx in skf.split(X, y_binned):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                from copy import deepcopy
                fold_gp = deepcopy(gp_model)
                fold_gp.fit(X_train, y_train)
                y_pred, _ = fold_gp.predict(X_val, return_std=True)
                r2 = r2_score(y_val, y_pred)
                cv_scores_stratified.append(r2)
            
            results_comparison.append({
                'strategy': f'Stratified ({n_bins} bins)',
                'bins': n_bins,
                'mean_r2': np.mean(cv_scores_stratified),
                'std_r2': np.std(cv_scores_stratified),
                'min_r2': np.min(cv_scores_stratified),
                'max_r2': np.max(cv_scores_stratified)
            })
            
        except Exception as e:
            print(f"  Failed for {n_bins} bins: {str(e)}")
    
    # Display comparison
    print(f"\nCROSS-VALIDATION STRATEGY COMPARISON:")
    print(f"{'Strategy':<20} {'Mean R²':<10} {'Std R²':<10} {'Range':<15} {'Consistency':<12}")
    print("-" * 75)
    
    for result in results_comparison:
        consistency = "High" if result['std_r2'] < 0.1 else "Medium" if result['std_r2'] < 0.2 else "Low"
        range_str = f"{result['min_r2']:.3f}-{result['max_r2']:.3f}"
        print(f"{result['strategy']:<20} {result['mean_r2']:<10.4f} {result['std_r2']:<10.4f} {range_str:<15} {consistency:<12}")
    
    # Recommendation
    best_strategy = min(results_comparison, key=lambda x: x['std_r2'])  # Lowest variability
    print(f"\nRECOMMENDED STRATEGY: {best_strategy['strategy']}")
    print(f"  Rationale: Lowest variability ({best_strategy['std_r2']:.4f}) indicates most consistent results")
    
    return results_comparison

# Conduct Stratified Cross-Validation with n=5
def perform_stratified_cv(optimizer, n_splits=5, auto_select_bins=True, manual_bins=None):
    """
    Perform stratified cross-validation on the trained Gaussian Process model
    with intelligent bin size selection
    """
    # Get the trained data
    X = np.array(optimizer.optimizer.Xi)
    y = np.array(optimizer.optimizer.yi)
    
    # Determine optimal bin size
    if auto_select_bins and manual_bins is None:
        print("AUTOMATIC BIN SIZE SELECTION:")
        optimal_bins, bin_analysis = analyze_optimal_bin_size(y, n_splits)
        visualize_binning_strategies(y, bin_sizes=[2, 3, optimal_bins] if optimal_bins not in [2, 3] else [2, 3, 5])
        
        # Compare different strategies
        comparison_results = compare_cv_strategies(optimizer, bin_sizes=[2, 3, optimal_bins], n_splits=n_splits)
        
        n_bins_to_use = optimal_bins
    elif manual_bins is not None:
        n_bins_to_use = manual_bins
        print(f"Using manually specified bin size: {n_bins_to_use}")
    else:
        n_bins_to_use = min(n_splits, len(y))
        print(f"Using default bin size: {n_bins_to_use}")
    
    # Convert continuous target to discrete bins for stratification
    # This is necessary because stratification requires categorical targets
    discretizer = KBinsDiscretizer(n_bins=n_bins_to_use, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    
    # Display binning results
    print(f"\nFINAL BINNING STRATEGY:")
    print(f"  Number of bins: {n_bins_to_use}")
    print(f"  Bin counts: {np.bincount(y_binned.astype(int))}")
    bin_edges = discretizer.bin_edges_[0]
    print(f"  Bin edges: {[f'{edge:.4f}' for edge in bin_edges]}")
    
    # Check bin adequacy
    bin_counts = np.bincount(y_binned.astype(int))
    min_bin_size = np.min(bin_counts)
    if min_bin_size < n_splits:
        print(f"⚠️  WARNING: Smallest bin has only {min_bin_size} samples (need ≥{n_splits} for {n_splits}-fold CV)")
        print("   Consider reducing number of bins or using regular k-fold CV")
    
    # Get the trained Gaussian Process model
    gp_model = optimizer.optimizer.models[-1]
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results
    cv_scores = []
    fold_results = []
    
    print(f"\nPerforming {n_splits}-fold Stratified Cross-Validation...")
    print("=" * 50)
    
    # First pass to identify best and worst folds
    temp_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        X_train_temp, X_val_temp = X[train_idx], X[val_idx]
        y_train_temp, y_val_temp = y[train_idx], y[val_idx]
        
        from copy import deepcopy
        temp_gp = deepcopy(gp_model)
        temp_gp.fit(X_train_temp, y_train_temp)
        y_pred_temp, _ = temp_gp.predict(X_val_temp, return_std=True)
        r2_temp = r2_score(y_val_temp, y_pred_temp)
        temp_results.append((fold + 1, r2_temp))
    
    # Find best and worst performing folds
    best_fold_num = max(temp_results, key=lambda x: x[1])[0]
    worst_fold_num = min(temp_results, key=lambda x: x[1])[0]
    
    print(f"Pre-analysis: Best fold = {best_fold_num}, Worst fold = {worst_fold_num}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        print(f"Fold {fold + 1}:")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create DataFrame for better visualization
        param_names = [dim.name for dim in dimensions]
        
        # Print detailed data for worst and best performing folds
        if fold + 1 == worst_fold_num:
            print(f"  *** DETAILED ANALYSIS OF FOLD {worst_fold_num} (WORST PERFORMING FOLD) ***")
            print(f"  Validation indices: {val_idx}")
            print(f"  Training indices: {train_idx}")
            
            print(f"\n  VALIDATION DATA (Fold {worst_fold_num}):")
            val_df = pd.DataFrame(X_val, columns=param_names)
            val_df['actual_result'] = y_val
            print(val_df.to_string(index=False, float_format='%.4f'))
            
            print(f"\n  TRAINING DATA (Fold {worst_fold_num}):")
            train_df = pd.DataFrame(X_train, columns=param_names)
            train_df['actual_result'] = y_train
            print(train_df.to_string(index=False, float_format='%.4f'))
            
            # Statistical comparison
            print(f"\n  STATISTICAL COMPARISON (Worst Fold):")
            print(f"  Validation set statistics:")
            for i, param in enumerate(param_names):
                val_mean = np.mean(X_val[:, i])
                val_std = np.std(X_val[:, i])
                train_mean = np.mean(X_train[:, i])
                train_std = np.std(X_train[:, i])
                print(f"    {param}: Val={val_mean:.3f}±{val_std:.3f}, Train={train_mean:.3f}±{train_std:.3f}")
            
            val_result_mean = np.mean(y_val)
            val_result_std = np.std(y_val)
            train_result_mean = np.mean(y_train)
            train_result_std = np.std(y_train)
            print(f"    Results: Val={val_result_mean:.3f}±{val_result_std:.3f}, Train={train_result_mean:.3f}±{train_result_std:.3f}")
        
        elif fold + 1 == best_fold_num:
            print(f"  *** DETAILED ANALYSIS OF FOLD {best_fold_num} (BEST PERFORMING FOLD) ***")
            print(f"  Validation indices: {val_idx}")
            print(f"  Training indices: {train_idx}")
            
            print(f"\n  VALIDATION DATA (Fold {best_fold_num}):")
            val_df = pd.DataFrame(X_val, columns=param_names)
            val_df['actual_result'] = y_val
            print(val_df.to_string(index=False, float_format='%.4f'))
            
            print(f"\n  TRAINING DATA (Fold {best_fold_num}):")
            train_df = pd.DataFrame(X_train, columns=param_names)
            train_df['actual_result'] = y_train
            print(train_df.to_string(index=False, float_format='%.4f'))
            
            # Statistical comparison
            print(f"\n  STATISTICAL COMPARISON (Best Fold):")
            print(f"  Validation set statistics:")
            for i, param in enumerate(param_names):
                val_mean = np.mean(X_val[:, i])
                val_std = np.std(X_val[:, i])
                train_mean = np.mean(X_train[:, i])
                train_std = np.std(X_train[:, i])
                print(f"    {param}: Val={val_mean:.3f}±{val_std:.3f}, Train={train_mean:.3f}±{train_std:.3f}")
            
            val_result_mean = np.mean(y_val)
            val_result_std = np.std(y_val)
            train_result_mean = np.mean(y_train)
            train_result_std = np.std(y_train)
            print(f"    Results: Val={val_result_mean:.3f}±{val_result_std:.3f}, Train={train_result_mean:.3f}±{train_result_std:.3f}")
        
        # Create a new GP model for this fold
        from copy import deepcopy
        fold_gp = deepcopy(gp_model)
        
        # Retrain on fold training data
        fold_gp.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred, y_std = fold_gp.predict(X_val, return_std=True)
        
        # Print predictions vs actual for worst and best folds
        if fold + 1 == worst_fold_num:
            print(f"\n  PREDICTIONS vs ACTUAL (Fold {worst_fold_num} - WORST):")
            pred_df = pd.DataFrame({
                'Actual': y_val,
                'Predicted': y_pred.flatten(),
                'Std_Dev': y_std.flatten(),
                'Error': y_val - y_pred.flatten(),
                'Abs_Error': np.abs(y_val - y_pred.flatten())
            })
            print(pred_df.to_string(index=False, float_format='%.4f'))
            print(f"  Mean Absolute Error: {np.mean(pred_df['Abs_Error']):.4f}")
            print(f"  Max Absolute Error: {np.max(pred_df['Abs_Error']):.4f}")
        
        elif fold + 1 == best_fold_num:
            print(f"\n  PREDICTIONS vs ACTUAL (Fold {best_fold_num} - BEST):")
            pred_df = pd.DataFrame({
                'Actual': y_val,
                'Predicted': y_pred.flatten(),
                'Std_Dev': y_std.flatten(),
                'Error': y_val - y_pred.flatten(),
                'Abs_Error': np.abs(y_val - y_pred.flatten())
            })
            print(pred_df.to_string(index=False, float_format='%.4f'))
            print(f"  Mean Absolute Error: {np.mean(pred_df['Abs_Error']):.4f}")
            print(f"  Max Absolute Error: {np.max(pred_df['Abs_Error']):.4f}")
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        # Store results
        fold_result = {
            'fold': fold + 1,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
        fold_results.append(fold_result)
        cv_scores.append(r2)
        
        print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  Mean prediction uncertainty: {np.mean(y_std):.6f}")
        print()
    
    # Calculate overall statistics
    cv_scores = np.array(cv_scores)
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print("Cross-Validation Summary:")
    print("=" * 30)
    print(f"Mean R² Score: {mean_score:.6f} ± {std_score:.6f}")
    print(f"Individual R² Scores: {cv_scores}")
    print(f"Best Fold R²: {np.max(cv_scores):.6f}")
    print(f"Worst Fold R²: {np.min(cv_scores):.6f}")
    
    # Create results DataFrame
    cv_df = pd.DataFrame(fold_results)
    
    return cv_df, mean_score, std_score

# Alternative: Time Series Cross-Validation (if your data has temporal order)
def perform_time_series_cv(optimizer, n_splits=5):
    """
    Perform time series cross-validation (walk-forward validation)
    Use this if your experiments were conducted sequentially in time
    """
    X = np.array(optimizer.optimizer.Xi)
    y = np.array(optimizer.optimizer.yi)
    
    n_samples = len(X)
    fold_size = n_samples // (n_splits + 1)
    
    cv_scores = []
    fold_results = []
    
    print(f"\nPerforming {n_splits}-fold Time Series Cross-Validation...")
    print("=" * 50)
    
    for fold in range(n_splits):
        # Expanding window: train on all data up to current point
        train_end = fold_size * (fold + 2)
        val_start = train_end
        val_end = min(train_end + fold_size, n_samples)
        
        if val_end <= val_start:
            break
            
        X_train, X_val = X[:train_end], X[val_start:val_end]
        y_train, y_val = y[:train_end], y[val_start:val_end]
        
        # Create and train GP model
        from copy import deepcopy
        fold_gp = deepcopy(optimizer.optimizer.models[-1])
        fold_gp.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred, y_std = fold_gp.predict(X_val, return_std=True)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        fold_result = {
            'fold': fold + 1,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
        fold_results.append(fold_result)
        cv_scores.append(r2)
        
        print(f"Fold {fold + 1}: R² = {r2:.6f}, RMSE = {rmse:.6f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"\nTime Series CV Mean R²: {mean_score:.6f} ± {std_score:.6f}")
    
    return pd.DataFrame(fold_results), mean_score, std_score

# Perform stratified cross-validation
try:
    cv_results, mean_cv_score, std_cv_score = perform_stratified_cv(optimizer, n_splits=N_CV_SPLITS, auto_select_bins=True)
    
    # Additional explanation of bin size impact
    print("\n" + "="*60)
    print("BIN SIZE IMPACT ON YOUR RESULTS")
    print("="*60)
    
    X = np.array(optimizer.optimizer.Xi)
    y = np.array(optimizer.optimizer.yi)
    
    print("UNDERSTANDING YOUR BINNING STRATEGY:")
    print("• Quantile-based binning ensures equal sample counts per bin")
    print("• This creates balanced folds for more reliable CV estimates")
    print("• Too few bins = poor stratification, too many bins = overfitting risk")
    
    # Show actual impact on your data
    current_bins = min(5, len(y))
    discretizer = KBinsDiscretizer(n_bins=current_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    
    print(f"\nYOUR DATA STRATIFICATION:")
    print(f"  Dataset size: {len(y)} samples")
    print(f"  Bins used: {current_bins}")
    
    for bin_id in range(current_bins):
        bin_mask = y_binned == bin_id
        bin_values = y[bin_mask]
        if len(bin_values) > 0:
            print(f"  Bin {bin_id}: {len(bin_values)} samples, mean={np.mean(bin_values):.4f}, range=[{np.min(bin_values):.4f}, {np.max(bin_values):.4f}]")
    
    # Recommendations for your specific case
    print(f"\nRECOMMENDATIONS FOR YOUR DATASET:")
    if len(y) < 20:
        print("• Small dataset: Consider 2-3 bins maximum")
        print("• Alternative: Use regular k-fold or leave-one-out CV")
    elif len(y) < 50:
        print("• Medium dataset: 3-5 bins is optimal")
        print("• Current stratification should work well")
    else:
        print("• Large dataset: Can use 5+ bins safely")
        print("• Consider more sophisticated stratification")
    
    if std_cv_score > 0.15:
        print("• High CV variability suggests bin size may need adjustment")
        print("• Try fewer bins (2-3) for more stable estimates")
    
    # Save cross-validation results
    cv_results.to_csv(r'C:\Users\cv_results.csv', index=False)
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(cv_results['fold'], cv_results['r2'])
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title('R² Score by Fold')
    plt.ylim([0, 1])
    
    plt.subplot(1, 2, 2)
    plt.bar(cv_results['fold'], cv_results['rmse'])
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('RMSE by Fold')
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Stratified CV failed (possibly due to insufficient data): {e}")
    print("Attempting time series cross-validation instead...")
    
    # Fallback to time series CV
    cv_results, mean_cv_score, std_cv_score = perform_time_series_cv(optimizer, n_splits=N_CV_SPLITS)
    print("Time Series CV completed successfully.")
    cv_results.to_csv(r'C:\Users\cv_results_timeseries.csv', index=False)

# Enhanced Model Reliability Analysis
def analyze_model_reliability(cv_results, mean_cv_score, std_cv_score):
    """
    Comprehensive analysis of model reliability based on CV results
    """
    print("\n" + "="*60)
    print("DETAILED MODEL RELIABILITY ANALYSIS")
    print("="*60)
    
    # Basic statistics
    r2_scores = cv_results['r2'].values
    min_r2 = np.min(r2_scores)
    max_r2 = np.max(r2_scores)
    cv_coefficient = std_cv_score / mean_cv_score if mean_cv_score > 0 else float('inf')
    
    print(f"Cross-Validation Performance Summary:")
    print(f"  Mean R²: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    print(f"  Coefficient of Variation: {cv_coefficient:.4f}")
    print(f"  Range: {min_r2:.4f} to {max_r2:.4f} (spread: {max_r2-min_r2:.4f})")
    
    # Reliability assessment
    print(f"\nReliability Assessment:")
    
    # Overall performance interpretation
    if mean_cv_score >= 0.8:
        performance_level = "Excellent"
    elif mean_cv_score >= 0.6:
        performance_level = "Good"
    elif mean_cv_score >= 0.4:
        performance_level = "Moderate"
    else:
        performance_level = "Poor"
    
    print(f"  Overall Performance: {performance_level} (R² = {mean_cv_score:.3f})")
    
    # Consistency assessment
    if cv_coefficient <= 0.1:
        consistency_level = "Very Consistent"
    elif cv_coefficient <= 0.2:
        consistency_level = "Consistent"
    elif cv_coefficient <= 0.3:
        consistency_level = "Moderately Consistent"
    else:
        consistency_level = "Inconsistent"
    
    print(f"  Model Consistency: {consistency_level} (CV = {cv_coefficient:.3f})")
    
    # Identify problematic folds
    mean_minus_std = mean_cv_score - std_cv_score
    problematic_folds = cv_results[cv_results['r2'] < mean_minus_std]
    
    if len(problematic_folds) > 0:
        print(f"  Problematic Folds: {len(problematic_folds)} fold(s) perform below mean-1std")
        for _, fold in problematic_folds.iterrows():
            print(f"    Fold {fold['fold']}: R² = {fold['r2']:.3f}, RMSE = {fold['rmse']:.3f}")
    
    # Overall reliability score
    reliability_score = mean_cv_score * (1 - cv_coefficient)
    print(f"\nOverall Reliability Score: {reliability_score:.3f}")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    if cv_coefficient > 0.25:
        print("  ⚠️  HIGH VARIABILITY DETECTED:")
        print("     - Consider collecting more training data")
        print("     - Check for outliers in your dataset")
        print("     - Consider feature engineering or parameter scaling")
        print("     - Try different GP kernel functions")
    
    if mean_cv_score < 0.6:
        print("  ⚠️  MODERATE PERFORMANCE:")
        print("     - Model may be underfitting")
        print("     - Consider more complex GP kernels")
        print("     - Check if important features are missing")
    
    if len(problematic_folds) > 1:
        print("  ⚠️  MULTIPLE POOR FOLDS:")
        print("     - Data distribution may be non-uniform")
        print("     - Consider stratification improvements")
        print("     - Check for temporal or batch effects")
    
    # Data adequacy check
    total_samples = cv_results['train_size'].iloc[0] + cv_results['val_size'].iloc[0]
    n_features = len([dim for dim in dimensions])  # Get number of features dynamically
    samples_per_param = total_samples / n_features
    
    print(f"\nData Adequacy Check:")
    print(f"  Total samples: {total_samples}")
    print(f"  Number of features: {n_features}")
    print(f"  Samples per parameter: {samples_per_param:.1f}")
    
    if samples_per_param < 5:
        print("  ⚠️  LIMITED DATA: Consider collecting more samples")
    elif samples_per_param < 10:
        print("  ⚪ ADEQUATE DATA: But more samples would improve reliability")
    else:
        print("  ✅ SUFFICIENT DATA: Good sample size for your parameter space")
    
    return reliability_score, performance_level, consistency_level

def plot_advanced_cv_diagnostics(cv_results, optimizer):
    """
    Create advanced diagnostic plots for cross-validation analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Advanced Cross-Validation Diagnostics', fontsize=16)
    
    # 1. R² distribution with confidence intervals
    r2_scores = cv_results['r2'].values
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    
    axes[0, 0].bar(cv_results['fold'], r2_scores, alpha=0.7, color='skyblue')
    axes[0, 0].axhline(y=mean_r2, color='red', linestyle='--', label=f'Mean: {mean_r2:.3f}')
    axes[0, 0].axhline(y=mean_r2 + std_r2, color='orange', linestyle=':', label=f'+1σ: {mean_r2 + std_r2:.3f}')
    axes[0, 0].axhline(y=mean_r2 - std_r2, color='orange', linestyle=':', label=f'-1σ: {mean_r2 - std_r2:.3f}')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score by Fold with Confidence Bounds')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    
    # 2. Performance vs Training Size
    axes[0, 1].scatter(cv_results['train_size'], cv_results['r2'], s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Training Set Size')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('Performance vs Training Size')
    
    # Add trend line
    z = np.polyfit(cv_results['train_size'], cv_results['r2'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(cv_results['train_size'], p(cv_results['train_size']), "r--", alpha=0.8)
    
    # 3. RMSE vs R² correlation
    axes[0, 2].scatter(cv_results['rmse'], cv_results['r2'], s=100, alpha=0.7, c=cv_results['fold'], cmap='viridis')
    axes[0, 2].set_xlabel('RMSE')
    axes[0, 2].set_ylabel('R² Score')
    axes[0, 2].set_title('RMSE vs R² Correlation')
    cbar = plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2])
    cbar.set_label('Fold')
    
    # 4. Residual analysis for the full model
    X = np.array(optimizer.optimizer.Xi)
    y = np.array(optimizer.optimizer.yi)
    gp_model = optimizer.optimizer.models[-1]
    y_pred_full, _ = gp_model.predict(X, return_std=True)
    residuals = y - y_pred_full
    
    axes[1, 0].scatter(y_pred_full, residuals, alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot (Full Model)')
    
    # 5. Q-Q plot for residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    # 6. Learning curve simulation
    train_sizes = np.array([fold['train_size'] for fold in cv_results.to_dict('records')])
    r2_scores = np.array([fold['r2'] for fold in cv_results.to_dict('records')])
    
    # Sort by training size
    sort_idx = np.argsort(train_sizes)
    train_sizes_sorted = train_sizes[sort_idx]
    r2_scores_sorted = r2_scores[sort_idx]
    
    axes[1, 2].plot(train_sizes_sorted, r2_scores_sorted, 'o-', linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('Training Set Size')
    axes[1, 2].set_ylabel('R² Score')
    axes[1, 2].set_title('Learning Curve')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run the enhanced analysis
reliability_score, performance_level, consistency_level = analyze_model_reliability(cv_results, mean_cv_score, std_cv_score)

# Comparative analysis between best and worst folds
def compare_best_worst_folds(optimizer, cv_results):
    """
    Compare characteristics between best and worst performing folds
    """
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: BEST vs WORST FOLD")
    print("="*60)
    
    # Get data
    X = np.array(optimizer.optimizer.Xi)
    y = np.array(optimizer.optimizer.yi)
    param_names = [dim.name for dim in dimensions]
    
    # Find best and worst folds
    best_fold_idx = cv_results['r2'].idxmax()
    worst_fold_idx = cv_results['r2'].idxmin()
    best_fold_num = cv_results.loc[best_fold_idx, 'fold']
    worst_fold_num = cv_results.loc[worst_fold_idx, 'fold']
    
    print(f"Best Fold: {best_fold_num} (R² = {cv_results.loc[best_fold_idx, 'r2']:.4f})")
    print(f"Worst Fold: {worst_fold_num} (R² = {cv_results.loc[worst_fold_idx, 'r2']:.4f})")
    print(f"Performance Gap: {cv_results.loc[best_fold_idx, 'r2'] - cv_results.loc[worst_fold_idx, 'r2']:.4f}")
    
    # Recreate fold splits to get exact data
    discretizer = KBinsDiscretizer(n_bins=min(5, len(y)), encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_val_data = None
    worst_val_data = None
    best_train_data = None
    worst_train_data = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        if fold + 1 == best_fold_num:
            best_val_data = (X[val_idx], y[val_idx], val_idx)
            best_train_data = (X[train_idx], y[train_idx], train_idx)
        elif fold + 1 == worst_fold_num:
            worst_val_data = (X[val_idx], y[val_idx], val_idx)
            worst_train_data = (X[train_idx], y[train_idx], train_idx)
    
    print(f"\nVALIDATION SET COMPARISON:")
    print(f"{'Parameter':<15} {'Best Fold':<15} {'Worst Fold':<15} {'Difference':<15}")
    print("-" * 65)
    
    # Compare validation sets
    for i, param in enumerate(param_names):
        best_mean = np.mean(best_val_data[0][:, i])
        worst_mean = np.mean(worst_val_data[0][:, i])
        diff = best_mean - worst_mean
        print(f"{param:<15} {best_mean:<15.4f} {worst_mean:<15.4f} {diff:<15.4f}")
    
    # Compare results
    best_result_mean = np.mean(best_val_data[1])
    worst_result_mean = np.mean(worst_val_data[1])
    result_diff = best_result_mean - worst_result_mean
    print(f"{'Results':<15} {best_result_mean:<15.4f} {worst_result_mean:<15.4f} {result_diff:<15.4f}")
    
    print(f"\nVALIDATION SET VARIABILITY:")
    print(f"{'Parameter':<15} {'Best Std':<15} {'Worst Std':<15} {'Ratio':<15}")
    print("-" * 65)
    
    for i, param in enumerate(param_names):
        best_std = np.std(best_val_data[0][:, i])
        worst_std = np.std(worst_val_data[0][:, i])
        ratio = worst_std / best_std if best_std > 0 else float('inf')
        print(f"{param:<15} {best_std:<15.4f} {worst_std:<15.4f} {ratio:<15.4f}")
    
    # Result variability
    best_result_std = np.std(best_val_data[1])
    worst_result_std = np.std(worst_val_data[1])
    result_ratio = worst_result_std / best_result_std if best_result_std > 0 else float('inf')
    print(f"{'Results':<15} {best_result_std:<15.4f} {worst_result_std:<15.4f} {result_ratio:<15.4f}")
    
    print(f"\nDATA RANGE ANALYSIS:")
    print(f"Best fold validation samples: {best_val_data[2]}")
    print(f"Worst fold validation samples: {worst_val_data[2]}")
    
    # Check if validation sets sample different regions of parameter space
    print(f"\nPARAMETER SPACE COVERAGE:")
    full_ranges = []
    for i, param in enumerate(param_names):
        param_min = np.min(X[:, i])
        param_max = np.max(X[:, i])
        full_range = param_max - param_min
        full_ranges.append(full_range)
        
        best_min = np.min(best_val_data[0][:, i])
        best_max = np.max(best_val_data[0][:, i])
        best_range = best_max - best_min
        best_coverage = best_range / full_range if full_range > 0 else 0
        
        worst_min = np.min(worst_val_data[0][:, i])
        worst_max = np.max(worst_val_data[0][:, i])
        worst_range = worst_max - worst_min
        worst_coverage = worst_range / full_range if full_range > 0 else 0
        
        print(f"{param:<15}: Best coverage = {best_coverage:.3f}, Worst coverage = {worst_coverage:.3f}")
    
    # Distance from dataset center
    dataset_center = np.mean(X, axis=0)
    
    best_distances = [np.linalg.norm(sample - dataset_center) for sample in best_val_data[0]]
    worst_distances = [np.linalg.norm(sample - dataset_center) for sample in worst_val_data[0]]
    
    print(f"\nDISTANCE FROM DATASET CENTER:")
    print(f"Best fold mean distance: {np.mean(best_distances):.4f} ± {np.std(best_distances):.4f}")
    print(f"Worst fold mean distance: {np.mean(worst_distances):.4f} ± {np.std(worst_distances):.4f}")
    
    # Key insights
    print(f"\nKEY INSIGHTS:")
    
    if np.mean(worst_distances) > np.mean(best_distances) * 1.2:
        print("• Worst fold samples are farther from dataset center (extrapolation issue)")
    elif np.mean(worst_distances) < np.mean(best_distances) * 0.8:
        print("• Worst fold samples are closer to dataset center (interpolation issue)")
    else:
        print("• Distance from center is similar between folds")
    
    if worst_result_std > best_result_std * 1.5:
        print("• Worst fold has much higher result variability (noisy data)")
    elif worst_result_std < best_result_std * 0.7:
        print("• Worst fold has lower result variability (limited range)")
    
    if abs(result_diff) > np.std(y):
        print(f"• Large difference in mean results ({result_diff:.3f}) suggests different data regimes")
    
    return {
        'best_fold': best_fold_num,
        'worst_fold': worst_fold_num,
        'performance_gap': cv_results.loc[best_fold_idx, 'r2'] - cv_results.loc[worst_fold_idx, 'r2'],
        'best_val_data': best_val_data,
        'worst_val_data': worst_val_data
    }

# Run comparative analysis
comparison_results = compare_best_worst_folds(optimizer, cv_results)

# Additional outlier analysis function
def detailed_outlier_analysis(optimizer, cv_results):
    """
    Perform detailed outlier analysis on the dataset
    """
    print("\n" + "="*60)
    print("DETAILED OUTLIER ANALYSIS")
    print("="*60)
    
    # Get all data
    X = np.array(optimizer.optimizer.Xi)
    y = np.array(optimizer.optimizer.yi)
    param_names = [dim.name for dim in dimensions]
    
    # Create full dataset DataFrame
    full_df = pd.DataFrame(X, columns=param_names)
    full_df['result'] = y
    full_df['experiment_id'] = range(len(y))
    
    print("COMPLETE DATASET:")
    print(full_df.to_string(index=False, float_format='%.4f'))
    
    # Identify outliers using multiple methods
    print(f"\nOUTLIER DETECTION:")
    
    # 1. Z-score method for results
    from scipy import stats
    z_scores = np.abs(stats.zscore(y))
    outliers_zscore = np.where(z_scores > 2)[0]  # |z| > 2
    
    print(f"  Z-score outliers (|z| > 2): {outliers_zscore}")
    if len(outliers_zscore) > 0:
        print("  Z-score outlier details:")
        for idx in outliers_zscore:
            print(f"    Exp {idx}: result={y[idx]:.4f}, z-score={z_scores[idx]:.3f}")
    
    # 2. IQR method for results
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = np.where((y < lower_bound) | (y > upper_bound))[0]
    
    print(f"  IQR outliers: {outliers_iqr}")
    print(f"    IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    if len(outliers_iqr) > 0:
        print("  IQR outlier details:")
        for idx in outliers_iqr:
            print(f"    Exp {idx}: result={y[idx]:.4f}")
    
    # 3. Parameter space outliers (Mahalanobis distance)
    try:
        from scipy.spatial.distance import mahalanobis
        cov_matrix = np.cov(X.T)
        cov_inv = np.linalg.inv(cov_matrix)
        mean_X = np.mean(X, axis=0)
        
        mahal_distances = []
        for i in range(len(X)):
            dist = mahalanobis(X[i], mean_X, cov_inv)
            mahal_distances.append(dist)
        
        mahal_distances = np.array(mahal_distances)
        mahal_threshold = np.percentile(mahal_distances, 95)  # Top 5% as outliers
        outliers_mahal = np.where(mahal_distances > mahal_threshold)[0]
        
        print(f"  Parameter space outliers (Mahalanobis > 95th percentile): {outliers_mahal}")
        if len(outliers_mahal) > 0:
            print("  Parameter space outlier details:")
            for idx in outliers_mahal:
                print(f"    Exp {idx}: Mahalanobis distance={mahal_distances[idx]:.3f}")
    except:
        print("  Parameter space outlier detection failed (possibly due to singular covariance matrix)")
    
    # 4. Cross-validation based outlier identification
    print(f"\nCROSS-VALIDATION BASED ANALYSIS:")
    
    # Identify which samples were in the worst performing fold
    worst_fold = cv_results.loc[cv_results['r2'].idxmin(), 'fold']
    print(f"  Worst performing fold: {worst_fold} (R² = {cv_results.loc[cv_results['r2'].idxmin(), 'r2']:.4f})")
    
    # Recreate the fold split to identify samples
    discretizer = KBinsDiscretizer(n_bins=min(5, len(y)), encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        if fold + 1 == worst_fold:
            print(f"  Samples in worst fold (validation): {val_idx}")
            print("  Worst fold sample details:")
            for idx in val_idx:
                print(f"    Exp {idx}: result={y[idx]:.4f}")
    
    # 5. Feature correlation analysis
    print(f"\nFEATURE CORRELATION ANALYSIS:")
    correlation_matrix = np.corrcoef(X.T)
    param_result_corr = []
    for i, param in enumerate(param_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        param_result_corr.append((param, corr))
        print(f"  {param} vs result correlation: {corr:.4f}")
    
    # Sort by absolute correlation
    param_result_corr.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Most influential parameter: {param_result_corr[0][0]} (|r| = {abs(param_result_corr[0][1]):.4f})")
    
    return full_df, outliers_zscore, outliers_iqr

# Run detailed outlier analysis
full_dataset, zscore_outliers, iqr_outliers = detailed_outlier_analysis(optimizer, cv_results)

# Final recommendation based on your specific results
print("\n" + "="*60)
print("FINAL ASSESSMENT FOR YOUR MODEL")
print("="*60)

print(f"Based on your cross-validation results:")
print(f"  • Your model shows {performance_level.lower()} predictive performance")
print(f"  • Model consistency is {consistency_level.lower()}")
print(f"  • Overall reliability score: {reliability_score:.3f}/1.0")

if reliability_score >= 0.6:
    recommendation = "✅ RELIABLE - Proceed with confidence"
    details = "Your model is sufficiently reliable for making predictions and guiding optimization decisions."
elif reliability_score >= 0.4:
    recommendation = "⚠️ MODERATELY RELIABLE - Use with caution"
    details = "The model has reasonable predictive power but shows some inconsistency. Consider collecting more data or investigating outliers."
else:
    recommendation = "❌ UNRELIABLE - Needs improvement"
    details = "The model shows significant instability. Strongly recommend collecting more data and/or revisiting the modeling approach."

print(f"\nRecommendation: {recommendation}")
print(f"Details: {details}")

# General insights based on CV results
print(f"\nGeneral Analysis of Cross-Validation Results:")
worst_fold = cv_results.loc[cv_results['r2'].idxmin(), 'fold']
worst_r2 = cv_results['r2'].min()
best_r2 = cv_results['r2'].max()
good_folds = len(cv_results[cv_results['r2'] > GOOD_PERFORMANCE_THRESHOLD])
total_folds = len(cv_results)

print(f"  • Fold {worst_fold} (R² = {worst_r2:.3f}) shows lowest performance - investigate this data subset")
print(f"  • {good_folds} out of {total_folds} folds perform well (R² > {GOOD_PERFORMANCE_THRESHOLD}), suggesting {'good' if good_folds/total_folds >= 0.8 else 'moderate'} general performance")
print(f"  • CV standard deviation ({std_cv_score:.3f}) indicates {'low' if std_cv_score < 0.1 else 'moderate' if std_cv_score < 0.2 else 'high'} data heterogeneity")
print(f"  • For optimization workflows, this performance level is {'excellent' if mean_cv_score > 0.8 else 'acceptable' if mean_cv_score > 0.6 else 'concerning'}")