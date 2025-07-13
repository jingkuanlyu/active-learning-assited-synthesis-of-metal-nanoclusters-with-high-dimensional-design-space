"""
Feature Correlation Analysis for Bayesian Optimization Data
===========================================================

This script performs comprehensive correlation analysis on the experimental parameters
and results from the Bayesian optimization process for L-pMBA synthesis.

Features analyzed:
- water_vol_ratio: Water volume ratio
- Au_Conc: Gold concentration
- SR_Au_ratio: SR:Au ratio  
- NaOH: NaOH volume
- NaBH4: NaBH4 volume
- result: Experimental result (target variable)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class CorrelationAnalyzer:
    """
    A comprehensive correlation analysis tool for experimental data
    """
    
    def __init__(self, data_path, target_column='result'):
        """
        Initialize the correlation analyzer
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing the data        target_column : str
            Name of the target variable column
        """
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.feature_names = {
            'water_vol_ratio': r'$VR_{water}$',
            'Au_Conc': r'$[Au]$',
            'SR_Au_ratio': r'$SR:Au$',
            'NaOH': r'$V_{NaOH}$',
            'NaBH4': r'$V_{NaBH_4}$',
            'result': 'Objective value'
        }
        self.load_data()
    
    def load_data(self):
        """Load and prepare the data"""
        try:
            # Load full dataset first
            full_df = pd.read_csv(self.data_path)
            print(f"Full dataset loaded: {full_df.shape[0]} samples, {full_df.shape[1]} features")
            print(f"All columns: {list(full_df.columns)}")
            
            # Select only the first 6 columns as requested
            self.df = full_df.iloc[:, :6].copy()
            print(f"\nFiltered to first 6 columns: {self.df.shape[0]} samples, {self.df.shape[1]} features")
            print(f"Analysis columns: {list(self.df.columns)}")
            
            # Update feature names to match actual column names
            actual_columns = list(self.df.columns)
            if len(actual_columns) >= 5:
                # Map to your known feature names if they match expected patterns
                expected_names = ['water_vol_ratio', 'Au_Conc', 'SR_Au_ratio', 'NaOH_tot', 'NaBH4', 'result']
                for i, col in enumerate(actual_columns):
                    if i < len(expected_names):
                        if col not in self.feature_names:
                            # Add the actual column name to feature_names
                            self.feature_names[col] = expected_names[i] if i < len(expected_names) else col
            
            # Check for missing values in the filtered dataset
            missing_values = self.df.isnull().sum()
            if missing_values.any():
                print("\nMissing values found in analysis columns:")
                for col, missing_count in missing_values[missing_values > 0].items():
                    display_name = self.feature_names.get(col, col)
                    print(f"  {display_name}: {missing_count} missing values")
            else:
                print("No missing values found in analysis columns.")
                
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found.")
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            return
            print(f"\nFiltered dataset: {self.df.shape[0]} samples, {self.df.shape[1]} features")
            print(f"Analysis columns: {list(self.df.columns)}")
            
            # Check for missing values in the filtered dataset
            missing_values = self.df.isnull().sum()
            if missing_values.any():
                print("\nMissing values found in analysis columns:")
                for col, missing_count in missing_values[missing_values > 0].items():
                    print(f"  {self.feature_names.get(col, col)}: {missing_count} missing values")
            else:
                print("No missing values found in analysis columns.")
                
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found.")
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    
    def validate_data(self):
        """Validate that all required columns are available for analysis"""
        if self.df is None:
            print("Error: No data loaded.")
            return False
        
        required_columns = list(self.feature_names.keys())
        available_columns = list(self.df.columns)
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {available_columns}")
            return False
        
        # Check if we have enough data points
        if len(self.df) < 3:
            print("Error: Need at least 3 data points for correlation analysis.")
            return False
        
        return True
    
    def basic_statistics(self):
        """Display basic statistics of the dataset"""
        print("\n" + "="*60)
        print("BASIC DATASET STATISTICS")
        print("="*60)
        
        # Numerical columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        print("\nDescriptive Statistics:")
        print(self.df[numeric_cols].describe().round(3))
        
        print(f"\nDataset shape: {self.df.shape}")
        print(f"Numerical features: {len(numeric_cols)}")
        
        return self.df[numeric_cols].describe()
    
    def pearson_correlation_analysis(self, threshold=0.7):
        """
        Perform Pearson correlation analysis
        
        Parameters:
        -----------
        threshold : float
            Threshold for identifying high correlations
        """
        print("\n" + "="*60)
        print("PEARSON CORRELATION ANALYSIS")
        print("="*60)
        
        # Calculate Pearson correlation matrix
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr(method='pearson')
        
        # Display correlation with target variable
        if self.target_column in corr_matrix.columns:
            target_corr = corr_matrix[self.target_column].abs().sort_values(ascending=False)
            print(f"\nFeatures ranked by correlation with {self.target_column}:")
            for feature, corr_val in target_corr.items():
                if feature != self.target_column:
                    feature_display = self.feature_names.get(feature, feature)
                    print(f"{feature_display:20s}: {corr_val:6.3f}")
          # Find highly correlated feature pairs
        high_corr_pairs = self.find_high_correlations(corr_matrix, threshold)
        
        if not high_corr_pairs.empty:
            print(f"\nFeature pairs with |correlation| > {threshold}:")
            for _, row in high_corr_pairs.iterrows():
                feat1_display = self.feature_names.get(row['Feature1'], row['Feature1'])
                feat2_display = self.feature_names.get(row['Feature2'], row['Feature2'])
                print(f"{feat1_display} ↔ {feat2_display}: {row['Correlation']:.3f}")
        else:
            print(f"\nNo feature pairs found with |correlation| > {threshold}")
        
        return corr_matrix, high_corr_pairs
    
    def find_high_correlations(self, corr_matrix, threshold=0.7):
        """Find pairs of features with correlation above threshold"""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr_pairs.append({
                        'Feature1': corr_matrix.columns[i],
                        'Feature2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        # Create DataFrame and sort only if we have data
        if high_corr_pairs:
            return pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['Feature1', 'Feature2', 'Correlation'])
    
    def correlation_with_significance(self):
        """Calculate correlations with statistical significance"""
        print("\n" + "="*60)
        print("CORRELATION WITH STATISTICAL SIGNIFICANCE")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_features = len(numeric_cols)
        
        # Initialize matrices
        corr_matrix = np.zeros((n_features, n_features))
        p_value_matrix = np.zeros((n_features, n_features))
        
        # Calculate correlations and p-values
        for i, feat1 in enumerate(numeric_cols):
            for j, feat2 in enumerate(numeric_cols):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_value_matrix[i, j] = 0.0
                else:
                    # Remove NaN values for correlation calculation
                    data1 = self.df[feat1].dropna()
                    data2 = self.df[feat2].dropna()
                    
                    # Find common indices
                    common_idx = data1.index.intersection(data2.index)
                    
                    if len(common_idx) > 2:  # Need at least 3 points for correlation
                        corr, p_val = pearsonr(self.df.loc[common_idx, feat1], 
                                             self.df.loc[common_idx, feat2])
                        corr_matrix[i, j] = corr
                        p_value_matrix[i, j] = p_val
                    else:
                        corr_matrix[i, j] = np.nan
                        p_value_matrix[i, j] = np.nan
        
        # Convert to DataFrames
        corr_df = pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)
        pvalue_df = pd.DataFrame(p_value_matrix, index=numeric_cols, columns=numeric_cols)
        
        # Display significant correlations
        print("\nSignificant correlations (p < 0.05) with target variable:")
        if self.target_column in pvalue_df.columns:
            for feature in numeric_cols:
                if feature != self.target_column:
                    p_val = pvalue_df.loc[feature, self.target_column]
                    corr_val = corr_df.loc[feature, self.target_column]
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    
                    feature_display = self.feature_names.get(feature, feature)
                    print(f"{feature_display:20s}: r={corr_val:6.3f}, p={p_val:.4f} {significance}")
        
        return corr_df, pvalue_df
    
    def compare_correlation_methods(self):
        """Compare Pearson, Spearman, and Kendall correlations"""
        print("\n" + "="*60)
        print("COMPARISON OF CORRELATION METHODS")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        methods = ['pearson', 'spearman', 'kendall']
        correlations = {}
        
        for method in methods:
            correlations[method] = self.df[numeric_cols].corr(method=method)
        
        # Compare correlations with target variable
        if self.target_column in numeric_cols:
            print(f"\nCorrelations with {self.target_column} using different methods:")
            print("-" * 50)
            
            for feature in numeric_cols:
                if feature != self.target_column:
                    feature_display = self.feature_names.get(feature, feature)
                    print(f"\n{feature_display}:")
                    for method in methods:
                        corr_val = correlations[method].loc[feature, self.target_column]
                        print(f"  {method.capitalize():10s}: {corr_val:6.3f}")
        
        return correlations
    
    def visualize_correlations(self, save_plots=True, output_dir=r'C:\Users\figures'): #MPA pMBSA
        """Create comprehensive correlation visualizations"""
        print("\n" + "="*60)
        print("CREATING CORRELATION VISUALIZATIONS")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(8,6))
        
        # 1. Correlation Heatmap
        # plt.subplot(2, 3, 1)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        
        # Create custom labels for the heatmap
        custom_labels = [self.feature_names.get(col, col) for col in corr_matrix.columns]
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8},
                    xticklabels=custom_labels, yticklabels=custom_labels)
        # Add a frame around the entire heatmap
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('black')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
           
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/comprehensive_correlation_analysis.png', 
                       dpi=600, bbox_inches='tight')
            print(f"Correlation plots saved to {output_dir}/comprehensive_correlation_analysis.png")
        
        plt.show()
        
        return fig
    
    def summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        # Overall statistics
        corr_values = corr_matrix.values
        corr_values = corr_values[np.triu_indices_from(corr_values, k=1)]  # Upper triangle only
        
        print(f"\nDataset Overview:")
        print(f"  • Total samples: {self.df.shape[0]}")
        print(f"  • Total features: {len(numeric_cols)}")
        print(f"  • Feature pairs analyzed: {len(corr_values)}")
        
        print(f"\nCorrelation Statistics:")
        print(f"  • Mean |correlation|: {np.mean(np.abs(corr_values)):.3f}")
        print(f"  • Max |correlation|: {np.max(np.abs(corr_values)):.3f}")
        print(f"  • Min |correlation|: {np.min(np.abs(corr_values)):.3f}")
        print(f"  • Std deviation: {np.std(corr_values):.3f}")
        
        # Correlation strength categories
        strong_corr = np.sum(np.abs(corr_values) > 0.7)
        moderate_corr = np.sum((np.abs(corr_values) > 0.3) & (np.abs(corr_values) <= 0.7))
        weak_corr = np.sum(np.abs(corr_values) <= 0.3)
        
        print(f"\nCorrelation Strength Distribution:")
        print(f"  • Strong correlations (|r| > 0.7): {strong_corr} ({strong_corr/len(corr_values)*100:.1f}%)")
        print(f"  • Moderate correlations (0.3 < |r| ≤ 0.7): {moderate_corr} ({moderate_corr/len(corr_values)*100:.1f}%)")
        print(f"  • Weak correlations (|r| ≤ 0.3): {weak_corr} ({weak_corr/len(corr_values)*100:.1f}%)")
        
        # Target variable analysis
        if self.target_column in corr_matrix.columns:
            target_corr = corr_matrix[self.target_column].drop(self.target_column)
            strongest_positive = target_corr.idxmax()
            strongest_negative = target_corr.idxmin()
            
            print(f"\nTarget Variable ({self.target_column}) Analysis:")
            print(f"  • Strongest positive correlation: {self.feature_names.get(strongest_positive, strongest_positive)} "
                  f"(r = {target_corr[strongest_positive]:.3f})")
            print(f"  • Strongest negative correlation: {self.feature_names.get(strongest_negative, strongest_negative)} "
                  f"(r = {target_corr[strongest_negative]:.3f})")
            print(f"  • Mean |correlation| with features: {np.mean(np.abs(target_corr)):.3f}")
        
        # Recommendations
        print(f"\nRecommendations:")
        high_corr_pairs = self.find_high_correlations(corr_matrix, 0.8)
        if not high_corr_pairs.empty:
            print(f"  • Consider feature selection: {len(high_corr_pairs)} highly correlated pairs found")
            print(f"  • Features that might be redundant:")
            for _, row in high_corr_pairs.head(3).iterrows():
                feat1_display = self.feature_names.get(row['Feature1'], row['Feature1'])
                feat2_display = self.feature_names.get(row['Feature2'], row['Feature2'])
                print(f"    - {feat1_display} and {feat2_display} (r = {row['Correlation']:.3f})")
        else:
            print(f"  • No highly correlated feature pairs found (threshold: 0.8)")
            print(f"  • Current feature set appears well-balanced for modeling")
        
        if self.target_column in corr_matrix.columns:
            weak_target_corr = target_corr[np.abs(target_corr) < 0.1]
            if len(weak_target_corr) > 0:
                print(f"  • Features with very weak target correlation ({len(weak_target_corr)} features):")
                for feat in weak_target_corr.index[:3]:  # Show top 3
                    feat_display = self.feature_names.get(feat, feat)
                    print(f"    - {feat_display} (r = {target_corr[feat]:.3f})")
            print("\n" + "="*60)
    
    def run_complete_analysis(self, correlation_threshold=0.7, save_plots=True):
        """Run the complete correlation analysis pipeline"""
        print("Starting Comprehensive Feature Correlation Analysis...")
        print("="*60)
        
        if self.df is None:
            print("Error: No data loaded. Please check the data path.")
            return None
        
        # Validate data before running analysis
        if not self.validate_data():
            print("Data validation failed. Please address the issues above.")
            return None
        
        print(f"Data validation successful. Proceeding with analysis of {len(self.df)} samples.")
        print(f"Analyzing columns: {list(self.df.columns)}")
        
        # Run all analysis steps
        self.basic_statistics()
        pearson_corr, high_corr_pairs = self.pearson_correlation_analysis(correlation_threshold)
        corr_with_sig, pvalues = self.correlation_with_significance()
        method_comparison = self.compare_correlation_methods()
        
        # Create visualizations
        self.visualize_correlations(save_plots=save_plots)
        
        # Generate summary report
        self.summary_report()
        
        return {
            'pearson_correlation': pearson_corr,
            'high_correlations': high_corr_pairs,
            'correlation_with_pvalues': corr_with_sig,
            'pvalues': pvalues,
            'method_comparison': method_comparison
        }


def main():
    """Main function to run the correlation analysis"""
    
    # Initialize the analyzer
    data_path = r'C:\Users\1_result_update.csv'
    analyzer = CorrelationAnalyzer(data_path, target_column='result')
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        correlation_threshold=0.7,
        save_plots=True
    )
    
    print("\nCorrelation analysis completed successfully!")
    print("Check the figures folder for visualization outputs.")
    
    return results


if __name__ == "__main__":
    # Run the analysis
    analysis_results = main()
