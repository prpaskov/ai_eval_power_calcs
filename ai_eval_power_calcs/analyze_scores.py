import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel
import warnings
from .run_power_calcs import calculate_minimum_detectable_effect, calculate_sample_size

class AnalyzeScores:
    def __init__(self, 
                 df: pd.DataFrame, 
                 test_cols: list, 
                 comparison_col: str, 
                 cluster_col: str = None, 
                 draws_col: str = None,
                 bonferroni_correction: bool = False, 
                 num_comparisons: int = 1,
                 alpha: float = 0.05,
                 beta: float = 0.20):
        """
        Initialize the analyzer for multiple paired t-tests.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        test_cols : list
            List of column names to test against the comparison column
        comparison_col : str
            Name of the comparison/baseline column
        cluster_col : str, optional
            Column name for clustering standard errors
        draws_col: str, optional
            Column name for draws
        bonferroni_correction : bool, default False
            Whether to apply Bonferroni correction across all tests
        num_comparisons : int, default 1
            Number of comparisons to run
        alpha : float, default 0.05
            Significance level
        beta : float, default 0.20
            Power level
        """
        self.df = df
        self.test_cols = test_cols if isinstance(test_cols, list) else [test_cols]
        self.comparison_col = comparison_col
        self.cluster_col = cluster_col
        self.draws_col = draws_col
        self.bonferroni_correction = bonferroni_correction
        self.num_comparisons = num_comparisons
        self.alpha = alpha
        self.beta = beta

    def _single_paired_ttest(self, col1: str, col2: str):
        """
        Perform a single paired t-test with optional clustering.
        
        Parameters:
        -----------
        col1 : str
            First column for comparison
        col2 : str
            Second column for comparison
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        # Clean data - remove rows with missing values
        cols_to_check = [col1, col2]
        if self.cluster_col is not None:
            cols_to_check.append(self.cluster_col)
        if self.draws_col is not None:
            cols_to_check.append(self.draws_col)
        
        clean_df = self.df[cols_to_check].dropna()
        
        if len(clean_df) == 0:
            raise ValueError(f"No valid observations after removing missing values for {col1} vs {col2}")
        
        x = clean_df[col1].values
        y = clean_df[col2].values
        differences = x - y
        differences_nonmiss = differences[~np.isnan(differences)]
        n = len(differences_nonmiss) ##total length of dataset
        
        mean_diff = np.mean(differences_nonmiss)
        
        # Standard paired t-test
        if self.cluster_col is None:
            # Regular paired t-test
            t_stat, p_value = ttest_rel(x, y)
            se_diff = np.std(differences, ddof=1) / np.sqrt(n)
            df_t = n - 1
            
        else:
            # Clustered standard errors
            clusters = clean_df[self.cluster_col].values
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters)
            
            if n_clusters == 1:
                warnings.warn(f"Only one cluster found for {col1} vs {col2}. Using regular standard errors.")
                t_stat, p_value = ttest_rel(x, y)
                se_diff = np.std(differences, ddof=1) / np.sqrt(n)
                df_t = n - 1
            else:
                # Calculate cluster-robust standard errors
                cluster_means = []
                cluster_sizes = []
                
                for cluster in unique_clusters:
                    cluster_mask = clusters == cluster
                    cluster_diff = differences[cluster_mask]
                    cluster_means.append(np.mean(cluster_diff))
                    cluster_sizes.append(len(cluster_diff))
                
                cluster_means = np.array(cluster_means)
                cluster_sizes = np.array(cluster_sizes)
                
                # Cluster-robust variance estimator
                total_obs = np.sum(cluster_sizes)
                cluster_var = np.sum(cluster_sizes * (cluster_means - mean_diff)**2) / (n_clusters - 1)
                se_diff = np.sqrt(cluster_var / total_obs)
                
                # Use cluster-adjusted degrees of freedom
                df_t = n_clusters - 1
                t_stat = mean_diff / se_diff
                p_value = 2 * stats.t.sf(np.abs(t_stat), df_t), 6
        
        # Apply Bonferroni correction if requested
        alpha_adj = self.alpha
        if self.bonferroni_correction:
            alpha_adj = self.alpha / self.num_comparisons
            p_value_adj = min(p_value * self.num_comparisons, 1.0)
        else:
            p_value_adj = p_value
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - alpha_adj/2, df_t)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Effect size (Cohen's d for paired samples)
        cohens_d = mean_diff / np.std(differences, ddof=1)
        
        # Calculate baseline column standard deviation for MDE calculations
        baseline_std = np.std(clean_df[col2], ddof=1)
        
        # Calculate MDE for current sample size
        # For paired t-test: omega_squared ≈ variance of differences, sigma terms = 0 for single samples
        omega_squared = np.var(differences, ddof=1)
        sigma_1_squared = 0  # Single sample per observation
        sigma_2_squared = 0  # Single sample per observation
        var_1 = np.var(clean_df[col1], ddof=1)
        var_2 = np.var(clean_df[col2], ddof=1)

        col1_subset = self.df[[self.draws_col, col1]].dropna(subset=[col1])
        K_1_df = col1_subset[self.draws_col].max()

        col2_subset = self.df[[self.draws_col, col2]].dropna(subset=[col2])
        K_2_df = col2_subset[self.draws_col].max()

        current_mde = calculate_minimum_detectable_effect(
            omega_squared=omega_squared,
            sigma_1_squared=sigma_1_squared,
            sigma_2_squared=sigma_2_squared,
            alpha=self.alpha,
            beta=self.beta,
            K_1=K_1_df,
            K_2=K_2_df,
            n=n,
            multiple_tests=self.num_comparisons if self.bonferroni_correction else 1
        )
        
        # Calculate sample sizes needed for 0.5 SD and 1.0 SD effects
        n_dict = {}
        for d in [.125, .25, .5, 1]:
            temp_n = calculate_sample_size(
                omega_squared=omega_squared,
                sigma_1_squared=sigma_1_squared,
                sigma_2_squared=sigma_2_squared,
                var_1=var_1,
                var_2=var_2,
                alpha=self.alpha,
                beta=self.beta,
                K_1=K_1_df,
                K_2=K_2_df,
                delta=d * baseline_std,
                multiple_tests=self.num_comparisons if self.bonferroni_correction else 1
            )
            n_dict[d] = temp_n
        
        results = {
            'n_observations': n,
            'K_judge': K_1_df,
            'K_baseline': K_2_df,
            'mean_difference': mean_diff,
            'std_difference': np.std(differences, ddof=1),
            't_statistic': t_stat,
            'alpha_adjusted': alpha_adj,
            'p_value': np.round(p_value, 4),
            'p_value_adjusted': np.round(p_value_adj, 4) if self.bonferroni_correction else None,
            'degrees_of_freedom': df_t,
            'standard_error': se_diff,
            'confidence_interval': (ci_lower, ci_upper),
            'alpha_level': alpha_adj,
            'significant': p_value_adj < alpha_adj,
            'clustering_applied': self.cluster_col is not None,
            'n_clusters': len(np.unique(clean_df[self.cluster_col])) if self.cluster_col else None,
            'bonferroni_applied': self.bonferroni_correction,
            'comparison': f"{col1} vs {col2}",
            # Power analysis additions
            'baseline_std': baseline_std,
            'current_mde': current_mde,
            'n_1_12_sd': n_dict[0.125],
            'n_1_4_sd': n_dict[0.25],
            'n_1_2_sd': n_dict[0.5],
            'n_1_sd': n_dict[1],
            'power_analysis_alpha': self.alpha,
            'power_analysis_beta': self.beta,
            'power_analysis_power': 1 - self.beta,
            'current_mde_relative_to_baseline_sd': current_mde / baseline_std,
        }
        
        return results

    def run_multiple_tests(self):
        """
        Run paired t-tests for all test columns against the comparison column.
        
        Returns:
        --------
        dict
            Dictionary with test column names as keys and test results as values
        """
        all_results = {}
        
        for test_col in self.test_cols:
            if test_col == self.comparison_col:
                warnings.warn(f"Skipping {test_col} as it's the same as comparison column")
                continue
                
            try:
                results = self._single_paired_ttest(test_col, self.comparison_col)
                all_results[test_col] = results
            except Exception as e:
                warnings.warn(f"Error testing {test_col} vs {self.comparison_col}: {str(e)}")
                all_results[test_col] = {'error': str(e)}
        
        return all_results

    def print_results(self, results_dict):
        """Pretty print the test results for multiple tests"""
        print("=== Multiple Paired T-Test Results ===")
        print(f"Comparison column: {self.comparison_col}")
        print(f"Number of tests: {len(results_dict)}")
        if self.bonferroni_correction:
            print(f"Bonferroni correction applied (α = {self.alpha}/{self.num_comparisons} = {self.alpha/self.num_comparisons:.4f})")
        print("="*60)
        
        for test_col, results in results_dict.items():
            if 'error' in results:
                print(f"\n{test_col} vs {self.comparison_col}: ERROR - {results['error']}")
                continue
                
            print(f"\n{test_col} vs {self.comparison_col}:")
            print(f"  Sample size: {results['n_observations']}")
            if results['clustering_applied']:
                print(f"  Number of clusters: {results['n_clusters']}")
            
            print(f"  Mean difference: {results['mean_difference']:.4f}")
            print(f"  Standard error: {results['standard_error']:.4f}")
            print(f"  t-statistic: {results['t_statistic']:.4f}")
            print(f"  Degrees of freedom: {results['degrees_of_freedom']}")
            
            print(f"  P-value: {results['p_value']:.6f}")
            if results['bonferroni_applied']:
                print(f"  P-value (Bonferroni adj.): {results['p_value_adjusted']:.6f}")
            
            print(f"  95% CI: ({results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f})")
            print(f"  Cohen's d: {results['cohens_d']:.4f}")
            
            significance = "Yes" if results['significant'] else "No"
            print(f"  Significant: {significance}")

    def get_summary_table(self, results_dict):
        """
        Create a summary table of all test results.
        
        Parameters:
        -----------
        results_dict : dict
            Results from run_multiple_tests()
            
        Returns:
        --------
        pd.DataFrame
            Summary table with key statistics
        """
        summary_data = []
        
        for test_col, results in results_dict.items():
            if 'error' in results:
                continue
                
            row = {
                'test_column': test_col,
                'n_obs': results['n_observations'],
                'K_judge': results['K_judge'],
                'K_baseline': results['K_baseline'],
                'mean_diff': results['mean_difference'],
                'std_error': results['standard_error'],
                't_stat': results['t_statistic'],
                'p_value': results['p_value'],
                'p_adj': results['p_value_adjusted'] if results['bonferroni_applied'] else None,
                'alpha_adjusted': results['alpha_adjusted'],
                'bonferroni_applied': results['bonferroni_applied'],
                'bonferroni_alpha': self.alpha/self.num_comparisons,
                'ci_lower': results['confidence_interval'][0],
                'ci_upper': results['confidence_interval'][1],
                'significant': results['significant'],
                'current_mde': results['current_mde'],
                'n_1_12_sd': results['n_1_12_sd'], ##holding K constant with what was used in the data analyzed here
                'n_1_4_sd': results['n_1_4_sd'],  ##holding K constant with what was used in the data analyzed here
                'n_1_2_sd': results['n_1_2_sd'], ##holding K constant with what was used in the data analyzed here
                'n_1_sd': results['n_1_sd'], ##holding K constant with what was used in the data analyzed here
                'current_mde_to_baseline_sd': results['current_mde_relative_to_baseline_sd'],

            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

    def export_latex_table(self, results_dict, output_file=None, caption="Test Results", label="tab:test_results", 
                        decimal_places=4):
        """
        Export test results to a LaTeX table format.
        
        Parameters:
        -----------
        results_dict : dict
            Results from run_multiple_tests()
        output_file : str, optional
            Path to save the LaTeX file. If None, returns LaTeX string
        caption : str, default "Test Results"
            Table caption
        label : str, default "tab:test_results"
            Table label for referencing
        decimal_places : int, default 4
            Number of decimal places to display
            
        Returns:
        --------
        str
            LaTeX table code
        """
        # Filter out error results
        valid_results = {k: v for k, v in results_dict.items() if 'error' not in v}
        
        if not valid_results:
            raise ValueError("No valid results to export")
        
        # Start building LaTeX table
        latex_lines = []
        
        # Table header
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append(f"\\caption{{{caption}}}")
        latex_lines.append(f"\\label{{{label}}}")
        
        # Determine number of columns
        n_cols = 4  # Test Column, Mean Diff, SE, t-stat
        if any(v.get('bonferroni_applied', False) for v in valid_results.values()):
            n_cols += 1  # Add p-value adjusted column
            col_spec = "lcccc"
            header = "Test Column & Mean Diff. & SE & t-stat & p-value (adj.) \\\\"
        else:
            col_spec = "lccc"
            header = "Test Column & Mean Diff. & SE & t-stat \\\\"
        
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\toprule")
        latex_lines.append(header)
        latex_lines.append("\\midrule")
        
        # Add data rows
        for test_col, results in valid_results.items():
            mean_diff = f"{results['mean_difference']:.{decimal_places}f}"
            se = f"{results['standard_error']:.{decimal_places}f}"
            t_stat = f"{results['t_statistic']:.{decimal_places}f}"
            
            # Clean up test column name for LaTeX (escape underscores)
            clean_test_col = test_col.replace('_', '\\_')
            
            if results.get('bonferroni_applied', False):
                p_adj = f"{results['p_value_adjusted']:.{decimal_places}f}"
                if results['significant']:
                    p_adj += "*"  # Add asterisk for significance
                row = f"{clean_test_col} & {mean_diff} & {se} & {t_stat} & {p_adj} \\\\"
            else:
                row = f"{clean_test_col} & {mean_diff} & {se} & {t_stat} \\\\"
            
            latex_lines.append(row)
        
        # Table footer
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        
        # Add notes if applicable
        if any(v.get('bonferroni_applied', False) for v in valid_results.values()):
            comparison_col_clean = self.comparison_col.replace('_', '\\_')
            alpha_adj = self.alpha / self.num_comparisons
            latex_lines.append("\\begin{tablenotes}")
            latex_lines.append("\\small")
            latex_lines.append(f"\\item Note: All tests compare against {comparison_col_clean}. ")
            latex_lines.append(f"Bonferroni correction applied (α = {alpha_adj:.4f}). ")
            latex_lines.append("* indicates significance at adjusted α level.")
            latex_lines.append("\\end{tablenotes}")
        
        latex_lines.append("\\end{table}")
        
        # Join all lines
        latex_code = "\n".join(latex_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(latex_code)
            print(f"LaTeX table saved to: {output_file}")
        
        return latex_code


    # Enhanced version with more customization options
    def export_latex_table_enhanced(self, results_dict, output_file=None, 
                                caption="Test Results", label="tab:test_results",
                                decimal_places=4, include_ci=False, 
                                include_sample_size=False, star_significance=True):
        """
        Export test results to a comprehensive LaTeX table format with more options.
        
        Parameters:
        -----------
        results_dict : dict
            Results from run_multiple_tests()
        output_file : str, optional
            Path to save the LaTeX file. If None, returns LaTeX string
        caption : str, default "Test Results"
            Table caption
        label : str, default "tab:test_results"
            Table label for referencing
        decimal_places : int, default 4
            Number of decimal places to display
        include_ci : bool, default False
            Whether to include confidence intervals
        include_sample_size : bool, default False
            Whether to include sample size column
        star_significance : bool, default True
            Whether to add stars for significant results
            
        Returns:
        --------
        str
            LaTeX table code
        """
        # Filter out error results
        valid_results = {k: v for k, v in results_dict.items() if 'error' not in v}
        
        if not valid_results:
            raise ValueError("No valid results to export")
        
        # Start building LaTeX table
        latex_lines = []
        
        # Add necessary packages at the top (as comment for user reference)
        latex_lines.append("% Required LaTeX packages:")
        latex_lines.append("% \\usepackage{booktabs}")
        latex_lines.append("% \\usepackage{threeparttable}")
        latex_lines.append("")
        
        # Table header with threeparttable for notes
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\begin{threeparttable}")
        latex_lines.append(f"\\caption{{{caption}}}")
        latex_lines.append(f"\\label{{{label}}}")
        
        # Build column specification and header dynamically
        columns = ["Test Column", "Mean Diff.", "SE", "t-stat"]
        col_spec = "lccc"
        
        if include_sample_size:
            columns.insert(1, "N")
            col_spec = "l" + "c" + col_spec[1:]
        
        if include_ci:
            columns.append("95\\% CI")
            col_spec += "c"

        # Add p-value column
        bonferroni_applied = any(v.get('bonferroni_applied', False) for v in valid_results.values())
        if bonferroni_applied:
            columns.append("p-value (adj.)")
        else:
            columns.append("p-value")
        col_spec += "c"
        
        header = " & ".join(columns) + " \\\\"
        
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\toprule")
        latex_lines.append(header)
        latex_lines.append("\\midrule")
        
        # Add data rows
        for test_col, results in valid_results.items():
            # Clean up test column name for LaTeX
            clean_test_col = test_col.replace('_', '\\_')
            
            # Build row data
            row_data = [clean_test_col]
            
            if include_sample_size:
                row_data.append(str(results['n_observations']))
            
            # Core statistics
            mean_diff = f"{results['mean_difference']:.{decimal_places}f}"
            se = f"{results['standard_error']:.{decimal_places}f}"
            t_stat = f"{results['t_statistic']:.{decimal_places}f}"
            
            row_data.extend([mean_diff, se, t_stat])
            
            if include_ci:
                ci_lower = f"{results['confidence_interval'][0]:.{decimal_places}f}"
                ci_upper = f"{results['confidence_interval'][1]:.{decimal_places}f}"
                ci_str = f"[{ci_lower}, {ci_upper}]"
                row_data.append(ci_str)
            
            # P-value with significance stars
            if bonferroni_applied:
                p_val = f"{results['p_value_adjusted']:.{decimal_places}f}"
            else:
                p_val = f"{results['p_value']:.{decimal_places}f}"
            
            if star_significance and results['significant']:
                if results['p_value'] < 0.001:
                    p_val += "***"
                elif results['p_value'] < 0.01:
                    p_val += "**"
                elif results['p_value'] < 0.05:
                    p_val += "*"
            
            row_data.append(p_val)
            
            # Join row and add to table
            row = " & ".join(row_data) + " \\\\"
            latex_lines.append(row)
        
        # Table footer
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        
        # Add comprehensive notes
        latex_lines.append("\\begin{tablenotes}")
        latex_lines.append("\\small")
        
        comparison_col_clean = self.comparison_col.replace('_', '\\_')
        latex_lines.append(f"\\item Note: All tests are paired comparisons against {comparison_col_clean}. ")
        
        if bonferroni_applied:
            alpha_adj = self.alpha / self.num_comparisons
            latex_lines.append(f"Bonferroni correction applied across {self.num_comparisons} tests (α = {alpha_adj:.4f}). ")
        
        if star_significance:
            latex_lines.append("*** p < 0.001, ** p < 0.01, * p < 0.05. ")
        
        if any(v.get('clustering_applied', False) for v in valid_results.values()):
            latex_lines.append("Cluster-robust standard errors applied. ")
        
        latex_lines.append("\\end{tablenotes}")
        latex_lines.append("\\end{threeparttable}")
        latex_lines.append("\\end{table}")
        
        # Join all lines
        latex_code = "\n".join(latex_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(latex_code)
            print(f"Enhanced LaTeX table saved to: {output_file}")
        
        return latex_code
