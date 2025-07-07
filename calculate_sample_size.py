"""
This module provides tools for statistical analysis of language model evaluations,
including variance decomposition, power analysis, and sample size calculations.

##say more about assumptions of dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. DATA GENERATION AND SIMULATION
# =============================================================================

def create_sample_data(n_question_types=1, n_questions_per_type=50, n_draws=3, 
                      add_correlation=True, seed=42):
    """
    Create sample data in long format for LLM evaluation analysis.
    
    Parameters:
    -----------
    n_question_types : int
        Number of question types/clusters (default: 1)
    n_questions_per_type : int
        Number of questions per type (default: 50)
    n_draws : int
        Number of draws/samples per question (default: 3)
    add_correlation : bool
        Whether to add correlation between scorers (default: True)
    seed : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    pd.DataFrame : Long format data with columns:
        - question_id: Unique identifier for each question
        - question_draw: Draw number (1, 2, 3, ...)
        - question_type: Type/cluster of question
        - scorer1_id: Always 'model_A'
        - scorer2_id: Always 'model_B'
        - score1: Score from model A
        - score2: Score from model B
    """
    np.random.seed(seed)
    
    data = []
    question_id = 0
    
    # Question types (clusters)
    question_types = [f'type_{i}' for i in range(n_question_types)]
    
    for question_type in question_types:
        # Each question type has different base difficulty
        type_difficulty_A = np.random.uniform(0.3, 0.8)
        type_difficulty_B = np.random.uniform(0.3, 0.8)
        
        for q in range(n_questions_per_type):
            question_id += 1
            
            # Individual question difficulty (varies around type difficulty)
            base_difficulty_A = type_difficulty_A + np.random.normal(0, 0.15)
            base_difficulty_A = np.clip(base_difficulty_A, 0.05, 0.95)
            
            if add_correlation:
                # Model B performance correlates with Model A (similar questions are hard/easy for both)
                correlation_strength = 0.6
                base_difficulty_B = (correlation_strength * base_difficulty_A + 
                                   (1 - correlation_strength) * type_difficulty_B + 
                                   np.random.normal(0, 0.1))
            else:
                base_difficulty_B = type_difficulty_B + np.random.normal(0, 0.15)
            
            base_difficulty_B = np.clip(base_difficulty_B, 0.05, 0.95)
            
            # Generate multiple draws for this question
            for draw in range(1, n_draws + 1):
                # Add some randomness to each draw (conditional variance)
                prob_A = np.clip(base_difficulty_A + np.random.normal(0, 0.05), 0, 1)
                prob_B = np.clip(base_difficulty_B + np.random.normal(0, 0.05), 0, 1)
                
                score_A = np.random.binomial(1, prob_A)
                score_B = np.random.binomial(1, prob_B)
                
                data.append({
                    'question_id': f'q_{question_id:03d}',
                    'question_draw': draw,
                    'question_type': question_type,
                    'scorer1_id': 'model_A',
                    'scorer2_id': 'model_B',
                    'score1': score_A,
                    'score2': score_B
                })
    
    return pd.DataFrame(data)

# =============================================================================
# 2. VARIANCE DECOMPOSITION - CONDITIONAL VARIANCES (σ²)
# =============================================================================

def calculate_simple_conditional_variances(df: pd.DataFrame,
                                         question_id_col: str,
                                         score1_col: str,
                                         score2_col: str) -> Tuple[float, float]:
    """Calculate simple (non-clustered) conditional variances."""
    
    # Calculate conditional variance for each question
    question_variances = df.groupby(question_id_col).agg({
        score1_col: lambda x: np.var(x, ddof=1) if len(x) > 1 else 0,
        score2_col: lambda x: np.var(x, ddof=1) if len(x) > 1 else 0
    })
    
    # Expected conditional variances
    sigma_A_squared = question_variances[score1_col].mean()
    sigma_B_squared = question_variances[score2_col].mean()
    
    return sigma_A_squared, sigma_B_squared

def calculate_clustered_conditional_variances(df: pd.DataFrame,
                                            question_id_col: str,
                                            question_draw_col: str,
                                            question_type_col: str,
                                            score1_col: str,
                                            score2_col: str,
                                            n_total: int) -> Tuple[float, float]:
    """Calculate clustered conditional variances using the paper's formula."""
    
    # Calculate conditional means for each question
    question_means = df.groupby([question_id_col, question_type_col]).agg({
        score1_col: 'mean',
        score2_col: 'mean'
    }).reset_index()
    
    # Merge back to get conditional means for each observation
    df_with_means = df.merge(question_means, on=[question_id_col, question_type_col], 
                            suffixes=('', '_mean'))
    
    K = df.groupby(question_id_col)[question_draw_col].nunique().max()
    
    # Calculate clustered conditional variances
    sigma_A_clustered = 0
    sigma_B_clustered = 0
    
    for question_type in df_with_means[question_type_col].unique():
        cluster_data = df_with_means[df_with_means[question_type_col] == question_type]
        
        for i, row_i in cluster_data.iterrows():
            for j, row_j in cluster_data.iterrows():
                if i != j:  # Only when i ≠ j
                    # Calculate epsilon values (residuals from conditional means)
                    epsilon_A_i = row_i[score1_col] - row_i[f'{score1_col}_mean']
                    epsilon_A_j = row_j[score1_col] - row_j[f'{score1_col}_mean']
                    epsilon_B_i = row_i[score2_col] - row_i[f'{score2_col}_mean']
                    epsilon_B_j = row_j[score2_col] - row_j[f'{score2_col}_mean']
                    
                    sigma_A_clustered += epsilon_A_i * epsilon_A_j
                    sigma_B_clustered += epsilon_B_i * epsilon_B_j
    
    sigma_A_clustered /= (n_total * (K - 1))
    sigma_B_clustered /= (n_total * (K - 1))
    
    return sigma_A_clustered, sigma_B_clustered

def calculate_conditional_variances(df: pd.DataFrame,
                                  question_id_col: str,
                                  question_draw_col: str, 
                                  question_type_col: str,
                                  score1_col: str,
                                  score2_col: str,
                                  is_clustered: bool) -> Tuple[float, float]:
    """Calculate σ²_A and σ²_B (conditional variances)."""
    
    n_total = df[question_id_col].nunique()
    
    if is_clustered:
        return calculate_clustered_conditional_variances(
            df, question_id_col, question_draw_col, question_type_col,
            score1_col, score2_col, n_total
        )
    else:
        return calculate_simple_conditional_variances(
            df, question_id_col, score1_col, score2_col
        )

# =============================================================================
# 3. VARIANCE DECOMPOSITION - OMEGA SQUARED (ω²)
# =============================================================================

def calculate_clustered_omega_squared(question_means: pd.DataFrame, 
                                    question_type_col: str,
                                    overall_mean_A: float,
                                    overall_mean_B: float, 
                                    impose_zero_cov: bool = False) -> float:
    """Calculate ω² using clustered variance/covariance formulas."""
    
    n = len(question_means)
    
    # Calculate clustered variances and covariance
    var_A_clustered = 0
    var_B_clustered = 0
    cov_AB_clustered = 0
    
    for question_type in question_means[question_type_col].unique():
            cluster_data = question_means[question_means[question_type_col] == question_type]
            
            for i, row_i in cluster_data.iterrows():
                for j, row_j in cluster_data.iterrows():
                    if i != j:  # Only when i ≠ j
                        # Variance A component
                        var_A_clustered += (row_i['mean_A'] - overall_mean_A) * (row_j['mean_A'] - overall_mean_A)
                        
                        # Variance B component
                        var_B_clustered += (row_i['mean_B'] - overall_mean_B) * (row_j['mean_B'] - overall_mean_B)
                        
                        # Covariance component
                        cov_AB_clustered += (row_i['mean_A'] - overall_mean_A) * (row_j['mean_B'] - overall_mean_B)
    
    var_A_clustered /= n
    var_B_clustered /= n
    
    if impose_zero_cov:
        cov_AB_clustered = 0
    else:
        cov_AB_clustered /= n
    
    omega_squared = var_A_clustered + var_B_clustered - 2 * cov_AB_clustered
    
    return omega_squared

# =============================================================================
# 4. MAIN VARIANCE ANALYSIS FUNCTION
# =============================================================================

def calculate_omega_squared(df: pd.DataFrame, 
                           score1_col: str = 'score1', 
                           score2_col: str = 'score2',
                           question_id_col: str = 'question_id',
                           question_draw_col: str = None, 
                           question_type_col: str = None, 
                           impose_zero_cov: bool = False) -> Dict[str, Any]:
    """
    Calculate ω² adaptively based on dataset structure.
    
    Automatically detects:
    - Whether questions are clustered (multiple question_types) based on whether there is >1 unique value of question_type
    - Whether there are multiple draws per question (multiple question_draws), based on whether there is >1 unique value of question_draw
    
    Parameters:
    -----------
    df : pd.DataFrame
        Long format data with columns for question_id, question_draw, question_type, scores
    score1_col : str
        Column name for scorer 1's scores
    score2_col : str  
        Column name for scorer 2's scores
    question_id_col : str
        Column name for question identifiers
    question_draw_col : str
        Column name for question draw numbers
    question_type_col : str
        Column name for question types/clusters
        
    Returns:
    --------
    dict : Results containing omega_squared and diagnostic information
    """
    print(question_type_col)
    # Detect dataset characteristics
    if question_type_col is not None:
        n_question_types = df[question_type_col].nunique()
        group_cols = [question_id_col, question_type_col]
    else:
        n_question_types = 1
        group_cols = [question_id_col]
    print(n_question_types)
    if question_draw_col is not None:
        n_draws_per_question = df.groupby(question_id_col)[question_draw_col].nunique().max()
    else:
        n_draws_per_question = 1   
    n_total_questions = df[question_id_col].nunique()
    
    is_clustered = n_question_types > 1
    has_multiple_draws = n_draws_per_question > 1
    
    print(f"Dataset characteristics:")
    print(f"  - Total unique questions: {n_total_questions}")
    print(f"  - Question types/clusters: {n_question_types}")
    print(f"  - Max draws per question: {n_draws_per_question}")
    print(f"  - Clustered analysis: {is_clustered}")
    print(f"  - Multiple draws: {has_multiple_draws}")
    print()
    
    # Calculate conditional means for each question (averaging across draws if multiple)
    question_means = df.groupby(group_cols).agg({
        score1_col: 'mean',
        score2_col: 'mean'
    }).reset_index()
    
    question_means.columns = group_cols + ['mean_A', 'mean_B']
    
    # Calculate overall means
    overall_mean_A = question_means['mean_A'].mean()
    overall_mean_B = question_means['mean_B'].mean()
    
    if is_clustered:
        # Use clustered variance/covariance calculations
        omega_squared = calculate_clustered_omega_squared(
            question_means, question_type_col, overall_mean_A, overall_mean_B
        )
        method = "clustered"
    else:
        # Use simple variance/covariance calculations
        var_A = np.var(question_means['mean_A'], ddof=1)
        var_B = np.var(question_means['mean_B'], ddof=1)
        if impose_zero_cov:
            cov_AB = 0
        else:
            cov_AB = np.cov(question_means['mean_A'], question_means['mean_B'], ddof=1)[0, 1]
        omega_squared = var_A + var_B - 2 * cov_AB
        method = "simple"
    
    # Calculate sigma_squared terms if multiple draws available
    if has_multiple_draws:
        sigma_A_squared, sigma_B_squared = calculate_conditional_variances(
            df, question_id_col, question_draw_col, question_type_col, 
            score1_col, score2_col, is_clustered
        )
    else:
        sigma_A_squared = 0.0
        sigma_B_squared = 0.0
        print("Note: Only single draw per question, so σ²_A = σ²_B = 0")
    
    # Compile results
    results = {
        'omega_squared': omega_squared,
        'sigma_A_squared': sigma_A_squared,
        'sigma_B_squared': sigma_B_squared,
        'method': method,
        'is_clustered': is_clustered,
        'has_multiple_draws': has_multiple_draws,
        'n_questions': n_total_questions,
        'n_question_types': n_question_types,
        'n_draws_per_question': n_draws_per_question,
        'var_A': np.var(question_means['mean_A'], ddof=1),
        'var_B': np.var(question_means['mean_B'], ddof=1),
        'cov_AB': np.cov(question_means['mean_A'], question_means['mean_B'], ddof=1)[0, 1]
    }
    
    print(f"Results using {method} method:")
    print(f"  ω² = {omega_squared:.6f}")
    print(f"  σ²_A = {sigma_A_squared:.6f}")
    print(f"  σ²_B = {sigma_B_squared:.6f}")
    
    return results

# =============================================================================
# 5. SAMPLE SIZE AND POWER CALCULATIONS
# =============================================================================

def calculate_sample_size(omega_squared: float, 
                        sigma_A_squared: float, 
                        sigma_B_squared: float, 
                        var_A: float,
                        var_B: float,
                        alpha: float = 0.05,
                        beta: float = 0.20,
                        K_A: int = 1, 
                        K_B: int = 1, 
                        delta: float = 0.05,
                        multiple_tests: int = 1, 
                        dime_version: bool=False) -> float:
    """
    Calculate required sample size

    Parameters:
    -----------
    omega_squared : float
        Variance of score differences (ω²)
    sigma_A_squared : float  
        Expected conditional variance for model A (σ²_A)
    sigma_B_squared : float
        Expected conditional variance for model B (σ²_B)
    var_A : float
        Variance of model A
    var_B : float
        Variance of model B
    alpha : float
        Significance level (before multiple testing correction)
    beta : float
        Type II error rate (1 - power)
    K_A : int
        Number of samples per question for model A
    K_B : int
        Number of samples per question for model B
    delta : float
        Minimum detectable effect size
    multiple_tests : int
        Number of hypothesis tests (for Bonferroni correction)
        
    Returns:
    --------
    n : float
        Required number of questions
    """
    # Apply Bonferroni correction if multiple tests
    alpha_corrected = alpha / multiple_tests if multiple_tests > 1 else alpha
    
    # Calculate z-scores with corrected alpha
    z_alpha_2 = stats.norm.ppf(1 - alpha_corrected/2)
    z_beta = stats.norm.ppf(1 - beta)
    
    if multiple_tests > 1:
        print(f"Applied Bonferroni correction: α = {alpha} / {multiple_tests} = {alpha_corrected:.6f}")
    
    if dime_version:
        numerator = (z_alpha_2 + z_beta)**2 * 4 * var_A
    else:
        numerator = (z_alpha_2 + z_beta)**2 * (omega_squared + sigma_A_squared/K_A + sigma_B_squared/K_B)
    denominator = delta**2
    
    return numerator / denominator

def calculate_minimum_detectable_effect(omega_squared, sigma_A_squared, sigma_B_squared,
                                      alpha: float = 0.05, beta: float = 0.20, 
                                      K_A: int = 1, K_B: int = 1, n: int = 1000,
                                      multiple_tests: int = 1):
    """
    Calculate minimum detectable effect for given sample size.
    Inverted version of the sample size formula.
    
    Parameters:
    -----------
    omega_squared : float
        Variance of score differences (ω²)
    sigma_A_squared : float  
        Expected conditional variance for model A (σ²_A)
    sigma_B_squared : float
        Expected conditional variance for model B (σ²_B)
    alpha : float
        Significance level (before multiple testing correction)
    beta : float
        Type II error rate (1 - power)
    K_A : int
        Number of samples per question for model A
    K_B : int
        Number of samples per question for model B
    n : int
        Number of questions available
    multiple_tests : int
        Number of hypothesis tests (for Bonferroni correction)
    
    Returns:
    --------
    delta : float
        Minimum detectable effect size
    """
    # Apply Bonferroni correction if multiple tests
    alpha_corrected = alpha / multiple_tests if multiple_tests > 1 else alpha
    
    # Calculate z-scores with corrected alpha
    z_alpha_2 = stats.norm.ppf(1 - alpha_corrected/2)
    z_beta = stats.norm.ppf(1 - beta)
    
    if multiple_tests > 1:
        print(f"Applied Bonferroni correction: α = {alpha} / {multiple_tests} = {alpha_corrected:.6f}")
    
    variance_term = omega_squared + sigma_A_squared/K_A + sigma_B_squared/K_B
    delta = (z_alpha_2 + z_beta) * np.sqrt(variance_term / n)
    
    return delta

# =============================================================================
# 6. POWER ANALYSIS GRID SETUP
# =============================================================================

def create_power_analysis_grid(multiple_tests: int = 1):
    """
    Create a comprehensive grid of parameters for power analysis.
    
    Parameters:
    -----------
    multiple_tests : int
        Number of hypothesis tests (for Bonferroni correction)
    
    Returns:
    --------
    dict : Parameter grid with all combinations
    """
    
    # Base significance levels (before correction)
    base_alphas = [0.01, 0.05, 0.10]
    
    # Apply Bonferroni correction if multiple tests
    if multiple_tests > 1:
        corrected_alphas = [alpha / multiple_tests for alpha in base_alphas]
        print(f"Applied Bonferroni correction for {multiple_tests} tests:")
        for base, corrected in zip(base_alphas, corrected_alphas):
            print(f"  α = {base} / {multiple_tests} = {corrected:.6f}")
    else:
        corrected_alphas = base_alphas
    
    # Standard significance levels and their z-values (two sided)
    significance_levels = {}
    for i, (base_alpha, corrected_alpha) in enumerate(zip(base_alphas, corrected_alphas)):
        key = f'alpha_{base_alpha:.2f}'
        if multiple_tests > 1:
            key += f'_corrected'
        significance_levels[key] = {
            'alpha': base_alpha,
            'alpha_corrected': corrected_alpha,
            'z_alpha_2': stats.norm.ppf(1 - corrected_alpha/2),
            'multiple_tests': multiple_tests
        }
    
    # Standard power levels and their z-values
    power_levels = {
        'power_0.80': {'beta': 0.20, 'power': 0.80, 'z_beta': stats.norm.ppf(1 - 0.20)},  # z ≈ 0.842
        'power_0.85': {'beta': 0.15, 'power': 0.85, 'z_beta': stats.norm.ppf(1 - 0.15)},  # z ≈ 1.036
        'power_0.90': {'beta': 0.10, 'power': 0.90, 'z_beta': stats.norm.ppf(1 - 0.10)},  # z ≈ 1.282
        'power_0.95': {'beta': 0.05, 'power': 0.95, 'z_beta': stats.norm.ppf(1 - 0.05)},  # z ≈ 1.645
    }
    
    # Sampling strategies (K values)
    K_values = [1, 2, 3, 5, 10]  # Number of samples per question
    
    # Create the parameter grid
    grid = {
        'significance_levels': significance_levels,
        'power_levels': power_levels, 
        'K_values': K_values,
        'multiple_tests': multiple_tests
    }
    
    return grid

def create_delta_grid(var_x_A, var_x_B, n_points=10, reference_delta=None):
    """
    Create a grid of delta values based on the standard deviations of x_A and x_B,
    or around a reference delta value.
    
    Parameters:
    -----------
    var_x_A : float
        Variance of conditional means for model A
    var_x_B : float  
        Variance of conditional means for model B
    n_points : int
        Number of delta values to generate
    reference_delta : float, optional
        Reference delta value to center the grid around. If provided,
        creates a grid around this value instead of using SD-based approach.
        
    Returns:
    --------
    list : Delta values ranging from small to large effects
    """
    if reference_delta is not None:
        # Create grid around reference delta
        # Generate values from 0.2x to 2.0x the reference delta
        multipliers = np.linspace(0.2, 2.0, n_points)
        delta_values = multipliers * reference_delta
        print(f"Using reference delta: {reference_delta:.4f}")
        print(f"Delta grid range: {delta_values.min():.4f} to {delta_values.max():.4f}")
    else:
        # Original SD-based approach
        # Calculate standard deviations
        sd_x_A = np.sqrt(var_x_A)
        sd_x_B = np.sqrt(var_x_B)
        avg_sd = (sd_x_A + sd_x_B) / 2
        
        # Create delta values as fractions of the average standard deviation
        # Small effect: 0.1 * SD, Medium: 0.5 * SD, Large: 1.0 * SD
        effect_sizes = np.linspace(0.05, 1.5, n_points)  # From 5% to 150% of SD
        delta_values = effect_sizes * avg_sd
        print(f"Using SD-based approach with avg_sd: {avg_sd:.4f}")
        print(f"Delta grid range: {delta_values.min():.4f} to {delta_values.max():.4f}")
    
    return delta_values.tolist()


# =============================================================================
# 7. COMPREHENSIVE POWER ANALYSIS GRID SEARCH
# =============================================================================

def run_power_analysis_grid(omega_squared, sigma_A_squared, sigma_B_squared,
                           var_A, var_B, max_n=10000, multiple_tests: int = 1, 
                           reference_delta=None, n_delta_points=10):
    """
    Run comprehensive power analysis across parameter grid.
    
    Parameters:
    -----------
    omega_squared : float
        Estimated ω² from your data
    sigma_A_squared : float
        Estimated σ²_A from your data  
    sigma_B_squared : float
        Estimated σ²_B from your data
    var_A : float
        Variance of conditional means for model A
    var_B : float
        Variance of conditional means for model B
    max_n : int
        Maximum sample size to consider practical
    multiple_tests : int
        Number of hypothesis tests (for Bonferroni correction)
    reference_delta : float, optional
        Reference delta value to center the grid around. If provided,
        creates a grid around this value instead of using SD-based approach.
    n_delta_points : int
        Number of delta values to generate in the grid
        
    Returns:
    --------
    pd.DataFrame : Results of grid search
    """
    
    # Create parameter grids
    param_grid = create_power_analysis_grid(multiple_tests)
    delta_values = create_delta_grid(var_A, var_B, n_points=n_delta_points, 
                                   reference_delta=reference_delta)
    
    results = []
    
    # Grid search over all combinations
    for sig_name, sig_params in param_grid['significance_levels'].items():
        for pow_name, pow_params in param_grid['power_levels'].items():
            for K in param_grid['K_values']:
                for delta in delta_values:
                    
                    # Calculate required sample size using corrected alpha
                    n_required = calculate_sample_size(
                        omega_squared, sigma_A_squared, sigma_B_squared,
                        var_A, var_B,
                        alpha=sig_params['alpha'], beta=pow_params['beta'],
                        K_A=K, K_B=K, delta=delta, multiple_tests=multiple_tests
                    )
                    
                    # Calculate effect size relative to average SD
                    avg_sd = (np.sqrt(var_A) + np.sqrt(var_B)) / 2
                    relative_effect = delta / avg_sd if avg_sd > 0 else np.nan
                    
                    # Calculate effect size relative to reference delta if provided
                    relative_to_ref = delta / reference_delta if reference_delta is not None else np.nan
                    
                    # Store results
                    results.append({
                        'alpha': sig_params['alpha'],
                        'alpha_corrected': sig_params.get('alpha_corrected', sig_params['alpha']),
                        'power': pow_params['power'],
                        'beta': pow_params['beta'],
                        'multiple_tests': multiple_tests,
                        'K': K,
                        'delta': delta,
                        'delta_relative_sd': relative_effect,
                        'delta_relative_ref': relative_to_ref,
                        'reference_delta': reference_delta,
                        'n_required': n_required,
                        'feasible': n_required <= max_n,
                        'z_alpha_2': sig_params['z_alpha_2'],
                        'z_beta': pow_params['z_beta']
                    })
    
    return pd.DataFrame(results)

def analyze_power_results(results_df, top_n=10):
    """
    Analyze and summarize power analysis results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_power_analysis_grid()
    top_n : int
        Number of top recommendations to show
        
    Returns:
    --------
    dict : Analysis summary
    """
    
    # Filter to feasible options
    feasible = results_df[results_df['feasible']].copy()
    
    if len(feasible) == 0:
        print("⚠️  No feasible options found! Consider:")
        print("   - Increasing max_n")
        print("   - Accepting larger effect sizes")
        print("   - Reducing power requirements")
        print("   - Reducing number of multiple tests")
        return None
    
    # Check if multiple testing correction was applied
    multiple_tests = feasible['multiple_tests'].iloc[0]
    correction_note = f" (Bonferroni corrected for {multiple_tests} tests)" if multiple_tests > 1 else ""
    
    # Find optimal configurations
    print("🎯 POWER ANALYSIS SUMMARY" + correction_note)
    print("=" * 50)
    
    # Most efficient (smallest n) for different effect sizes
    print("\n📊 Most Efficient Configurations (Smallest Sample Size):")
    display_cols = ['alpha', 'power', 'K', 'delta_relative_sd', 'n_required']
    if multiple_tests > 1:
        display_cols.insert(1, 'alpha_corrected')
    
    efficient = feasible.nsmallest(top_n, 'n_required')[display_cols].round(3)
    print(efficient.to_string(index=False))
    
    # Different effect size categories
    small_effect = feasible[feasible['delta_relative_sd'] <= 0.3]
    medium_effect = feasible[(feasible['delta_relative_sd'] > 0.3) & (feasible['delta_relative_sd'] <= 0.7)]
    large_effect = feasible[feasible['delta_relative_sd'] > 0.7]
    
    print(f"\n📈 Sample Size Ranges:")
    if len(small_effect) > 0:
        print(f"Small effects (≤0.3 SD): {small_effect['n_required'].min():.0f} - {small_effect['n_required'].max():.0f} questions")
    if len(medium_effect) > 0:
        print(f"Medium effects (0.3-0.7 SD): {medium_effect['n_required'].min():.0f} - {medium_effect['n_required'].max():.0f} questions")  
    if len(large_effect) > 0:
        print(f"Large effects (>0.7 SD): {large_effect['n_required'].min():.0f} - {large_effect['n_required'].max():.0f} questions")
    
    # Impact of K (sampling strategy)
    print(f"\n🔄 Impact of Multiple Sampling (K):")
    k_impact = feasible.groupby('K')['n_required'].agg(['mean', 'min']).round(0)
    print(k_impact)
    
    return {
        'feasible_options': len(feasible),
        'min_n': feasible['n_required'].min(),
        'max_n': feasible['n_required'].max(),
        'multiple_tests': multiple_tests,
        'recommended': efficient.iloc[0].to_dict() if len(efficient) > 0 else None
    }

# =============================================================================
# 8. VISUALIZATION
# =============================================================================

def plot_power_analysis(results_df, save_path=None):
    """
    Create visualizations of power analysis results.
    """
    feasible = results_df[results_df['feasible']].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Power Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Sample size vs effect size
    ax1 = axes[0, 0]
    for alpha in feasible['alpha'].unique():
        for power in feasible['power'].unique():
            subset = feasible[(feasible['alpha'] == alpha) & 
                            (feasible['power'] == power) & 
                            (feasible['K'] == 1)]
            if len(subset) > 0:
                ax1.plot(subset['delta_relative_sd'], subset['n_required'], 
                        'o-', label=f'α={alpha}, power={power}', alpha=0.7)
    
    ax1.set_xlabel('Effect Size (relative to SD)')
    ax1.set_ylabel('Required Sample Size')
    ax1.set_title('Sample Size vs Effect Size (K=1)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # 2. Impact of K
    ax2 = axes[0, 1]
    for K in sorted(feasible['K'].unique()):
        subset = feasible[(feasible['K'] == K) & 
                         (feasible['alpha'] == 0.05) & 
                         (feasible['power'] == 0.8)]
        if len(subset) > 0:
            ax2.plot(subset['delta_relative_sd'], subset['n_required'], 
                    'o-', label=f'K={K}', alpha=0.7)
    
    ax2.set_xlabel('Effect Size (relative to SD)')
    ax2.set_ylabel('Required Sample Size')
    ax2.set_title('Impact of Multiple Sampling (α=0.05, power=0.8)')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap of feasible configurations
    ax3 = axes[1, 0]
    pivot = feasible.pivot_table(values='n_required', 
                               index=['alpha', 'power'], 
                               columns='K', 
                               aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis_r', ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Average Sample Size by Configuration')
    ax3.set_xlabel('K (Samples per Question)')
    ax3.set_ylabel('Alpha, Power')
    
    # 4. Effect size recommendations
    ax4 = axes[1, 1]
    bins = [0, 0.3, 0.7, 1.0, 2.0]
    labels = ['Small\n(≤0.3 SD)', 'Medium\n(0.3-0.7 SD)', 'Large\n(0.7-1.0 SD)', 'Very Large\n(>1.0 SD)']
    feasible['effect_category'] = pd.cut(feasible['delta_relative_sd'], bins=bins, labels=labels, include_lowest=True)
    
    effect_summary = feasible.groupby('effect_category')['n_required'].agg(['mean', 'min', 'max'])
    effect_summary.plot(kind='bar', ax=ax4)
    ax4.set_title('Sample Size by Effect Category')
    ax4.set_xlabel('Effect Size Category')
    ax4.set_ylabel('Required Sample Size')
    ax4.legend(fontsize='small')
    ax4.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig, axes

def run_mde_analysis_grid(omega_squared, sigma_A_squared, sigma_B_squared,
                         var_A, var_B, n_fixed, multiple_tests: int = 1, 
                         beta: float = 0.20):
    """
    Run MDE analysis with fixed sample size, varying alpha and K.
    
    Parameters:
    -----------
    omega_squared : float
        Estimated ω² from your data
    sigma_A_squared : float
        Estimated σ²_A from your data  
    sigma_B_squared : float
        Estimated σ²_B from your data
    var_A : float
        Variance of conditional means for model A
    var_B : float
        Variance of conditional means for model B
    n_fixed : int
        Fixed sample size (number of questions)
    multiple_tests : int
        Number of hypothesis tests (for Bonferroni correction)
    beta : float
        Type II error rate (1 - power), default 0.20 (80% power)
        
    Returns:
    --------
    pd.DataFrame : Results of MDE analysis
    """
    
    # Create parameter grids
    param_grid = create_power_analysis_grid(multiple_tests)
    
    results = []
    
    # Grid search over significance levels and K values
    for sig_name, sig_params in param_grid['significance_levels'].items():
        for K in param_grid['K_values']:
            
            # Calculate MDE using the minimum detectable effect function
            mde = calculate_minimum_detectable_effect(
                omega_squared, sigma_A_squared, sigma_B_squared,
                alpha=sig_params['alpha'], beta=beta,
                K_A=K, K_B=K, n=n_fixed, multiple_tests=multiple_tests
            )
            
            # Calculate effect size relative to average SD
            avg_sd = (np.sqrt(var_A) + np.sqrt(var_B)) / 2
            mde_relative_sd = mde / avg_sd if avg_sd > 0 else np.nan
            
            # Store results
            results.append({
                'alpha': sig_params['alpha'],
                'alpha_corrected': sig_params.get('alpha_corrected', sig_params['alpha']),
                'power': 1 - beta,
                'beta': beta,
                'multiple_tests': multiple_tests,
                'K': K,
                'n_fixed': n_fixed,
                'mde': mde,
                'mde_relative_sd': mde_relative_sd,
                'z_alpha_2': sig_params['z_alpha_2'],
                'z_beta': stats.norm.ppf(1 - beta)
            })
    
    return pd.DataFrame(results)

def analyze_mde_results(results_df):
    """
    Analyze and summarize MDE analysis results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_mde_analysis_grid()
        
    Returns:
    --------
    dict : Analysis summary
    """
    
    # Check if multiple testing correction was applied
    multiple_tests = results_df['multiple_tests'].iloc[0]
    n_fixed = results_df['n_fixed'].iloc[0]
    power = results_df['power'].iloc[0]
    
    correction_note = f" (Bonferroni corrected for {multiple_tests} tests)" if multiple_tests > 1 else ""
    
    print("🎯 MINIMUM DETECTABLE EFFECT ANALYSIS" + correction_note)
    print("=" * 60)
    print(f"Fixed sample size: {n_fixed} questions")
    print(f"Fixed power: {power:.0%}")
    
    # Show all configurations
    print("\n📊 MDE for All Configurations:")
    display_cols = ['alpha', 'K', 'mde', 'mde_relative_sd']
    if multiple_tests > 1:
        display_cols.insert(1, 'alpha_corrected')
    
    summary = results_df[display_cols].round(4)
    print(summary.to_string(index=False))
    
    # Best configurations (smallest MDE)
    print(f"\n🏆 Best Configurations (Smallest MDE):")
    best = results_df.nsmallest(5, 'mde')[display_cols].round(4)
    print(best.to_string(index=False))
    
    # Impact of K for each alpha level
    print(f"\n🔄 Impact of K by Alpha Level:")
    k_impact = results_df.pivot_table(values='mde', index='K', columns='alpha', aggfunc='mean').round(4)
    print(k_impact)
    
    # MDE ranges
    print(f"\n📈 MDE Summary:")
    print(f"Best MDE: {results_df['mde'].min():.4f} ({results_df['mde_relative_sd'].min():.2f} SD)")
    print(f"Worst MDE: {results_df['mde'].max():.4f} ({results_df['mde_relative_sd'].max():.2f} SD)")
    print(f"Range: {results_df['mde'].max() - results_df['mde'].min():.4f}")
    
    return {
        'n_fixed': n_fixed,
        'power': power,
        'multiple_tests': multiple_tests,
        'best_mde': results_df['mde'].min(),
        'worst_mde': results_df['mde'].max(),
        'best_config': results_df.loc[results_df['mde'].idxmin()].to_dict()
    }

def plot_mde_analysis(results_df, save_path=None):
    """
    Create visualizations of MDE analysis results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_mde_analysis_grid()
    save_path : str, optional
        Path to save the plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Minimum Detectable Effect Analysis (n={results_df["n_fixed"].iloc[0]})', 
                 fontsize=16, fontweight='bold')
    
    # 1. MDE vs K for different alpha levels
    ax1 = axes[0, 0]
    for alpha in sorted(results_df['alpha'].unique()):
        subset = results_df[results_df['alpha'] == alpha]
        ax1.plot(subset['K'], subset['mde'], 'o-', label=f'α={alpha}', alpha=0.8)
    
    ax1.set_xlabel('K (Samples per Question)')
    ax1.set_ylabel('Minimum Detectable Effect (MDE)')
    ax1.set_title('MDE vs K by Alpha Level')
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # 2. MDE (relative to SD) vs K for different alpha levels
    ax2 = axes[0, 1]
    for alpha in sorted(results_df['alpha'].unique()):
        subset = results_df[results_df['alpha'] == alpha]
        ax2.plot(subset['K'], subset['mde_relative_sd'], 'o-', label=f'α={alpha}', alpha=0.8)
    
    ax2.set_xlabel('K (Samples per Question)')
    ax2.set_ylabel('MDE (relative to SD)')
    ax2.set_title('MDE (Relative to SD) vs K by Alpha Level')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap of MDE values
    ax3 = axes[1, 0]
    pivot = results_df.pivot_table(values='mde', index='alpha', columns='K', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis_r', ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('MDE Heatmap')
    ax3.set_xlabel('K (Samples per Question)')
    ax3.set_ylabel('Alpha Level')
    
    # 4. Bar plot comparing configurations
    ax4 = axes[1, 1]
    # Create combined labels for alpha and K
    results_df['config_label'] = results_df.apply(lambda x: f"α={x['alpha']}, K={x['K']}", axis=1)
    results_sorted = results_df.sort_values('mde')
    
    bars = ax4.bar(range(len(results_sorted)), results_sorted['mde'], alpha=0.7)
    ax4.set_xlabel('Configuration (sorted by MDE)')
    ax4.set_ylabel('Minimum Detectable Effect')
    ax4.set_title('MDE by Configuration (Best to Worst)')
    ax4.set_xticks(range(len(results_sorted)))
    ax4.set_xticklabels(results_sorted['config_label'], rotation=45, ha='right', fontsize='small')
    
    # Color bars by alpha level for easier interpretation
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df['alpha'].unique())))
    alpha_colors = dict(zip(sorted(results_df['alpha'].unique()), colors))
    for i, (_, row) in enumerate(results_sorted.iterrows()):
        bars[i].set_color(alpha_colors[row['alpha']])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig, axes