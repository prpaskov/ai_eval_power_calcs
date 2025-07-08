"""
AI Evaluation Power Calculations

This package provides tools for statistical power analysis and sample size calculations 
for AI evaluation experiments, particularly for comparing paired benchmark scores 
between different models or judges.

Main Components:
- AggregateScores: Load and aggregate multiple CSV files from judge experiments
- AnalyzeScores: Perform paired t-tests with Bonferroni correction and power calculations
- Power calculation functions: Sample size calculations and variance decomposition
"""

from .aggregate_scores import AggregateScores
from .analyze_scores import AnalyzeScores
from .run_power_calcs import (
    calculate_sample_size,
    calculate_omega_squared,
    run_power_analysis_grid,
    plot_power_analysis,
    calculate_minimum_detectable_effect,
    create_power_analysis_grid,
    create_sample_data
)

__version__ = "0.1.0"
__author__ = "Patricia Paskov"

__all__ = [
    "AggregateScores",
    "AnalyzeScores", 
    "calculate_sample_size",
    "calculate_omega_squared",
    "run_power_analysis_grid",
    "plot_power_analysis",
    "calculate_minimum_detectable_effect",
    "create_power_analysis_grid",
    "create_sample_data"
] 