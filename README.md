# ai_eval_power_calcs

This repository provides tools for statistical power analysis and sample size calculations for AI evaluation experiments, particularly for comparing paired benchmark scores between different models or judges.

## Installation

### From source (recommended for development)
```bash
git clone https://github.com/yourusername/ai_eval_power_calcs.git
cd ai_eval_power_calcs
pip install -e .
```

### Basic installation
```bash
pip install -r requirements.txt
```

## Usage

### Basic Import
```python
from ai_eval_power_calcs import AggregateScores, AnalyzeScores, calculate_sample_size
```

### Quick Start
```python
# Load and aggregate experimental data
aggregator = AggregateScores(
    eval_name="your_eval",
    max_score=6.0,
    comparison_judge_id="baseline",
    question_id_col="question_id",
    folder_path="./data"
)
df = aggregator.load_reshaped_df()

# Analyze results with paired t-tests
analyzer = AnalyzeScores(
    df=df,
    test_cols=["judge_2c2s", "judge_2c_3s"],
    comparison_col="baseline",
    bonferroni_correction=True,
    num_comparisons=2
)
results = analyzer.run_multiple_tests()
analyzer.print_results(results)
```

## Project Structure

```
ai_eval_power_calcs/
├── ai_eval_power_calcs/           # Main package directory
│   ├── __init__.py                # Package initialization
│   ├── aggregate_scores.py        # Data aggregation utilities
│   ├── analyze_scores.py          # Statistical analysis tools
│   ├── run_power_calcs.py         # Power calculation functions
│   └── ...
├── notebooks/                     # Jupyter notebooks
│   ├── persuade_test_results.ipynb # Example analysis notebook
│   └── ...
├── data/                          # Data files
├── results/                       # Output files
├── sims/                          # Simulation scripts
├── README.md
├── requirements.txt
└── setup.py
```

## Main Components

### `aggregate_scores.py`
Contains the `AggregateScores` class that:
- Loads multiple CSV files from judge experiments matching a specified pattern
- Aggregates results from different judges and experimental runs
- Reshapes data from long format to wide format for analysis
- Calculates Mean Absolute Error (MAE) metrics normalized by score range
- Handles baseline comparisons and multiple experimental draws

### `analyze_scores.py` 
Contains the `AnalyzeScores` class that:
- Performs paired t-tests between test columns and a comparison baseline
- Supports clustered standard errors for correlated observations
- Applies Bonferroni correction for multiple comparisons
- Calculates effect sizes, confidence intervals, and power statistics
- Computes minimum detectable effects and required sample sizes for different effect sizes
- Exports results to LaTeX tables and summary formats

### `run_power_calcs.py`
Contains functions for comprehensive power analysis:
- `calculate_sample_size()` - Calculates required sample size with optional Bonferroni adjustment
- `calculate_omega_squared()` - Estimates variance components (ω²) from paired data
- `run_power_analysis_grid()` - Runs grid search over multiple parameter combinations
- `plot_power_analysis()` - Creates visualizations of power analysis results
- Supports both simple and clustered variance calculations

### `notebooks/persuade_test_results.ipynb`
Example Jupyter notebook that demonstrates:
- How to load and aggregate experimental data using `AggregateScores`
- Performing statistical analysis with `AnalyzeScores` including Bonferroni correction
- Exporting results to LaTeX tables for publication
- Real-world application on persuasive essay scoring data