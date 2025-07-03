# ai_eval_power_calcs

[[work in progress]]

run_power_calcs.ipynb takes in a sample of paired benchmark data and calculates stats (e.g. variance, covariance) needed for power calculations. It then runs grid search over parameters to report sample size requirements under a range of assumptions. This file is designed for researchers looking to run t-tests paired scores of a test. 

generate_simulation_data.py takes in a sample of paired benchmark data and generates a larger set of synthetic data with which simulations can be run to determine power implications of different parameters (e.g. number of clusters, number of draws, etc.) This file is more niche (eventually to be generalized) and is designed to compare non-expert to LLM scores. 
