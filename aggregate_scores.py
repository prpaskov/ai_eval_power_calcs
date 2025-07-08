import os
import numpy as np
import glob
from numpy.matlib import float32
import pandas as pd
from collections import Counter

class AggregateScores:
    def __init__(self, 
                    eval_name: str, 
                    max_score: float32, ##Note: Dev, if this is observation, specific, will need to tweak the code
                    comparison_judge_id: str, 
                    question_id_col: str,
                    folder_path: str = '.',
                    verbose: bool = False,
                    ground_truth_col: str = None,
                    judge_score_col: str = None, 
                    ):
        self.eval_name = eval_name
        self.max_score = max_score
        self.folder_path = folder_path
        self.question_id_col = question_id_col
        self.verbose = verbose
        self.ground_truth_col = ground_truth_col
        self.judge_score_col = judge_score_col
        self.comparison_judge_id = comparison_judge_id

    def load_reshaped_df(self):
        df_long = self.load_long_file()
        df_reshaped = self.reshape_long_to_wide(df_long)
        return df_reshaped
    
    def reshape_long_to_wide(self, 
                        df_long: pd.DataFrame):
        """
        Aggregate all decomposition judge experiment .csv files for a given eval into one dataset.
        """
        df_pc = df_long.copy()
        df_pc['mae'] = (df_pc[self.ground_truth_col] - df_pc[self.judge_score_col]).abs() / self.max_score
        df_pc[df_pc['judge_name'] == 'baseline'].head()
        df_reshaped = df_pc.pivot_table(
            index=[self.question_id_col, 'draws'],
            columns='judge_name',  # column to spread   
            values='mae',      # values to fill
            aggfunc='first'            # how to aggregate if there are duplicates
        ).reset_index()
        return df_reshaped

    def load_long_file(self):
        """
        Aggregate all decomposition judge experiment .csv files for a given eval into one dataset.
        """
        
        # Find all files matching the pattern
        experiment_pattern = f'{self.eval_name}_experiment_*.csv'
        experiment_pattern_path = os.path.join(self.folder_path, experiment_pattern)    
        csv_files = glob.glob(experiment_pattern_path)
        
        if not csv_files:
            if self.verbose:
                print(f"No files found matching pattern: {experiment_pattern}")
            return None
        
        if self.verbose:
            print(f"Found {len(csv_files)} files")
        
        # Dictionary to store dataframes and judge info
        dataframes = []
        judge_counts = Counter()
        
        # Process each file
        for file_path in csv_files:
            try:
                # Extract filename without path
                filename = os.path.basename(file_path)
                if self.verbose:
                    print(f"Processing: {filename}")
                
                # Extract judge_name from filename
                # Format: persuade_experiment_4c_2s_20250706_224756.csv
                if self.comparison_judge_id in filename:
                    judge_name = 'baseline'
                else:
                    # Split by underscore and find the pattern like "4c_2s"
                    parts = filename.split('_')
                    judge_name = None
                    
                    for i, part in enumerate(parts):
                        if part == 'experiment' and i + 1 < len(parts):
                            # Next part should be like "4c" 
                            next_part = parts[i + 1]
                            if i + 2 < len(parts):
                                # Part after that should be like "2s"
                                after_next = parts[i + 2]
                                # Check if they match pattern (digit+letter)
                                if (len(next_part) >= 2 and next_part[:-1].isdigit() and next_part[-1].isalpha() and
                                    len(after_next) >= 2 and after_next[:-1].isdigit() and after_next[-1].isalpha()):
                                    judge_name = 'judge_'+f"{next_part}_{after_next}"
                                    break
                    
                    if judge_name is None:
                        if self.verbose:
                            print(f"  - Skipping {filename}: couldn't extract judge_name")
                        continue
                
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Count this judge + add draws
                judge_counts[judge_name] += 1
                df['draws'] = judge_counts[judge_name]
                
                # Add judge_name column
                df['judge_name'] = judge_name
                
                # Add to list
                dataframes.append(df)
                
                if self.verbose:
                    print(f"  - Judge: {judge_name}, Rows: {len(df)}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  - Error processing {filename}: {e}")
        
        if not dataframes:
            if self.verbose:
                print("No valid files processed")
            return None
        
        # Combine all dataframes
        long_df = pd.concat(dataframes, ignore_index=True)
        
        # Rename columns and add baseline score
        long_df.rename(columns={'Pipeline_root.block.unit[Map]_points': 'judge_score'}, inplace=True)
        long_df['judge_score'] = np.where(long_df['judge_name'] == 'baseline', long_df['pred_label'], long_df['judge_score'])

        # Print summary
        if self.verbose:
            print(f"\nSummary:")
            print(f"Total rows: {len(combined_df)}")
            print(f"Unique judges: {len(judge_counts)}")
            print(f"Judge counts:")
            for judge, count in sorted(judge_counts.items()):
                print(f"  {judge}: {count} files")
        
        return long_df




