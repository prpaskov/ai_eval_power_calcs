import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataGenerator:
    """
    Simulates and visualizes a dataset of LLM and non-expert scores compared to expert-graded questions.

    Parameters:
        pilot_df (DataFrame): Must contain, in wide format:
            ['question_type', 'q', 'exp_score', 'llm_score', 'nonexp_score', 'nonexp_id'].
            Scores must be normalized.
        n_llms (int): Number of distinct LLMs to simulate.
        n_nonexperts (int): Number of distinct non-experts to simulate.
        n_items (int): Total number of rows to simulate (each = 1 (LLM, q, qtype) with 10 nonexp scores).
        n_expert_questions (int or None): Number of unique expert-graded questions to sample.
        simulate_exp_scores (bool): If True, simulate expert scores using distribution by question_type.
    """

    def __init__(self, 
                 pilot_df: pd.DataFrame, 
                 n_llms: int = 2, 
                 n_nonexperts: int = 10, 
                 n_items: int = 10000, 
                 n_expert_questions: int = None, 
                 simulate_exp_scores: bool = False):
        
        self.pilot_df = pilot_df.copy()
        self.n_llms = n_llms
        self.n_nonexperts = n_nonexperts
        self.n_items = n_items
        self.n_expert_questions = n_expert_questions
        self.simulate_exp_scores = simulate_exp_scores
        self.required_cols = {'question_type', 'q', 'exp_score', 'llm_score', 'nonexp_score', 'nonexp_id'}
        
        if not self.required_cols.issubset(self.pilot_df.columns):
            raise ValueError(f"pilot_df must include columns: {self.required_cols}")

    def simulate(self, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)

        # LLM behavior
        llm_bias_mean_by_qtype = self.pilot_df.groupby('question_type').apply(
            lambda g: (g['llm_score'] - g['exp_score']).mean()
        )
        llm_noise_std_by_qtype = self.pilot_df.groupby('question_type').apply(
            lambda g: (g['llm_score'] - g['exp_score']).std()
        )

        # Non-expert behavior
        self.pilot_df['resid'] = self.pilot_df['nonexp_score'] - self.pilot_df['exp_score']
        nonexp_bias_mean_by_grader = self.pilot_df.groupby('nonexp_id')['resid'].mean()
        nonexp_bias_mean = nonexp_bias_mean_by_grader.mean()
        nonexp_bias_std_across_graders = nonexp_bias_mean_by_grader.std()
        nonexp_noise_std_within_graders = self.pilot_df.groupby('nonexp_id')['resid'].std().mean()

        # Expert pool
        expert_pool = self.pilot_df[['question_type', 'q', 'exp_score']].drop_duplicates()

        if self.n_expert_questions is not None:
            if self.simulate_exp_scores:
                expert_means = self.pilot_df.groupby('question_type')['exp_score'].mean()
                expert_sds = self.pilot_df.groupby('question_type')['exp_score'].std()
                sampled_qtypes = np.random.choice(
                    expert_means.index, size=self.n_expert_questions, replace=True
                )
                expert_pool = pd.DataFrame({
                    'question_type': sampled_qtypes,
                    'q': [f'q{i}' for i in range(self.n_expert_questions)],
                    'exp_score': [
                        np.random.normal(loc=expert_means[qtype], scale=expert_sds[qtype])
                        for qtype in sampled_qtypes
                    ]
                })
            else:
                expert_pool = expert_pool.sample(
                    n=min(self.n_expert_questions, len(expert_pool)), 
                    replace=False, 
                    random_state=seed
                )

        # Simulate data
        llms = [f"llm_{i}" for i in range(self.n_llms)]
        nonexperts = [f"nonexp_{i}" for i in range(self.n_nonexperts)]
        expert_pool = expert_pool.reset_index(drop=True)

        rows = []
        for _ in range(self.n_items):
            row_expert = expert_pool.sample(1).iloc[0]
            qtype, q, true_score = row_expert['question_type'], row_expert['q'], row_expert['exp_score']
            llm = np.random.choice(llms)

            llm_score = (
                true_score +
                llm_bias_mean_by_qtype[qtype] +
                np.random.normal(loc=0, scale=llm_noise_std_by_qtype[qtype])
            )

            for nonexp_id in nonexperts:
                bias_shift = np.random.normal(loc=nonexp_bias_mean, scale=nonexp_bias_std_across_graders)
                noise = np.random.normal(loc=0, scale=nonexp_noise_std_within_graders)
                nonexp_score = true_score + bias_shift + noise
                y = abs(nonexp_score - llm_score)

                rows.append({
                    'question_type': qtype,
                    'q': q,
                    'LLM': llm,
                    'nonexp_id': nonexp_id,
                    'true_score': true_score,
                    'llm_score': llm_score,
                    'nonexp_score': nonexp_score,
                    'y': y
                })

        return pd.DataFrame(rows)

    @staticmethod
    def visualize_distributions(sim_df):
        plt.figure(figsize=(12, 6))
        sns.kdeplot(sim_df['true_score'], label='True Score (Expert)', linewidth=2)
        sns.kdeplot(sim_df['llm_score'], label='LLM Score', linewidth=2)
        sns.kdeplot(sim_df['nonexp_score'], label='Non-Expert Score (Pooled)', linewidth=2)
        sns.kdeplot(sim_df['y'], label='|Non-Expert - LLM| (y)', linestyle='--', linewidth=2)
        plt.title("Distributions of Simulated Scores")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def transform_to_wide_format(sim_df):
        base_cols = ['question_type', 'q', 'LLM', 'true_score', 'llm_score']
        wide_df = sim_df.pivot_table(
            index=base_cols,
            columns='nonexp_id',
            values='nonexp_score'
        ).reset_index()

        wide_df.columns.name = None
        wide_df = wide_df.rename(columns=lambda x: f'nonexp_score_{x}' if isinstance(x, str) and x.startswith('nonexp_') else x)
        return wide_df
