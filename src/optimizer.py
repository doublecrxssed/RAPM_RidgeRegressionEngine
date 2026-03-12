import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

class ParameterOptimizer:
    """
    Maximum Likelihood Estimation (MLE) Engine.
    Mathematically derives the optimal positional weights and Pythagorean 
    exponent by minimizing the historical Log-Loss of actual game outcomes.
    """
    def __init__(self, raw_data_path="../data/raw/whl_2025.csv", rapm_path="../data/outputs/rapm_ratings.csv"):
        self.raw_data_path = raw_data_path
        self.rapm_path = rapm_path
        self.BASE_GOALS = 2.5

    def run_optimization(self):
        print("Running Maximum Likelihood Parameter Optimization (MLE)...")
        
        if not os.path.exists(self.raw_data_path) or not os.path.exists(self.rapm_path):
            print("Error: Missing raw data or RAPM ratings. Run previous steps first.")
            return

        # Load historical outcomes and RAPM ratings
        raw_df = pd.read_csv(self.raw_data_path)
        games = raw_df.groupby(['game_id', 'home_team', 'away_team']).agg({'home_goals': 'sum', 'away_goals': 'sum'}).reset_index()
        games['home_win'] = (games['home_goals'] > games['away_goals']).astype(int)

        rapm = pd.read_csv(self.rapm_path)
        skater_rapm = rapm[rapm['Role'] != 'Goalie'].groupby('Team')['RAPM_Rating'].sum().to_dict()
        goalie_rapm = rapm[rapm['Role'] == 'Goalie'].groupby('Team')['RAPM_Rating'].max().to_dict()

        def objective_function(weights):
            w_skater, w_goalie, gamma = weights
            
            # Calculate total team strength based on current iteration weights
            h_str = games['home_team'].map(lambda x: w_skater * skater_rapm.get(x, 0) + w_goalie * goalie_rapm.get(x, 0))
            a_str = games['away_team'].map(lambda x: w_skater * skater_rapm.get(x, 0) + w_goalie * goalie_rapm.get(x, 0))

            # Convert to expected goals margin
            xGF_h = np.maximum(0.1, self.BASE_GOALS + (h_str - a_str) / 2)
            xGF_a = np.maximum(0.1, self.BASE_GOALS - (h_str - a_str) / 2)

            # Pythagorean probability
            prob_h = (xGF_h**gamma) / ((xGF_h**gamma) + (xGF_a**gamma))
            
            # Clip probabilities to prevent log(0) errors
            prob_h = np.clip(prob_h, 1e-5, 1 - 1e-5) 

            # Return Negative Log-Loss (We want to minimize this)
            return -np.mean(games['home_win'] * np.log(prob_h) + (1 - games['home_win']) * np.log(1 - prob_h))

        # Initial guess: standard NHL exponent and equal 1.0 weights
        initial_guess = [1.0, 1.0, 2.15]
        
        # Set boundaries so the optimizer doesn't try impossible physics
        bounds = ((0.1, 3.0), (0.1, 3.0), (1.0, 4.0))
        
        print("Executing L-BFGS-B algorithm. Please wait...")
        res = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        opt_w_s, opt_w_g, opt_gamma = res.x

        print("\n--- OPTIMIZATION SUCCESS ---")
        print(f"Optimized Skater Weight (W_s): {opt_w_s:.3f}")
        print(f"Optimized Goalie Weight (W_g): {opt_w_g:.3f}")
        print(f"Optimized Pythagorean Exp (γ): {opt_gamma:.3f}")
        print("Update simulator.py with these values if they differ from the defaults.")

if __name__ == "__main__":
    optimizer = ParameterOptimizer()
    optimizer.run_optimization()
