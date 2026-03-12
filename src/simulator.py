import pandas as pd
import numpy as np
import os

class WHLSimulator:
    def __init__(self, rapm_path="../data/outputs/rapm_ratings.csv", matchups_path="../data/raw/matchups.csv", out_dir="../data/outputs/"):
        self.rapm_path = rapm_path
        self.matchups_path = matchups_path
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        # MLE Optimized Parameters
        # Derived via L-BFGS-B optimization to minimize historical Log-Loss
        self.W_SKATER = 0.319
        self.W_GOALIE = 0.660
        self.GAMMA = 1.823
        self.BASE_GOALS = 2.5
        
    def clean_name(self, name):
        """Format team names for presentation."""
        if not isinstance(name, str): return name
        if name.lower() in ['uk', 'usa']: return name.upper()
        return name.replace('_', ' ').title()

    def run(self):
        print("Running Pythagorean Simulator...")
        try:
            rapm = pd.read_csv(self.rapm_path)
        except FileNotFoundError:
            print("Error: rapm_ratings.csv not found. Please run RAPM engine first.")
            return
            
        # ==========================================
        # 1. POWER RANKINGS
        # ==========================================
        print("Generating power rankings...")
        
        # Calculate Skater Strength
        skaters = rapm[rapm['Role'] != 'Goalie'].groupby('Team')['RAPM_Rating'].sum().reset_index()
        skaters['Skater_Strength'] = skaters['RAPM_Rating'] * self.W_SKATER
        
        # Calculate Goalie Strength (using starting goalie max rating)
        goalies = rapm[rapm['Role'] == 'Goalie'].groupby('Team')['RAPM_Rating'].max().reset_index()
        goalies['Goalie_Strength'] = goalies['RAPM_Rating'] * self.W_GOALIE
        
        power = pd.merge(skaters[['Team', 'Skater_Strength']], goalies[['Team', 'Goalie_Strength']], on='Team', how='outer').fillna(0)
        power['Total_Strength'] = power['Skater_Strength'] + power['Goalie_Strength']
        power['Team'] = power['Team'].apply(self.clean_name)
        
        power = power.sort_values(by='Total_Strength', ascending=False).reset_index(drop=True)
        power.index += 1
        power.index.name = 'Rank'
        
        # Normalize to 0-100 scale
        min_str = power['Total_Strength'].min()
        max_str = power['Total_Strength'].max()
        power['Power_Score_100'] = ((power['Total_Strength'] - min_str) / (max_str - min_str)) * 100
        
        power.to_csv(os.path.join(self.out_dir, "power_rankings.csv"))
        print("Saved power_rankings.csv")

        # ==========================================
        # 2. LINE DISPARITY
        # ==========================================
        print("Calculating line disparity...")
        lines = rapm[rapm['Role'].isin(['first_off', 'second_off'])].copy()
        lines['Team'] = lines['Team'].apply(self.clean_name)
        pivot_lines = lines.pivot(index='Team', columns='Role', values='RAPM_Rating').reset_index()
        
        # Calculate Absolute Expected Goals ratio
        pivot_lines['Disparity_Ratio'] = (self.BASE_GOALS + pivot_lines['first_off']) / (self.BASE_GOALS + pivot_lines['second_off'])
        top_10 = pivot_lines.sort_values(by='Disparity_Ratio', ascending=False).head(10)
        
        top_10.to_csv(os.path.join(self.out_dir, "top_10_disparity.csv"), index=False)
        print("Saved top_10_disparity.csv")

        # ==========================================
        # 3. MATCHUP PROBABILITIES
        # ==========================================
        print("Simulating bracket probabilities...")
        if not os.path.exists(self.matchups_path):
            print(f"Warning: Matchups file not found at {self.matchups_path}. Skipping bracket prediction.")
            print("Please provide a matchups.csv file in the raw data folder to generate predictions.")
        else:
            matchups = pd.read_csv(self.matchups_path)
            team_dict = dict(zip(power['Team'].str.lower().str.replace(' ', '_'), power['Total_Strength']))
            
            predictions = []
            for idx, row in matchups.iterrows():
                cols = matchups.columns
                t1, t2 = str(row[cols[-2]]), str(row[cols[-1]])
                
                t1_norm, t2_norm = t1.lower().replace(' ', '_'), t2.lower().replace(' ', '_')
                margin = team_dict.get(t1_norm, 0) - team_dict.get(t2_norm, 0)
                
                # Convert margin to absolute goals
                xGF_t1 = max(0.1, self.BASE_GOALS + (margin / 2))
                xGF_t2 = max(0.1, self.BASE_GOALS - (margin / 2))
                
                # Pythagorean Expectation Formula
                prob_t1 = (xGF_t1**self.GAMMA) / ((xGF_t1**self.GAMMA) + (xGF_t2**self.GAMMA))
                
                winner = self.clean_name(t1) if prob_t1 > 0.5 else self.clean_name(t2)
                predictions.append({
                    'Matchup_ID': idx + 1,
                    'Team_1': self.clean_name(t1),
                    'Team_2': self.clean_name(t2),
                    'Predicted_Winner': winner,
                    'Team_1_Win_Prob': round(prob_t1, 4),
                    'Team_2_Win_Prob': round(1 - prob_t1, 4)
                })
            
            pd.DataFrame(predictions).to_csv(os.path.join(self.out_dir, "bracket_predictions.csv"), index=False)
            print("Saved bracket_predictions.csv")
            
        print("Phase 3 complete. All deliverables generated.")

if __name__ == "__main__":
    sim = WHLSimulator()
    sim.run()
