import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeCV
import os

class RAPMEngine:
    def __init__(self, clean_data_path="../data/processed/whl_5v5_cleaned.csv", out_path="../data/outputs/rapm_ratings.csv"):
        self.clean_data_path = clean_data_path
        self.out_path = out_path
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def run(self):
        print(f"--> booting up RAPM engine. loading {self.clean_data_path}")
        try:
            df = pd.read_csv(self.clean_data_path)
        except FileNotFoundError:
            print("bro you need to run the data pipeline first to get the clean data.")
            return

        print("--> building the massive sparse matrix (+1 for home, -1 for away)...")
        observations = []
        
        # formatting entity IDs so they are super easy to parse later
        # using the format: Team|Role (e.g., "usa|first_off" or "uk|Goalie|player_id")
        for _, row in df.iterrows():
            obs = {
                f"{row['home_team']}|{row['home_off_line']}": 1,
                f"{row['home_team']}|{row['home_def_pairing']}": 1,
                f"{row['home_team']}|Goalie|{row['home_goalie']}": 1,
                f"{row['away_team']}|{row['away_off_line']}": -1,
                f"{row['away_team']}|{row['away_def_pairing']}": -1,
                f"{row['away_team']}|Goalie|{row['away_goalie']}": -1
            }
            observations.append(obs)

        # DictVectorizer is magic. it turns our dictionaries into a huge sparse matrix instantly
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(observations)
        
        # what are we trying to predict? net expected goals per 60
        y = df['net_xg_per_60'].values
        
        # longer shifts = more reliable data. weight them heavier.
        weights = (df['toi'] / 3600).values

        print("--> running RidgeCV regression. finding the optimal bayesian prior...")
        # testing a bunch of alphas so we don't overfit or underfit
        alphas = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        model = RidgeCV(alphas=alphas, fit_intercept=True)
        model.fit(X, y, sample_weight=weights)

        print(f"--> math done. optimal alpha chosen: {model.alpha_}")
        print(f"--> home ice advantage (intercept): {model.intercept_:.4f} xG/60")

        # extract the true talent coefficients
        features = vectorizer.get_feature_names_out()
        coefs = model.coef_

        results = pd.DataFrame({
            'Entity_ID': features,
            'RAPM_Rating': coefs
        })

        # parse out the team and role from that ID string we made earlier
        def get_role(entity):
            if '|Goalie|' in entity: return 'Goalie'
            return entity.split('|')[1]

        results['Team'] = results['Entity_ID'].apply(lambda x: x.split('|')[0])
        results['Role'] = results['Entity_ID'].apply(get_role)

        # sort them so the best units are at the top
        results = results.sort_values(by='RAPM_Rating', ascending=False)
        
        results.to_csv(self.out_path, index=False)
        print(f"--> success! isolated {len(results)} unit ratings. saved to {self.out_path} 🎯")

if __name__ == "__main__":
    engine = RAPMEngine()
    engine.run()