import pandas as pd
import numpy as np
import os

# Dynamically resolve absolute paths to prevent working-directory bugs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "whl_2025.csv")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

class HockeyPipeline:
    """
    Data Preprocessing Pipeline.
    Cleans raw shift data, filters to pure 5v5 even-strength play, 
    and engineers the Net Expected Goals (xG) per 60 metrics.
    """
    def __init__(self, raw_path=DEFAULT_RAW_PATH, out_dir=DEFAULT_OUT_DIR):
        self.raw_path = raw_path
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        print(f"Loading raw data from: {os.path.abspath(self.raw_path)}")
        try:
            df = pd.read_csv(self.raw_path)
        except FileNotFoundError:
            print(f"Error: Raw data file not found at {self.raw_path}. Please ensure the file exists or run the dummy data generator.")
            return

        print(f"Initial row count: {len(df)}")

        # Filter out overtime periods to ensure pure 5v5 evaluation
        df = df[df['went_ot'] == 0].copy()

        # Filter out special teams (Power Plays and Penalty Kills)
        special_teams_flags = ['PP', 'PK', 'kill', 'up', 'dwn', 'Empty']
        
        for col in ['home_off_line', 'away_off_line', 'home_def_pairing', 'away_def_pairing']:
            # Using na=False to safely handle missing categorical values
            df = df[~df[col].astype(str).str.contains('|'.join(special_teams_flags), case=False, na=False)]
            
        print(f"Filtered to 5v5 even-strength shifts. Remaining rows: {len(df)}")

        # Ensure strictly positive TOI to prevent division by zero errors
        df = df[df['toi'] > 0].copy()

        # Engineer target variable: Net Expected Goals per 60 minutes
        print("Calculating Net xG/60 target variables...")
        df['net_xg'] = df['home_xg'] - df['away_xg']
        df['net_xg_per_60'] = (df['net_xg'] * 3600) / df['toi']

        # Cap extreme statistical outliers (e.g., from volatile micro-shifts)
        df['net_xg_per_60'] = df['net_xg_per_60'].clip(lower=-15.0, upper=15.0)

        # Export clean data
        out_file = os.path.join(self.out_dir, "whl_5v5_cleaned.csv")
        df.to_csv(out_file, index=False)
        print(f"Pipeline execution complete. Clean data saved to: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    pipe = HockeyPipeline()
    pipe.run()