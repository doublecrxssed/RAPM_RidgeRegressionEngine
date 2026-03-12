import pandas as pd
import numpy as np
import os
import random

# Dynamically resolve absolute paths to prevent working-directory bugs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "whl_2025.csv")

def make_fake_data(num_rows=1000, out_path=DEFAULT_OUT_PATH):
    print(f"Generating {num_rows} rows of dummy WHL data...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    teams = [f"team_{i}" for i in range(1, 33)]
    lines = ["first_off", "second_off", "first_def", "second_def", "PP_up", "PK_kill_dwn"]
    goalies = [f"goalie_{i}" for i in range(1, 65)]

    data = []
    for i in range(num_rows):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        
        row = {
            "game_id": f"game_{i // 50}",
            "record_id": f"rec_{i}",
            "home_team": home,
            "away_team": away,
            "went_ot": random.choices([0, 1], weights=[0.9, 0.1])[0],
            
            "home_off_line": random.choice(lines),
            "home_def_pairing": random.choice(lines),
            "away_off_line": random.choice(lines),
            "away_def_pairing": random.choice(lines),
            
            "home_goalie": random.choice(goalies),
            "away_goalie": random.choice(goalies),
            
            "toi": round(random.uniform(10.0, 120.0), 1),
            "home_xg": round(random.uniform(0, 0.2), 4),
            "away_xg": round(random.uniform(0, 0.2), 4),
            
            "home_shots": random.randint(0, 3),
            "away_shots": random.randint(0, 3),
            "home_goals": random.choices([0, 1], weights=[0.9, 0.1])[0],
            "away_goals": random.choices([0, 1], weights=[0.9, 0.1])[0]
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    print(f"Saved dummy dataset successfully to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    make_fake_data()
