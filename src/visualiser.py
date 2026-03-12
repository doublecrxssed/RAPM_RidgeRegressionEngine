import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class DashboardGenerator:
    """
    Visualization suite. 
    Generates presentation-ready graphics to summarize econometric findings.
    """
    def __init__(self, rapm_path="../data/outputs/rapm_ratings.csv", power_path="../data/outputs/power_rankings.csv", out_dir="../data/outputs/"):
        self.rapm_path = rapm_path
        self.power_path = power_path
        self.out_dir = out_dir
        self.BASE_GOALS = 2.5
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_disparity_dashboard(self):
        print("Generating Executive Disparity Dashboard...")
        if not os.path.exists(self.rapm_path) or not os.path.exists(self.power_path):
            print("Error: Missing CSVs. Ensure the simulator has been run.")
            return

        rapm = pd.read_csv(self.rapm_path)
        power = pd.read_csv(self.power_path)

        # Calculate disparity natively
        lines = rapm[rapm['Role'].isin(['first_off', 'second_off'])]
        pivot = lines.pivot(index='Team', columns='Role', values='RAPM_Rating').reset_index()
        pivot['Disparity_Ratio'] = (self.BASE_GOALS + pivot['first_off']) / (self.BASE_GOALS + pivot['second_off'])
        
        # Format team names
        def clean_name(name):
            if name.lower() in ['uk', 'usa']: return name.upper()
            return str(name).replace('_', ' ').title()
            
        pivot['Team'] = pivot['Team'].apply(clean_name)
        merged = pd.merge(pivot, power, on='Team')

        # Global aesthetics
        plt.rcParams['figure.facecolor'] = '#F8F9FA'
        plt.rcParams['axes.facecolor'] = '#F8F9FA'
        plt.rcParams['font.family'] = 'sans-serif'
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate correlation
        corr = merged['Disparity_Ratio'].corr(merged['Power_Score_100'])

        # Regression Plot
        sns.regplot(
            data=merged, x='Disparity_Ratio', y='Power_Score_100', ax=ax,
            scatter_kws={"s": 100, "color": "#002D72", "edgecolors": "white", "linewidths": 1.5, "alpha": 0.8},
            line_kws={"color": "#990000", "linewidth": 2, "linestyle": "--", "label": f"Linear Trend (r = {corr:.2f})"},
            ci=95
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color='#E2E8F0', linestyle='-', linewidth=1, alpha=0.7)
        ax.set_xlabel('Offensive Line Disparity Ratio (L1 xG / L2 xG)', fontweight='bold', labelpad=10)
        ax.set_ylabel('Total Team Strength (Power Score)', fontweight='bold', labelpad=10)
        ax.set_title('Strategic Roster Construction: Line Parity vs Overall Strength', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')

        # Annotate extreme outliers for context
        for i in range(len(merged)):
            x, y, team = merged['Disparity_Ratio'].iloc[i], merged['Power_Score_100'].iloc[i], merged['Team'].iloc[i]
            if x > merged['Disparity_Ratio'].quantile(0.9) or y > merged['Power_Score_100'].quantile(0.9):
                ax.annotate(team, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color='#334155')

        out_path = os.path.join(self.out_dir, "disparity_dashboard.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dashboard saved successfully to: {out_path}")

if __name__ == "__main__":
    viz = DashboardGenerator()
    viz.generate_disparity_dashboard()
