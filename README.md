# RAPM_RidgeRegressionEngine
A production-grade econometric pipeline for hockey analytics. Isolates true player talent using Regularized Adjusted Plus-Minus (RAPM) via Ridge Regression, and simulates calibrated win probabilities using Maximum Likelihood Estimation (MLE) and Pythagorean Expectation. Built by the winners of WHSDC 25'

# Generative Structural Inference for Hockey Analytics
**An open-source, econometric pipeline for isolating true talent and calibrating win probabilities.**

This repository contains the end-to-end data science architecture initially engineered for the **2026 Wharton Sports Data Science Competition (WHSDSC)**. 

Instead of relying on "black-box" machine learning classifiers (e.g., Random Forests, XGBoost) that are prone to overfitting on small sample sizes, this project utilizes **Regularized Adjusted Plus-Minus (RAPM)** and **Maximum Likelihood Estimation (MLE)** to mathematically isolate player impact and generate strictly calibrated game probabilities.

---

## The Mathematical Architecture

This pipeline solves the core problem in hockey analytics: **Collinearity**. Players share the ice with 5 other teammates, making it difficult to isolate individual contributions using traditional box-score statistics.

Our framework uses two distinct econometric engines:

### 1. The RAPM Engine (Ridge Regression)
We convert every 5v5 shift into a massive sparse design matrix (+1 for home units, -1 for away units) and regress it against Net Expected Goals (xG) per 60 minutes.
* By applying an **L2 Ridge Penalty (`RidgeCV`)**, we mathematically isolate the "True Quality" coefficient of all 161 individual lines, defensive pairings, and goaltenders.
* This inherently solves the "Strength of Schedule" problem and prevents small-sample outliers from corrupting the model.

### 2. The Generative Simulator (Pythagorean Expectation & MLE)
To translate RAPM coefficients into game predictions, we do not use ML classifiers. Instead, we use the Pythagorean Expectation formula. 
* We utilized **Maximum Likelihood Estimation (via L-BFGS-B optimization)** on historical game data to mathematically derive the league's optimal macro-parameters. 
* The optimizer proved that in this dataset, **Goalies (W_g = 0.660) are worth approximately twice as much as Skaters (W_s = 0.319)**, and established a custom Pythagorean exponent (gamma = 1.823).

---

## Project Structure

    whl-hockey-analytics/
    ├── data/
    │   ├── raw/                 # Contains raw datasets (ignored via .gitignore)
    │   ├── processed/           # Filtered 5v5 output data
    │   └── outputs/             # Generated deliverables, CSVs, and dashboards
    ├── src/
    │   ├── generate_dummy_data.py # Generates fake data for testing without proprietary WHL data
    │   ├── data_pipeline.py     # ETL: Filters 5v5, engineers Net xG/60 target variables
    │   ├── rapm_engine.py       # Sparse matrix construction & RidgeCV regression
    │   ├── optimizer.py         # L-BFGS-B optimization for Log-Loss minimization
    │   ├── simulator.py         # Pythagorean probability generation & Power Rankings
    │   ├── validation.py        # Monte Carlo stress testing & Brier score calibration
    │   └── visualizer.py        # Publication-ready Seaborn/Matplotlib dashboards
    └── README.md


---

## Statistical Validation Suite

A model is only as good as its stress tests. Running `src/validation.py` executes three custom statistical evaluations to prove the model's structural integrity:
1. **Probabilistic Calibration (Expected Brier Score = 0.2436):** Proves the model correctly identifies favorites without generating overconfident, overfit predictions.
2. **Ranking Stability (Spearman Rank r_s = 0.994):** Injects random statistical noise into 1,000 alternate universes to prove the 1-32 hierarchy is driven by true talent, not "puck-luck."
3. **Bootstrap Retention Rate:** Validates the persistence of top-heavy roster structures across Monte Carlo simulations.

---

## How to Run Locally

*Note: The original Wharton 2026 dataset is proprietary and excluded from this repository. We have provided a dummy data generator so you can test the pipeline locally.*

**1. Clone the repository and setup the environment:**
```bash
git clone [https://github.com/YourUsername/whl-hockey-analytics.git](https://github.com/YourUsername/whl-hockey-analytics.git)
cd whl-hockey-analytics
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```
**2. Generate the test data and run the pipeline sequentially:
```
python src/generate_dummy_data.py
python src/data_pipeline.py
python src/rapm_engine.py
python src/optimizer.py
python src/simulator.py
python src/validation.py
python src/visualizer.py
```
