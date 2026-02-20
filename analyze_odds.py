import pandas as pd
import matplotlib.pyplot as plt

def analyze_brier_score_trend():
    """
    Analyzes the Brier score for B365 home win odds over seasons.
    """
    try:
        df = pd.read_csv('all_seasons.csv')
    except FileNotFoundError:
        print("Error: all_seasons.csv not found. Please run fetch_data.py first.")
        return

    # --- Data Cleaning and Preparation ---
    # Select relevant columns
    relevant_cols = ['Season', 'FTR', 'B365H']
    df = df[relevant_cols].copy()

    # Drop rows with missing values in the essential columns
    df.dropna(subset=relevant_cols, inplace=True)
    
    # Convert odds to probabilities
    # The implied probability is 1 / decimal odds
    df['B365H_Prob'] = 1 / df['B365H']

    # Create the outcome column (1 if Home Team Won, 0 otherwise)
    df['HomeWin_Outcome'] = (df['FTR'] == 'H').astype(int)

    # --- Brier Score Calculation ---
    # Calculate the squared error for each match
    df['SquaredError'] = (df['B365H_Prob'] - df['HomeWin_Outcome'])**2

    # Group by season and calculate the mean squared error (Brier Score)
    brier_scores_by_season = df.groupby('Season')['SquaredError'].mean().reset_index()
    brier_scores_by_season.rename(columns={'SquaredError': 'BrierScore'}, inplace=True)

    # Sort by season to ensure correct plotting order
    brier_scores_by_season = brier_scores_by_season.sort_values('Season')

    print("Brier Scores per Season:")
    print(brier_scores_by_season)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(brier_scores_by_season['Season'], brier_scores_by_season['BrierScore'], marker='o', linestyle='-', color='dodgerblue')

    # Formatting the plot
    ax.set_title('B365 Home Win Odds Efficiency (Brier Score Trend)', fontsize=16)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Brier Score (Lower is Better)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the figure
    output_filename = 'efficiency_trend.png'
    plt.savefig(output_filename)
    print(f"Chart saved as {output_filename}")

if __name__ == "__main__":
    analyze_brier_score_trend()
