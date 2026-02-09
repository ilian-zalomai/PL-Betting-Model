import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DATA_URL = 'https://raw.githubusercontent.com/ilian-zalomai/PL-Betting-Model/main/data/all_seasons.csv'
VALUE_BET_PROB_THRESHOLD = 0.4 # Default threshold for value bets (implied prob < this)

# --- Load Data (with caching for performance) ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding='latin1', on_bad_lines='skip')
    # Pre-process data for Brier Score and Value Bets
    df.dropna(subset=['Season', 'FTR', 'B365H'], inplace=True)
    df['B365H_Prob'] = 1 / df['B365H']
    df['HomeWin_Outcome'] = (df['FTR'] == 'H').astype(int)
    return df

# --- Brier Score Calculation ---
@st.cache_data
def calculate_brier_scores(df_data):
    df_data['SquaredError'] = (df_data['B365H_Prob'] - df_data['HomeWin_Outcome'])**2
    brier_scores = df_data.groupby('Season')['SquaredError'].mean().reset_index()
    brier_scores.rename(columns={'SquaredError': 'BrierScore'}, inplace=True)
    brier_scores = brier_scores.sort_values('Season')
    return brier_scores

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="PL Betting Efficiency Model")

st.title("Premier League Betting Efficiency Analysis")

df = load_data()

if df.empty:
    st.error("Could not load data. Please ensure 'all_seasons.csv' exists and contains data.")
else:
    # --- Sidebar for Season Selection ---
    st.sidebar.header("Filter Options")
    selected_season = st.sidebar.selectbox(
        "Select a Season:",
        options=df['Season'].unique().tolist(),
        index=len(df['Season'].unique()) - 1 # Default to the latest season
    )
    
    # Value bet threshold adjustment
    value_bet_threshold = st.sidebar.slider(
        "Value Bet Implied Probability Threshold (Lower is 'Better'):",
        min_value=0.1, max_value=0.9, value=VALUE_BET_PROB_THRESHOLD, step=0.01,
        help="Matches where (1 / Bookmaker_Home_Odds) < this threshold AND Home team won."
    )

    # Filter data for the selected season - used by multiple tabs
    season_df = df[df['Season'] == selected_season].copy()

    # Define tabs
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Efficiency Analysis", "Team Explorer"])

    with tab1:
        st.header(f"Market Overview for Season: {selected_season}")
        st.write("Displaying raw data for the selected season.")
        st.dataframe(season_df) # Display the raw data for the season

    with tab2:
        st.header("Bookmaker Efficiency (Brier Score) Trend")
        brier_scores_df = calculate_brier_scores(df)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(brier_scores_df['Season'], brier_scores_df['BrierScore'], marker='o', linestyle='-', color='dodgerblue')
        ax.set_title('B365 Home Win Odds Efficiency (Brier Score Trend)', fontsize=14)
        ax.set_xlabel('Season', fontsize=10)
        ax.set_ylabel('Brier Score (Lower is Better)', fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(
            """
            The Brier Score measures the accuracy of probabilistic predictions.
            A lower Brier Score indicates better predictive accuracy by the bookmaker.
            This chart shows the historical trend of B365 Home Win odds accuracy.
            """
        )

        st.header(f"Value Bets for Season: {selected_season}")

        # Identify "value bets" - Home Win where implied probability is below threshold
        # and the home team actually won.
        value_bets = season_df[
            (season_df['HomeWin_Outcome'] == 1) &
            (season_df['B365H_Prob'] < value_bet_threshold)
        ].copy()
        
        if not value_bets.empty:
            # Display relevant columns for value bets
            display_cols = [
                'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365H_Prob', 'HomeWin_Outcome'
            ]
            st.dataframe(value_bets[display_cols].sort_values(by='Date'))
            st.ma