import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from predictive_model import load_and_clean_data, calculate_rolling_stats, train_and_evaluate, get_latest_stats

warnings.filterwarnings('ignore')

# --- Configuration & Styling ---
st.set_page_config(layout="wide", page_title="PL Betting Evolution: Project 1 & 2")

# Custom CSS for a senior engineer look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #e0e2e6; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading (Shared and Cached) ---
@st.cache_data
def load_all_dashboard_data():
    # 1. Load for Predictive Model (ML)
    df_ml_base = load_and_clean_data('all_seasons.csv')
    df_with_stats = calculate_rolling_stats(df_ml_base)
    model, le, features, test_data = train_and_evaluate(df_with_stats)
    latest_stats = get_latest_stats(df_with_stats)
    
    # 2. Load for Original App Logic (Historical)
    df_hist = pd.read_csv('all_seasons.csv', low_memory=False)
    df_hist.columns = df_hist.columns.str.replace('ï»¿', '').str.strip()
    df_hist.dropna(subset=['Season', 'FTR', 'B365H'], inplace=True)
    df_hist['B365H_Prob'] = 1 / df_hist['B365H']
    df_hist['HomeWin_Outcome'] = (df_hist['FTR'] == 'H').astype(int)
    
    return df_ml_base, df_with_stats, model, le, features, test_data, latest_stats, df_hist

# --- Original Project 1 Logic Function ---
def calculate_brier_scores(df_data):
    df_data['SquaredError'] = (df_data['B365H_Prob'] - df_data['HomeWin_Outcome'])**2
    brier_scores = df_data.groupby('Season')['SquaredError'].mean().reset_index()
    brier_scores.rename(columns={'SquaredError': 'BrierScore'}, inplace=True)
    return brier_scores.sort_values('Season')

# --- App Layout ---
st.title("⚽ Premier League Betting Model Dashboard")
st.markdown("### Integrated Project 1 (Historical) & Project 2 (Predictive)")

try:
    with st.spinner("Initializing models and loading 20+ seasons of data..."):
        df_ml, df_with_stats, model, le, features, test_data, latest, df_hist = load_all_dashboard_data()

    # Create Tabs for the two projects
    tab_p1, tab_p2 = st.tabs(["📊 Project 1: Historical Efficiency", "🤖 Project 2: Advanced Prediction"])

    # --- TAB 1: PROJECT 1 (Historical Logic) ---
    with tab_p1:
        st.header("Phase 1: Market Efficiency & Brier Scores")
        st.info("This section preserves your original analysis of how bookmaker odds performed historically.")
        
        # Brier Score Plot
        brier_scores_df = calculate_brier_scores(df_hist)
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(brier_scores_df['Season'], brier_scores_df['BrierScore'], marker='o', linestyle='-', color='#3498db')
        ax1.set_title('B365 Home Win Odds Efficiency Trend', fontsize=12)
        ax1.set_ylabel('Brier Score (Lower is More Accurate)')
        plt.xticks(rotation=45, fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)
        
        # Historical Explorer
        st.divider()
        st.subheader("Historical Season Explorer")
        sel_season = st.selectbox("Select Season to Review", df_hist['Season'].unique()[::-1], key="p1_season")
        season_df = df_hist[df_hist['Season'] == sel_season]
        
        all_teams = sorted(pd.unique(season_df[['HomeTeam', 'AwayTeam']].values.ravel('K')))
        sel_team = st.selectbox("Select Team", all_teams, key="p1_team")
        
        team_matches = season_df[(season_df['HomeTeam'] == sel_team) | (season_df['AwayTeam'] == sel_team)]
        
        c1, c2, c3, c4 = st.columns(4)
        wins = len(team_matches[((team_matches['HomeTeam'] == sel_team) & (team_matches['FTR'] == 'H')) | 
                                ((team_matches['AwayTeam'] == sel_team) & (team_matches['FTR'] == 'A'))])
        c1.metric("Wins", wins)
        c2.metric("Matches", len(team_matches))
        c3.metric("Goals For", int(team_matches[team_matches['HomeTeam'] == sel_team]['FTHG'].sum() + 
                                  team_matches[team_matches['AwayTeam'] == sel_team]['FTAG'].sum()))
        c4.metric("Avg Odds", f"{team_matches[team_matches['HomeTeam'] == sel_team]['B365H'].mean():.2f}")
        
        st.dataframe(team_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].sort_values('Date', ascending=False), use_container_width=True)

    # --- TAB 2: PROJECT 2 (Predictive ML Logic) ---
    with tab_p2:
        st.header("Phase 2: Machine Learning & Value Finder")
        st.markdown("This advanced analysis uses a **Random Forest Classifier** to find betting value by comparing model probabilities to bookmaker odds.")
        
        ml_sub_tab1, ml_sub_tab2, ml_sub_tab3 = st.tabs(["Model Accuracy", "Current Value Bets", "Upcoming Match Predictor"])
        
        with ml_sub_tab1:
            st.subheader("Model Validation (Recent Season)")
            last_season = df_ml['Season'].unique()[-1]
            
            y_test = test_data['Target']
            y_pred = model.predict(test_data[features])
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, y_pred)
            
            col_acc1, col_acc2 = st.columns(2)
            col_acc1.metric("Predictive Accuracy", f"{acc:.1%}")
            col_acc1.write(f"**Test Season:** {last_season}")
            col_acc1.write(f"**ML Features:** Goals, Shots, Shots on Target, Corners (Rolling 5-game Averages)")
            
            # Simple bar chart for class distribution
            st.write("**Outcome Prediction Accuracy (Recent Matches)**")
            test_results = test_data.copy()
            test_results['Predicted'] = le.inverse_transform(y_pred)
            correct_count = (test_results['Predicted'] == test_results['FTR']).sum()
            st.progress(acc)
            st.write(f"Model correctly predicted {correct_count} out of {len(test_data)} matches in the test set.")

        with ml_sub_tab2:
            st.subheader("Value Discovery Analysis")
            st.write("Matches where our model's probability exceeds the bookmaker's implied probability.")
            
            ev_threshold = st.slider("Select Min Expected Value (EV) %", 0, 50, 10, key="ml_ev_slider") / 100
            
            # Calculate EVs
            probs = model.predict_proba(test_data[features])
            class_map = {cls: i for i, cls in enumerate(le.classes_)}
            
            td = test_data.copy()
            value_bets = []
            for outcome in ['H', 'D', 'A']:
                p_col = probs[:, class_map[outcome]]
                ev_col = (p_col * td[f'B365{outcome}']) - 1
                
                for i, ev in enumerate(ev_col):
                    if ev > ev_threshold:
                        row = td.iloc[i]
                        value_bets.append({
                            'Date': row['Date'],
                            'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                            'Pick': outcome,
                            'Odds': row[f'B365{outcome}'],
                            'Model Prob': f"{p_col[i]:.1%}",
                            'EV': f"{ev:.1%}",
                            'Actual FTR': row['FTR']
                        })
            
            v_df = pd.DataFrame(value_bets)
            if not v_df.empty:
                v_df['Won'] = v_df.apply(lambda r: r['Pick'] == r['Actual FTR'], axis=1)
                st.dataframe(v_df.sort_values('Date', ascending=False), use_container_width=True)
                st.success(f"Found {len(v_df)} value bets with >{ev_threshold:.0%} EV.")
            else:
                st.warning("No value bets found with the selected threshold.")

        with ml_sub_tab3:
            st.subheader("Interactive Match Predictor")
            st.write("Input any upcoming fixture and odds to find 'Value'.")
            
            teams = sorted(latest['Team'].unique())
            c1, c2 = st.columns(2)
            h_team = c1.selectbox("Home Team", teams, index=teams.index('Man United') if 'Man United' in teams else 0)
            a_team = c2.selectbox("Away Team", teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 1)
            
            st.write("**Market Odds (Closing/Current)**")
            o1, o2, o3 = st.columns(3)
            h_odds = o1.number_input("Home Odds", value=2.0, min_value=1.01)
            d_odds = o2.number_input("Draw Odds", value=3.4, min_value=1.01)
            a_odds = o3.number_input("Away Odds", value=3.5, min_value=1.01)
            
            if st.button("Generate ML Forecast"):
                # Get stats for teams
                h_s = latest[latest['Team'] == h_team].iloc[0]
                a_s = latest[latest['Team'] == a_team].iloc[0]
                
                input_row = pd.DataFrame([{
                    'Home_Rolling_GoalsFor': h_s['Rolling_GoalsFor'],
                    'Home_Rolling_GoalsAgainst': h_s['Rolling_GoalsAgainst'],
                    'Home_Rolling_Shots': h_s['Rolling_Shots'],
                    'Home_Rolling_ShotsOnTarget': h_s['Rolling_ShotsOnTarget'],
                    'Home_Rolling_Corners': h_s['Rolling_Corners'],
                    'Away_Rolling_GoalsFor': a_s['Rolling_GoalsFor'],
                    'Away_Rolling_GoalsAgainst': a_s['Rolling_GoalsAgainst'],
                    'Away_Rolling_Shots': a_s['Rolling_Shots'],
                    'Away_Rolling_ShotsOnTarget': a_s['Rolling_ShotsOnTarget'],
                    'Away_Rolling_Corners': a_s['Rolling_Corners']
                }])
                
                res_probs = model.predict_proba(input_row[features])[0]
                p_h, p_d, p_a = res_probs[class_map['H']], res_probs[class_map['D']], res_probs[class_map['A']]
                
                st.divider()
                st.write(f"### Forecast for {h_team} vs {a_team}")
                
                res_c1, res_c2 = st.columns([2, 1])
                with res_c1:
                    # Probabilities display
                    fig_res, ax_res = plt.subplots(figsize=(6, 3))
                    ax_res.bar(['Home', 'Draw', 'Away'], [p_h, p_d, p_a], color=['#e74c3c', '#3498db', '#2ecc71'])
                    ax_res.set_ylabel("Probability")
                    st.pyplot(fig_res)
                
                with res_c2:
                    st.write("**Value Analysis**")
                    ev_h = (p_h * h_odds) - 1
                    ev_d = (p_d * d_odds) - 1
                    ev_a = (p_a * a_odds) - 1
                    
                    st.write(f"Home EV: {ev_h:.1%}")
                    st.write(f"Draw EV: {ev_d:.1%}")
                    st.write(f"Away EV: {ev_a:.1%}")
                    
                    best_ev = max(ev_h, ev_d, ev_a)
                    if best_ev > 0.05:
                        pick = ['Home', 'Draw', 'Away'][np.argmax([ev_h, ev_d, ev_a])]
                        st.success(f"**PICK: {pick}**")
                    else:
                        st.info("No significant value.")

except Exception as e:
    st.error(f"Critical Error: {e}")
    st.info("Ensure all_seasons.csv is available and predictive_model.py is in the same directory.")
