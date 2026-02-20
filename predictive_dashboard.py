import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from predictive_model import (load_and_clean_data, calculate_elo, calculate_rolling_stats, 
                              train_and_evaluate, get_latest_stats, get_calibration_data)
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

# --- Page Config & Styling ---
st.set_page_config(layout="wide", page_title="PL Predictive Engine v2.0")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    div[data-testid="metric-container"] {
        background-color: #1e293b; border: 1px solid #334155; padding: 20px; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab-list"] { background-color: #1e293b; border-radius: 12px; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_full_engine():
    df = load_and_clean_data('all_seasons.csv')
    df = calculate_elo(df)
    df_with_stats = calculate_rolling_stats(df)
    model, le, features, test_data = train_and_evaluate(df_with_stats)
    latest = get_latest_stats(df_with_stats)
    return df, df_with_stats, model, le, features, test_data, latest

# --- Dashboard Layout ---
try:
    with st.spinner("Powering up Predictive Engine v2.0..."):
        df_raw, df_stats, model, le, features, test_data, latest = load_full_engine()

    st.title("⚽ Premier League Predictive Engine v2.0")
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("View", ["Advanced Dashboard", "Calibration & Reliability", "Market vs Model", "Instructor Documentation"])

    if mode == "Advanced Dashboard":
        # Multi-tab dashboard
        t1, t2 = st.tabs(["Active Prediction Terminal", "Elo Strength Tracker"])
        
        with t1:
            st.header("Predictive Match Terminal")
            c1, c2 = st.columns(2)
            teams = sorted(latest['Team'].unique())
            h_team = c1.selectbox("Home Team", teams, index=teams.index('Liverpool'))
            a_team = c2.selectbox("Away Team", teams, index=teams.index('Arsenal'))
            
            st.markdown("#### Market Odds Comparison")
            o1, o2, o3 = st.columns(3)
            h_odds = o1.number_input("Best Home Odds", value=2.0)
            d_odds = o2.number_input("Best Draw Odds", value=3.4)
            a_odds = o3.number_input("Best Away Odds", value=3.5)
            
            if st.button("Generate Intelligence Report"):
                h_s, a_s = latest[latest['Team'] == h_team].iloc[0], latest[latest['Team'] == a_team].iloc[0]
                inp = pd.DataFrame([{
                    'Home_Rolling_GoalsFor': h_s['Rolling_GoalsFor'], 'Home_Rolling_GoalsAgainst': h_s['Rolling_GoalsAgainst'],
                    'Home_Rolling_Shots': h_s['Rolling_Shots'], 'Home_Rolling_ShotsOnTarget': h_s['Rolling_ShotsOnTarget'], 'Home_Rolling_Corners': h_s['Rolling_Corners'],
                    'Away_Rolling_GoalsFor': a_s['Rolling_GoalsFor'], 'Away_Rolling_GoalsAgainst': a_s['Rolling_GoalsAgainst'],
                    'Away_Rolling_Shots': a_s['Rolling_Shots'], 'Away_Rolling_ShotsOnTarget': a_s['Rolling_ShotsOnTarget'], 'Away_Rolling_Corners': a_s['Rolling_Corners'],
                    'Home_Elo': h_s['Elo'], 'Away_Elo': a_s['Elo'], 'Elo_Diff': h_s['Elo'] - a_s['Elo']
                }])
                
                probs = model.predict_proba(inp[features])[0]
                class_map = {cls: i for i, cls in enumerate(le.classes_)}
                p_h, p_d, p_a = probs[class_map['H']], probs[class_map['D']], probs[class_map['A']]
                
                st.divider()
                col_res1, col_res2 = st.columns([2, 1])
                with col_res1:
                    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#0e1117')
                    ax.set_facecolor('#0e1117')
                    ax.bar(['Home', 'Draw', 'Away'], [p_h, p_d, p_a], color=['#ef4444', '#3b82f6', '#10b981'])
                    ax.tick_params(colors='#94a3b8')
                    st.pyplot(fig)
                
                with col_res2:
                    st.write("**Elo Rating Check**")
                    st.write(f"{h_team}: {int(h_s['Elo'])}")
                    st.write(f"{a_team}: {int(a_s['Elo'])}")
                    st.write(f"Diff: {int(h_s['Elo'] - a_s['Elo'])}")
                    
                    ev_h = p_h * h_odds - 1
                    if ev_h > 0.05: st.success(f"VALUE FOUND: Home (EV: {ev_h:.1%})")
                    else: st.info("No significant value detected.")

        with t2:
            st.header("Elo Rating Trajectory")
            team_to_track = st.selectbox("Select Team to track Elo", teams)
            # Filter raw df for this team
            team_elo = df_raw[(df_raw['HomeTeam'] == team_to_track) | (df_raw['AwayTeam'] == team_to_track)].copy()
            team_elo['Current_Elo'] = team_elo.apply(lambda r: r['Home_Elo'] if r['HomeTeam'] == team_to_track else r['Away_Elo'], axis=1)
            
            fig_elo, ax_elo = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
            ax_elo.set_facecolor('#0e1117')
            ax_elo.plot(team_elo['Date'], team_elo['Current_Elo'], color='#3b82f6')
            ax_elo.tick_params(colors='#94a3b8')
            plt.grid(alpha=0.2)
            st.pyplot(fig_elo)

    elif mode == "Calibration & Reliability":
        st.header("Model Reliability & Calibration")
        st.markdown("""
        **Reliability Diagram:** This plot checks if the model's predicted probabilities match the real-world frequency of outcomes. 
        If a model says a team has a 70% chance of winning, they should actually win 70% of the time.
        """)
        
        y_test = test_data['Target']
        prob_true, prob_pred = get_calibration_data(model, test_data[features], y_test, le)
        
        fig_cal, ax_cal = plt.subplots(figsize=(6, 6), facecolor='#0e1117')
        ax_cal.set_facecolor('#0e1117')
        ax_cal.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax_cal.plot(prob_pred, prob_true, "s-", color='#3b82f6', label="Random Forest")
        ax_cal.set_ylabel("Fraction of positives", color='white')
        ax_cal.set_xlabel("Mean predicted probability", color='white')
        ax_cal.tick_params(colors='white')
        st.pyplot(fig_cal)

    elif mode == "Market vs Model":
        st.header("Comparative Market Analysis")
        st.write("Comparing our ML model against multiple bookmakers (B365, Bwin, Interwetten, etc.)")
        
        # Calculate Brier Scores for different bookies on the test set
        bookies = {'B365': 'B365H', 'Bwin': 'BWH', 'Interwetten': 'IWH', 'VCBet': 'VCH', 'Model': 'Prob_H'}
        
        probs = model.predict_proba(test_data[features])
        class_map = {cls: i for i, cls in enumerate(le.classes_)}
        test_data['Prob_H'] = probs[:, class_map['H']]
        test_data['HomeWin_Actual'] = (test_data['FTR'] == 'H').astype(int)
        
        brier_results = []
        for name, col in bookies.items():
            if col in test_data.columns:
                temp_df = test_data.dropna(subset=[col])
                if name == 'Model':
                    b_score = np.mean((temp_df[col] - temp_df['HomeWin_Actual'])**2)
                else:
                    implied_prob = 1 / temp_df[col]
                    b_score = np.mean((implied_prob - temp_df['HomeWin_Actual'])**2)
                brier_results.append({'Bookmaker': name, 'Brier Score': b_score})
        
        st.table(pd.DataFrame(brier_results).sort_values('Brier Score'))
        st.info("A lower Brier score represents higher accuracy. If the Model is close to or better than the bookmakers, it demonstrates significant depth.")

    elif mode == "Instructor Documentation":
        st.header("Project 2: Technical Documentation & AI Usage")
        st.markdown("""
        ### 1. Improvements over Project 1
        - **Transition from Retrospective to Predictive:** Project 1 calculated value based on known outcomes. Project 2 uses a **Random Forest Classifier** to estimate probabilities *before* the match occurs.
        - **Deeper Feature Engineering:** Added **Elo Ratings** (dynamic team strength) and **Rolling Performance Stats** (last 5 games of goals, shots, corners).
        - **Reliability Testing:** Implemented Calibration Curves to verify model overconfidence.
        - **Market Benchmarking:** Compared model performance against 4+ global bookmakers.

        ### 2. AI Leverage Documentation
        - **Architecture Design:** AI (Gemini CLI) was used to design the feature engineering pipeline and the rolling statistics logic.
        - **Mathematical Validation:** AI assisted in implementing the Elo rating formula and the Brier score comparative analysis.
        - **UI/UX Engineering:** AI provided the CSS and layout structure for this Streamlit Pro dashboard.
        - **Debugging:** AI assisted in resolving Streamlit deprecation warnings and environment dependency issues.

        ### 3. Reproducibility
        - Code is structured into `predictive_model.py` (Engine) and `predictive_dashboard.py` (Interface).
        - Environment managed via `requirements.txt`.
        - Version history maintained on GitHub.
        """)

except Exception as e:
    st.error(f"System Error: {e}")
