import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from predictive_model import load_and_clean_data, calculate_rolling_stats, train_and_evaluate, get_latest_stats

warnings.filterwarnings('ignore')

# --- Configuration & Styling ---
st.set_page_config(layout="wide", page_title="PL Betting Evolution: Project 1 to 2")

# CSS to make it look polished
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_all_with_code=True)

# --- Data Loading (Cached) ---
@st.cache_data
def load_all_data():
    # Load for ML (predictive_model logic)
    df_ml = load_and_clean_data('all_seasons.csv')
    df_with_stats = calculate_rolling_stats(df_ml)
    model, le, features, test_data, latest_stats = None, None, None, None, None
    
    try:
        model, le, features, test_data = train_and_evaluate(df_with_stats)
        latest_stats = get_latest_stats(df_with_stats)
    except Exception as e:
        st.error(f"ML Training Error: {e}")
    
    # Load for Project 1 logic (Historical/Brier)
    df_hist = pd.read_csv('all_seasons.csv', low_memory=False)
    df_hist.columns = df_hist.columns.str.replace('ï»¿', '').str.strip()
    df_hist.dropna(subset=['Season', 'FTR', 'B365H'], inplace=True)
    df_hist['B365H_Prob'] = 1 / df_hist['B365H']
    df_hist['HomeWin_Outcome'] = (df_hist['FTR'] == 'H').astype(int)
    
    return df_ml, df_with_stats, model, le, features, test_data, latest_stats, df_hist

# --- Original Project 1 Logic ---
def calculate_brier_scores(df_data):
    df_data['SquaredError'] = (df_data['B365H_Prob'] - df_data['HomeWin_Outcome'])**2
    brier_scores = df_data.groupby('Season')['SquaredError'].mean().reset_index()
    brier_scores.rename(columns={'SquaredError': 'BrierScore'}, inplace=True)
    return brier_scores.sort_values('Season')

# --- Main App Execution ---
st.title("⚽ Premier League Betting Model Evolution")
st.markdown("### From Historical Efficiency (Project 1) to Predictive ML (Project 2)")

try:
    with st.spinner("Processing 20 years of data and training ML model..."):
        df_ml, df_with_stats, model, le, features, test_data, latest, df_hist = load_all_data()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Analysis Phase", 
                                ["Phase 1: Historical Efficiency", 
                                 "Phase 2: Predictive Machine Learning"])

    if app_mode == "Phase 1: Historical Efficiency":
        st.sidebar.info("This section contains your original Project 1 logic analyzing past data.")
        
        tab1, tab2, tab3 = st.tabs(["Market Efficiency", "Team Explorer", "Raw Data Overview"])
        
        with tab1:
            st.header("Bookmaker Efficiency Analysis")
            brier_scores_df = calculate_brier_scores(df_hist)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(brier_scores_df['Season'], brier_scores_df['BrierScore'], marker='o', color='#2ecc71', linewidth=2)
            ax.set_title('B365 Home Win Efficiency (Brier Score Trend)', fontsize=14)
            ax.set_ylabel('Brier Score (Lower = More Accurate)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.info("**Brier Score Analysis:** This shows how 'correct' the bookmakers were historically.")

        with tab2:
            st.header("Historical Team Explorer")
            selected_season = st.selectbox("Select Season", df_hist['Season'].unique()[::-1])
            season_df = df_hist[df_hist['Season'] == selected_season]
            all_teams = sorted(pd.unique(season_df[['HomeTeam', 'AwayTeam']].values.ravel('K')))
            selected_team = st.selectbox("Select Team", all_teams)
            team_matches = season_df[(season_df['HomeTeam'] == selected_team) | (season_df['AwayTeam'] == selected_team)]
            
            col1, col2, col3 = st.columns(3)
            wins = len(team_matches[((team_matches['HomeTeam'] == selected_team) & (team_matches['FTR'] == 'H')) | 
                                    ((team_matches['AwayTeam'] == selected_team) & (team_matches['FTR'] == 'A'))])
            col1.metric("Wins", wins)
            col2.metric("Total Goals", int(team_matches[team_matches['HomeTeam'] == selected_team]['FTHG'].sum() + 
                                       team_matches[team_matches['AwayTeam'] == selected_team]['FTAG'].sum()))
            col3.metric("Avg. Closing Odds", f"{team_matches[team_matches['HomeTeam'] == selected_team]['B365H'].mean():.2f}")
            st.dataframe(team_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].sort_values('Date', ascending=False))

        with tab3:
            st.header("Project 1: Raw Data Overview")
            st.dataframe(df_hist.head(100))

    else: # Phase 2: Predictive Machine Learning
        st.sidebar.warning("This section contains your new Project 2 logic using a Random Forest Classifier.")
        
        if model is None:
            st.error("ML Model could not be initialized. Check data quality.")
        else:
            tab_ml1, tab_ml2, tab_ml3 = st.tabs(["Model Diagnostics", "Value Discovery", "Match Predictor"])
            
            with tab_ml1:
                st.header("Random Forest Model Performance")
                last_season = df_ml['Season'].unique()[-1]
                y_test = test_data['Target']
                y_pred = model.predict(test_data[features])
                from sklearn.metrics import accuracy_score, confusion_matrix
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
                col2.metric("Test Season", last_season)
                col3.metric("Features Used", len(features))
                
                st.subheader("Confusion Matrix: Predicted vs Actual")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                st.pyplot(fig)

            with tab_ml2:
                st.header("Finding Betting Value")
                ev_threshold = st.slider("Min Expected Value (EV) %", 0, 50, 10) / 100
                probs = model.predict_proba(test_data[features])
                class_map = {cls: i for i, cls in enumerate(le.classes_)}
                
                td = test_data.copy()
                for outcome in ['H', 'D', 'A']:
                    td[f'Prob_{outcome}'] = probs[:, class_map[outcome]]
                    td[f'EV_{outcome}'] = (td[f'Prob_{outcome}'] * td[f'B365{outcome}']) - 1
                
                value_list = []
                for _, row in td.iterrows():
                    for outcome in ['H', 'D', 'A']:
                        if row[f'EV_{outcome}'] > ev_threshold:
                            value_list.append({
                                'Date': row['Date'],
                                'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                                'Pick': outcome,
                                'Odds': row[f'B365{outcome}'],
                                'Model Prob': f"{row[f'Prob_{outcome}']:.1%}",
                                'EV': f"{row[f'EV_{outcome}']:.1%}",
                                'Result': 'Won' if row['FTR'] == outcome else 'Lost'
                            })
                
                v_df = pd.DataFrame(value_list)
                if not v_df.empty:
                    st.dataframe(v_df.sort_values('Date', ascending=False), use_container_width=True)
                else:
                    st.warning("No value bets found with current EV threshold.")

            with tab_ml3:
                st.header("Live Match Prediction Tool")
                teams = sorted(latest['Team'].unique())
                c1, c2 = st.columns(2)
                h_team = c1.selectbox("Home Team", teams, index=teams.index('Liverpool') if 'Liverpool' in teams else 0)
                a_team = c2.selectbox("Away Team", teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 1)
                
                st.subheader("Current Market Odds")
                o1, o2, o3 = st.columns(3)
                h_o = o1.number_input("Home Odds", value=2.0)
                d_o = o2.number_input("Draw Odds", value=3.4)
                a_o = o3.number_input("Away Odds", value=3.5)
                
                if st.button("Run ML Prediction"):
                    h_stats = latest[latest['Team'] == h_team].iloc[0]
                    a_stats = latest[latest['Team'] == a_team].iloc[0]
                    
                    input_row = pd.DataFrame([{
                        'Home_Rolling_GoalsFor': h_stats['Rolling_GoalsFor'],
                        'Home_Rolling_GoalsAgainst': h_stats['Rolling_GoalsAgainst'],
                        'Home_Rolling_Shots': h_stats['Rolling_Shots'],
                        'Home_Rolling_ShotsOnTarget': h_stats['Rolling_ShotsOnTarget'],
                        'Home_Rolling_Corners': h_stats['Rolling_Corners'],
                        'Away_Rolling_GoalsFor': a_stats['Rolling_GoalsFor'],
                        'Away_Rolling_GoalsAgainst': a_stats['Rolling_GoalsAgainst'],
                        'Away_Rolling_Shots': a_stats['Rolling_Shots'],
                        'Away_Rolling_ShotsOnTarget': a_stats['Rolling_ShotsOnTarget'],
                        'Away_Rolling_Corners': a_stats['Rolling_Corners']
                    }])
                    
                    res_probs = model.predict_proba(input_row[features])[0]
                    p_h, p_d, p_a = res_probs[class_map['H']], res_probs[class_map['D']], res_probs[class_map['A']]
                    
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    sns.barplot(x=['Home', 'Draw', 'Away'], y=[p_h, p_d, p_a], palette='viridis', ax=ax2)
                    st.pyplot(fig2)
                    
                    evs = [p_h*h_o - 1, p_d*d_o - 1, p_a*a_o - 1]
                    best_idx = np.argmax(evs)
                    if evs[best_idx] > 0.05:
                        st.success(f"**Best Value Found:** {['Home', 'Draw', 'Away'][best_idx]} (EV: {evs[best_idx]:.1%})")
                    else:
                        st.info("No significant value found.")

except Exception as e:
    st.error(f"Dashboard Error: {e}")
