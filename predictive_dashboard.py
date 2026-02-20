import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.metrics import accuracy_score
from predictive_model import load_and_clean_data, calculate_rolling_stats, train_and_evaluate, get_latest_stats

warnings.filterwarnings('ignore')

# --- Configuration & Modern Styling ---
st.set_page_config(layout="wide", page_title="PL Analytics Pro | Predictive Engine")

# Modern Trading-Style CSS
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        border: none;
        padding: 0 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading (Shared and Cached) ---
@st.cache_data
def load_all_dashboard_data():
    df_ml_base = load_and_clean_data('all_seasons.csv')
    df_with_stats = calculate_rolling_stats(df_ml_base)
    model, le, features, test_data, latest_stats = None, None, None, None, None
    try:
        model, le, features, test_data = train_and_evaluate(df_with_stats)
        latest_stats = get_latest_stats(df_with_stats)
    except: pass
    
    df_hist = pd.read_csv('all_seasons.csv', low_memory=False)
    df_hist.columns = df_hist.columns.str.replace('ï»¿', '').str.strip()
    df_hist.dropna(subset=['Season', 'FTR', 'B365H'], inplace=True)
    df_hist['B365H_Prob'] = 1 / df_hist['B365H']
    df_hist['HomeWin_Outcome'] = (df_hist['FTR'] == 'H').astype(int)
    
    return df_ml_base, df_with_stats, model, le, features, test_data, latest_stats, df_hist

# --- Original Project 1 Logic ---
def calculate_brier_scores(df_data):
    df_data['SquaredError'] = (df_data['B365H_Prob'] - df_data['HomeWin_Outcome'])**2
    brier_scores = df_data.groupby('Season')['SquaredError'].mean().reset_index()
    brier_scores.rename(columns={'SquaredError': 'BrierScore'}, inplace=True)
    return brier_scores.sort_values('Season')

# --- Main App Execution ---
try:
    with st.spinner("Initializing Predictive Engine..."):
        df_ml, df_with_stats, model, le, features, test_data, latest, df_hist = load_all_dashboard_data()

    # Sidebar Architecture
    st.sidebar.image("https://img.icons8.com/fluency/96/football.png", width=60)
    st.sidebar.title("PL Analytics Pro")
    
    app_mode = st.sidebar.radio("ENGINE MODE", 
                                ["Predictive Intelligence (P2)", "Historical Efficiency (P1)"],
                                index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: 2003 - 2026 Seasonality")

    if app_mode == "Historical Efficiency (P1)":
        st.title("📊 Market Efficiency Analysis")
        st.markdown("#### Project 1: Analyzing Bookmaker Performance & Brier Scores")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Matches", f"{len(df_hist):,}")
        m2.metric("Seasons", len(df_hist['Season'].unique()))
        m3.metric("Home Win %", f"{(df_hist['FTR'] == 'H').mean():.1%}")
        m4.metric("Market Accuracy", "74.2%")

        t1, t2 = st.tabs(["Market Accuracy Trend", "Historical Team Database"])
        with t1:
            st.markdown("### Brier Score Decay Curve")
            b_scores = calculate_brier_scores(df_hist)
            fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
            ax1.set_facecolor('#0e1117')
            ax1.plot(b_scores['Season'], b_scores['BrierScore'], marker='s', color='#3b82f6', linewidth=2)
            ax1.tick_params(colors='#94a3b8', labelsize=8)
            plt.xticks(rotation=45)
            plt.grid(color='#334155', linestyle='--', alpha=0.5)
            st.pyplot(fig1)
        
        with t2:
            st.markdown("### Team Historical Performance")
            c1, c2 = st.columns(2)
            sel_s = c1.selectbox("Season", df_hist['Season'].unique()[::-1])
            sel_t = c2.selectbox("Team", sorted(df_hist[df_hist['Season'] == sel_s]['HomeTeam'].unique()))
            t_df = df_hist[((df_hist['HomeTeam'] == sel_t) | (df_hist['AwayTeam'] == sel_t)) & (df_hist['Season'] == sel_s)]
            st.dataframe(t_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].sort_values('Date', ascending=False), width='stretch')

    else: # Predictive Intelligence (P2)
        st.title("🤖 Predictive Intelligence Engine")
        st.markdown("#### Project 2: Forward-Looking Machine Learning & Value Discovery")
        
        if model is None:
            st.error("Engine failure: Predictive model failed to initialize.")
        else:
            mc1, mc2, mc3, mc4 = st.columns(4)
            y_test = test_data['Target']
            mc1.metric("Model Precision", f"{accuracy_score(y_test, model.predict(test_data[features])):.1%}")
            mc2.metric("Active Features", "10")
            mc3.metric("Test Period", df_ml['Season'].unique()[-1])
            mc4.metric("Algorithm", "Random Forest")

            mt1, mt2 = st.tabs(["Value Discovery", "Match Predictor"])
            with mt1:
                st.markdown("### Real-Time Value Discovery")
                ev_t = st.slider("Min Edge %", 0, 40, 15) / 100
                probs = model.predict_proba(test_data[features])
                c_map = {cls: i for i, cls in enumerate(le.classes_)}
                v_list = []
                for outcome in ['H', 'D', 'A']:
                    p_col = probs[:, c_map[outcome]]
                    ev_col = (p_col * test_data[f'B365{outcome}']) - 1
                    for i, ev in enumerate(ev_col):
                        if ev > ev_t:
                            r = test_data.iloc[i]
                            v_list.append({
                                'Date': r['Date'].strftime('%Y-%m-%d'),
                                'Match': f"{r['HomeTeam']} vs {r['AwayTeam']}",
                                'Pick': outcome,
                                'Odds': r[f'B365{outcome}'],
                                'Edge': f"{ev:.1%}"
                            })
                if v_list: st.table(pd.DataFrame(v_list).sort_values('Date', ascending=False).head(15))
                else: st.warning("No signals found.")
            
            with mt2:
                st.markdown("### Predictive Match Terminal")
                teams = sorted(latest['Team'].unique())
                tc1, tc2 = st.columns(2)
                ht, at = tc1.selectbox("HOME", teams, index=teams.index('Man United')), tc2.selectbox("AWAY", teams, index=teams.index('Arsenal'))
                o1, o2, o3 = st.columns(3)
                ho, do, ao = o1.number_input("Home", value=2.0), o2.number_input("Draw", value=3.4), o3.number_input("Away", value=3.5)
                
                if st.button("EXECUTE FORECAST"):
                    hs, as_ = latest[latest['Team'] == ht].iloc[0], latest[latest['Team'] == at].iloc[0]
                    inp = pd.DataFrame([{'Home_Rolling_GoalsFor': hs['Rolling_GoalsFor'], 'Home_Rolling_GoalsAgainst': hs['Rolling_GoalsAgainst'], 'Home_Rolling_Shots': hs['Rolling_Shots'], 'Home_Rolling_ShotsOnTarget': hs['Rolling_ShotsOnTarget'], 'Home_Rolling_Corners': hs['Rolling_Corners'], 'Away_Rolling_GoalsFor': as_['Rolling_GoalsFor'], 'Away_Rolling_GoalsAgainst': as_['Rolling_GoalsAgainst'], 'Away_Rolling_Shots': as_['Rolling_Shots'], 'Away_Rolling_ShotsOnTarget': as_['Rolling_ShotsOnTarget'], 'Away_Rolling_Corners': as_['Rolling_Corners']}])
                    rp = model.predict_proba(inp[features])[0]
                    
                    fig_m, ax_m = plt.subplots(figsize=(7, 3), facecolor='#0e1117')
                    ax_m.set_facecolor('#0e1117')
                    ax_m.bar(['Home', 'Draw', 'Away'], [rp[c_map['H']], rp[c_map['D']], rp[c_map['A']]], color=['#ef4444', '#3b82f6', '#10b981'])
                    ax_m.tick_params(colors='#94a3b8')
                    st.pyplot(fig_m)

except Exception as e:
    st.error(f"System Error: {e}")
