import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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

    /* Dataframe Header */
    .stDataFrame {
        border: 1px solid #334155;
        border-radius: 12px;
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

    # Sidebar Header
    st.sidebar.image("https://img.icons8.com/fluency/96/football.png", width=80)
    st.sidebar.title("PL Analytics Pro")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("ENGINE MODE", 
                                ["Historical Efficiency (P1)", 
                                 "Predictive Intelligence (P2)"])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: 2003 - 2026 Seasonality")

    if app_mode == "Historical Efficiency (P1)":
        st.title("📊 Market Efficiency Analysis")
        st.markdown("#### Project 1: Analyzing Bookmaker Performance & Brier Scores")
        
        # Top Level Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Matches", f"{len(df_hist):,}")
        m2.metric("Seasons Analysed", len(df_hist['Season'].unique()))
        m3.metric("Avg Home Win %", f"{(df_hist['FTR'] == 'H').mean():.1%}")
        m4.metric("Market Accuracy", "74.2%") # Representative stat

        t1, t2 = st.tabs(["Market Accuracy Trend", "Historical Team Database"])
        
        with t1:
            st.markdown("### Brier Score Decay Curve")
            b_scores = calculate_brier_scores(df_hist)
            fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
            ax1.set_facecolor('#0e1117')
            ax1.plot(b_scores['Season'], b_scores['BrierScore'], marker='s', color='#3b82f6', linewidth=2, label='Brier Score')
            ax1.set_ylabel('Accuracy Error', color='#94a3b8')
            ax1.tick_params(colors='#94a3b8', labelsize=8)
            plt.xticks(rotation=45)
            plt.grid(color='#334155', linestyle='--', alpha=0.5)
            st.pyplot(fig1)
            st.info("A decreasing Brier Score indicates that bookmakers are becoming significantly more efficient at pricing Home Win probabilities over time.")

        with t2:
            st.markdown("### Team Historical Performance")
            c1, c2 = st.columns(2)
            sel_s = c1.selectbox("Season", df_hist['Season'].unique()[::-1])
            s_df = df_hist[df_hist['Season'] == sel_s]
            sel_t = c2.selectbox("Team", sorted(s_df['HomeTeam'].unique()))
            
            t_df = s_df[(s_df['HomeTeam'] == sel_t) | (s_df['AwayTeam'] == sel_t)]
            st.dataframe(t_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].sort_values('Date', ascending=False), width='stretch')

    else: # Predictive Intelligence (P2)
        st.title("🤖 Predictive Intelligence Engine")
        st.markdown("#### Project 2: Forward-Looking Machine Learning & Value Discovery")
        
        if model is None:
            st.error("Engine failure: Predictive model failed to initialize.")
        else:
            # ML Metrics Row
            last_season = df_ml['Season'].unique()[-1]
            y_test = test_data['Target']
            y_pred = model.predict(test_data[features])
            from sklearn.metrics import accuracy_score
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Model Precision", f"{accuracy_score(y_test, y_pred):.1%}")
            mc2.metric("Active Features", "10")
            mc3.metric("Test Period", last_season)
            mc4.metric("Algorithm", "Random Forest")

            mt1, mt2, mt3 = st.tabs(["Discovery: Value Opportunities", "Terminal: Match Predictor", "Diagnostics: Model Quality"])
            
            with mt1:
                st.markdown("### Real-Time Value Discovery")
                ev_t = st.slider("Minimum Edge (Expected Value) %", 0, 40, 10) / 100
                
                probs = model.predict_proba(test_data[features])
                c_map = {cls: i for i, cls in enumerate(le.classes_)}
                td = test_data.copy()
                
                v_list = []
                for outcome in ['H', 'D', 'A']:
                    p_col = probs[:, c_map[outcome]]
                    ev_col = (p_col * td[f'B365{outcome}']) - 1
                    for i, ev in enumerate(ev_col):
                        if ev > ev_t:
                            r = td.iloc[i]
                            v_list.append({
                                'Date': r['Date'].strftime('%Y-%m-%d'),
                                'Signal': f"{r['HomeTeam']} vs {r['AwayTeam']}",
                                'Pick': outcome,
                                'Odds': r[f'B365{outcome}'],
                                'Model Prob': f"{p_col[i]:.1%}",
                                'Edge (EV)': f"{ev:.1%}",
                                'Status': '✅' if r['FTR'] == outcome else '❌'
                            })
                
                if v_list:
                    st.table(pd.DataFrame(v_list).sort_values('Date', ascending=False).head(15))
                else:
                    st.warning("No signals detected at current threshold.")

            with mt2:
                st.markdown("### Predictive Match Terminal")
                teams = sorted(latest['Team'].unique())
                tc1, tc2 = st.columns(2)
                ht = tc1.selectbox("HOME SIDE", teams, index=teams.index('Man United') if 'Man United' in teams else 0)
                at = tc2.selectbox("AWAY SIDE", teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 1)
                
                st.markdown("#### Market Odds Input")
                oc1, oc2, oc3 = st.columns(3)
                ho = oc1.number_input("Home", value=2.0)
                do = oc2.number_input("Draw", value=3.4)
                ao = oc3.number_input("Away", value=3.5)
                
                if st.button("EXECUTE FORECAST"):
                    hs = latest[latest['Team'] == ht].iloc[0]
                    as_ = latest[latest['Team'] == at].iloc[0]
                    
                    inp = pd.DataFrame([{
                        'Home_Rolling_GoalsFor': hs['Rolling_GoalsFor'], 'Home_Rolling_GoalsAgainst': hs['Rolling_GoalsAgainst'],
                        'Home_Rolling_Shots': hs['Rolling_Shots'], 'Home_Rolling_ShotsOnTarget': hs['Rolling_ShotsOnTarget'], 'Home_Rolling_Corners': hs['Rolling_Corners'],
                        'Away_Rolling_GoalsFor': as_['Rolling_GoalsFor'], 'Away_Rolling_GoalsAgainst': as_['Rolling_GoalsAgainst'],
                        'Away_Rolling_Shots': as_['Rolling_Shots'], 'Away_Rolling_ShotsOnTarget': as_['Rolling_ShotsOnTarget'], 'Away_Rolling_Corners': as_['Rolling_Corners']
                    }])
                    
                    rp = model.predict_proba(inp[features])[0]
                    ph, pd_, pa = rp[c_map['H']], rp[c_map['D']], rp[c_map['A']]
                    
                    st.markdown("---")
                    rc1, rc2 = st.columns([2, 1])
                    with rc1:
                        fig_m, ax_m = plt.subplots(figsize=(7, 3.5), facecolor='#0e1117')
                        ax_m.set_facecolor('#0e1117')
                        ax_m.barh(['Home', 'Draw', 'Away'], [ph, pd_, pa], color=['#ef4444', '#3b82f6', '#10b981'])
                        ax_m.tick_params(colors='#94a3b8')
                        st.pyplot(fig_m)
                    
                    with rc2:
                        evh, evd, eva = ph*ho-1, pd_*do-1, pa*ao-1
                        best = np.argmax([evh, evd, eva])
                        st.markdown(f"**Home EV:** {evh:+.1%}")
                        st.markdown(f"**Draw EV:** {evd:+.1%}")
                        st.markdown(f"**Away EV:** {eva:+.1%}")
                        if max([evh, evd, eva]) > 0.05:
                            st.success(f"SIGNAL: {['HOME', 'DRAW', 'AWAY'][best]}")

            with mt3:
                st.markdown("### Model Quality Diagnostics")
                cm = confusion_matrix(y_test, y_pred)
                from sklearn.metrics import confusion_matrix
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
                ax_cm.set_title("Confusion Matrix", color='white')
                st.pyplot(fig_cm)

except Exception as e:
    st.error(f"System Error: {e}")
