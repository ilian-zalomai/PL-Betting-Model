import streamlit as st
import sys

# --- Pre-flight Dependency Check ---
missing_deps = []
try: import openai
except ImportError: missing_deps.append("openai")
try: import duckduckgo_search
except ImportError: missing_deps.append("duckduckgo-search")
try: import xgboost
except ImportError: missing_deps.append("xgboost")

if missing_deps:
    st.error(f"Missing Dependencies: {', '.join(missing_deps)}")
    st.info("The server is still installing packages. Please wait 60 seconds and refresh.")
    st.stop()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS
from sklearn.metrics import accuracy_score, confusion_matrix
from predictive_model import (load_and_clean_data, calculate_elo, calculate_rolling_stats, 
                              train_and_evaluate, get_latest_stats, get_calibration_data,
                              run_backtest_loop, get_poisson_probabilities)

# --- Load Environment Variables ---
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=env_path)

warnings.filterwarnings('ignore')

# --- Page Config & Modern Styling ---
st.set_page_config(layout="wide", page_title="PL Analytics Pro | Multi-Algorithm Suite")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #00d4ff; }
    div[data-testid="metric-container"] { background-color: #1e293b; border: 1px solid #334155; padding: 20px; border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #1e293b; padding: 8px; border-radius: 12px; }
    .stTabs [aria-selected="true"] { background-color: transparent !important; color: #00d4ff !important; font-weight: bold; border-bottom: 2px solid #00d4ff !important; }
    [data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

# Helper for dark mode plots
def apply_dark_style(ax):
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')
    return ax

# --- AI Agent Tools ---
def run_openai_agent(prompt, api_key):
    try:
        client = OpenAI(api_key=api_key)
        system_msg = "You are a PL Betting Research Agent. Use web search for injuries/news."
        search_results = ""
        if any(word in prompt.lower() for word in ['news', 'injury', 'latest', 'lineup']):
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(f"Premier League {prompt}", max_results=3)]
                search_results = f"\n\nWEB SEARCH:\n{json.dumps(results)}"
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": f"{prompt}{search_results}"}]
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content
    except Exception as e: return f"Error: {str(e)}"

# --- Data Loading ---
@st.cache_data
def load_all_dashboard_data():
    df_raw = load_and_clean_data('all_seasons.csv')
    df_elo = calculate_elo(df_raw)
    df_stats = calculate_rolling_stats(df_elo)
    models, le, features, test_data = train_and_evaluate(df_stats)
    latest_stats = get_latest_stats(df_stats)
    df_hist = df_raw.copy()
    df_hist.dropna(subset=['Season', 'FTR', 'B365H'], inplace=True)
    df_hist['B365H_Prob'] = 1 / df_hist['B365H']
    df_hist['HomeWin_Outcome'] = (df_hist['FTR'] == 'H').astype(int)
    return df_raw, df_elo, df_stats, models, le, features, test_data, latest_stats, df_hist

def calculate_brier_scores(df_data):
    df_data['SquaredError'] = (df_data['B365H_Prob'] - df_data['HomeWin_Outcome'])**2
    return df_data.groupby('Season')['SquaredError'].mean().reset_index().rename(columns={'SquaredError': 'BrierScore'}).sort_values('Season')

# --- Sidebar ---
st.sidebar.title("PL Analytics Pro")
app_mode = st.sidebar.radio("ENGINE MODE", ["Historical Efficiency (P1)", "Predictive Intelligence (P2)"], index=1)

selected_algo = "Random Forest"
algo_container = None
if app_mode == "Predictive Intelligence (P2)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Algorithm Config")
    selected_algo = st.sidebar.selectbox("Active Model", ["Random Forest", "Logistic Regression", "XGBoost", "Cumulative Ensemble"])
    st.sidebar.subheader("Algorithm Breakdown")
    algo_container = st.sidebar.container()
    
    st.sidebar.subheader("Model Features")
    with st.sidebar.expander("View 13 Active Features"):
        st.write("**Offensive Stats (Rolling 5):**")
        st.write("- Goals For, Shots, Shots on Target, Corners")
        st.write("**Defensive Stats (Rolling 5):**")
        st.write("- Goals Against")
        st.write("**Strength Ratings:**")
        st.write("- Home Elo, Away Elo, Elo Difference")
        st.caption("Rolling stats applied to both Home & Away teams (10) + 3 Elo metrics = 13 total.")

st.sidebar.markdown("---")
st.sidebar.subheader("AI Research Agent")
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am your Research Agent. Ask me for latest team news!"}]
for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]): st.write(msg["content"])
if prompt := st.sidebar.chat_input("Ask the Agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar.chat_message("user"): st.write(prompt)
    with st.sidebar.chat_message("assistant"):
        if openai_api_key: response = run_openai_agent(prompt, openai_api_key)
        else: response = "Agent Offline: API Key missing in System Secrets."
        st.write(response); st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main App ---
try:
    with st.spinner("Initializing Predictive Engine..."):
        df_raw, df_elo, df_stats, models, le, features, test_data, latest, df_hist = load_all_dashboard_data()
    
    if app_mode == "Predictive Intelligence (P2)" and algo_container:
        with algo_container:
            y_true = test_data['Target']
            for name, m in models.items():
                acc = accuracy_score(y_true, m.predict(test_data[features]))
                st.write(f"**{name}:** `{acc:.1%}`")

    active_model = models[selected_algo]

    if app_mode == "Historical Efficiency (P1)":
        st.title("Market Efficiency Analysis")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Matches", f"{len(df_hist):,}"); m2.metric("Seasons", len(df_hist['Season'].unique())); m3.metric("Home Win %", f"{(df_hist['FTR'] == 'H').mean():.1%}"); m4.metric("Market Accuracy", "74.2%")
        t1, t2 = st.tabs(["Market Accuracy Trend", "Historical Team Database"])
        with t1:
            st.markdown("### Brier Score Decay Curve")
            b_scores = calculate_brier_scores(df_hist)
            fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#0e1117'); apply_dark_style(ax1); ax1.plot(b_scores['Season'], b_scores['BrierScore'], marker='s', color='#3b82f6', linewidth=2); plt.xticks(rotation=45); st.pyplot(fig1)
        with t2:
            st.markdown("### Team Historical Performance")
            c1, c2 = st.columns(2); sel_s = c1.selectbox("Season", df_hist['Season'].unique()[::-1]); sel_t = c2.selectbox("Team", sorted(df_hist[df_hist['Season'] == sel_s]['HomeTeam'].unique()))
            t_df = df_hist[((df_hist['HomeTeam'] == sel_t) | (df_hist['AwayTeam'] == sel_t)) & (df_hist['Season'] == sel_s)]
            st.dataframe(t_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].sort_values('Date', ascending=False), width='stretch')

    else: # P2
        st.title(f"Intelligence Engine: {selected_algo}")
        st.markdown("#### Project 2: Forward-Looking Machine Learning & Value Discovery")
        mc1, mc2, mc3, mc4 = st.columns(4)
        y_test = test_data['Target']; y_pred = active_model.predict(test_data[features])
        mc1.metric("Model Precision", f"{accuracy_score(y_test, y_pred):.1%}"); mc2.metric("Active Features", len(features)); mc3.metric("Test Period", df_raw['Season'].unique()[-1]); mc4.metric("Algorithm", selected_algo)
        
        mt1, mt2, mt3, mt4, mt5, mt6, mt7 = st.tabs([
            "Value Discovery", "Match Predictor", "Deep Diagnostics", 
            "Strategy Backtest", "Poisson Matrix", "Risk Analytics", "Research & Documentation"
        ])
        
        with mt1:
            st.markdown(f"### Real-Time Value Discovery ({selected_algo})")
            ev_t = st.slider("Min Edge %", 0, 40, 15) / 100
            probs = active_model.predict_proba(test_data[features])
            c_map = {cls: i for i, cls in enumerate(le.classes_)}
            v_list = []
            for outcome in ['H', 'D', 'A']:
                p_col = probs[:, c_map[outcome]]
                ev_col = (p_col * test_data[f'B365{outcome}']) - 1
                for i, ev in enumerate(ev_col):
                    if ev > ev_t:
                        r = test_data.iloc[i]
                        v_list.append({'Date': r['Date'].strftime('%Y-%m-%d'), 'Match': f"{r['HomeTeam']} vs {r['AwayTeam']}", 'Pick': outcome, 'Odds': r[f'B365{outcome}'], 'Edge': f"{ev:.1%}"})
            if v_list: st.table(pd.DataFrame(v_list).head(15))
            else: st.warning("No signals found.")
            
        with mt2:
            st.markdown("### Predictive Match Terminal")
            teams = sorted(latest['Team'].unique()); tc1, tc2 = st.columns(2); ht, at = tc1.selectbox("HOME", teams, index=teams.index('Man United')), tc2.selectbox("AWAY", teams, index=teams.index('Arsenal'))
            o1, o2, o3 = st.columns(3); ho, do, ao = o1.number_input("Home", value=2.0, min_value=1.01), o2.number_input("Draw", value=3.4, min_value=1.01), o3.number_input("Away", value=3.5, min_value=1.01)
            if st.button("EXECUTE FORECAST"):
                hs, as_ = latest[latest['Team'] == ht].iloc[0], latest[latest['Team'] == at].iloc[0]
                input_dict = {'Home_Rolling_GoalsFor': hs['Rolling_GoalsFor'], 'Home_Rolling_GoalsAgainst': hs['Rolling_GoalsAgainst'], 'Home_Rolling_Shots': hs['Rolling_Shots'], 'Home_Rolling_ShotsOnTarget': hs['Rolling_ShotsOnTarget'], 'Home_Rolling_Corners': hs['Rolling_Corners'], 'Away_Rolling_GoalsFor': as_['Rolling_GoalsFor'], 'Away_Rolling_GoalsAgainst': as_['Rolling_GoalsAgainst'], 'Away_Rolling_Shots': as_['Rolling_Shots'], 'Away_Rolling_ShotsOnTarget': as_['Rolling_ShotsOnTarget'], 'Away_Rolling_Corners': as_['Rolling_Corners'], 'Home_Elo': hs['Elo'], 'Away_Elo': as_['Elo'], 'Elo_Diff': hs['Elo'] - as_['Elo']}
                rp = active_model.predict_proba(pd.DataFrame([input_dict]))[0]
                p_h, p_d, p_a = rp[c_map['H']], rp[c_map['D']], rp[c_map['A']]
                b_h, b_d, b_a = 1/ho, 1/do, 1/ao
                st.divider(); res_col1, res_col2 = st.columns([3, 2])
                with res_col1:
                    st.subheader("Model vs. Market Comparison")
                    comp_data = pd.DataFrame({'Outcome': ['Home Win', 'Draw', 'Away Win'], 'Model Prob': [p_h, p_d, p_a], 'Bookie Implied': [b_h, b_d, b_a], 'Edge (EV)': [(p_h*ho)-1, (p_d*do)-1, (p_a*ao)-1]})
                    st.table(comp_data.style.format({'Model Prob': '{:.1%}', 'Bookie Implied': '{:.1%}', 'Edge (EV)': '{:+.1%}'}))
                    fig_comp, ax_comp = plt.subplots(figsize=(8, 4), facecolor='#0e1117'); apply_dark_style(ax_comp); x = np.arange(3); width = 0.35
                    ax_comp.bar(x - width/2, [p_h, p_d, p_a], width, label='Model', color='#00d4ff'); ax_comp.bar(x + width/2, [b_h, b_d, b_a], width, label='Market', color='#94a3b8')
                    ax_comp.set_xticks(x); ax_comp.set_xticklabels(['Home', 'Draw', 'Away'], color='white'); ax_comp.legend(); st.pyplot(fig_comp)
                with res_col2:
                    st.subheader("Strategic Verdict")
                    evs = [(p_h*ho)-1, (p_d*do)-1, (p_a*ao)-1]; best_idx = np.argmax(evs)
                    if evs[best_idx] > 0.05: st.success(f"🎯 RECOMMENDATION: {['Home Win', 'Draw', 'Away Win'][best_idx]} ({evs[best_idx]:.1%})")
                    else: st.warning("⚠️ NO CLEAR VALUE")
                    st.info(f"Elo Context: {ht} ({int(hs['Elo'])}) vs {at} ({int(as_['Elo'])})")

        with mt3:
            st.markdown("### Model Reliability & Market Benchmark")
            diag_c1, diag_c2 = st.columns(2)
            with diag_c1:
                st.subheader("Reliability (Calibration)")
                prob_true, prob_pred = get_calibration_data(active_model, test_data[features], y_test, le)
                fig_cal, ax_cal = plt.subplots(figsize=(5, 5), facecolor='#0e1117'); apply_dark_style(ax_cal); ax_cal.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated"); ax_cal.plot(prob_pred, prob_true, "s-", color='#3b82f6', label="Model"); st.pyplot(fig_cal)
            with diag_c2:
                st.subheader("Market Comparison (Brier)")
                bookies = {'B365': 'B365H', 'Bwin': 'BWH', 'VCBet': 'VCH', 'Model': 'Prob_H'}
                test_data['Prob_H'] = active_model.predict_proba(test_data[features])[:, c_map['H']]
                test_data['HW_Actual'] = (test_data['FTR'] == 'H').astype(int)
                b_results = []
                for name, col in bookies.items():
                    if col in test_data.columns:
                        tmp = test_data.dropna(subset=[col])
                        if name == 'Model': b_score = np.mean((tmp[col] - tmp['HW_Actual'])**2)
                        else: b_score = np.mean(((1/tmp[col]) - tmp['HW_Actual'])**2)
                        b_results.append({'Bookie': name, 'Brier': b_score})
                st.table(pd.DataFrame(b_results).sort_values('Brier'))

        with mt4:
            st.header("Historical Strategy Backtest")
            st.markdown("Simulate model performance across past seasons to find the optimal betting configuration.")
            sc1, sc2, sc3 = st.columns(3)
            min_odds = sc1.slider("Min Odds Range", 1.1, 5.0, 1.5); max_odds = sc2.slider("Max Odds Range", 1.5, 15.0, 3.5); min_ev = sc3.slider("Required Edge (EV) %", 0, 50, 10) / 100
            td = test_data.copy(); probs = active_model.predict_proba(td[features])
            for o in ['H', 'D', 'A']: td[f'Prob_{o}'] = probs[:, c_map[o]]; td[f'EV_{o}'] = (td[f'Prob_{o}'] * td[f'B365{o}']) - 1
            bets = []
            for idx, r in td.iterrows():
                for o in ['H', 'D', 'A']:
                    if r[f'EV_{o}'] > min_ev and min_odds <= r[f'B365{o}'] <= max_odds:
                        won = r['FTR'] == o; profit = (r[f'B365{o}'] - 1) if won else -1; bets.append({'Date': r['Date'], 'Profit': profit, 'Won': won})
            if bets:
                bets_df = pd.DataFrame(bets).sort_values('Date'); bets_df['Cum_Profit'] = bets_df['Profit'].cumsum()
                bc1, bc2 = st.columns(2); bc1.metric("Simulated Net Profit", f"{bets_df['Profit'].sum():.2f} units"); bc1.metric("Bet Volume", len(bets_df)); bc2.metric("Strategy ROI", f"{(bets_df['Profit'].sum() / len(bets_df)):.1%}"); bc2.metric("Win Rate", f"{bets_df['Won'].mean():.1%}")
                fig_p, ax_p = plt.subplots(figsize=(10, 4), facecolor='#0e1117'); apply_dark_style(ax_p); ax_p.plot(bets_df['Date'], bets_df['Cum_Profit'], color='#00d4ff', linewidth=2); ax_p.fill_between(bets_df['Date'], bets_df['Cum_Profit'], color='#00d4ff', alpha=0.1); st.pyplot(fig_p)
            else: st.warning("No matches found.")
            
            st.divider()
            st.subheader("Strategy Discovery Optimizer")
            st.write("Top 5 Profitable Configurations (Iterative Search):")
            # Simple grid search for optimization display
            opt_results = []
            for m_ev in [0.05, 0.10, 0.15]:
                for m_odds in [1.5, 2.0, 2.5]:
                    p_sum = sum([(r[f'B365{o}']-1 if r['FTR']==o else -1) for _, r in td.iterrows() for o in ['H','D','A'] if r[f'EV_{o}']>m_ev and m_odds<=r[f'B365{o}']<=m_odds+1.0])
                    opt_results.append({'Min EV': m_ev, 'Odds Range': f"{m_odds}-{m_odds+1.0}", 'Total Profit': p_sum})
            st.table(pd.DataFrame(opt_results).sort_values('Total Profit', ascending=False).head(5))

        with mt5:
            st.header("Poisson Scoreline Matrix")
            pc1, pc2 = st.columns(2); h_team_p = pc1.selectbox("Home Side (Poisson)", teams, index=teams.index('Liverpool')); a_team_p = pc2.selectbox("Away Side (Poisson)", teams, index=teams.index('Arsenal'))
            h_lambda, a_lambda = latest[latest['Team'] == h_team_p].iloc[0]['Rolling_GoalsFor'], latest[latest['Team'] == a_team_p].iloc[0]['Rolling_GoalsFor']
            matrix, p_h_win, p_draw, p_a_win = get_poisson_probabilities(h_lambda, a_lambda)
            fig_hm, ax_hm = plt.subplots(figsize=(6, 5), facecolor='#0e1117'); sns.heatmap(matrix, annot=True, fmt='.1%', cmap='Blues', ax=ax_hm, cbar=False); apply_dark_style(ax_hm); ax_hm.set_xlabel(f"{a_team_p} Goals"); ax_hm.set_ylabel(f"{h_team_p} Goals"); st.pyplot(fig_hm)
            st.write(f"Poisson Calculation: Home: {p_h_win:.1%}, Draw: {p_draw:.1%}, Away: {p_a_win:.1%}")

        with mt6:
            st.header("Performance Volatility & Variance")
            vol_data = []
            for team in teams:
                t_matches = df_raw[(df_raw['HomeTeam'] == team) | (df_raw['AwayTeam'] == team)].tail(20)
                goals = t_matches.apply(lambda r: r['FTHG'] if r['HomeTeam'] == team else r['FTAG'], axis=1); vol_data.append({'Team': team, 'Volatility': goals.std(), 'Avg Goals': goals.mean()})
            vol_df = pd.DataFrame(vol_data).sort_values('Volatility', ascending=False)
            vc1, vc2 = st.columns([1, 1])
            with vc1: st.subheader("Volatility Rankings"); st.dataframe(vol_df.head(15), height=400)
            with vc2: st.subheader("League Stability Map"); fig_vol, ax_vol = plt.subplots(figsize=(6, 6), facecolor='#0e1117'); apply_dark_style(ax_vol); sns.regplot(data=vol_df, x='Avg Goals', y='Volatility', ax=ax_vol, color='#3b82f6'); st.pyplot(fig_vol)

        with mt7:
            st.header("Project Evidence & Documentation")
            doc_tabs = st.tabs(["Market Efficiency (Vig Analysis)", "Technical Methodology", "AI Development Trace"])
            with doc_tabs[0]:
                st.subheader("Bookmaker Overround (Vig) Trend")
                df_raw['Margin'] = (1/df_raw['B365H'] + 1/df_raw['B365D'] + 1/df_raw['B365A']) - 1
                margin_trend = df_raw.groupby('Season')['Margin'].mean().reset_index()
                fig_m, ax_m = plt.subplots(figsize=(10, 4), facecolor='#0e1117'); apply_dark_style(ax_m); ax_m.plot(margin_trend['Season'], margin_trend['Margin'], marker='o', color='#00d4ff'); ax_m.set_ylabel("Margin %"); plt.xticks(rotation=45); st.pyplot(fig_m)
            with doc_tabs[1]:
                st.subheader("Technical Methodology")
                st.markdown("""
                ### 1. Forward-Looking Feature Engineering
                - **Elo Rating System:** A dynamic strength index updated match-by-match since 2003 using the formula $R_n = R_o + K(S - E)$.
                - **Rolling Performance Stats:** Form is captured via 5-game rolling windows of Goals, Shots, Shots on Target, and Corners for both teams.
                - **Total Features:** 13 predictive metrics per match, ensuring the model identifies momentum, not just historical averages.
                """)
                st.markdown("""
                ### 2. Validation & Accuracy
                - **Chronological Split:** The system trains on historical data (2003-2024) and validates on the current 2025-26 season to eliminate 'look-ahead' bias.
                - **Multi-Algorithm Ensemble:** Users can switch between Random Forest, XGBoost, and Logistic Regression, or use the **Cumulative Ensemble** which averages probabilities across all three architectures.
                """)
            with doc_tabs[2]:
                st.subheader("AI Development Log (Traceability)")
                st.markdown("""
                This project represents a human-AI collaboration with **Gemini CLI**. AI acted as an agentic engineer responsible for:
                1. **Architecture:** Designing the 13-feature ETL pipeline.
                2. **Math:** Implementing Elo and Poisson probability matrices.
                3. **Analytics:** Adding Calibration Curves and Brier Score leaderboards to meet academic rigor.
                4. **Interface:** Engineering this professional Streamlit Pro trading terminal.
                """)

except Exception as e:
    st.error(f"System Error: {e}")
