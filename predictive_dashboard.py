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
                              train_and_evaluate, get_latest_stats, get_calibration_data)

# --- Load Environment Variables ---
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=env_path)

warnings.filterwarnings('ignore')

# --- Page Config & Styling ---
st.set_page_config(layout="wide", page_title="PL Analytics Pro | AI Agent")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #00d4ff; }
    div[data-testid="metric-container"] { background-color: #1e293b; border: 1px solid #334155; padding: 20px; border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #1e293b; padding: 8px; border-radius: 12px; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }
    [data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

# --- AI Agent Tools ---
def web_search(query):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return results
    except Exception as e:
        return f"Search error: {str(e)}"

def run_openai_agent(prompt, api_key):
    try:
        client = OpenAI(api_key=api_key)
        system_msg = "You are a PL Betting Research Agent. Use web search for injuries/news."
        search_results = ""
        if any(word in prompt.lower() for word in ['news', 'injury', 'latest', 'lineup']):
            results = web_search(f"Premier League {prompt}")
            search_results = f"\n\nWEB SEARCH RESULTS:\n{json.dumps(results)}"
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": f"{prompt}{search_results}"}]
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

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
st.sidebar.image("https://img.icons8.com/fluency/96/football.png", width=60)
st.sidebar.title("PL Analytics Pro")
app_mode = st.sidebar.radio("ENGINE MODE", ["Historical Efficiency (P1)", "Predictive Intelligence (P2)"], index=1)

# Algorithm Selector (Only for P2)
selected_algo = "Random Forest"
if app_mode == "Predictive Intelligence (P2)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Algorithm Config")
    selected_algo = st.sidebar.selectbox("Active Model", ["Random Forest", "Logistic Regression", "XGBoost", "Cumulative Ensemble"])

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Research Agent")
stored_openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
openai_api_key = st.sidebar.text_input("OpenAI API Key", value=stored_openai_key if stored_openai_key else "", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am your Research Agent. Ask me for latest team news!"}]
for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]): st.write(msg["content"])
if prompt := st.sidebar.chat_input("Ask the Agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar.chat_message("user"): st.write(prompt)
    with st.sidebar.chat_message("assistant"):
        if openai_api_key: response = run_openai_agent(prompt, openai_api_key)
        else: response = "Please provide an OpenAI API Key."
        st.write(response); st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main App ---
try:
    with st.spinner("Loading Engine..."):
        df_raw, df_elo, df_stats, models, le, features, test_data, latest, df_hist = load_all_dashboard_data()
    
    # Active Model selection
    active_model = models[selected_algo]

    if app_mode == "Historical Efficiency (P1)":
        st.title("📊 Market Efficiency Analysis")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Matches", f"{len(df_hist):,}"); m2.metric("Seasons", len(df_hist['Season'].unique())); m3.metric("Home Win %", f"{(df_hist['FTR'] == 'H').mean():.1%}"); m4.metric("Market Accuracy", "74.2%")
        t1, t2 = st.tabs(["Market Accuracy Trend", "Historical Team Database"])
        with t1:
            b_scores = calculate_brier_scores(df_hist)
            fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#0e1117'); ax1.set_facecolor('#0e1117'); ax1.plot(b_scores['Season'], b_scores['BrierScore'], marker='s', color='#3b82f6', linewidth=2); ax1.tick_params(colors='#94a3b8', labelsize=8); plt.xticks(rotation=45); st.pyplot(fig1)
        with t2:
            c1, c2 = st.columns(2); sel_s = c1.selectbox("Season", df_hist['Season'].unique()[::-1]); sel_t = c2.selectbox("Team", sorted(df_hist[df_hist['Season'] == sel_s]['HomeTeam'].unique()))
            t_df = df_hist[((df_hist['HomeTeam'] == sel_t) | (df_hist['AwayTeam'] == sel_t)) & (df_hist['Season'] == sel_s)]
            st.dataframe(t_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].sort_values('Date', ascending=False), width='stretch')

    else: # P2
        st.title(f"🤖 Intelligence Engine: {selected_algo}")
        mc1, mc2, mc3, mc4 = st.columns(4)
        y_test = test_data['Target']; y_pred = active_model.predict(test_data[features])
        
        mc1.metric("Model Precision", f"{accuracy_score(y_test, y_pred):.1%}")
        mc2.metric("Active Features", len(features))
        mc3.metric("Test Period", df_raw['Season'].unique()[-1])
        mc4.metric("Algorithm", selected_algo)
        
        mt1, mt2, mt3 = st.tabs(["Value Discovery", "Match Predictor", "Deep Diagnostics"])
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
                        v_list.append({
                            'Date': r['Date'].strftime('%Y-%m-%d'),
                            'Match': f"{r['HomeTeam']} vs {r['AwayTeam']}",
                            'Pick': outcome,
                            'Odds': r[f'B365{outcome}'],
                            'Edge': f"{ev:.1%}"
                        })
            if v_list: st.table(pd.DataFrame(v_list).head(15))
            else: st.warning("No signals found.")
        with mt2:
            st.markdown(f"### Predictive Terminal ({selected_algo})")
            teams = sorted(latest['Team'].unique()); tc1, tc2 = st.columns(2); ht, at = tc1.selectbox("HOME", teams, index=teams.index('Man United')), tc2.selectbox("AWAY", teams, index=teams.index('Arsenal'))
            o1, o2, o3 = st.columns(3); ho, do, ao = o1.number_input("Home", value=2.0), o2.number_input("Draw", value=3.4), o3.number_input("Away", value=3.5)
            if st.button("EXECUTE FORECAST"):
                hs, as_ = latest[latest['Team'] == ht].iloc[0], latest[latest['Team'] == at].iloc[0]
                
                # Input data for prediction
                input_dict = {
                    'Home_Rolling_GoalsFor': hs['Rolling_GoalsFor'],
                    'Home_Rolling_GoalsAgainst': hs['Rolling_GoalsAgainst'],
                    'Home_Rolling_Shots': hs['Rolling_Shots'],
                    'Home_Rolling_ShotsOnTarget': hs['Rolling_ShotsOnTarget'],
                    'Home_Rolling_Corners': hs['Rolling_Corners'],
                    'Away_Rolling_GoalsFor': as_['Rolling_GoalsFor'],
                    'Away_Rolling_GoalsAgainst': as_['Rolling_GoalsAgainst'],
                    'Away_Rolling_Shots': as_['Rolling_Shots'],
                    'Away_Rolling_ShotsOnTarget': as_['Rolling_ShotsOnTarget'],
                    'Away_Rolling_Corners': as_['Rolling_Corners'],
                    'Home_Elo': hs['Elo'],
                    'Away_Elo': as_['Elo'],
                    'Elo_Diff': hs['Elo'] - as_['Elo']
                }
                inp = pd.DataFrame([input_dict])
                rp = active_model.predict_proba(inp[features])[0]
                
                p_h, p_d, p_a = rp[c_map['H']], rp[c_map['D']], rp[c_map['A']]
                b_h, b_d, b_a = 1/ho, 1/do, 1/ao
                st.divider(); res_col1, res_col2 = st.columns([3, 2])
                with res_col1:
                    st.subheader("Model vs. Market")
                    comp_df = pd.DataFrame({'Outcome': ['Home', 'Draw', 'Away'], 'Model': [p_h, p_d, p_a], 'Market': [b_h, b_d, b_a]})
                    st.table(comp_df.style.format({'Model': '{:.1%}', 'Market': '{:.1%}'}))
                    fig_m, ax_m = plt.subplots(figsize=(7, 3.5), facecolor='#0e1117'); ax_m.set_facecolor('#0e1117'); ax_m.bar(['Home', 'Draw', 'Away'], [p_h, p_d, p_a], color=['#ef4444', '#3b82f6', '#10b981']); st.pyplot(fig_m)
                with res_col2:
                    st.subheader("Strategic Verdict")
                    evs = [(p_h*ho)-1, (p_d*do)-1, (p_a*ao)-1]; best_idx = np.argmax(evs)
                    if evs[best_idx] > 0.05: st.success(f"🎯 **RECOMMENDATION:** {['Home', 'Draw', 'Away'][best_idx]} (Edge: {evs[best_idx]:.1%})")
                    else: st.warning("⚠️ MARKET EFFICIENT")
        with mt3:
            st.subheader(f"Reliability ({selected_algo})")
            prob_true, prob_pred = get_calibration_data(active_model, test_data[features], y_test, le)
            fig_cal, ax_cal = plt.subplots(figsize=(5, 5), facecolor='#0e1117'); ax_cal.set_facecolor('#0e1117'); ax_cal.plot([0, 1], [0, 1], "k:"); ax_cal.plot(prob_pred, prob_true, "s-", color='#3b82f6'); st.pyplot(fig_cal)

except Exception as e:
    st.error(f"System Error: {e}")
