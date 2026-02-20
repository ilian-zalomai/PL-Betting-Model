import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from dotenv import load_dotenv
from google import genai
from predictive_model import load_and_clean_data, calculate_rolling_stats, train_and_evaluate, get_latest_stats

# --- Load Environment Variables ---
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=env_path)

warnings.filterwarnings('ignore')

# --- Page Config & Styling ---
st.set_page_config(layout="wide", page_title="AI Predictive Engine | PL Analytics")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1e293b; }
    .stMetric { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 12px; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LLM Helper Functions ---
def get_llm_response(prompt, api_key):
    try:
        client = genai.Client(api_key=api_key)
        system_context = "You are a Premier League Sports Betting AI Agent. You are an expert in Kelly Criterion, Elo Ratings, and football statistics. Provide concise, professional advice."
        full_prompt = f"{system_context}\n\nUser: {prompt}"
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini: {str(e)}"

@st.cache_data
def load_all_data():
    df = load_and_clean_data('all_seasons.csv')
    df_stats = calculate_rolling_stats(df)
    model, le, features, test_data = train_and_evaluate(df_stats)
    latest = get_latest_stats(df_stats)
    return df, df_stats, model, le, features, test_data, latest

# --- Sidebar Architecture ---
st.sidebar.title("PL Analytics Pro")
st.sidebar.image("https://img.icons8.com/fluency/96/football.png", width=60)

# Debug info for API Key
if os.getenv("GEMINI_API_KEY"):
    st.sidebar.caption("✅ System: .env key detected")
else:
    st.sidebar.caption("⚠️ System: No .env key found")

# LLM Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 LLM Configuration")
stored_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=stored_key if stored_key else "", 
    type="password", 
    help="Get one at https://ai.google.dev/"
)

if not gemini_api_key:
    st.sidebar.warning("Agent Offline: Enter your Gemini API Key.")
else:
    st.sidebar.success("Agent Online: Live LLM Active")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Strategy Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am your live LLM-powered Strategy Agent. Ask me anything about betting logic!"}]

for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.sidebar.chat_input("Ask the Agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar.chat_message("user"):
        st.write(prompt)

    with st.sidebar.chat_message("assistant"):
        if gemini_api_key:
            with st.spinner("Analyzing..."):
                response = get_llm_response(prompt, gemini_api_key)
        else:
            response = "Please provide a Gemini API Key to enable live LLM reasoning."
        
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main App Execution ---
try:
    with st.spinner("Initializing Predictive Engine..."):
        df_raw, df_stats, ml_model, le, features, test_data, latest = load_all_data()

    st.title("🧠 AI-Powered Predictive Analytics")
    st.markdown("##### Project 2: Combining Random Forest Logic with Live LLM Intelligence")

    tab1, tab2, tab3 = st.tabs(["🔮 Match Prediction", "💰 Value Discovery", "📊 Model Performance"])

    with tab1:
        st.header("Match Intelligence Terminal")
        teams = sorted(latest['Team'].unique())
        c1, c2 = st.columns(2)
        h_team = c1.selectbox("Home Side", teams, index=teams.index('Man United'))
        a_team = c2.selectbox("Away Side", teams, index=teams.index('Arsenal'))
        
        st.divider()
        o1, o2, o3 = st.columns(3)
        h_odds = o1.number_input("Home Odds", value=2.0)
        d_odds = o2.number_input("Draw Odds", value=3.4)
        a_odds = o3.number_input("Away Odds", value=3.5)
        
        if st.button("RUN PREDICTION"):
            h_s, a_s = latest[latest['Team'] == h_team].iloc[0], latest[latest['Team'] == a_team].iloc[0]
            input_row = pd.DataFrame([{
                'Home_Rolling_GoalsFor': h_s['Rolling_GoalsFor'], 'Home_Rolling_GoalsAgainst': h_s['Rolling_GoalsAgainst'],
                'Home_Rolling_Shots': h_s['Rolling_Shots'], 'Home_Rolling_ShotsOnTarget': h_s['Rolling_ShotsOnTarget'], 'Home_Rolling_Corners': h_s['Rolling_Corners'],
                'Away_Rolling_GoalsFor': a_s['Rolling_GoalsFor'], 'Away_Rolling_GoalsAgainst': a_s['Rolling_GoalsAgainst'],
                'Away_Rolling_Shots': a_s['Rolling_Shots'], 'Away_Rolling_ShotsOnTarget': a_s['Rolling_ShotsOnTarget'], 'Away_Rolling_Corners': a_s['Rolling_Corners']
            }])
            
            probs = ml_model.predict_proba(input_row[features])[0]
            class_map = {cls: i for i, cls in enumerate(le.classes_)}
            p_h, p_d, p_a = probs[class_map['H']], probs[class_map['D']], probs[class_map['A']]
            
            res_c1, res_c2 = st.columns([1, 1])
            with res_c1:
                st.subheader("Forecast Probabilities")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=['Home', 'Draw', 'Away'], y=[p_h, p_d, p_a], palette='Blues_d')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
            
            with res_c2:
                st.subheader("Value Assessment")
                ev_h, ev_d, ev_a = (p_h * h_odds)-1, (p_d * d_odds)-1, (p_a * a_odds)-1
                st.write(f"Home EV: {ev_h:+.1%}")
                st.write(f"Draw EV: {ev_d:+.1%}")
                st.write(f"Away EV: {ev_a:+.1%}")
                
                if max(ev_h, ev_d, ev_a) > 0.1:
                    st.success("🎯 SIGNAL DETECTED")
                else:
                    st.warning("⚠️ MARKET EFFICIENT")

    with tab2:
        st.header("Value Discovery")
        ev_thresh = st.slider("Min Edge %", 0, 50, 15) / 100
        probs_all = ml_model.predict_proba(test_data[features])
        class_map = {cls: i for i, cls in enumerate(le.classes_)}
        
        value_bets = []
        for outcome in ['H', 'D', 'A']:
            p_col = probs_all[:, class_map[outcome]]
            ev_col = (p_col * test_data[f'B365{outcome}']) - 1
            for i, ev in enumerate(ev_col):
                if ev > ev_thresh:
                    row = test_data.iloc[i]
                    value_bets.append({
                        'Date': row['Date'].strftime('%Y-%m-%d'),
                        'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                        'Pick': outcome,
                        'Odds': row[f'B365{outcome}'],
                        'EV': f"{ev:.1%}"
                    })
        st.table(pd.DataFrame(value_bets).sort_values('Date', ascending=False).head(20))

    with tab3:
        st.header("Performance Diagnostics")
        y_test = test_data['Target']
        acc = accuracy_score(y_test, ml_model.predict(test_data[features]))
        st.metric("Model Accuracy", f"{acc:.1%}")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, ml_model.predict(test_data[features]))
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        st.pyplot(fig_cm)

except Exception as e:
    st.error(f"Engine Error: {e}")
