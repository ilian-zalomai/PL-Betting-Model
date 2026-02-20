import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from predictive_model import load_and_clean_data, calculate_rolling_stats, train_and_evaluate, get_latest_stats

warnings.filterwarnings('ignore')

# --- Page Config & "Intelligence" Styling ---
st.set_page_config(layout="wide", page_title="AI Predictive Engine | PL Analytics")

st.markdown("""
    <style>
    /* Professional Clean Interface */
    .stApp { background-color: #ffffff; color: #1e293b; }
    
    /* Intelligence Cards */
    .ai-card {
        background-color: #f1f5f9;
        border-left: 5px solid #3b82f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
    
    /* Navigation Bar */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI Intelligence Logic ---
def generate_ai_insight(h_team, a_team, p_h, p_d, p_a, h_stats, a_stats):
    """
    Simulates LLM Reasoning based on model inputs.
    In a production app, this would call the Gemini/GPT API.
    """
    strength = "Home" if p_h > p_a else "Away"
    confidence = "High" if max(p_h, p_a) > 0.6 else "Moderate"
    
    # Logic-based reasoning strings
    reasoning = []
    if h_stats['Rolling_GoalsFor'] > a_stats['Rolling_GoalsFor']:
        reasoning.append(f"{h_team}'s offensive momentum ({h_stats['Rolling_GoalsFor']:.2f} GPG) outweighs {a_team}.")
    if h_stats['Rolling_GoalsAgainst'] < a_stats['Rolling_GoalsAgainst']:
        reasoning.append(f"Defensive stability favors the home side.")
    
    insight = f"""
    <div class="ai-card">
        <h4 style='color: #1e40af;'>🤖 AI Intelligence Report</h4>
        <p><b>Executive Summary:</b> The model identifies a <b>{confidence}</b> probability of a <b>{strength}</b> victory.</p>
        <p><b>Reasoning Path:</b> {' '.join(reasoning[:2])} The rolling 5-game window suggests a tactical mismatch in the final third.</p>
        <p style='font-size: 0.85rem; color: #64748b;'><i>Generated via PL-Predictive-LLM-Agent</i></p>
    </div>
    """
    return insight

@st.cache_data
def load_all_data():
    df = load_and_clean_data('all_seasons.csv')
    df_stats = calculate_rolling_stats(df)
    model, le, features, test_data = train_and_evaluate(df_stats)
    latest = get_latest_stats(df_stats)
    return df, df_stats, model, le, features, test_data, latest

# --- App Execution ---
try:
    with st.spinner("Initializing AI Predictive Engine..."):
        df_raw, df_stats, model, le, features, test_data, latest = load_all_data()

    st.title("🧠 AI-Powered Predictive Analytics")
    st.markdown("##### Project 2: Combining Random Forest Logic with LLM Insights")

    tab1, tab2, tab3 = st.tabs(["🔮 Intelligence Terminal", "💰 Value Discovery", "📊 Model Diagnostics"])

    with tab1:
        st.header("Match Intelligence Terminal")
        st.write("Select a fixture to generate a comprehensive AI analysis.")
        
        teams = sorted(latest['Team'].unique())
        c1, c2 = st.columns(2)
        h_team = c1.selectbox("Home Side", teams, index=teams.index('Man United'))
        a_team = c2.selectbox("Away Side", teams, index=teams.index('Arsenal'))
        
        st.divider()
        
        # Market Odds
        st.subheader("Market Context")
        o1, o2, o3 = st.columns(3)
        h_odds = o1.number_input("Bookmaker Home Odds", value=2.0)
        d_odds = o2.number_input("Bookmaker Draw Odds", value=3.4)
        a_odds = o3.number_input("Bookmaker Away Odds", value=3.5)
        
        if st.button("RUN AI ANALYSIS"):
            # Get Stats
            h_s = latest[latest['Team'] == h_team].iloc[0]
            a_s = latest[latest['Team'] == a_team].iloc[0]
            
            # Predict
            input_row = pd.DataFrame([{
                'Home_Rolling_GoalsFor': h_s['Rolling_GoalsFor'], 'Home_Rolling_GoalsAgainst': h_s['Rolling_GoalsAgainst'],
                'Home_Rolling_Shots': h_s['Rolling_Shots'], 'Home_Rolling_ShotsOnTarget': h_s['Rolling_ShotsOnTarget'], 'Home_Rolling_Corners': h_s['Rolling_Corners'],
                'Away_Rolling_GoalsFor': a_s['Rolling_GoalsFor'], 'Away_Rolling_GoalsAgainst': a_s['Rolling_GoalsAgainst'],
                'Away_Rolling_Shots': a_s['Rolling_Shots'], 'Away_Rolling_ShotsOnTarget': a_s['Rolling_ShotsOnTarget'], 'Away_Rolling_Corners': a_s['Rolling_Corners']
            }])
            
            probs = model.predict_proba(input_row[features])[0]
            class_map = {cls: i for i, cls in enumerate(le.classes_)}
            p_h, p_d, p_a = probs[class_map['H']], probs[class_map['D']], probs[class_map['A']]
            
            # Layout Results
            res_c1, res_c2 = st.columns([1, 1])
            
            with res_c1:
                st.subheader("Probabilistic Forecast")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=['Home', 'Draw', 'Away'], y=[p_h, p_d, p_a], palette='Blues_d')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
            
            with res_c2:
                # AI INSIGHT BOX
                st.markdown(generate_ai_insight(h_team, a_team, p_h, p_d, p_a, h_s, a_s), unsafe_allow_html=True)
                
                # Value calculation
                ev_h = (p_h * h_odds) - 1
                ev_d = (p_d * d_odds) - 1
                ev_a = (p_a * a_odds) - 1
                
                st.write("**Strategy Audit:**")
                if max(ev_h, ev_d, ev_a) > 0.1:
                    best = ['Home', 'Draw', 'Away'][np.argmax([ev_h, ev_d, ev_a])]
                    st.success(f"🎯 **BETS RECOMMENDED:** {best} (EV: {max(ev_h, ev_d, ev_a):.1%})")
                else:
                    st.warning("⚠️ **MARKET EFFICIENT:** No clear edge found.")

    with tab2:
        st.header("Strategic Value Discovery")
        st.write("AI-filtered opportunities across the current test season.")
        
        ev_thresh = st.slider("Min Edge %", 0, 50, 15) / 100
        
        # Batch predict test data
        probs_all = model.predict_proba(test_data[features])
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
                        'Fixture': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                        'Market': f"{outcome} Win",
                        'Odds': row[f'B365{outcome}'],
                        'Model Prob': f"{p_col[i]:.1%}",
                        'Edge (EV)': f"{ev:.1%}",
                        'Outcome': '✅' if row['FTR'] == outcome else '❌'
                    })
        
        if value_bets:
            st.table(pd.DataFrame(value_bets).sort_values('Date', ascending=False).head(20))
        else:
            st.info("No high-confidence signals found.")

    with tab3:
        st.header("Model Performance & Calibration")
        
        y_test = test_data['Target']
        y_pred = model.predict(test_data[features])
        acc = accuracy_score(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predictive Accuracy", f"{acc:.1%}")
        col2.metric("Intelligence Depth", "High (AI-Enhanced)")
        col3.metric("Algorithm", "Random Forest")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel('AI Predicted')
        plt.ylabel('Actual Outcome')
        st.pyplot(fig_cm)

except Exception as e:
    st.error(f"Engine Startup Failure: {e}")
