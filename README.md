# Premier League Predictive Betting Engine v2.0

## 🏆 Project Overview
This repository contains a sophisticated sports analytics engine designed to identify "Value" in the Premier League betting market. It represents the evolution from historical data analysis (Project 1) to a forward-looking, machine-learning-driven predictive system (Project 2).

### 🚀 Key Features
- **Machine Learning Engine:** Uses a Random Forest Classifier trained on 20+ seasons of PL data.
- **Advanced Feature Engineering:** 
    - **Elo Rating System:** Dynamic team strength tracking based on the physics of competition.
    - **Rolling Performance Stats:** 5-game rolling averages for goals, shots, shots on target, and corners.
- **Market Comparison:** Benchmarks model probabilities against 4+ global bookmakers (B365, Bwin, etc.).
- **Reliability Diagnostics:** Includes Calibration Curves (Reliability Diagrams) to ensure the model isn't overconfident.
- **Value Discovery:** Identifies "Edge" by finding discrepancies between model forecasts and market prices.

## 🛠 Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ilian-zalomai/PL-Betting-Model.git
   cd PL-Betting-Model
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Dashboard:**
   ```bash
   streamlit run predictive_dashboard.py
   ```

## 📊 Methodology (Project 2 Improvements)
As per instructor feedback from Project 1, this version moves away from retrospective "value" (knowing the outcome) to true **Predictive Value**.
1. **Data:** 8,500+ matches from 2003 to 2026.
2. **Validation:** Chronological split (Training: 2003-2024, Testing: 2025-2026 Season).
3. **Calibration:** Model probabilities are validated against actual frequencies to ensure statistical reliability.

## 🤖 AI Development Log
This project utilized **Gemini CLI** (an AI-agentic coding partner) to:
- Architect the rolling statistics pipeline.
- Implement complex mathematical formulas (Elo ratings, Brier scores).
- Design the professional Streamlit Pro interface.
- Manage dependency and deployment troubleshooting.

*See `AI_DEVELOPMENT_LOG.md` for a full trace of the development process.*
