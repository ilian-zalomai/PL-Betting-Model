# AI Development Log: PL Predictive Betting Engine v2.0

## 🛠 Overview
This log documents the iterative development of the **PL Predictive Betting Engine v2.0**, developed in partnership with **Gemini CLI** (an agentic software engineering AI). This collaboration allowed the project to move from a basic 159-line historical analyzer to a 500+ line predictive machine learning system.

## 📅 Development Timeline (Project 2 Evolution)

### Phase 1: Predictive Engine Architecture (Feb 20, 2026)
- **AI Task:** Transition from retrospective "value" logic to predictive modeling.
- **AI Strategy:** Architected a **Random Forest Classifier** pipeline.
- **Logic Implemented:** 
    - Designed a **Rolling Statistics** function (`calculate_rolling_stats`) to capture 5-game form (goals, shots, corners).
    - Established a **Chronological Split** (Seasons 2003-2024 for training, 2025-2026 for testing) to prevent data leakage.
- **AI Tool Used:** `write_file`, `run_shell_command` (Python script generation).

### Phase 2: Professional Interface Design (Feb 20, 2026)
- **AI Task:** Create a Streamlit dashboard that unifies Project 1 and Project 2 work.
- **AI Strategy:** Implemented a **Multi-Tab Dashboard** using custom **CSS Styling** to mimic professional trading terminals (dark mode, custom metrics).
- **AI Tool Used:** `write_file` (Streamlit frontend engineering).

### Phase 3: Deepening Analytics (Project 2 Enhancements)
- **AI Task:** Address instructor feedback by adding "Deep Analytics" (Elo, Calibration, Multi-Bookie Comparison).
- **Logic Implemented:**
    - **Elo Rating System:** AI implemented a dynamic Elo calculation function to track team strength over 20+ seasons.
    - **Calibration Curves:** AI integrated `sklearn.calibration.calibration_curve` to generate Reliability Diagrams.
    - **Market Benchmark:** AI designed a **Brier Score Comparative Analysis** between the ML model and 4+ global bookmakers.
- **AI Tool Used:** `replace`, `thinkdeep` (Complex math and architecture validation).

### Phase 4: Deployment & Compliance
- **AI Task:** Resolve Streamlit Cloud errors and ensure repository reproducibility.
- **Logic Implemented:** 
    - Fixed `ModuleNotFoundError` by updating `requirements.txt`.
    - Resolved Streamlit deprecation warnings (`use_container_width` -> `width='stretch'`).
    - Standardized repository structure for public submission.
- **AI Tool Used:** `git status`, `git commit`, `git push`.

## 🤖 AI Leverage Assessment
The AI acted as a **Senior Engineering Partner**, not just a code generator. It was responsible for:
1. **Mathematical Accuracy:** Implementing Elo and Brier Score logic correctly.
2. **Architecture:** Ensuring the ML pipeline prevented data leakage through chronological splitting.
3. **UI/UX:** Elevating the interface to a professional grade.
4. **Validation:** Proposing calibration curves to test for model overconfidence.

*This project represents a human-AI collaboration that prioritized predictive rigor and statistical transparency.*
