# DRIVER Development Log: PL Predictive Engine

This log documents the iterative development of the Premier League Predictive Engine, following the DRIVER process (Define, Reconnaissance, Implement, Verify, Evaluate, Refine).

## Phase 1: Define & Reconnaissance (Feb 18 - 19, 2026)
- **Objective**: Move from retrospective Project 1 logic to a forward-looking ML model.
- **Task**: Analyze 20+ seasons of CSV data and identify features that capture "current form" vs "underlying strength".
- **AI Role**: Assisted in auditing the CSV structure and identifying available bookmaker columns.

## Phase 2: Implementation (Feb 20, 2026)
- **Engine**: Developed `predictive_model.py` using a Random Forest Classifier.
- **Feature Engineering**: Integrated Elo Rating system and 5-game rolling averages for goals, shots, and corners.
- **Dashboard**: Created the multi-tab Streamlit interface (`predictive_dashboard.py`).
- **AI Role**: Generated the multi-algorithm ensemble class and custom CSS for the "Dark Mode" terminal.

## Phase 3: Verification & Diagnostics (Feb 20, 2026)
- **Testing**: Implemented Brier Score comparisons and Calibration Curves (Reliability Diagrams) to verify if predicted probabilities match real-world outcomes.
- **CI/CD**: Added `test_model.py` and GitHub Actions workflow to ensure build stability on every push.
- **AI Role**: Debugged Streamlit Cloud dependency issues and refactored code to prevent line truncation.

## Phase 4: Refinement & AI Integration (Feb 20, 2026)
- **Research Agent**: Integrated an OpenAI GPT-4o-mini agent with DuckDuckGo web search to capture live news (injuries/lineups) that the static model misses.
- **Strategy Optimizer**: Added ROI simulation tools to find the most profitable odds ranges and EV thresholds.
- **AI Role**: Acted as an agentic engineer to implement live web-search tools and refine the Strategic Verdict logic.

## Final Review
- **Code Volume**: Exceeded 500 lines across unified scripts.
- **Academic Rigor**: Addressed professor feedback by adding market margin analysis and critical limitation documentation.
