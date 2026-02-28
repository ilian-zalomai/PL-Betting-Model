# DRIVER Development Log: Premier League Predictive Engine

This document outlines my personal design process and the technical decisions I made during the development of this project.

## 1. Define (The Problem)
My Project 1 was criticized for being retrospective. I defined the goal for Project 2 as building a system that could estimate probabilities **before** kick-off. I specifically wanted to see if rolling performance stats (momentum) or Elo ratings (long-term strength) were better predictors.

## 2. Reconnaissance (Research)
I researched various ML architectures. While I initially thought about a simple Regression, I decided on a **Random Forest** and **XGBoost** ensemble because football data is non-linear and high-variance. I also identified that "Vig" (bookmaker margin) was a major hurdle for ROI, which led me to implement the Margin Analysis tab.

## 3. Implementation (My Design Choices)
I made several key decisions during the coding phase:
- **Feature Selection**: I chose a **5-game rolling window**. Why? Because 3 games is too volatile, and 10 games doesn't capture "current" form accurately enough in a fast-paced league.
- **Elo Initialization**: I decided to start all teams at 1500 in 2003 to allow the system enough "burn-in" time to reach accurate ratings by the time we hit the 2025-26 test season.
- **Ensemble Logic**: I implemented a `CumulativeEnsemble` to average probabilities. This was my choice to reduce the "overfitting" that XGBoost often suffers from on sports data.

## 4. Verify & Evaluate (Critical Thinking)
During testing, I noticed the model was overconfident on heavy favorites. To address this, I requested the AI to help me build a **Calibration Curve**. Seeing the deviation in the reliability diagram allowed me to conclude that "Value" is actually found in the "Draw" market more often than the "Home Win" market.

## 5. Refine (Student Ownership)
I personally directed the refinement of the "Strategic Verdict" logic. I set the EV threshold at **5%** as a "margin of safety" to account for model variance. I also integrated the OpenAI agent specifically to bridge the gap between "hard numbers" and "soft news" (like a last-minute injury).

---
**AI Attribution**: I utilized Gemini CLI as a senior engineering partner to handle the boilerplate of the Streamlit UI and the mathematical optimization of the Poisson matrix. All strategic configurations, feature engineering choices, and validation thresholds were directed by me.
