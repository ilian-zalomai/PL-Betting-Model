import pytest
import pandas as pd
import numpy as np
from predictive_model import load_and_clean_data, calculate_elo, calculate_rolling_stats, train_and_evaluate

def test_data_pipeline():
    """Verify that the CSV data loads and cleans correctly."""
    df = load_and_clean_data('all_seasons.csv')
    assert not df.empty, "Dataframe should not be empty"
    assert 'HomeTeam' in df.columns, "Missing required columns"
    assert 'FTR' in df.columns, "Missing target column"

def test_elo_logic():
    """Verify that Elo ratings are calculated and varied."""
    df = load_and_clean_data('all_seasons.csv')
    df_elo = calculate_elo(df)
    assert 'Home_Elo' in df_elo.columns
    assert df_elo['Home_Elo'].iloc[0] == 1500
    assert df_elo['Home_Elo'].std() > 0, "Elo ratings should vary over time"

def test_model_ensemble():
    """Verify that the multi-algorithm training returns the expected dictionary."""
    df = load_and_clean_data('all_seasons.csv')
    df = calculate_elo(df)
    df = calculate_rolling_stats(df)
    
    models, le, features, test_data = train_and_evaluate(df)
    
    expected_models = ["Random Forest", "Logistic Regression", "XGBoost", "Cumulative Ensemble"]
    for model_name in expected_models:
        assert model_name in models, f"Model {model_name} missing from training output"
    
    assert len(features) == 13, "Should have 13 active features"

def test_prediction_output():
    """Verify that the ensemble produces valid probability distributions."""
    df = load_and_clean_data('all_seasons.csv')
    df = calculate_elo(df)
    df = calculate_rolling_stats(df)
    models, le, features, test_data = train_and_evaluate(df)
    
    sample_row = test_data[features].iloc[:1]
    probs = models["Cumulative Ensemble"].predict_proba(sample_row)[0]
    
    assert len(probs) == 3, "Should output 3 probabilities (H, D, A)"
    assert np.isclose(np.sum(probs), 1.0), "Probabilities must sum to 1"
