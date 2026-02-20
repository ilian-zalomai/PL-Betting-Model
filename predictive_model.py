import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report
import warnings
from scipy.stats import poisson

warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    """
    Standard ETL process for Premier League historical data.
    Handles encoding, column cleaning, and type conversion.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.replace('ï»¿', '').str.strip()
    core_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'Season']
    bookie_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'AvgH', 'AvgD', 'AvgA']
    present_cols = [c for c in core_cols + bookie_cols if c in df.columns]
    df = df[present_cols]
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
    df = df.sort_values('Date')
    return df

def calculate_elo(df):
    """
    Implements a dynamic Elo rating system to track team strength relative to opponents.
    Updates ratings match-by-match since 2003.
    """
    elo_ratings = {team: 1500 for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()}
    k_factor = 32
    home_elos, away_elos = [], []
    for idx, row in df.iterrows():
        h_team, a_team, result = row['HomeTeam'], row['AwayTeam'], row['FTR']
        h_elo, a_elo = elo_ratings[h_team], elo_ratings[a_team]
        home_elos.append(h_elo); away_elos.append(a_elo)
        e_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        s_h = 1 if result == 'H' else (0.5 if result == 'D' else 0)
        elo_ratings[h_team] = h_elo + k_factor * (s_h - e_h)
        elo_ratings[a_team] = a_elo + k_factor * ((1 - s_h) - (1 - e_h))
    df['Home_Elo'], df['Away_Elo'] = home_elos, away_elos
    df['Elo_Diff'] = df['Home_Elo'] - df['Away_Elo']
    return df

def calculate_rolling_stats(df, rolling_n=5):
    """
    Generates time-series features using rolling windows.
    Captures recent offensive and defensive form.
    """
    home_stats = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'HC']].copy()
    home_stats.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Shots', 'ShotsOnTarget', 'Corners']
    home_stats['IsHome'] = 1
    away_stats = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'AST', 'AC']].copy()
    away_stats.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Shots', 'ShotsOnTarget', 'Corners']
    away_stats['IsHome'] = 0
    team_stats = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date'])
    stats_to_roll = ['GoalsFor', 'GoalsAgainst', 'Shots', 'ShotsOnTarget', 'Corners']
    for stat in stats_to_roll:
        team_stats[f'Rolling_{stat}'] = team_stats.groupby('Team')[stat].transform(lambda x: x.shift(1).rolling(window=rolling_n, min_periods=rolling_n).mean())
    team_stats = team_stats.dropna(subset=[f'Rolling_{s}' for s in stats_to_roll])
    df_merged = df.merge(team_stats[team_stats['IsHome'] == 1][['Date', 'Team'] + [f'Rolling_{stat}' for stat in stats_to_roll]], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='inner').drop(columns=['Team']).rename(columns={f'Rolling_{s}': f'Home_Rolling_{s}' for s in stats_to_roll})
    df_merged = df_merged.merge(team_stats[team_stats['IsHome'] == 0][['Date', 'Team'] + [f'Rolling_{stat}' for stat in stats_to_roll]], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='inner').drop(columns=['Team']).rename(columns={f'Rolling_{s}': f'Away_Rolling_{s}' for s in stats_to_roll})
    return df_merged

class CumulativeEnsemble:
    """
    Ensemble model that averages prediction probabilities across multiple algorithms.
    Implements standard scikit-learn interface.
    """
    def __init__(self, models): self.models = models
    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)
    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)

def train_and_evaluate(df):
    """
    Trains 3 core ML models and 1 ensemble.
    Validates on the most recent season.
    """
    feature_cols = ['Home_Rolling_GoalsFor', 'Home_Rolling_GoalsAgainst', 'Home_Rolling_Shots', 'Home_Rolling_ShotsOnTarget', 'Home_Rolling_Corners', 'Away_Rolling_GoalsFor', 'Away_Rolling_GoalsAgainst', 'Away_Rolling_Shots', 'Away_Rolling_ShotsOnTarget', 'Away_Rolling_Corners', 'Home_Elo', 'Away_Elo', 'Elo_Diff']
    le = LabelEncoder(); df['Target'] = le.fit_transform(df['FTR'])
    seasons = sorted(df['Season'].unique())
    test_season = seasons[-1]
    train_df, test_df = df[df['Season'] != test_season], df[df['Season'] == test_season]
    X_train, y_train = train_df[feature_cols], train_df['Target']
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=12, max_depth=10, random_state=42).fit(X_train, y_train)
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42).fit(X_train, y_train)
    ensemble = CumulativeEnsemble([rf, lr, xgb])
    models = {"Random Forest": rf, "Logistic Regression": lr, "XGBoost": xgb, "Cumulative Ensemble": ensemble}
    return models, le, feature_cols, test_df

def run_backtest_loop(df, feature_cols, le):
    """
    Sophisticated iterative backtest. 
    For each season, trains on ALL previous data and tests on that season.
    """
    seasons = sorted(df['Season'].unique())
    results = []
    # Start from the 5th season to have enough training data
    for i in range(5, len(seasons)):
        train_s = seasons[:i]
        test_s = seasons[i]
        train_data = df[df['Season'].isin(train_s)]
        test_data = df[df['Season'] == test_s]
        
        # Train simple RF for speed in backtest
        model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        model.fit(train_data[feature_cols], train_data['Target'])
        
        preds = model.predict(test_data[feature_cols])
        acc = accuracy_score(test_data['Target'], preds)
        
        results.append({
            'Season': test_s,
            'Accuracy': acc,
            'Matches': len(test_data),
            'Training_Size': len(train_data)
        })
    return pd.DataFrame(results)

def get_poisson_probabilities(h_exp, a_exp, max_goals=6):
    """
    Calculates exact score probabilities using Poisson distribution.
    """
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = poisson.pmf(i, h_exp) * poisson.pmf(j, a_exp)
    
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    
    return matrix, home_win, draw, away_win

def get_calibration_data(model, X_test, y_test, le):
    """Generates calibration curve points."""
    probs = model.predict_proba(X_test); class_idx = list(le.classes_).index('H'); y_true_h = (y_test == class_idx).astype(int); prob_h = probs[:, class_idx]
    prob_true, prob_pred = calibration_curve(y_true_h, prob_h, n_bins=10)
    return prob_true, prob_pred

def get_latest_stats(df):
    """Aggregates most recent performance metrics for live inference."""
    cols = ['Rolling_GoalsFor', 'Rolling_GoalsAgainst', 'Rolling_Shots', 'Rolling_ShotsOnTarget', 'Rolling_Corners', 'Elo']
    home = df[['Date', 'HomeTeam', 'Home_Rolling_GoalsFor', 'Home_Rolling_GoalsAgainst', 'Home_Rolling_Shots', 'Home_Rolling_ShotsOnTarget', 'Home_Rolling_Corners', 'Home_Elo']].copy()
    home.columns, away = ['Date', 'Team'] + cols, df[['Date', 'AwayTeam', 'Away_Rolling_GoalsFor', 'Away_Rolling_GoalsAgainst', 'Away_Rolling_Shots', 'Away_Rolling_ShotsOnTarget', 'Away_Rolling_Corners', 'Away_Elo']].copy()
    away.columns = ['Date', 'Team'] + cols
    return pd.concat([home, away]).sort_values(['Team', 'Date']).groupby('Team').last().reset_index()
