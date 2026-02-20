import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Data loaded: {len(df)} rows.")
    
    # Clean column names
    df.columns = df.columns.str.replace('ï»¿', '').str.strip()
    
    cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'B365H', 'B365D', 'B365A', 'Season']
    
    # Ensure all required columns exist
    present_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[present_cols]
    print(f"Columns filtered: {present_cols}")
    
    # Convert Date to datetime
    print("Converting dates...")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
    print(f"Dates converted. Remaining rows: {len(df)}")
    
    # Sort by date
    df = df.sort_values('Date')
    
    return df

def calculate_rolling_stats(df, rolling_n=5):
    print("Calculating rolling stats...")
    home_stats = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'HC']].copy()
    home_stats.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Shots', 'ShotsOnTarget', 'Corners']
    home_stats['IsHome'] = 1
    
    away_stats = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'AST', 'AC']].copy()
    away_stats.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Shots', 'ShotsOnTarget', 'Corners']
    away_stats['IsHome'] = 0
    
    team_stats = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date'])
    
    stats_to_roll = ['GoalsFor', 'GoalsAgainst', 'Shots', 'ShotsOnTarget', 'Corners']
    for stat in stats_to_roll:
        print(f"Rolling {stat}...")
        team_stats[f'Rolling_{stat}'] = team_stats.groupby('Team')[stat].transform(
            lambda x: x.shift(1).rolling(window=rolling_n, min_periods=rolling_n).mean()
        )
    
    print("Merging stats back...")
    # Drop rows with NaN rolling stats
    team_stats = team_stats.dropna(subset=[f'Rolling_{s}' for s in stats_to_roll])
    
    # Merge Home stats
    df_merged = df.merge(
        team_stats[team_stats['IsHome'] == 1][['Date', 'Team'] + [f'Rolling_{s}' for s in stats_to_roll]],
        left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='inner'
    ).drop(columns=['Team'])
    df_merged = df_merged.rename(columns={f'Rolling_{s}': f'Home_Rolling_{s}' for s in stats_to_roll})
    
    # Merge Away stats
    df_merged = df_merged.merge(
        team_stats[team_stats['IsHome'] == 0][['Date', 'Team'] + [f'Rolling_{s}' for s in stats_to_roll]],
        left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='inner'
    ).drop(columns=['Team'])
    df_merged = df_merged.rename(columns={f'Rolling_{s}': f'Away_Rolling_{s}' for s in stats_to_roll})
    print(f"Rolling stats complete: {len(df_merged)} rows.")
    
    return df_merged

def train_and_evaluate(df):
    feature_cols = [
        'Home_Rolling_GoalsFor', 'Home_Rolling_GoalsAgainst', 'Home_Rolling_Shots', 'Home_Rolling_ShotsOnTarget', 'Home_Rolling_Corners',
        'Away_Rolling_GoalsFor', 'Away_Rolling_GoalsAgainst', 'Away_Rolling_Shots', 'Away_Rolling_ShotsOnTarget', 'Away_Rolling_Corners'
    ]
    
    # Encode target (A, D, H)
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['FTR'])
    
    # Train/Test Split by Season
    seasons = sorted(df['Season'].unique())
    test_season = seasons[-1]
    train_df = df[df['Season'] != test_season]
    test_df = df[df['Season'] == test_season]
    
    print(f"\nTraining on seasons up to {seasons[-2]}")
    print(f"Testing on season {test_season} ({len(test_df)} matches)")
    
    X_train = train_df[feature_cols]
    y_train = train_df['Target']
    X_test = test_df[feature_cols]
    y_test = test_df['Target']
    
    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on {test_season}: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return rf, le, feature_cols, test_df

def value_analysis(test_df, model, le, feature_cols):
    if 'B365H' not in test_df.columns:
        print("Odds data (B365H) not found in test set.")
        return
    
    probs = model.predict_proba(test_df[feature_cols])
    
    # Get index for each class
    class_indices = {cls: i for i, cls in enumerate(le.classes_)}
    
    test_df['Prob_H'] = probs[:, class_indices['H']]
    test_df['Prob_D'] = probs[:, class_indices['D']]
    test_df['Prob_A'] = probs[:, class_indices['A']]
    
    # Calculate Expected Value (EV)
    # EV = Prob * Odds - 1
    test_df['EV_H'] = (test_df['Prob_H'] * test_df['B365H']) - 1
    test_df['EV_D'] = (test_df['Prob_D'] * test_df['B365D']) - 1
    test_df['EV_A'] = (test_df['Prob_A'] * test_df['B365A']) - 1
    
    # Threshold for value
    threshold = 0.10 # 10% expected value
    
    print(f"\n--- Value Betting Opportunities (EV > {threshold*100}%) ---")
    
    # Collect value bets
    value_bets = []
    for idx, row in test_df.iterrows():
        if row['EV_H'] > threshold:
            value_bets.append({'Date': row['Date'], 'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}", 'Pick': 'Home', 'Odds': row['B365H'], 'Prob': row['Prob_H'], 'EV': row['EV_H'], 'Result': row['FTR']})
        if row['EV_D'] > threshold:
            value_bets.append({'Date': row['Date'], 'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}", 'Pick': 'Draw', 'Odds': row['B365D'], 'Prob': row['Prob_D'], 'EV': row['EV_D'], 'Result': row['FTR']})
        if row['EV_A'] > threshold:
            value_bets.append({'Date': row['Date'], 'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}", 'Pick': 'Away', 'Odds': row['B365A'], 'Prob': row['Prob_A'], 'EV': row['EV_A'], 'Result': row['FTR']})
            
    vb_df = pd.DataFrame(value_bets)
    if not vb_df.empty:
        # Check if the bet won
        vb_df['Won'] = vb_df.apply(lambda r: (r['Pick'][0] == r['Result']), axis=1)
        print(vb_df[['Date', 'Match', 'Pick', 'Odds', 'Prob', 'EV', 'Won']].sort_values('EV', ascending=False).head(10))
        win_rate = vb_df['Won'].mean()
        print(f"\nValue Bet Win Rate: {win_rate:.2%}")
    else:
        print("No value bets found with current threshold.")

def get_latest_stats(df):
    """
    Get the most recent rolling stats for every team.
    """
    home_stats = df[['Date', 'HomeTeam', 'Home_Rolling_GoalsFor', 'Home_Rolling_GoalsAgainst', 'Home_Rolling_Shots', 'Home_Rolling_ShotsOnTarget', 'Home_Rolling_Corners']].copy()
    home_stats.columns = ['Date', 'Team', 'Rolling_GoalsFor', 'Rolling_GoalsAgainst', 'Rolling_Shots', 'Rolling_ShotsOnTarget', 'Rolling_Corners']
    
    away_stats = df[['Date', 'AwayTeam', 'Away_Rolling_GoalsFor', 'Away_Rolling_GoalsAgainst', 'Away_Rolling_Shots', 'Away_Rolling_ShotsOnTarget', 'Away_Rolling_Corners']].copy()
    away_stats.columns = ['Date', 'Team', 'Rolling_GoalsFor', 'Rolling_GoalsAgainst', 'Rolling_Shots', 'Rolling_ShotsOnTarget', 'Rolling_Corners']
    
    combined = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date'])
    
    # Get the latest row for each team
    latest = combined.groupby('Team').last().reset_index()
    return latest

def predict_upcoming(model, le, features, latest_stats, home_team, away_team, h_odds=None, d_odds=None, a_odds=None):
    if home_team not in latest_stats['Team'].values or away_team not in latest_stats['Team'].values:
        print(f"Error: One of the teams ({home_team} or {away_team}) not found in dataset.")
        return
    
    h_data = latest_stats[latest_stats['Team'] == home_team].iloc[0]
    a_data = latest_stats[latest_stats['Team'] == away_team].iloc[0]
    
    input_data = pd.DataFrame([{
        'Home_Rolling_GoalsFor': h_data['Rolling_GoalsFor'],
        'Home_Rolling_GoalsAgainst': h_data['Rolling_GoalsAgainst'],
        'Home_Rolling_Shots': h_data['Rolling_Shots'],
        'Home_Rolling_ShotsOnTarget': h_data['Rolling_ShotsOnTarget'],
        'Home_Rolling_Corners': h_data['Rolling_Corners'],
        'Away_Rolling_GoalsFor': a_data['Rolling_GoalsFor'],
        'Away_Rolling_GoalsAgainst': a_data['Rolling_GoalsAgainst'],
        'Away_Rolling_Shots': a_data['Rolling_Shots'],
        'Away_Rolling_ShotsOnTarget': a_data['Rolling_ShotsOnTarget'],
        'Away_Rolling_Corners': a_data['Rolling_Corners']
    }])
    
    # Predict
    probs = model.predict_proba(input_data[features])[0]
    class_indices = {cls: i for i, cls in enumerate(le.classes_)}
    
    p_h, p_d, p_a = probs[class_indices['H']], probs[class_indices['D']], probs[class_indices['A']]
    
    print(f"\n--- PREDICTION: {home_team} vs {away_team} ---")
    print(f"Model Probabilities: Home Win: {p_h:.1%}, Draw: {p_d:.1%}, Away Win: {p_a:.1%}")
    
    if h_odds and d_odds and a_odds:
        ev_h = (p_h * h_odds) - 1
        ev_d = (p_d * d_odds) - 1
        ev_a = (p_a * a_odds) - 1
        
        print("\n--- VALUE FINDER ---")
        print(f"Home (Odds {h_odds}): EV: {ev_h:.2%}")
        print(f"Draw (Odds {d_odds}): EV: {ev_d:.2%}")
        print(f"Away (Odds {a_odds}): EV: {ev_a:.2%}")
        
        best_pick = None
        max_ev = -1
        for pick, ev in [('Home', ev_h), ('Draw', ev_d), ('Away', ev_a)]:
            if ev > max_ev:
                max_ev = ev
                best_pick = pick
        
        if max_ev > 0.05:
            print(f"\nBEST BET: {best_pick} (Expected Value: {max_ev:.2%})")
        else:
            print("\nNo significant value found for this match.")

if __name__ == "__main__":
    print("PL Betting Model - Project 2: Predictive Analysis")
    
    data = load_and_clean_data('all_seasons.csv')
    data_with_stats = calculate_rolling_stats(data)
    model, le, features, test_data = train_and_evaluate(data_with_stats)
    
    # Value analysis on test set
    value_analysis(test_data, model, le, features)
    
    # Predict hypothetical upcoming games
    latest = get_latest_stats(data_with_stats)
    
    print("\n" + "="*40)
    print("UPCOMING PREDICTIONS (Feb 21-22 2026)")
    print("="*40)
    
    # Example games (Fixtures for the upcoming weekend)
    predict_upcoming(model, le, features, latest, 'Man United', 'Arsenal', 3.2, 3.5, 2.2)
    predict_upcoming(model, le, features, latest, 'Chelsea', 'Liverpool', 2.8, 3.4, 2.5)
    predict_upcoming(model, le, features, latest, 'Man City', 'Everton', 1.25, 6.0, 12.0)
    predict_upcoming(model, le, features, latest, 'Tottenham', 'Leeds', 1.8, 3.75, 4.5)
