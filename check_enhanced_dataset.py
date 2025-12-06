"""Check the enhanced dataset for NaN issues."""
import pandas as pd

df = pd.read_csv('data/processed/match_dataset_enhanced.csv', low_memory=False)
print(f"Total rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()

# Exactly match select_feature_columns logic
feature_cols = []
for col in df.columns:
    if col.startswith("home_avg_") or col.startswith("away_avg_"):
        feature_cols.append(col)
    elif col.startswith("home_ema_") or col.startswith("away_ema_"):
        feature_cols.append(col)
    elif col.startswith("home_recent_") or col.startswith("away_recent_"):
        feature_cols.append(col)
    elif col.startswith("league_avg_"):
        feature_cols.append(col)
    elif col.startswith("home_avg_xg_") or col.startswith("away_avg_xg_"):
        feature_cols.append(col)
    elif col in {"home_matches_played", "away_matches_played"}:
        feature_cols.append(col)
    elif col.startswith("home_adv_") or col.startswith("away_adv_"):
        feature_cols.append(col)
    elif col.startswith("weather_"):
        feature_cols.append(col)
    elif col in {"home_injury_count", "away_injury_count", "home_injury_severity", "away_injury_severity",
                 "home_suspended_count", "away_suspended_count", "injury_count_diff", "injury_severity_diff"}:
        feature_cols.append(col)
    elif col in {"home_manager_tenure_days", "away_manager_tenure_days", "home_new_manager", "away_new_manager",
                 "home_experienced_manager", "away_experienced_manager", "manager_tenure_diff"}:
        feature_cols.append(col)

print(f"Total feature columns: {len(feature_cols)}")

# Define optional vs core features
optional_prefixes = ('xg', 'xG', 'home_adv_', 'away_adv_', 'weather_', 
                     'home_injuries_', 'away_injuries_', 'home_manager_', 'away_manager_')

optional_features = [c for c in feature_cols if any(c.startswith(p) or p in c.lower() for p in optional_prefixes)]
core_features = [c for c in feature_cols if c not in optional_features]

print(f"Core features: {len(core_features)}")
print(f"Optional features: {len(optional_features)}")
print()

# Check NaN in core features
print("NaN counts in core features (top 20 with most NaNs):")
core_nan = {col: df[col].isna().sum() for col in core_features}
for col, nan_count in sorted(core_nan.items(), key=lambda x: -x[1])[:20]:
    if nan_count > 0:
        print(f"  {col}: {nan_count} / {len(df)} ({100*nan_count/len(df):.1f}%)")

# Check which columns are missing entirely from the dataframe
print()
missing_cols = [c for c in feature_cols if c not in df.columns]
print(f"Feature columns not in dataframe: {len(missing_cols)}")
for col in missing_cols[:5]:
    print(f"  {col}")

print()
print(f"Complete cases (core features only): {df[core_features].dropna().shape[0]} / {len(df)}")

