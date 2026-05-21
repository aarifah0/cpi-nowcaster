"""
MIDAS + XGBoost model for CPI inflation nowcasting.
Handles mixed-frequency data and produces current-month inflation estimates.
"""

import datetime

import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


def create_midas_weights(num_lags=22, degree=2):
    """
    Create MIDAS (Mixed Data Sampling) polynomial weights.
    
    MIDAS uses a polynomial to weight high-frequency (daily) lags
    when predicting a low-frequency (monthly) target. Instead of
    using 22 separate daily lags as features, we compress them
    into a few weighted sums using an Almon polynomial.
    
    Parameters:
        num_lags (int): Number of daily lags (approx 22 trading days/month).
        degree (int): Degree of the Almon polynomial.
    
    Returns:
        numpy array of shape (num_lags, degree + 1): MIDAS weights.
    """
    # Normalized time index from 0 to 1
    t = np.linspace(0, 1, num_lags)
    
    # Almon polynomial: each column is t^d
    weights = np.column_stack([t ** d for d in range(degree + 1)])
    
    return weights


def apply_midas(daily_series, weights):
    """
    Apply MIDAS weighting to a daily series to create compressed features.
    
    Parameters:
        daily_series (array): Daily values, most recent first.
        weights (array): MIDAS weight matrix.
    
    Returns:
        array: Compressed MIDAS features (one per polynomial degree).
    """
    if len(daily_series) < len(weights):
        # Not enough data, pad with the first value
        padding = [daily_series[0]] * (len(weights) - len(daily_series))
        daily_series = np.concatenate([daily_series, padding])
    
    # Take the most recent num_lags values
    recent = daily_series[-len(weights):]
    
    # Apply polynomial weights
    midas_features = recent.dot(weights)
    
    return midas_features


def prepare_training_data(ragged_df, labels, daily_cols=None):
    """
    Convert the ragged edge matrix into training features and targets.
    
    Parameters:
        ragged_df (DataFrame): Output from ragged_edge.get_latest_available().
        labels (Series): YoY inflation rates, indexed by month.
        daily_cols (list): Column names of daily data series.
    
    Returns:
        X (DataFrame): Feature matrix.
        y (Series): Target vector (inflation).
        feature_names (list): Names of all features.
    """
    if daily_cols is None:
        daily_cols = ["T5YIE", "DTWEXBGS", "DCOILWTICO"]
    
    # Create a copy to avoid modifying the original
    df = ragged_df.copy()
    
    # Add lagged CPI (inflation inertia)
    df["CPI_lag1"] = df["CPIAUCSL"].shift(1)
    df["CPI_lag2"] = df["CPIAUCSL"].shift(2)
    df["CPI_lag3"] = df["CPIAUCSL"].shift(3)
    
    # Calculate month-over-month changes for monthly data
    for col in ["UNRATE", "INDPRO", "PAYEMS", "RSAFS"]:
        if col in df.columns:
            df[f"{col}_mom"] = df[col].pct_change() * 100
    
    # Use the current daily values as features
    # These represent the latest market snapshot
    current_features = [c for c in df.columns if c.endswith("_current") or 
                                                     c.endswith("_5d_avg") or 
                                                     c.endswith("_21d_ago")]
    
    # Base monthly features
    monthly_features = ["UNRATE", "INDPRO", "PAYEMS", "RSAFS"]
    mom_features = [f"{col}_mom" for col in monthly_features]
    
    # Daily end-of-month features
    daily_features = ["T5YIE", "DTWEXBGS", "DCOILWTICO"]
    
    # Combine all features
    feature_names = (["CPI_lag1", "CPI_lag2", "CPI_lag3"] + 
                     monthly_features + 
                     mom_features + 
                     daily_features + 
                     current_features)
    
    # Keep only features that exist in the DataFrame
    feature_names = [f for f in feature_names if f in df.columns]
    
    X = df[feature_names].copy()
    
    # Align with labels
    y = labels.reindex(X.index)
    
    # Drop rows where target is NaN
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]
    
    # Fill remaining NaN features with column median
    X = X.fillna(X.median())
    
    return X, y, feature_names


def expanding_window_cv(X, y, model=None, min_train=24):
    """
    Perform expanding window cross-validation.
    
    Unlike standard K-fold, this respects time ordering.
    For each month t:
        Train on all data up to month t-1
        Test on month t
    This mimics how the model would perform in real-time.
    
    Parameters:
        X (DataFrame): Feature matrix with datetime index.
        y (Series): Target vector.
        model: XGBoost model (or None to create default).
        min_train (int): Minimum number of training months.
    
    Returns:
        predictions (DataFrame): Actual vs predicted for each test month.
        metrics (dict): Summary performance metrics.
    """
    if model is None:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    # Sort by date
    X = X.sort_index()
    y = y.sort_index()
    
    predictions = []
    
    for i in range(min_train, len(X)):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[i:i+1]
        y_test = y.iloc[i:i+1]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        
        predictions.append({
            "date": X_test.index[0],
            "actual": y_test.values[0],
            "predicted": pred
        })
    
    # Handle case where there aren't enough samples
    if len(predictions) == 0:
        print("  Warning: Not enough data for cross-validation.")
        print(f"  Need at least {min_train + 1} samples, got {len(X)}.")
        empty_df = pd.DataFrame(columns=["actual", "predicted"])
        empty_metrics = {"RMSE": None, "MAE": None, "num_test_months": 0}
        return empty_df, empty_metrics
    
    pred_df = pd.DataFrame(predictions).set_index("date")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(pred_df["actual"], pred_df["predicted"]))
    mae = mean_absolute_error(pred_df["actual"], pred_df["predicted"])
    
    metrics = {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "num_test_months": len(pred_df)
    }
    
    return pred_df, metrics


def train_final_model(X, y):
    """
    Train the final model on all available data.
    
    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target vector.
    
    Returns:
        model: Trained XGBoost model.
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X, y)
    return model


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from trained model.
    
    Parameters:
        model: Trained XGBoost model.
        feature_names (list): Names of features.
    
    Returns:
        DataFrame: Feature importance sorted by importance score.
    """
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return importance


if __name__ == "__main__":
    # Test the full model pipeline
    from data_fetcher import fetch_data
    from ragged_edge import get_latest_available, create_training_labels
    
    print("=" * 60)
    print("PHASE 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # Step 1: Get data
    print("\n[Step 1] Fetching data...")
    data = fetch_data(start_date="2010-01-01")
    monthly = data["monthly"]
    daily = data["daily"]
    
    # Step 2: Build ragged edge matrix
    print("\n[Step 2] Building ragged edge matrix...")
    
    today = datetime.now().strftime("%Y-%m-%d")
    ragged = get_latest_available(today, monthly, daily)
    # Step 3: Create labels
    print("\n[Step 3] Creating inflation labels...")
    labels = create_training_labels(monthly)
    print(f"  Labels: {len(labels)} months")
    print(f"  Range: {labels.min():.2f}% to {labels.max():.2f}%")
    
    # Step 4: Prepare training data
    print("\n[Step 4] Preparing features...")
    X, y, feature_names = prepare_training_data(ragged, labels)
    print(f"  Features: {X.shape[1]} columns")
    print(f"  Training samples: {X.shape[0]} months")
    print(f"  Feature names: {feature_names}")
    
    # Step 5: Cross-validation
    print("\n[Step 5] Expanding window cross-validation...")
    pred_df, metrics = expanding_window_cv(X, y)
    print(f"  RMSE: {metrics['RMSE']}%")
    print(f"  MAE: {metrics['MAE']}%")
    print(f"  Test months: {metrics['num_test_months']}")
    
    # Step 6: Train final model
    print("\n[Step 6] Training final model on all data...")
    model = train_final_model(X, y)
    
    # Step 7: Feature importance
    print("\n[Step 7] Feature importance:")
    print("-" * 40)
    importance = get_feature_importance(model, feature_names)
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"  {row['feature']:<25s} {row['importance']:.3f}  {bar}")
    
    # Step 8: Show predictions
    print("\n" + "=" * 60)
    print("PREDICTIONS VS ACTUALS (last 12 months)")
    print("=" * 60)
    print(pred_df.tail(12).round(2).to_string())
    
    # Step 9: Current nowcast
    print("\n" + "=" * 60)
    print("CURRENT NOWCAST")
    print("=" * 60)
    try:
        current_features = X.iloc[-1:].copy()
        current_nowcast = model.predict(current_features)[0]
        latest_actual = y.iloc[-1]
        print(f"  Latest actual CPI inflation: {latest_actual:.2f}%")
        print(f"  Current nowcast:             {current_nowcast:.2f}%")
    except Exception as e:
        print(f"  Could not generate nowcast: {e}")