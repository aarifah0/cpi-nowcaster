"""
Ragged edge processor for CPI Nowcaster.
Handles the mixed-frequency data alignment problem.

The core challenge: at any given moment, monthly economic data has different
publication lags, while daily financial data is available through yesterday.
This module creates a clean matrix with the most recent available data for
each reference month, respecting actual release calendars.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Publication lag (in months) for each monthly series.
# Example: CPI for March is released in mid-April, so on April 1st,
# the latest CPI is February -> lag of 1 month from the reference month.
# These are approximate. Major releases happen ~2-3 weeks into the next month.
PUBLICATION_LAG = {
    "CPIAUCSL": 1,      # CPI: released ~2 weeks into next month
    "UNRATE": 1,         # Unemployment: released ~1 week into next month
    "INDPRO": 1,         # Industrial Production: mid next month
    "PAYEMS": 1,         # Nonfarm Payrolls: ~1 week into next month
    "RSAFS": 1,          # Retail Sales: mid next month
}


def get_latest_available(as_of_date, monthly_df, daily_df):
    """
    Build the ragged-edge matrix for a given date.

    Parameters:
        as_of_date (str or datetime): The date we're "standing on" (today).
        monthly_df (DataFrame): Monthly data with datetime index.
        daily_df (DataFrame): Daily data with datetime index.

    Returns:
        DataFrame with index = reference months, columns = all features,
        values = most recent available data as of as_of_date.
    """
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date)

    # Determine the reference month (the month we're nowcasting)
    # If we're past the 15th of the month, the previous month's data
    # is mostly available and we nowcast the current month.
    # If before the 15th, we're still nowcasting the previous month.
    if as_of_date.day >= 15:
        reference_month = as_of_date.replace(day=1)
    else:
        # We're early in the month, target previous month
        reference_month = (as_of_date.replace(day=1) - pd.DateOffset(months=1))

    # Create a range of reference months for the matrix
# Go back as far as the labels allow (for training)
# First find the earliest date in monthly data
    earliest_monthly = monthly_df.index.min()
    if pd.isna(earliest_monthly):
        earliest_monthly = reference_month - pd.DateOffset(months=119)

    months = pd.date_range(
        start=earliest_monthly,
        end=reference_month,
        freq="MS"
)

    # Initialize the output DataFrame
    ragged = pd.DataFrame(index=months)
    ragged.index.name = "reference_month"

    # ---- Fill monthly data ----
    for col in monthly_df.columns:
        lag = PUBLICATION_LAG.get(col, 1)
        ragged[col] = np.nan

        for month in months:
            # The latest data available for this reference month
            # is from (reference_month - lag) or earlier
            cutoff = month - pd.DateOffset(months=lag)
            
            # Get all available data up to the cutoff
            available = monthly_df.loc[:cutoff, col].dropna()
            
            if not available.empty:
                # Use the most recent available value
                ragged.loc[month, col] = available.iloc[-1]

    # ---- Fill daily data ----
    for col in daily_df.columns:
        ragged[col] = np.nan

        for month in months:
            # Get daily data from the last 5 trading days of the month
            # This captures the "end-of-month" snapshot for each month
            month_end = month + pd.DateOffset(months=1) - pd.DateOffset(days=1)
            
            # Look back from month_end to find available data
            # But not past month_end (no peeking into the future)
            available = daily_df.loc[:month_end, col].dropna()
            
            if not available.empty:
                # Take the last available value on or before month_end
                ragged.loc[month, col] = available.iloc[-1]

    # ---- Also add current-month daily data ----
    # These are the most recent daily values (as of as_of_date)
    # They capture the latest market movements
    current_daily = {}
    for col in daily_df.columns:
        available = daily_df.loc[:as_of_date, col].dropna()
        if not available.empty:
            current_daily[f"{col}_current"] = available.iloc[-1]
            
            # Also get the average over the last 5 trading days
            recent = available.iloc[-5:]
            current_daily[f"{col}_5d_avg"] = recent.mean()
            
            # Get value from 21 trading days ago (1 month) for momentum
            if len(available) > 21:
                current_daily[f"{col}_21d_ago"] = available.iloc[-22]
            else:
                current_daily[f"{col}_21d_ago"] = available.iloc[0]

    # Add current daily data to the last row (reference month)
    for key, value in current_daily.items():
        ragged.loc[reference_month, key] = value

    return ragged


def create_training_labels(monthly_df, target_col="CPIAUCSL"):
    """
    Create year-over-year inflation labels for training.

    Parameters:
        monthly_df (DataFrame): Monthly data with CPI.
        target_col (str): The column to create labels from.

    Returns:
        Series with YoY inflation rates, indexed by month.
    """
    cpi = monthly_df[target_col].dropna()
    yoy_inflation = cpi.pct_change(periods=12) * 100
    return yoy_inflation.dropna()


if __name__ == "__main__":
    # Test the ragged edge processor
    from data_fetcher import fetch_data

    print("=" * 60)
    print("PHASE 2: RAGGED EDGE PROCESSOR TEST")
    print("=" * 60)

    # Fetch data
    print("\n[Step 1] Fetching data from FRED...")
    data = fetch_data(start_date="2015-01-01")
    monthly = data["monthly"]
    daily = data["daily"]

    print(f"  Monthly data: {monthly.shape}")
    print(f"  Daily data:   {daily.shape}")

    # Test with today's date
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n[Step 2] Building ragged edge matrix as of {today}...")
    
    ragged = get_latest_available(today, monthly, daily)

    print(f"\n[Step 3] Ragged edge matrix shape: {ragged.shape}")
    print(f"\nColumns: {list(ragged.columns)}")
    
    print("\n" + "=" * 60)
    print("RAGGED EDGE MATRIX (last 6 months)")
    print("=" * 60)
    print(ragged.tail(6).to_string())

    # Show missing pattern
    print("\n" + "=" * 60)
    print("MISSING DATA PATTERN (NaN count per column)")
    print("=" * 60)
    missing = ragged.isna().sum()
    print(missing[missing > 0].to_string())

    # Create labels
    print("\n" + "=" * 60)
    print("INFLATION LABELS (last 6 months)")
    print("=" * 60)
    labels = create_training_labels(monthly)
    print(labels.tail(6).to_string())