import datetime
import pytz
import numpy as np
import pandas as pd
from scipy.stats import norm

RISK_FREE_RATE = 0.045  

def calculate_exact_dte(expiration_str):
    tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(tz)
    exp_date = datetime.datetime.strptime(expiration_str, "%Y-%m-%d")
    exp_datetime = tz.localize(datetime.datetime(exp_date.year, exp_date.month, exp_date.day, 16, 0, 0))
    time_delta = exp_datetime - now
    return max(time_delta.total_seconds() / (3600 * 24) / 365.25, 0.00001)

def black_scholes_gamma(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

def project_volatility_path(spot_price, daily_move, buy_zone, sell_zone, gamma_flip, horizon_days=5):
    """
    Builds a predictive multi-day horizon data mapping path
    using institutional friction boundaries.
    """
    days = ["Today"] + [f"Day {i}" for i in range(2, horizon_days + 1)]
    time_steps = np.sqrt(np.arange(1, horizon_days + 1))
    
    path_data = []
    for idx, day in enumerate(days):
        step = time_steps[idx]
        
        upper_band = spot_price + (daily_move * step)
        lower_band = spot_price - (daily_move * step)
        
        pinned_upper = min(upper_band, sell_zone) if sell_zone else upper_band
        pinned_lower = max(lower_band, buy_zone) if buy_zone else lower_band
        
        path_data.append({
            "Horizon": day,
            "Bullish Path Target": upper_band,
            "Pinned Path Upper": pinned_upper,
            "Pinned Path Lower": pinned_lower,
            "Bearish Cascade Target": lower_band
        })
        
    return pd.DataFrame(path_data)
