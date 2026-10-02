import sys
import os

# ─── PATH ANCHOR ROUTING (MUST SIT AT TOP) ───
PROJECT_DIR = r"c:\Users\faiza\OneDrive\Desktop\GexVex"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from calculations import calculate_exact_dte, black_scholes_gamma, RISK_FREE_RATE, project_volatility_path

st.set_page_config(page_title="GEX Advanced Analytics Dashboard", layout="wide")

st.title("📊 Institutional Options Heat Engine")
st.markdown("Query custom ticker assets and explore multi-week structural dealer positioning walls in real-time.")

st.sidebar.header("🎯 Target Selection Controls")
ticker_options = ["SPY", "QQQ", "IWM", "DIA", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD"]
user_ticker = st.sidebar.selectbox("Select Equity Ticker Symbol:", options=ticker_options, index=0)
range_slider = st.sidebar.slider("Strike Boundary View Window (%)", min_value=1, max_value=25, value=5)

# DYNAMIC PRICE HIGHWAY TIMELINES DROPDOWN
horizon_selection = st.sidebar.selectbox(
    "Predictive Path Forecast Horizon:",
    options=[5, 10, 15],
    format_func=lambda x: f"{x} Days Outlook",
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Liquidity Matrix Engine")
matrix_mode = st.sidebar.radio(
    "Choose Active Data Layer:",
    options=["Open Interest Architecture", "Live Intraday Volume Flows"]
)

if user_ticker:
    with st.spinner(f"Extracting live option matrix records for {user_ticker}..."):
        asset = yf.Ticker(user_ticker)
        
        try:
            historical_data = asset.history(period="1d")
            if historical_data.empty:
                st.error(f"Could not locate pricing data for '{user_ticker}'.")
                st.stop()
            spot_price = historical_data["Close"].iloc[-1]
        except Exception:
            st.error(f"Data fetching connection failure for token symbol '{user_ticker}'.")
            st.stop()
            
        raw_yield = asset.info.get("dividendYield", 0.0) or 0.0
        div_yield = raw_yield if raw_yield < 0.15 else (raw_yield / 100.0 if raw_yield > 1.0 else 0.0)
        expirations = asset.options
        
        if not expirations:
            st.warning(f"No active option chain architectures detected for '{user_ticker}'.")
            st.stop()

        selected_expiries = st.multiselect(
            "Select Expiration Horizons to Map:",
            options=expirations,
            default=expirations[:4]
        )
        
        if not selected_expiries:
            st.info("Please select at least one expiration date column to map the data matrix layout.")
            st.stop()
            
        all_contracts = []
        range_pct = range_slider / 100.0
        data_col = 'openInterest' if matrix_mode == "Open Interest Architecture" else 'volume'
        
        for exp in selected_expiries:
            T = calculate_exact_dte(exp)
            try: opt_chain = asset.option_chain(exp)
            except Exception: continue
            
            for _, row in opt_chain.calls.iterrows():
                if pd.isna(row[data_col]) or row[data_col] < 50 or row['volume'] == 0: continue
                if abs(row['strike'] - spot_price) / spot_price > range_pct: continue 
                gamma = black_scholes_gamma(spot_price, row['strike'], T, RISK_FREE_RATE, div_yield, row['impliedVolatility'])
                all_contracts.append({
                    "strike": row['strike'], "expiry": exp, 
                    "gex_m": (row[data_col] * 100 * gamma * spot_price * 0.1) / 1_000_000
                })
                
            for _, row in opt_chain.puts.iterrows():
                if pd.isna(row[data_col]) or row[data_col] < 50 or row['volume'] == 0: continue
                if abs(row['strike'] - spot_price) / spot_price > range_pct: continue
                gamma = black_scholes_gamma(spot_price, row['strike'], T, RISK_FREE_RATE, div_yield, row['impliedVolatility'])
                all_contracts.append({
                    "strike": row['strike'], "expiry": exp, 
                    "gex_m": (row[data_col] * 100 * gamma * spot_price * -0.1) / 1_000_000
                })

        if not all_contracts:
            st.error("No option contracts matched your liquidity flow filters inside this boundary window.")
        else:
            df = pd.DataFrame(all_contracts)
            grid = df.groupby(["strike", "expiry"])["gex_m"].sum().unstack(fill_value=0.0)
            grid = grid.sort_index(ascending=False)

            total_net_gex_b = (df["gex_m"].sum() * 1_000_000) / 1_000_000_000
            strike_totals = df.groupby("strike")["gex_m"].sum().reset_index()
            
            puts_only = strike_totals[strike_totals["gex_m"] < 0]
            calls_only = strike_totals[strike_totals["gex_m"] > 0]
            
            # 🔥 FIXED: Explicit positional .iloc[0] placement used below to resolve key indexing errors
            buy_zone_strike = puts_only.sort_values(by="gex_m").iloc[0]["strike"] if not puts_only.empty else spot_price * 0.98
            sell_zone_strike = calls_only.sort_values(by="gex_m", ascending=False).iloc[0]["strike"] if not calls_only.empty else spot_price * 1.02

            matrix_totals = strike_totals.copy()
            matrix_totals['cross_check'] = matrix_totals['gex_m'].shift(1)
            flip_rows = matrix_totals[((matrix_totals['gex_m'] >= 0) & (matrix_totals['cross_check'] < 0)) | 
                                      ((matrix_totals['gex_m'] < 0) & (matrix_totals['cross_check'] >= 0))]
            
            if not flip_rows.empty:
                gamma_flip_strike = flip_rows.iloc[(flip_rows['strike'] - spot_price).abs().argsort()[:1]]["strike"].values[0]
            else:
                gamma_flip_strike = None

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Spot Price", f"${spot_price:.2f}")
            
            gex_label = "🟢 Long Gamma" if total_net_gex_b >= 0 else "🔴 Short Gamma"
            col2.metric("Total Net GEX", f"{total_net_gex_b:+.3f}B", help=gex_label)
            col3.metric("🔵 INSTITUTIONAL BUY ZONE", f"${buy_zone_strike:.1f}")
            col4.metric("🟢 TAKE PROFIT / SELL ZONE", f"${sell_zone_strike:.1f}")
            
            flip_text = f"${gamma_flip_strike:.1f}" if gamma_flip_strike else "N/A"
            col5.metric("Gamma Flip", flip_text)
            st.markdown("---")

            fig, ax = plt.subplots(figsize=(14, max(9, len(grid) * 0.36)), facecolor='#0D1117')
            ax.set_facecolor('#0D1117')

            colors_step = ["#D9381E", "#161B22", "#1E6091"]
            cmap = mcolors.LinearSegmentedColormap.from_list("HighContrastGEX", colors_step, N=256)

            max_val = max(abs(grid.values.min()), abs(grid.values.max()), 1.0)
            sns.heatmap(
                grid, annot=True, fmt=".1f", cmap=cmap, center=0.0, vmin=-max_val, vmax=max_val,
                cbar_kws={'label': 'Dealer Gamma Profile (Millions $)'},
                linewidths=1.0, linecolor='#0D1117',
                annot_kws={"size": 10, "weight": "bold"}, ax=ax
            )

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid.values[i, j]
                    cell_text = ax.texts[i * grid.shape[1] + j]
                    if val > 0.5 or val < -0.5: cell_text.set_color('#FFFFFF')
                    else: cell_text.set_color('#484F58')

            strike_array = np.array(grid.index)
            closest_strike = strike_array[(np.abs(strike_array - spot_price)).argsort()][0]
            spot_idx = list(grid.index).index(closest_strike)

            ax.add_patch(plt.Rectangle((0, spot_idx), len(grid.columns), 1, fill=False, edgecolor='#FFD700', lw=4.0, zorder=5))

            if gamma_flip_strike and gamma_flip_strike in grid.index:
                flip_idx = list(grid.index).index(gamma_flip_strike)
                ax.axhline(flip_idx + 0.5, color='#FF007F', linestyle='--', lw=3.0, label='Gamma Flip Level', zorder=6)

            if buy_zone_strike in grid.index:
                b_idx = list(grid.index).index(buy_zone_strike)
                ax.text(len(grid.columns) + 0.1, b_idx + 0.6, "⬅️ BUY FLOOR", color='#52B788', weight='bold', fontsize=11, va='center')
                
            if sell_zone_strike in grid.index:
                s_idx = list(grid.index).index(sell_zone_strike)
                ax.text(len(grid.columns) + 0.1, s_idx + 0.6, "⬅️ SELL CEILING", color='#4EA8DE', weight='bold', fontsize=11, va='center')

            ax.tick_params(colors='#FFFFFF', which='both', labelsize=11)
            ax.set_yticklabels([f"{float(label.get_text()):.1f}" for label in ax.get_yticklabels()], rotation=0)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            plt.xticks(rotation=15, ha='left')
            plt.ylabel("Option Strike Values ($)", fontsize=12, color='#8B949E')
            plt.xlabel(f"Selected Expiration Horizons Matrix Layer ({matrix_mode})", fontsize=12, color='#8B949E', labelpad=15)
            
            if len(ax.collections) > 0:
                heatmap_obj = ax.collections[0]
                if hasattr(heatmap_obj, 'colorbar') and heatmap_obj.colorbar:
                    heatmap_obj.colorbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

            st.pyplot(fig)

            # ─── VOLATILITY PATH TRAJECTORY FORECASTER ───
            st.markdown(f"### 🗺️ Future Volatility Path Estimator ({horizon_selection}-Day Trajectory)")
            st.markdown("This line matrix graphs the highest-probability paths for the asset based on option-implied volatility models.")

            try:
                t_symbol = "^VXN" if user_ticker == "QQQ" else "^VIX"
                vix_close = yf.Ticker(t_symbol).history(period="1d")["Close"].iloc[-1] / 100
            except Exception:
                vix_close = 0.16
            daily_move = spot_price * (vix_close / np.sqrt(252))
            # Pass our active 'horizon_selection' choice directly to the mapping engine
            path_df = project_volatility_path(spot_price, daily_move, buy_zone_strike, sell_zone_strike, gamma_flip_strike, horizon_days=horizon_selection)
            fig_path, ax_path = plt.subplots(figsize=(14, 5), facecolor='#0D1117')
            ax_path.set_facecolor('#0D1117')
            ax_path.plot(path_df["Horizon"], path_df["Bullish Path Target"], color='#1E6091', linestyle=':', lw=2, label='Maximum Upside Breakout Route')
            ax_path.plot(path_df["Horizon"], path_df["Pinned Path Upper"], color='#22C55E', lw=3, label='Standard Expected Ceiling (Pinned Track)')
            ax_path.plot(path_df["Horizon"], path_df["Pinned Path Lower"], color='#EF4444', lw=3, label='Standard Expected Floor (Pinned Track)')
            ax_path.plot(path_df["Horizon"], path_df["Bearish Cascade Target"], color='#D9381E', linestyle=':', lw=2, label='Maximum Volatility Cascade Breakdown')
            ax_path.axhline(spot_price, color='#FFD700', linestyle='-', lw=1, alpha=0.5, label='Current Entry Spot Price')
            ax_path.set_title(f"{user_ticker} {horizon_selection}-DAY EXPECTED VOLATILITY TRAJECTORY HIGHWAY", color='white', fontsize=12, pad=15, weight='bold')
            ax_path.tick_params(colors='white', which='both', labelsize=10)
            ax_path.set_ylabel("Projected Asset Price ($)", color='#8B949E')
            ax_path.grid(color='#1E293B', linestyle='--', alpha=0.5)
            legend = ax_path.legend(facecolor='#0D1117', edgecolor='#1E293B', labelcolor='white', loc='upper left', fontsize=9)
            st.pyplot(fig_path)