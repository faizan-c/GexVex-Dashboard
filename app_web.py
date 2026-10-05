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
import plotly.express as px
import plotly.graph_objects as go

from calculations import calculate_exact_dte, black_scholes_gamma, RISK_FREE_RATE, project_volatility_path

st.set_page_config(page_title="GEX Advanced Analytics Dashboard", layout="wide")

# ─── PREMIUM CUSTOM CSS DESIGN SYSTEM ───
st.markdown("""
    <style>
    /* Main Background Overrides */
    .stApp { background-color: #0D1117; }
    
    /* Institutional Metric Cards */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 11px;
        color: #8B949E;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #C9D1D9;
        margin-top: 4px;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ARCHITECTURE CONTROLS ───
st.sidebar.header("🎯 Target Selection Controls")
ticker_options = [
    # Indices
    "SPY", "QQQ",

    # MyFavs
    "CRDO", "AAOI", "CRWV", "NBIS", "LITE",

    # Top Tech & Mega-Caps
    "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AVGO", 
    "AMD", "NFLX", "PLTR", "INTC", "MU", "QCOM", "ADBE", "CSCO",
    
    # High-Volume Growth & Semis
    "AMAT", "LRCX", "KLAC", "MRVL", "PANW", "CRWD", "TXN", "ADI", 
    "ASML", "WDC", "STX", "SNPS", "CDNS", "DDOG", "FTNT", "ARM", 
    
    # Prominent S&P 500 Consumer, Finance & Industrials
    "WMT", "COST", "PEP", "SBUX", "BKNG", "MCD", "NKE", "LULU", 
    "JPM", "BAC", "GS", "MS", "CAT", "GE", "HON", "NOW"
]

# 1. Filter out 'SPY' and 'QQQ' from the original list, then sort the rest
remaining_sorted = sorted([t for t in ticker_options if t not in ('SPY', 'QQQ')])

# 2. Put 'SPY' and 'QQQ' at the front, followed by the sorted remaining tickers
custom_options = ['SPY', 'QQQ'] + remaining_sorted

with st.sidebar:
    with st.expander("🎯 Target Profiles", expanded=True):
        user_ticker = st.selectbox("Select Equity Ticker Symbol:", options=custom_options, index=0)
        range_slider = st.slider("Strike Boundary View Window (%)", min_value=1, max_value=25, value=5)

    # DYNAMIC PRICE HIGHWAY TIMELINES DROPDOWN
    with st.expander("⏱️ Horizon Settings", expanded=True):
        horizon_selection = st.selectbox(
            "Predictive Path Forecast Horizon:",
            options=[5, 10, 15],
            format_func=lambda x: f"{x} Days Outlook",
            index=0
        )

    st.markdown("---")
    with st.expander("⚙️ Liquidity Matrix Engine", expanded=True):
        matrix_mode = st.radio(
            "Choose Active Data Layer:",
            options=["Open Interest Architecture", "Live Intraday Volume Flows"]
        )

# ─── MAIN HEADER DISPLAY LAYER ───
if user_ticker:
    st.title("📊 Institutional Options Heat Engine")
    st.markdown(f"Query custom ticker assets and explore multi-week structural dealer positioning walls for **{user_ticker}** in real-time.")
    st.markdown("---")

    with st.status(f"Extracting live option matrix records for {user_ticker}...", expanded=False) as status:
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

        status.update(label="Live Options Chain Retrieved successfully!", state="complete")

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
        grid = grid.sort_index(ascending=True)

        total_net_gex_b = (df["gex_m"].sum() * 1_000_000) / 1_000_000_000
        strike_totals = df.groupby("strike")["gex_m"].sum().reset_index()
        
        puts_only = strike_totals[strike_totals["gex_m"] < 0]
        calls_only = strike_totals[strike_totals["gex_m"] > 0]
        
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

        # ─── METRIC CARD LAYOUT ENGINE ───
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Spot Price</div><div class="metric-value">${spot_price:.2f}</div></div>', unsafe_allow_html=True)
        with col2:
            gex_color = "#22C55E" if total_net_gex_b >= 0 else "#EF4444"
            gex_txt = "🟢 Long Gamma" if total_net_gex_b >= 0 else "🔴 Short Gamma"
            st.markdown(f'<div class="metric-card" title="{gex_txt}"><div class="metric-label">Total Net GEX</div><div class="metric-value" style="color: {gex_color};">{total_net_gex_b:+.3f}B</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">🔵 BUY FLOOR</div><div class="metric-value" style="color: #52B788;">${buy_zone_strike:.1f}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">🟢 SELL CEILING</div><div class="metric-value" style="color: #4EA8DE;">${sell_zone_strike:.1f}</div></div>', unsafe_allow_html=True)
        with col5:
            flip_text = f"${gamma_flip_strike:.1f}" if gamma_flip_strike else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Gamma Flip</div><div class="metric-value" style="color: #FF007F;">{flip_text}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")

        # ─── INTERACTIVE PLOTLY HEATMAP ENGINE (CUSTOM DARK TERMINAL PALETTE) ───
        max_val = max(abs(grid.values.min()), abs(grid.values.max()), 1.0)
        
        # Custom high-contrast dark palette: Negative is Red, Zero is Dark Slate, Positive is Blue
        dark_terminal_cmap = [
            [0.0, "#FF4444"],   # Max Negative Gamma = Bright Red
            [0.4, "#8B2626"],   # Low Negative Gamma = Dark Red
            [0.5, "#161B22"],   # Zero Neutral Gamma = Dark Slate Background
            [0.6, "#0F4C81"],   # Low Positive Gamma = Dark Blue
            [1.0, "#00E5FF"]    # Max Positive Gamma = Electric Cyan/Blue
        ]
        
        fig_heatmap = px.imshow(
            grid,
            labels=dict(x="Expiration Date", y="Strike Price ($)", color="GEX ($M)"),
            x=grid.columns,
            y=grid.index,
            color_continuous_scale=dark_terminal_cmap,
            color_continuous_midpoint=0.0,
            range_color=[-max_val, max_val],
            text_auto=".1f",
            aspect="auto"
        )
        
        fig_heatmap.update_layout(
            paper_bgcolor="#0D1117",  
            plot_bgcolor="#0D1117",   
            xaxis=dict(
                side="top", 
                tickangle=15, 
                title=None, 
                gridcolor="#21262D",   # Dark subtle grid lines
                zerolinecolor="#21262D",
                tickfont=dict(color="#C9D1D9", size=11, weight="bold")
            ),
            yaxis=dict(
                title=dict(text="Option Strike Values ($)", font=dict(color="#8B949E", size=12)), 
                gridcolor="#21262D",   
                zerolinecolor="#21262D",
                tickformat=".1f",
                tickfont=dict(color="#C9D1D9", size=11, weight="bold"),
                autorange=True
            ),
            coloraxis_colorbar=dict(
                title=dict(text="Gamma ($M)", font=dict(color="#C9D1D9")),
                tickfont=dict(color="#C9D1D9"),
                bgcolor="#0D1117"
            ),
            margin=dict(l=90, r=90, t=80, b=50),
            height=max(550, len(grid) * 26)
        )
        
        # Now white text will beautifully pop on the dark background cells!
        fig_heatmap.update_traces(
            textfont=dict(size=11, weight="bold", color="#FFFFFF"),
            hovertemplate="Strike: %{y}<br>Expiry: %{x}<br>Gamma Position: %{z:.2f}M<extra></extra>"
        )
        
        # Find closest strike to spot to outline it cleanly
        strike_array = np.array(grid.index)
        closest_strike = strike_array[(np.abs(strike_array - spot_price)).argsort()][0]  # Extraction fixed here
        
        # Add premium thick horizontal indicator anchors
        fig_heatmap.add_hline(y=closest_strike, line_color="#FFD700", line_width=3, 
                                annotation_text="★ SPOT LEVEL", annotation_position="left", annotation_font=dict(color="#FFD700", size=12, weight="bold"))
        
        if gamma_flip_strike and gamma_flip_strike in grid.index:
            fig_heatmap.add_hline(y=gamma_flip_strike, line_color="#FF007F", line_width=2.5, line_dash="dash", 
                                    annotation_text="GAMMA FLIP", annotation_position="right", annotation_font=dict(color="#FF007F", size=11, weight="bold"))
        if buy_zone_strike in grid.index:
            fig_heatmap.add_hline(y=buy_zone_strike, line_color="#52B788", line_width=2.5, 
                                    annotation_text="⬅️ BUY FLOOR", annotation_position="right", annotation_font=dict(color="#52B788", size=11, weight="bold"))
        if sell_zone_strike in grid.index:
            fig_heatmap.add_hline(y=sell_zone_strike, line_color="#4EA8DE", line_width=2.5, 
                                    annotation_text="⬅️ SELL CEILING", annotation_position="right", annotation_font=dict(color="#4EA8DE", size=11, weight="bold"))

        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # ─── INTERACTIVE VOLATILITY PATH FORECASTER ───
        st.markdown(f"### 🗺️ Future Volatility Path Estimator ({horizon_selection}-Day Trajectory)")
        st.markdown("This line matrix graphs the highest-probability paths for the asset based on option-implied volatility models.")
        
        try:
            t_symbol = "^VXN" if user_ticker == "QQQ" else "^VIX"
            vix_close = yf.Ticker(t_symbol).history(period="1d")["Close"].iloc[-1] / 100
        except Exception:
            vix_close = 0.16
        
        daily_move = spot_price * (vix_close / np.sqrt(252))
        path_df = project_volatility_path(spot_price, daily_move, buy_zone_strike, sell_zone_strike, gamma_flip_strike, horizon_days=horizon_selection)
        fig_path = go.Figure()
        fig_path.add_trace(go.Scatter(x=path_df["Horizon"], y=path_df["Bullish Path Target"], name="Maximum Upside Breakout Route", line=dict(color='#1E6091', dash='dot')))
        fig_path.add_trace(go.Scatter(x=path_df["Horizon"], y=path_df["Pinned Path Upper"], name="Standard Expected Ceiling (Pinned Track)", line=dict(color='#22C55E', width=3)))
        fig_path.add_trace(go.Scatter(x=path_df["Horizon"], y=path_df["Pinned Path Lower"], name="Standard Expected Floor (Pinned Track)", line=dict(color='#EF4444', width=3)))
        fig_path.add_trace(go.Scatter(x=path_df["Horizon"], y=path_df["Bearish Cascade Target"], name="Maximum Volatility Cascade Breakdown", line=dict(color='#D9381E', dash='dot')))
        fig_path.add_hline(y=spot_price, line_color="#FFD700", line_width=1.5, annotation_text="Current Entry Spot Price", annotation_position="left", annotation_font=dict(color="#FFD700"))
        fig_path.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            title=dict(
            text=f"{user_ticker} {horizon_selection}-DAY EXPECTED VOLATILITY TRAJECTORY HIGHWAY",
            font=dict(size=14, weight="bold")
            ),
            xaxis=dict(gridcolor="#1F2937"),
            yaxis=dict(
            title=dict(text="Projected Asset Price ($)", font=dict(color="#8B949E")),
            gridcolor="#1F2937"
            ),
            margin=dict(l=50, r=50, t=60, b=50),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_path, use_container_width=True)
