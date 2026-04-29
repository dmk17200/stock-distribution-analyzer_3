"""
Multi-Page Streamlit Stock Analysis App with Yahoo Finance Integration
Combines distribution analysis, automated stock scanning, and dashboard generation
NOW WITH HOURLY DATA ANALYSIS AND ENHANCED TECHNICAL ANALYSIS

To run:
    pip install streamlit yfinance pandas numpy matplotlib scipy plotly
    streamlit run app_hourly_fixed.py
"""

import streamlit as st
import streamlit as st
if not hasattr(st, 'cache_data'):
    st.cache_data = st.cache 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import ndimage, stats
from datetime import datetime, timedelta
import io
import os
import glob
import warnings
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Distribution Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for caching
if 'spy_data' not in st.session_state:
    st.session_state.spy_data = None
    st.session_state.spy_last_updated = None

# Initialize session state for hourly caching
if 'spy_data_hourly' not in st.session_state:
    st.session_state.spy_data_hourly = None
    st.session_state.spy_last_updated_hourly = None

# Stock list from your automated analyzer
# Portfolio file path - will be loaded to get stock list
PORTFOLIO_FILE = 'RH_Apr_2026.csv'

def load_portfolio_data():
    """Load portfolio data from CSV and return both data and stock symbols"""
    try:
        possible_paths = [PORTFOLIO_FILE, f'data/{PORTFOLIO_FILE}']
        portfolio_df = None
        for path in possible_paths:
            try:
                portfolio_df = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        if portfolio_df is None:
            return None, []
        portfolio_df['Shares'] = portfolio_df['Shares'].str.replace(',', '').astype(float)
        portfolio_df['Average Cost'] = portfolio_df['Average cost'].str.replace('$', '').str.replace(',', '').astype(float)
        portfolio_df['Equity'] = portfolio_df['Average Cost'] * portfolio_df['Shares']
        portfolio_df['Symbol'] = portfolio_df['Symbol'].str.strip()
        total_equity = (portfolio_df['Average Cost'] * portfolio_df['Shares']).sum()
        stock_symbols = portfolio_df['Symbol'].tolist()
        return portfolio_df[['Symbol', 'Shares', 'Equity', 'Average Cost']], stock_symbols
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        return None, []

portfolio_data, STOCK_SYMBOLS = load_portfolio_data()
if not STOCK_SYMBOLS:
    st.error("Could not load portfolio. Place Robinhood_December_1_-_Sheet1.csv in same directory.")
    STOCK_SYMBOLS = []


# Create data directories if they don't exist
DATA_DIR = 'data'
DATA_DIR_HOURLY = 'data_hourly'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(DATA_DIR_HOURLY):
    os.makedirs(DATA_DIR_HOURLY)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# NEW ENHANCED HELPER FUNCTIONS
def detect_parabolic_move(price_data, lookback=20):
    """
    Detect if price is in parabolic acceleration.
    Returns: (is_parabolic: bool, acceleration_rate: float)
    """
    if len(price_data) < lookback:
        return False, 0
    
    recent_prices = price_data[-lookback:]
    
    # Calculate rate of change acceleration
    first_half = recent_prices[:lookback//2]
    second_half = recent_prices[lookback//2:]
    
    first_half_change = (first_half[-1] - first_half[0]) / first_half[0] if first_half[0] != 0 else 0
    second_half_change = (second_half[-1] - second_half[0]) / second_half[0] if second_half[0] != 0 else 0
    
    # Parabolic if second half gains much faster than first half
    if second_half_change > first_half_change * 1.5 and second_half_change > 0.15:
        return True, second_half_change
    
    # Also check for parabolic decline
    if second_half_change < first_half_change * 1.5 and second_half_change < -0.15:
        return True, second_half_change
    
    return False, 0

def detect_liquidity_sweep(price_data, volume_data, lookback=60):
    """
    Detect potential liquidity sweeps:
    - Sharp spike to new high/low
    - On relatively thin volume (compared to recent average)
    - Classic manipulation pattern before reversal
    
    Returns: (has_sweep: bool, sweep_direction: str or None)
    """
    if len(price_data) < lookback or len(volume_data) < lookback:
        return False, None
    
    recent_prices = price_data[-lookback:]
    recent_volumes = volume_data[-lookback:]
    current_price = price_data[-1]
    
    # Check for spike to new high
    prior_high = max(recent_prices[:-5])  # Exclude last 5 days
    if current_price > prior_high * 1.05:  # 5% above prior high
        # Check if on lower volume (volume sweep characteristic)
        recent_avg_volume = np.mean(recent_volumes[-20:-5])
        spike_volume = np.mean(recent_volumes[-5:])
        
        if spike_volume < recent_avg_volume * 1.2:  # Not a high volume breakout
            return True, "UPSIDE_SWEEP"
    
    # Check for spike to new low
    prior_low = min(recent_prices[:-5])
    if current_price < prior_low * 0.95:  # 5% below prior low
        recent_avg_volume = np.mean(recent_volumes[-20:-5])
        spike_volume = np.mean(recent_volumes[-5:])
        
        if spike_volume < recent_avg_volume * 1.2:
            return True, "DOWNSIDE_SWEEP"
    
    return False, None

def calculate_tsmom(price_data, windows=[5, 10, 20, 60, 126, 252]):
    """
    Time Series Momentum across multiple look-back windows.
    Replaces check_trend_momentum with academically-validated signal.
    Returns dict with raw return and signal for each window, plus alignment.
    """
    results = {}
    current_price = price_data[-1]
    
    for window in windows:
        if len(price_data) >= window:
            past_price = price_data[-window]
            momentum = (current_price / past_price) - 1
            signal = "BUY" if momentum > 0 else "SELL"
        else:
            momentum = None
            signal = "N/A"
        
        results[window] = {
            "return": momentum,
            "signal": signal
        }
    
    signals = [v["signal"] for v in results.values() if v["signal"] != "N/A"]
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    total = len(signals)
    
    if total == 0:
        alignment = "INSUFFICIENT_DATA"
    elif buy_count == total:
        alignment = "FULL_BULL"
    elif sell_count == total:
        alignment = "FULL_BEAR"
    elif buy_count > sell_count:
        alignment = "LEANING_BULL"
    elif sell_count > buy_count:
        alignment = "LEANING_BEAR"
    else:
        alignment = "MIXED"
    
    results["alignment"] = alignment
    results["buy_count"] = buy_count
    results["sell_count"] = sell_count
    results["total_windows"] = total
    
    return results

def get_cached_spy_data():
    """Get SPY data with session-level caching + 1 hour refresh."""
    now = datetime.now()
    
    # Fetch if no data or > 1 hour old
    if (st.session_state.spy_data is None or 
        st.session_state.spy_last_updated is None or
        (now - st.session_state.spy_last_updated).total_seconds() > 3600):
        
        st.session_state.spy_data = fetch_stock_data("SPY", days=720)
        st.session_state.spy_last_updated = now
    
    return st.session_state.spy_data

def get_cached_spy_data_hourly():
    """Get SPY hourly data with session-level caching + 1 hour refresh."""
    now = datetime.now()
    
    # Fetch if no data or > 1 hour old
    if (st.session_state.spy_data_hourly is None or 
        st.session_state.spy_last_updated_hourly is None or
        (now - st.session_state.spy_last_updated_hourly).total_seconds() > 3600):
        
        st.session_state.spy_data_hourly = fetch_stock_data_hourly("SPY", days=30)
        st.session_state.spy_last_updated_hourly = now
    
    return st.session_state.spy_data_hourly

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, days=720):
    """Fetch stock data from Yahoo Finance for the past N days."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        # Prepare data in the format expected by analysis
        df = df.reset_index()
        df['stock'] = ticker
        df['date'] = df['Date']
        df['price'] = df['Close']
        df['volume'] = df['Volume']
        
        df['open'] = df['Open']
        return df[['stock', 'date', 'price', 'open', 'volume']]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def create_price_volume_chart(df, stock_symbol):
    """
    Price chart with:
    - Stacked buy/sell volume profile on right (horizontal bars, blue=up days, gold=down days)
    - Volume-by-date bar chart below (blue=up, gold=down)
    """
    if df is None or df.empty:
        return None

    stock_data = df[df['stock'] == stock_symbol].copy()
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values('date')

    # ── Moving averages ───────────────────────────────────────────────
    stock_data['MA_20']  = stock_data['price'].rolling(window=20,  min_periods=1).mean()
    stock_data['MA_50']  = stock_data['price'].rolling(window=50,  min_periods=1).mean()
    stock_data['MA_200'] = stock_data['price'].rolling(window=200, min_periods=1).mean()

    # ── Classify each bar as up-day (buy proxy) or down-day (sell proxy) ──
    # We need open price; yfinance supplies it — re-fetch with OHLC if available.
    # Fallback: compare close to prior close (shift).




    # ── Volume profile: group by rounded price ────────────────────────
    stock_data['price'] = stock_data['price'].replace([np.inf, -np.inf], np.nan)
    stock_data = stock_data.dropna(subset=['price'])

    if 'open' in stock_data.columns:
        is_up = stock_data['price'] >= stock_data['open']
    else:
        is_up = stock_data['price'] >= stock_data['price'].shift(1).fillna(stock_data['price'])

    stock_data['buy_vol']  = np.where(is_up,  stock_data['volume'], 0)
    stock_data['sell_vol'] = np.where(~is_up, stock_data['volume'], 0)

    stock_data['price_rounded'] = stock_data['price'].round().astype(int)

    vp = stock_data.groupby('price_rounded')[['buy_vol', 'sell_vol']].sum()
    vp_prices  = vp.index.values
    vp_buy     = vp['buy_vol'].values
    vp_sell    = vp['sell_vol'].values

    # ── Layout: 2 rows, 2 cols; price+profile share a row, vol-by-date below ─
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1],   # left (price) much wider than right (profile)
        height_ratios=[3, 1],  # top (price) taller than bottom (vol-by-date)
        hspace=0.08,
        wspace=0.03
    )

    ax_price   = fig.add_subplot(gs[0, 0])   # price line + MAs
    ax_profile = fig.add_subplot(gs[0, 1], sharey=ax_price)  # volume profile
    ax_vol     = fig.add_subplot(gs[1, 0], sharex=ax_price)  # daily vol bars
    ax_empty   = fig.add_subplot(gs[1, 1])   # blank corner
    ax_empty.axis('off')

    dates  = stock_data['date'].values
    prices = stock_data['price'].values

    # ── Panel 1: Price + MAs ──────────────────────────────────────────
    ax_price.plot(dates, prices,                          color='#2196F3', lw=2,   label='Price')
    ax_price.plot(dates, stock_data['MA_20'].values,      color='#4CAF50', lw=1.2, ls='--', label='20-Day MA')
    ax_price.plot(dates, stock_data['MA_50'].values,      color='#FF9800', lw=1.2, ls='--', label='50-Day MA')
    ax_price.plot(dates, stock_data['MA_200'].values,     color='#F44336', lw=1.2, ls='--', label='200-Day MA')

    ax_price.set_ylabel('Price ($)')
    ax_price.set_title(f'{stock_symbol} — Price, Volume Profile & Daily Volume', fontweight='bold')
    ax_price.legend(loc='upper left', fontsize=8)
    ax_price.grid(True, alpha=0.25)
    ax_price.tick_params(labelbottom=False)   # date labels shown on ax_vol instead

    # ── Panel 2: Stacked horizontal volume profile ────────────────────
    bar_height = max(1, (vp_prices.max() - vp_prices.min()) / len(vp_prices) * 0.85)

    ax_profile.barh(vp_prices, vp_buy  / 1e6, height=bar_height,
                    color='#2196F3', alpha=0.75, label='Up-day vol')
    ax_profile.barh(vp_prices, vp_sell / 1e6, height=bar_height,
                    left=vp_buy / 1e6,            # stacked to the right of buy bars
                    color='#FFB300', alpha=0.75, label='Down-day vol')

    ax_profile.set_xlabel('Volume (M)', fontsize=8)
    ax_profile.tick_params(labelleft=False, labelsize=7)
    ax_profile.tick_params(axis='x', labelsize=7)
    ax_profile.legend(fontsize=7, loc='upper right')
    ax_profile.grid(True, alpha=0.2, axis='x')
    ax_profile.set_title('Vol Profile', fontsize=9)

    # Highlight current price on profile
    current_price = prices[-1]
    ax_profile.axhline(current_price, color='black', lw=1.2, ls='-', alpha=0.8)
    ax_profile.annotate(f'${current_price:.0f}',
                        xy=(0, current_price),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=7, color='black')

    # ── Panel 3: Daily volume bars coloured by direction ─────────────
    buy_mask  = is_up.values
    sell_mask = ~buy_mask

    ax_vol.bar(dates[buy_mask],  stock_data['volume'].values[buy_mask]  / 1e6,
               color='#2196F3', alpha=0.75, width=1.0, label='Up-day vol')
    ax_vol.bar(dates[sell_mask], stock_data['volume'].values[sell_mask] / 1e6,
               color='#FFB300', alpha=0.75, width=1.0, label='Down-day vol')

    ax_vol.set_ylabel('Vol (M)', fontsize=8)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}M'))
    ax_vol.tick_params(axis='x', rotation=30, labelsize=8)
    ax_vol.grid(True, alpha=0.2, axis='y')
    ax_vol.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    return fig

def analyze_distributions_with_valleys(df, stock_symbol):
    """Generate distribution analysis chart for a stock."""
    
    try:
        # Filter and prepare data
        stock_data = df[df['stock'] == stock_symbol].copy()
        if stock_data.empty:
            return None, None
        
        # Convert date column
        stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')
        stock_data = stock_data.dropna(subset=['date'])
        
        if stock_data.empty:
            return None, None
        
        # Ensure price data is clean
        stock_data = stock_data.dropna(subset=['price', 'volume'])
        stock_data = stock_data[stock_data['price'] > 0]  # Remove zero or negative prices
        stock_data = stock_data[stock_data['volume'] > 0]  # Remove zero volume
        
        if stock_data.empty:
            return None, None
        
        # Get current price
        try:
            max_date_idx = stock_data['date'].idxmax()
            current_price = stock_data.loc[max_date_idx, 'price']
            current_vol = stock_data.loc[max_date_idx, 'volume'] / 1000000
            current_date = stock_data.loc[max_date_idx, 'date']
        except:
            current_price = stock_data.iloc[-1]['price']
            current_vol = stock_data.iloc[-1]['volume'] / 1000000
            current_date = stock_data.iloc[-1]['date']

        # Get last 7 days of prices
        stock_data_sorted = stock_data.sort_values('date')
        last_7_days = stock_data_sorted.tail(7)
        last_7_prices = last_7_days['price'].values
        last_7_dates = last_7_days['date'].values
        
        # Create rounded prices with proper handling of NaN/infinite values
        stock_data['price_rounded'] = stock_data['price'].round().astype(float)
        stock_data = stock_data.dropna(subset=['price_rounded'])
        
        if stock_data.empty:
            return None, None
        
        # Convert to integer safely
        stock_data['price_rounded'] = stock_data['price_rounded'].astype(int)
        
        # Create volume histogram by price
        volume_by_price = stock_data.groupby('price_rounded')['volume'].sum().sort_index()
        
        if volume_by_price.empty:
            return None, None
            
        prices = volume_by_price.index.values
        volumes = volume_by_price.values
        
        # Check for invalid data
        if len(prices) == 0 or len(volumes) == 0:
            return None, None
            
        # Smooth the data
        try:
            smoothed_volumes = ndimage.gaussian_filter1d(volumes, sigma=2.5)
        except:
            smoothed_volumes = volumes  # Use original if smoothing fails
        
        # Find peaks with error handling
        try:
            peaks, _ = find_peaks(smoothed_volumes,
                                 height=np.max(volumes) * 0.08,
                                 distance=12,
                                 prominence=np.max(volumes) * 0.05,
                                 width=3)
        except:
            peaks = np.array([])
        
        # Find valleys
        try:
            valleys, _ = find_peaks(-smoothed_volumes, distance=15)
        except:
            valleys = np.array([])
        
        # Create distribution boundaries
        distributions = []
        
        for i, peak_idx in enumerate(peaks):
            try:
                peak_price = prices[peak_idx]
                peak_volume = volumes[peak_idx]
                
                left_valleys = valleys[valleys < peak_idx]
                right_valleys = valleys[valleys > peak_idx]
                
                left_boundary = left_valleys[-1] if len(left_valleys) > 0 else 0
                right_boundary = right_valleys[0] if len(right_valleys) > 0 else len(prices) - 1
                
                dist_prices = prices[left_boundary:right_boundary+1]
                dist_volumes = volumes[left_boundary:right_boundary+1]
                
                if len(dist_volumes) == 0:
                    continue
                    
                total_volume = np.sum(dist_volumes)
                
                # Handle case where all volumes are zero
                if total_volume == 0:
                    continue
                    
                weighted_mean = np.average(dist_prices, weights=dist_volumes)
                weighted_variance = np.average((dist_prices - weighted_mean)**2, weights=dist_volumes)
                weighted_std = np.sqrt(weighted_variance)
                
                # Create weighted price points for distribution fitting
                weighted_price_points = []
                for price, volume in zip(dist_prices, dist_volumes):
                    count = max(1, int(volume / 1000000)) if volume > 0 else 1
                    weighted_price_points.extend([price] * count)
                
                if len(weighted_price_points) > 1:
                    try:
                        fitted_mean, fitted_std = stats.norm.fit(weighted_price_points)
                    except:
                        fitted_mean, fitted_std = weighted_mean, weighted_std
                else:
                    fitted_mean, fitted_std = weighted_mean, weighted_std
                
                distribution = {
                    'peak_idx': peak_idx,
                    'peak_price': peak_price,
                    'peak_volume': peak_volume,
                    'left_boundary': prices[left_boundary],
                    'right_boundary': prices[right_boundary],
                    'boundary_indices': (left_boundary, right_boundary),
                    'weighted_mean': weighted_mean,
                    'weighted_std': weighted_std,
                    'fitted_mean': fitted_mean,
                    'fitted_std': fitted_std,
                    'total_volume': total_volume,
                    'price_range': prices[right_boundary] - prices[left_boundary],
                    'distribution_prices': dist_prices,
                    'distribution_volumes': dist_volumes
                }
                distributions.append(distribution)
            except Exception as e:
                st.warning(f"Error processing distribution {i} for {stock_symbol}: {str(e)}")
                continue
        
        # If no distributions found, return None
        if not distributions:
            return None, None
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Peak and Valley Detection
        ax1 = axes[0, 0]
        ax1.bar(prices, volumes, alpha=0.6, color='lightblue', label='Original Volume')
        ax1.plot(prices, smoothed_volumes, 'b-', linewidth=2, label='Smoothed Volume')
        if len(peaks) > 0:
            ax1.plot(prices[peaks], smoothed_volumes[peaks], 'ro', markersize=10, label=f'Peaks ({len(peaks)})')
        if len(valleys) > 0:
            ax1.plot(prices[valleys], smoothed_volumes[valleys], 'go', markersize=8, label=f'Valleys ({len(valleys)})')
        ax1.axvline(current_price, color='black', linestyle='-', linewidth=3, alpha=0.8, label=f'Current Price: ${current_price:.2f}')
        
        for i, peak_idx in enumerate(peaks):
            ax1.annotate(f'P{i+1}: ${prices[peak_idx]}', 
                        xy=(prices[peak_idx], smoothed_volumes[peak_idx]),
                        xytext=(5, 10), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Volume')
        ax1.set_title(f'{stock_symbol} - Peak and Valley Detection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # Plot 2: Distribution Boundaries
        ax2 = axes[0, 1]
        ax2.bar(prices, volumes, alpha=0.6, color='lightblue')
        
        colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
        for i, dist in enumerate(distributions):
            color = colors[i % len(colors)]
            
            ax2.bar(dist['distribution_prices'], dist['distribution_volumes'], 
                   alpha=0.8, color=color, 
                   label=f"Dist {i+1}: ${dist['left_boundary']:.2f}-${dist['right_boundary']:.2f}")
            
            ax2.axvline(dist['left_boundary'], color=color, linestyle='--', alpha=0.7)
            ax2.axvline(dist['right_boundary'], color=color, linestyle='--', alpha=0.7)
            
            mean = dist['fitted_mean']
            std = dist['fitted_std']
            ax2.axvline(mean, color=color, linestyle='-', linewidth=2, alpha=0.9)
            ax2.axvspan(mean - std, mean + std, alpha=0.2, color=color)
            ax2.axvspan(mean - 2*std, mean + 2*std, alpha=0.1, color=color)
        
        ax2.axvline(current_price, color='black', linestyle='-', linewidth=3, alpha=0.8, label=f'Current: ${current_price:.2f}')
        ax2.set_xlabel('Price ($)')
        ax2.set_ylabel('Volume')
        ax2.set_title('Distribution Boundaries and Standard Deviations')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # Plot 3: Normal Distribution Overlays
        ax3 = axes[1, 0]
        ax3.bar(prices, volumes, alpha=0.4, color='lightgray', label='Actual Volume')
        
        for i, dist in enumerate(distributions):
            color = colors[i % len(colors)]
            mean = dist['fitted_mean']
            std = dist['fitted_std']
            
            x_range = np.linspace(mean - 3*std, mean + 3*std, 100)
            scale_factor = dist['peak_volume'] / stats.norm.pdf(mean, mean, std) if std > 0 else 1
            y_normal = stats.norm.pdf(x_range, mean, std) * scale_factor
            
            ax3.plot(x_range, y_normal, color=color, linewidth=2, 
                    label=f'Dist {i+1}: μ=${mean:.1f}, σ=${std:.1f}')
            ax3.axvline(mean, color=color, linestyle='--', alpha=0.7)
        
        ax3.axvline(current_price, color='black', linestyle='-', linewidth=3, alpha=0.8, label=f'Current: ${current_price:.2f}')
        ax3.set_xlabel('Price ($)')
        ax3.set_ylabel('Volume')
        ax3.set_title('Fitted Normal Distributions')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Determine current distribution
        current_distribution = None
        for i, dist in enumerate(distributions):
            if (current_price >= dist['left_boundary'] and 
                current_price <= dist['right_boundary']):
                current_distribution = i + 1
                break
        
        summary_text = f"""{stock_symbol} Distribution Analysis

Current: ${current_price:.2f} | Vol: {current_vol:.1f}M
Date: {current_date.strftime('%Y-%m-%d')}
Distribution: {'#' + str(current_distribution) if current_distribution else 'Outside ranges'}

Distributions: {len(distributions)}

"""
        
        for i, dist in enumerate(distributions):
            if current_distribution == i + 1:
                std_devs_away = abs(current_price - dist['fitted_mean']) / dist['fitted_std'] if dist['fitted_std'] > 0 else 0
                current_info = f"  >>> {std_devs_away:.1f}σ from mean <<<"
            else:
                current_info = ""
            
            summary_text += f"""Dist {i+1}:
  Peak: ${dist['peak_price']:.2f} ({dist['peak_volume']/1e6:.1f}M)
  Range: ${dist['left_boundary']:.2f}-${dist['right_boundary']:.2f}
  Mean: ${dist['fitted_mean']:.1f}
  Std: ${dist['fitted_std']:.1f}
  68%: ${dist['fitted_mean']-dist['fitted_std']:.1f}-${dist['fitted_mean']+dist['fitted_std']:.1f}
  95%: ${dist['fitted_mean']-2*dist['fitted_std']:.1f}-${dist['fitted_mean']+2*dist['fitted_std']:.1f}{current_info}

"""
        
        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Calculate metrics for return
        metrics = {
            'stock': stock_symbol,
            'current_price': current_price,
            'current_vol': current_vol,
            'current_date': current_date,
            'num_distributions': len(distributions),
            'distributions': distributions,
            'current_distribution': current_distribution
        }
        
        # Use current distribution if price is within one, otherwise use closest distribution
        if distributions:
            # Find which distribution to use for metrics
            if current_distribution is not None:
                target_dist = distributions[current_distribution - 1]
                metrics['using_distribution'] = 'current'
            else:
                closest_dist = min(distributions, key=lambda d: abs(current_price - d['fitted_mean']))
                target_dist = closest_dist
                metrics['using_distribution'] = 'closest'
            
            # Calculate metrics based on the target distribution
            metrics['peak_price'] = target_dist['peak_price']
            metrics['peak_volume_M'] = target_dist['peak_volume'] / 1e6
            metrics['fitted_std'] = target_dist['fitted_std']
            metrics['fitted_mean'] = target_dist['fitted_mean']
            metrics['std_devs_from_mean'] = (current_price - target_dist['fitted_mean']) / target_dist['fitted_std'] if target_dist['fitted_std'] > 0 else 0
            metrics['range_95_lower'] = target_dist['fitted_mean'] - 2 * target_dist['fitted_std']
            metrics['range_95_upper'] = target_dist['fitted_mean'] + 2 * target_dist['fitted_std']
            metrics['price_high_52w'] = float(stock_data['price'].max())
            metrics['price_low_52w'] = float(stock_data['price'].min())
            metrics['pct_from_high'] = round(((current_price - stock_data['price'].max()) / stock_data['price'].max()) * 100, 2)
            
            # Second peak if exists
            if len(distributions) > 1:
                sorted_dists = sorted(distributions, key=lambda d: d['peak_volume'], reverse=True)
                metrics['second_min_peak'] = sorted_dists[1]['peak_price']
            else:
                metrics['second_min_peak'] = None
        
        return fig, metrics
        
    except Exception as e:
        st.warning(f"Error in analyze_distributions_with_valleys for {stock_symbol}: {str(e)}")
        return None, None

def batch_analyze_stocks(stock_list, is_hourly=False, progress_bar=None):
    """Analyze multiple stocks and return metrics."""
    results = []
    
    for i, symbol in enumerate(stock_list):
        if progress_bar:
            progress_bar.progress((i + 1) / len(stock_list))
        
        try:
            if is_hourly:
                df = fetch_stock_data_hourly(symbol, days=30)
            else:
                df = fetch_stock_data(symbol, days=720)
            if df is not None:
                _, metrics = analyze_distributions_with_valleys(df, symbol)
                if metrics:
                    # Calculate moving averages
                    stock_data = df[df['stock'] == symbol].copy()
                    if not stock_data.empty:
                        stock_data = stock_data.sort_values('date')
                        stock_data['MA_20'] = stock_data['price'].rolling(window=20, min_periods=1).mean()
                        stock_data['MA_50'] = stock_data['price'].rolling(window=50, min_periods=1).mean()
                        stock_data['MA_200'] = stock_data['price'].rolling(window=200, min_periods=1).mean()
                        avg_volume_20d = stock_data['volume'].tail(20).mean()
                        metrics['avg_volume_20d'] = avg_volume_20d
                        metrics['rel_vol'] = (stock_data['volume'].iloc[-1] / avg_volume_20d) if avg_volume_20d > 0 else None

                        
                        # Get the latest moving average values
                        metrics['MA_20'] = stock_data['MA_20'].iloc[-1] if not stock_data['MA_20'].isna().iloc[-1] else None
                        metrics['MA_50'] = stock_data['MA_50'].iloc[-1] if not stock_data['MA_50'].isna().iloc[-1] else None
                        metrics['MA_200'] = stock_data['MA_200'].iloc[-1] if not stock_data['MA_200'].isna().iloc[-1] else None

                        # Day-over-day price change
                        if len(stock_data) >= 2:
                            prev_close = float(stock_data['price'].iloc[-2])
                            curr_close = float(stock_data['price'].iloc[-1])
                            metrics['price_change_1d'] = round(curr_close - prev_close, 2)
                            metrics['price_change_1d_pct'] = round(((curr_close - prev_close) / prev_close) * 100, 2)
                        else:
                            metrics['price_change_1d'] = None
                            metrics['price_change_1d_pct'] = None
                    
                    results.append(metrics)
        except Exception as e:
            st.warning(f"Error analyzing {symbol}: {str(e)}")
            continue
    
    return results


def analyze_stock_technical(ticker, df):
    """
    Comprehensive technical analysis.
    Scoring: MA position (2pts) + TSMOM 20/60/126/252d (1pt each) = max ±6
    Z-score and volume profile are display-only warnings, not scored.
    """
    try:
        if df is None or df.empty:
            return None

        stock_data = df[df['stock'] == ticker].copy()
        if stock_data.empty:
            return None

        stock_data = stock_data.sort_values('date')
        stock_data = stock_data.replace([np.inf, -np.inf], np.nan)
        stock_data = stock_data.dropna(subset=['price', 'volume'])

        if len(stock_data) < 50:
            return None

        current_price = float(stock_data['price'].iloc[-1])
        price_data    = stock_data['price'].values.astype(float)
        volume_data   = stock_data['volume'].values.astype(float)

        # ── Initialize flags early so nothing below can raise UnboundLocalError ──
        warning_flags = []

        # ==================================================================
        # STEP 1: Moving Averages
        # ==================================================================
        ma_50  = stock_data['price'].rolling(window=50,  min_periods=25).mean().iloc[-1]
        ma_200 = stock_data['price'].rolling(window=200, min_periods=100).mean().iloc[-1]

        if pd.isna(ma_50):  ma_50  = current_price
        if pd.isna(ma_200): ma_200 = current_price

        percent_above_ma50  = ((current_price - ma_50)  / ma_50)  * 100 if ma_50  != 0 else 0
        percent_above_ma200 = ((current_price - ma_200) / ma_200) * 100 if ma_200 != 0 else 0

        # ==================================================================
        # STEP 2: MA Position
        # ==================================================================
        if   current_price > ma_50 and current_price > ma_200: ma_position = "ABOVE_BOTH"
        elif current_price < ma_50 and current_price < ma_200: ma_position = "BELOW_BOTH"
        elif current_price > ma_200 and current_price < ma_50: ma_position = "BETWEEN"
        else:                                                   ma_position = "MIXED"

        # ==================================================================
        # STEP 3: Volume Profile + Distribution-Anchored Volume Metrics
        # ==================================================================
        stock_data['price_rounded'] = stock_data['price'].round().astype(float)
        stock_data = stock_data.dropna(subset=['price_rounded'])
        stock_data['price_rounded'] = stock_data['price_rounded'].astype(int)
        volume_profile = stock_data.groupby('price_rounded')['volume'].sum().sort_index()

        if not volume_profile.empty:
            high_volume_nodes = volume_profile[volume_profile >= volume_profile.quantile(0.80)]
        else:
            high_volume_nodes = pd.Series(dtype=float)

        try:
            # ------------------------------------------------------------------
            # 1. Find the nearest local distribution by expanding search band
            # ------------------------------------------------------------------
            nearest_node_price = None
            local_distribution = None
            search_band_used   = None

            for band_pct in [0.02, 0.05, 0.10, 0.20, 0.35]:
                lower = current_price * (1 - band_pct)
                upper = current_price * (1 + band_pct)
                local = volume_profile[
                    (volume_profile.index >= lower) &
                    (volume_profile.index <= upper)
                ]
                if not local.empty and local.sum() > 0:
                    nearest_node_price = float(local.idxmax())
                    local_distribution = local
                    search_band_used   = band_pct
                    break

            if nearest_node_price is None:
                nearest_node_price = float(min(high_volume_nodes.index,
                                               key=lambda x: abs(float(x) - current_price)))
                local_distribution = volume_profile
                search_band_used   = None

            # ------------------------------------------------------------------
            # 2. Characterize the local distribution (node-level)
            # ------------------------------------------------------------------
            local_vals       = local_distribution.values.astype(float)
            local_median_vol = float(np.median(local_vals))
            local_mean_vol   = float(np.mean(local_vals))

            # ------------------------------------------------------------------
            # 3. Pin today's volume before any further transforms
            # ------------------------------------------------------------------
            today_volume        = float(stock_data['volume'].iloc[-1])
            vol_node_ratio      = today_volume / local_median_vol if local_median_vol > 0 else 0.0
            vol_node_percentile = float(stats.percentileofscore(
                volume_profile.values.astype(float), today_volume
            ))
            pct_from_nearest_node = ((current_price - nearest_node_price)
                                      / nearest_node_price) * 100

            # ------------------------------------------------------------------
            # 4. Distribution-anchored metrics
            #    Find the distribution (from analyze_distributions_with_valleys)
            #    whose fitted_mean is closest to current price, then compute:
            #    - std devs from that distribution's mean
            #    - avg daily volume on days price was inside that distribution
            #    - number of days price was inside that distribution
            # ------------------------------------------------------------------

            # Re-derive distributions inline (same logic as analyze_distributions_with_valleys)
            volume_by_price = stock_data.groupby('price_rounded')['volume'].sum().sort_index()
            vp_prices  = volume_by_price.index.values
            vp_volumes = volume_by_price.values

            try:
                smoothed = ndimage.gaussian_filter1d(vp_volumes, sigma=2.5)
                peaks, _ = find_peaks(smoothed,
                                      height=np.max(vp_volumes) * 0.08,
                                      distance=12,
                                      prominence=np.max(vp_volumes) * 0.05,
                                      width=3)
                valleys, _ = find_peaks(-smoothed, distance=15)
            except Exception:
                peaks   = np.array([])
                valleys = np.array([])

            inline_distributions = []
            for peak_idx in peaks:
                try:
                    left_valleys  = valleys[valleys < peak_idx]
                    right_valleys = valleys[valleys > peak_idx]
                    left_boundary_idx  = left_valleys[-1]  if len(left_valleys)  > 0 else 0
                    right_boundary_idx = right_valleys[0]  if len(right_valleys) > 0 else len(vp_prices) - 1

                    dist_prices  = vp_prices[left_boundary_idx:right_boundary_idx + 1]
                    dist_volumes = vp_volumes[left_boundary_idx:right_boundary_idx + 1]
                    if dist_volumes.sum() == 0:
                        continue

                    weighted_mean = np.average(dist_prices, weights=dist_volumes)
                    weighted_var  = np.average((dist_prices - weighted_mean) ** 2, weights=dist_volumes)
                    weighted_std  = np.sqrt(weighted_var)

                    # Fit normal to get fitted_mean / fitted_std
                    pts = []
                    for p, v in zip(dist_prices, dist_volumes):
                        pts.extend([p] * max(1, int(v / 1_000_000)))
                    if len(pts) > 1:
                        fitted_mean, fitted_std = stats.norm.fit(pts)
                    else:
                        fitted_mean, fitted_std = weighted_mean, weighted_std

                    inline_distributions.append({
                        'fitted_mean':    fitted_mean,
                        'fitted_std':     fitted_std,
                        'left_boundary':  float(vp_prices[left_boundary_idx]),
                        'right_boundary': float(vp_prices[right_boundary_idx]),
                    })
                except Exception:
                    continue

            # Find closest distribution to current price
            if inline_distributions:
                closest_dist = min(inline_distributions,
                                   key=lambda d: abs(current_price - d['fitted_mean']))

                # Std devs from closest distribution mean
                dist_std_devs = (
                    (current_price - closest_dist['fitted_mean']) / closest_dist['fitted_std']
                    if closest_dist['fitted_std'] > 0 else 0.0
                )

                # Days price was inside closest distribution's price range
                in_dist_mask = (
                    (stock_data['price'] >= closest_dist['left_boundary']) &
                    (stock_data['price'] <= closest_dist['right_boundary'])
                )
                days_in_dist        = int(in_dist_mask.sum())
                avg_vol_in_dist     = float(stock_data.loc[in_dist_mask, 'volume'].mean()) if days_in_dist > 0 else 0.0
                vol_vs_dist_avg     = today_volume / avg_vol_in_dist if avg_vol_in_dist > 0 else 0.0
                closest_dist_mean   = round(closest_dist['fitted_mean'], 2)
                closest_dist_std    = round(closest_dist['fitted_std'],  2)

            else:
                dist_std_devs     = 0.0
                days_in_dist      = 0
                avg_vol_in_dist   = 0.0
                vol_vs_dist_avg   = 0.0
                closest_dist_mean = round(weighted_mean, 2) if 'weighted_mean' in dir() else 0.0
                closest_dist_std  = 0.0

        except Exception:
            nearest_node_price  = None
            local_median_vol    = 0.0
            local_mean_vol      = 0.0
            today_volume        = 0.0
            vol_node_ratio      = 0.0
            vol_node_percentile = 0.0
            pct_from_nearest_node = 0.0
            search_band_used    = None
            dist_std_devs       = 0.0
            days_in_dist        = 0
            avg_vol_in_dist     = 0.0
            vol_vs_dist_avg     = 0.0
            closest_dist_mean   = 0.0
            closest_dist_std    = 0.0


        # ==================================================================
        # STEP 4: Z-Score (display-only)
        # ==================================================================
        total_volume = volume_data.sum()
        try:
            weighted_mean     = np.average(price_data, weights=volume_data) if total_volume > 0 else np.mean(price_data)
            weighted_variance = np.average((price_data - weighted_mean)**2, weights=volume_data) if total_volume > 0 else np.var(price_data)
            std_dev           = np.sqrt(weighted_variance) if weighted_variance > 0 else 1.0
        except Exception:
            weighted_mean = np.mean(price_data)
            std_dev       = np.std(price_data) if np.std(price_data) > 0 else 1.0

        z_score = (current_price - weighted_mean) / std_dev if std_dev > 0 else 0

        # ==================================================================
        # STEP 5: TSMOM (single call — scored on 20/60/126/252d only)
        # ==================================================================
        try:
            tsmom          = calculate_tsmom(price_data)
            momentum_state = tsmom["alignment"]

            scoring_windows   = [20, 60, 126, 252]
            buy_count_scored  = sum(1 for w in scoring_windows if tsmom.get(w, {}).get("signal") == "BUY")
            sell_count_scored = sum(1 for w in scoring_windows if tsmom.get(w, {}).get("signal") == "SELL")
            tsmom["scored_alignment"] = (
                "BULL" if buy_count_scored > sell_count_scored else
                "BEAR" if sell_count_scored > buy_count_scored else
                "MIXED"
            )

            # SHORT_TSMOM_REVERSAL: 5d+10d both flipping SELL while 60d+ still mostly bullish
            short_bearish    = (tsmom.get(5,  {}).get("signal") == "SELL" and
                                tsmom.get(10, {}).get("signal") == "SELL")
            long_still_bull  = buy_count_scored >= 3
            if short_bearish and long_still_bull:
                warning_flags.append("SHORT_TSMOM_REVERSAL")

        except Exception:
            tsmom          = {"alignment": "UNKNOWN", "buy_count": 0, "sell_count": 0, "total_windows": 0}
            momentum_state = "UNKNOWN"
            buy_count_scored  = 0
            sell_count_scored = 0

        # ==================================================================
        # STEP 6: Support / Resistance from high-volume nodes
        # ==================================================================
        support_levels    = sorted(
            [float(p) for p in high_volume_nodes.index if float(p) < current_price * 0.98],
            reverse=True
        )
        resistance_levels = sorted(
            [float(p) for p in high_volume_nodes.index if float(p) > current_price * 1.02]
        )
        nearest_support    = support_levels[0]    if support_levels    else None
        nearest_resistance = resistance_levels[0] if resistance_levels else None

        # ==================================================================
        # STEP 7: Trend Strength
        # ==================================================================
        recent_prices = price_data[-20:] if len(price_data) >= 20 else price_data
        if len(recent_prices) > 0:
            if   current_price > max(recent_prices) * 0.98:            trend_strength = "STRONG_UPTREND"
            elif current_price < min(recent_prices) * 1.02:            trend_strength = "STRONG_DOWNTREND"
            elif ma_50 > ma_200 and current_price > ma_50:             trend_strength = "MODERATE_UPTREND"
            elif ma_50 < ma_200 and current_price < ma_50:             trend_strength = "MODERATE_DOWNTREND"
            else:                                                        trend_strength = "SIDEWAYS"
        else:
            trend_strength = "UNKNOWN"

        # ==================================================================
        # STEP 8: Scoring
        # ==================================================================
        bullish_score = 0
        bearish_score = 0

        # MA position — 2pts
        if   ma_position == "ABOVE_BOTH": bullish_score += 2
        elif ma_position == "BELOW_BOTH": bearish_score += 2

        # TSMOM 20/60/126/252d — 1pt each
        for w in [20, 60, 126, 252]:
            sig = tsmom.get(w, {}).get("signal")
            if   sig == "BUY":  bullish_score += 1
            elif sig == "SELL": bearish_score += 1

        # Warnings (display-only, no score impact)
        if abs(pct_from_nearest_node) > 15:
            warning_flags.append(f"VP_EXTENDED_{pct_from_nearest_node:+.1f}%_FROM_NODE")
        if z_score > 2:
            warning_flags.append(f"Z_SCORE_HIGH_{z_score:.1f}σ")
        elif z_score < -2:
            warning_flags.append(f"Z_SCORE_LOW_{z_score:.1f}σ")

        # ==================================================================
        # STEP 9: Signal
        # ==================================================================
        net_score = bullish_score - bearish_score

        if   net_score >= 5:  signal = "STRONG_BUY"
        elif net_score >= 3:  signal = "BUY"
        elif net_score >= 1:  signal = "HOLD_WATCH_FOR_BUY"
        elif net_score <= -5: signal = "STRONG_SELL"
        elif net_score <= -3: signal = "SELL"
        elif net_score <= -1: signal = "HOLD_WATCH_FOR_SELL"
        else:                 signal = "HOLD_NEUTRAL"

        # Override: extended-from-value is a warning, not a reversal, when trend is intact
        if signal in ["STRONG_SELL", "SELL"] and ma_position == "ABOVE_BOTH" and momentum_state in ["FULL_BULL", "LEANING_BULL"]:
            signal = "WAIT_FOR_PULLBACK"
            warning_flags.append("TREND_INTACT_DESPITE_SCORE")

        # ==================================================================
        # STEP 10: Action Recommendation
        # ==================================================================
        action_map = {
            "STRONG_BUY":          "ADD / ENTER POSITION",
            "BUY":                 "CONSIDER ENTRY",
            "HOLD_WATCH_FOR_BUY":  "WATCH — no entry yet",
            "HOLD_NEUTRAL":        "HOLD — no action",
            "HOLD_WATCH_FOR_SELL": "WATCH — monitor closely",
            "SELL":                "CONSIDER REDUCING",
            "STRONG_SELL":         "REDUCE / EXIT",
            "WAIT_FOR_PULLBACK":   "TREND INTACT — wait for better entry",
        }
        recommendations = {
            "action": action_map.get(signal, "HOLD — no action"),
            "note":   "Use ATR-based stops. Set price alerts manually at key levels."
        }

        # ==================================================================
        # STEP 11: Return
        # ==================================================================
        return {
            "ticker":        ticker,
            "current_price": round(current_price, 2),
            "signal":        signal,

            "scores": {
                "bullish": bullish_score,
                "bearish": bearish_score,
                "net":     net_score,
            },

            "technicals": {
                "ma_50":              round(ma_50,  2),
                "ma_200":             round(ma_200, 2),
                "ma_position":        ma_position,
                "trend_strength":     trend_strength,
                "percent_above_ma50":  round(percent_above_ma50,  2),
                "percent_above_ma200": round(percent_above_ma200, 2),
            },

            "tsmom": tsmom,

            "volume_analysis": {
                "nearest_node":           nearest_node_price,
                "pct_from_nearest_node":  round(pct_from_nearest_node, 2),
                "vol_node_ratio":         round(vol_node_ratio, 2),
                "vol_node_percentile":    round(vol_node_percentile, 1),
                "local_median_vol":       round(local_median_vol, 0),
                "local_mean_vol":         round(local_mean_vol, 0),
                "search_band_used":       search_band_used,
                "dist_std_devs":          round(dist_std_devs, 2),
                "days_in_dist":           days_in_dist,
                "avg_vol_in_dist":        round(avg_vol_in_dist, 0),
                "vol_vs_dist_avg":        round(vol_vs_dist_avg, 2),
                "closest_dist_mean":      closest_dist_mean,
                "closest_dist_std":       closest_dist_std,
            },

            "distribution": {
                "z_score":    round(z_score, 2),
                "mean_price": round(weighted_mean, 2),
                "std_dev":    round(std_dev, 2),
            },

            "key_levels": {
                "nearest_support":    round(nearest_support,    2) if nearest_support    else None,
                "nearest_resistance": round(nearest_resistance, 2) if nearest_resistance else None,
            },

            "warnings":        warning_flags,
            "recommendations": recommendations,
        }

    except Exception as e:
        st.warning(f"Error in technical analysis for {ticker}: {str(e)}")
        return None
# ============================================================================
# HELPER FUNCTION: Sort DataFrame by any column
# ============================================================================
def sort_dataframe_by_columns(df, sort_columns, ascending=True):
    """
    Sort a DataFrame by any column(s) with proper handling of numeric and string columns.
    
    Parameters:
    - df: pandas DataFrame to sort
    - sort_columns: list of column names to sort by
    - ascending: boolean or list of booleans for sort order
    
    Returns:
    - Sorted DataFrame
    """
    if df.empty or not sort_columns:
        return df
    
    # Make a copy to avoid modifying the original
    sorted_df = df.copy()
    
    # Convert any string representations of numbers to numeric where possible
    for col in sort_columns:
        if col in sorted_df.columns:
            # Try to convert to numeric for sorting
            try:
                sorted_df[col + '_sort'] = pd.to_numeric(sorted_df[col], errors='coerce')
                # Use the numeric column for sorting if conversion was successful
                if not sorted_df[col + '_sort'].isna().all():
                    sorted_df = sorted_df.sort_values(by=[col + '_sort'], ascending=ascending)
                    sorted_df = sorted_df.drop(columns=[col + '_sort'])
                    return sorted_df
            except:
                pass
    
    # Fall back to regular sorting
    return sorted_df.sort_values(by=sort_columns, ascending=ascending)

# ============================================================================
# HELPER FUNCTION: Create sortable dataframe display
# ============================================================================
def display_sortable_dataframe(df, column_config=None, height=400, key_prefix="df"):
    """
    Display a dataframe with column sorting capability.
    
    Parameters:
    - df: DataFrame to display
    - column_config: Optional column configuration for data_editor
    - height: Height of the dataframe display
    - key_prefix: Unique prefix for session state keys
    
    Returns:
    - None (displays the dataframe in Streamlit)
    """
    if df.empty:
        st.warning("No data to display")
        return
    
    # Create sort controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        all_columns = df.columns.tolist()
        default_sort = 'std_devs_from_mean' if 'std_devs_from_mean' in all_columns else all_columns[0]
        
        sort_column = st.selectbox(
            "Sort by:",
            options=all_columns,
            index=all_columns.index(default_sort) if default_sort in all_columns else 0,
            key=f"{key_prefix}_sort_col"
        )
    
    with col2:
        sort_ascending = st.checkbox("Ascending", value=True, key=f"{key_prefix}_sort_asc")

    # DEBUG: Show column names and types
    st.write(f"Sorting by column: '{sort_column}'")
    st.write(f"Column exists in dataframe: {sort_column in df.columns}")
    if sort_column in df.columns:
        st.write(f"Data type: {df[sort_column].dtype}")
        st.write(f"Sample values: {df[sort_column].head(3).tolist()}")
    
    # Sort the dataframe using the original numeric data
    # IMPORTANT: Sort the original dataframe, not a formatted version
    sorted_df = sort_dataframe_by_columns(df, [sort_column], sort_ascending)
    
    # NOW create a formatted copy for display AFTER sorting
    display_df = sorted_df.copy()
    
    # ===== APPLY FORMATTING AFTER SORTING =====
    # Format columns with comma for thousands
    comma_format_columns = [
        'Current Value',
        'Cost Value ($)',
        'Delta',
        '$ Impact (10% move)',
        'Volume',
        'Peak Volume'
        # Add any other columns that need comma formatting
    ]
    
    for col in comma_format_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )
    
    # Format columns to nearest round number (integers) without commas
    columns_to_round = [
        'Shares Owned',
        '52W High',
        '52W Low',
        '20-Day MA',
        '50-Day MA',
        '200-Day MA',
        'Mean',
        'Price @ Peak Vol',
        'Current Price',
        'Current Price (2)',
        'Current Price (3)',
        'Avg Cost ($)'
        # Add any other columns you want to round
    ]
    
    for col in columns_to_round:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{int(round(x))}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )
    
    # Format percentage columns with 1 decimal place
    percentage_columns = ['Portfolio %', 'pct From High', 'Confidence']
    for col in percentage_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else x
            )
    
    # Format standard deviation columns with 2 decimals
    std_columns = ['Std Devs', 'Rel Std Dev', 'Rel Vol Ratio', 'Z-Score']
    for col in std_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )
    # ===== END OF FORMATTING LOGIC =====
    
    # Display the formatted dataframe
    st.data_editor(
        display_df,
        disabled=True,
        width=2000,
        height=height,
        hide_index=True,
        use_container_width=True
    )
    
    return sorted_df

# ============================================================================
# HELPER FUNCTION: Create interactive Plotly chart with filtering
# ============================================================================
def create_interactive_scatter(df, x_col, y_col, color_col, text_col, hover_data=None, 
                               title="Interactive Scatter Plot", height=600):
    """
    Create an interactive Plotly scatter plot with filtering capability.
    
    Parameters:
    - df: DataFrame with data
    - x_col: Column name for x-axis
    - y_col: Column name for y-axis
    - color_col: Column name for color coding
    - text_col: Column name for point labels
    - hover_data: Additional columns for hover information
    - title: Plot title
    - height: Plot height
    
    Returns:
    - Plotly figure
    """
    if df.empty:
        return None
    
    # Define color mapping for signals
    color_map = {
        'STRONG_BUY': 'darkgreen',
        'BUY': 'green',
        'HOLD_WATCH_FOR_BUY': 'lightgreen',
        'HOLD_NEUTRAL': 'gray',
        'HOLD_WATCH_FOR_SELL': 'orange',
        'SELL': 'red',
        'STRONG_SELL': 'darkred',
        'WAIT_FOR_PULLBACK': 'yellow',
        'WATCH_SUPPORT_BREAK': 'purple',
        'N/A': 'black'
    }
    
    fig = go.Figure()
    
    # Add scatter points for each category
    if color_col in df.columns:
        unique_values = df[color_col].unique()
        for value in unique_values:
            subset = df[df[color_col] == value]
            if not subset.empty:
                color = color_map.get(value, 'gray')
                
                # Prepare hover text
                hover_template = '<b>%{text}</b><br>'
                if hover_data:
                    for col in hover_data:
                        if col in subset.columns:
                            hover_template += f'{col}: %{{customdata[{hover_data.index(col)}]}}<br>'
                hover_template += '<extra></extra>'
                
                fig.add_trace(go.Scatter(
                    x=subset[x_col] if x_col in subset.columns else [],
                    y=subset[y_col] if y_col in subset.columns else [],
                    mode='markers+text',
                    text=subset[text_col] if text_col in subset.columns else [],
                    textposition="top center",
                    marker=dict(
                        color=color,
                        size=12,
                        line=dict(width=2, color='darkgray')
                    ),
                    name=str(value),
                    customdata=subset[hover_data].values if hover_data else None,
                    hovertemplate=hover_template
                ))
    
    # Add reference lines
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Mean")
    if 'relative_vol_ratio' in df.columns:
        fig.add_hline(y=1, line_dash="dot", line_color="green", annotation_text="Normal Vol")
    
    # Add threshold lines if they exist in column names
    if x_col in ['relative_std_dev', 'std_devs_from_mean']:
        fig.add_vline(x=1, line_dash="dot", line_color="blue", opacity=0.5)
        fig.add_vline(x=-1, line_dash="dot", line_color="blue", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        height=height,
        hovermode='closest',
        showlegend=True
    )
    
    return fig

def render_tsmom_widget(tsmom, compact=False):
    """Render a TSMOM alignment indicator."""
    windows = [5, 10, 20, 60, 126, 252]
    labels = ["5d", "10d","20d", "60d", "126d", "252d"]
    alignment = tsmom.get("alignment", "UNKNOWN")
    
    alignment_colors = {
        "FULL_BULL":    "#1B5E20",
        "LEANING_BULL": "#4CAF50",
        "MIXED":        "#9E9E9E",
        "LEANING_BEAR": "#EF9A9A",
        "FULL_BEAR":    "#B71C1C",
        "INSUFFICIENT_DATA": "#9E9E9E",
        "UNKNOWN":      "#9E9E9E"
    }
    alignment_color = alignment_colors.get(alignment, "#9E9E9E")
    
    if compact:
        bars = ""
        for w, label in zip(windows, labels):
            entry = tsmom.get(w, {})
            sig = entry.get("signal", "N/A")
            ret = entry.get("return")
            if sig == "BUY":
                color = "#4CAF50"; arrow = "▲"
            elif sig == "SELL":
                color = "#EF5350"; arrow = "▼"
            else:
                color = "#9E9E9E"; arrow = "–"
            ret_str = f"{ret*100:+.1f}%" if ret is not None else "N/A"
            bars += f"""
            <span style='background:{color};color:white;padding:2px 6px;
                         border-radius:4px;margin:2px;font-size:11px;font-weight:bold;
                         display:inline-block;'>
                {arrow} {label}: {ret_str}
            </span>"""
        
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin:4px 0;'>
            {bars}
            <span style='background:{alignment_color};color:white;padding:3px 10px;
                         border-radius:12px;font-size:11px;font-weight:bold;margin-left:4px;'>
                {alignment.replace("_", " ")}
            </span>
        </div>""", unsafe_allow_html=True)
    
    else:
        # Full version - 4 window cards + alignment card
        cols = st.columns(7)
        for i, (w, label) in enumerate(zip(windows, labels)):
            with cols[i]:
                entry = tsmom.get(w, {})
                sig = entry.get("signal", "N/A")
                ret = entry.get("return")
                ret_str = f"{ret*100:+.1f}%" if ret is not None else "N/A"
                if sig == "BUY":
                    color = "#4CAF50"; arrow = "▲"; bg = "#E8F5E9"
                elif sig == "SELL":
                    color = "#EF5350"; arrow = "▼"; bg = "#FFEBEE"
                else:
                    color = "#9E9E9E"; arrow = "–"; bg = "#F5F5F5"
                
                st.markdown(f"""
                <div style='background:{bg};border-left:4px solid {color};
                            padding:12px;border-radius:6px;text-align:center;'>
                    <div style='font-size:26px;color:{color};font-weight:bold;'>{arrow}</div>
                    <div style='font-size:13px;font-weight:bold;color:#333;'>{label}</div>
                    <div style='font-size:18px;color:{color};font-weight:bold;'>{ret_str}</div>
                    <div style='font-size:11px;color:{color};'>{sig}</div>
                </div>""", unsafe_allow_html=True)
        
        with cols[6]:
            buy_count = tsmom.get("buy_count", 0)
            total = tsmom.get("total_windows", 0)
            st.markdown(f"""
            <div style='background:{alignment_color};padding:12px;border-radius:6px;
                        text-align:center;height:100%;'>
                <div style='font-size:11px;color:white;font-weight:bold;'>ALIGNMENT</div>
                <div style='font-size:24px;color:white;font-weight:bold;'>{buy_count}/{total}</div>
                <div style='font-size:11px;color:white;'>{alignment.replace("_"," ")}</div>
            </div>""", unsafe_allow_html=True)

# Main App Navigation
st.title("📈 Stock Distribution Analyzer Pro")
st.markdown("Complete stock analysis suite with distribution analysis, automated scanning, and dashboard generation")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Daily - Single Stock", 
    "🔍 Daily - Batch Scanner", 
    "📈 Daily - Dashboard",
    "⚙️ Daily Data Management",
    "⏰ Hourly - Single Stock",
    "🔍 Hourly - Batch Scanner", 
    "📈 Hourly - Dashboard",
    "⚙️ Hourly Data Management"
])

# DAILY TABS
with tab1:
    st.header("Single Stock Distribution Analysis")
    st.markdown("Analyze price distributions and identify support/resistance levels")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_ticker = st.text_input(
            "Enter Stock Ticker",
            value="AAPL",
            help="Enter any valid stock ticker (e.g., AAPL, TSLA, MSFT)"
        ).upper()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze Stock")
    
    if analyze_button and stock_ticker:
        with st.spinner(f"Fetching data for {stock_ticker}..."):
            stock_df = fetch_stock_data(stock_ticker, days=720)
            spy_df = get_cached_spy_data()
            
            if stock_df is None:
                st.error(f"Could not fetch data for {stock_ticker}. Please check the ticker symbol.")
            elif spy_df is None:
                st.error("Could not fetch SPY data for benchmark comparison.")
            else:
                # Analyze both stocks
                with st.spinner(f"Analyzing {stock_ticker}..."):
                    stock_fig, stock_metrics = analyze_distributions_with_valleys(stock_df, stock_ticker)
                
                with st.spinner("Analyzing SPY benchmark..."):
                    spy_fig, spy_metrics = analyze_distributions_with_valleys(spy_df, "SPY")
                
                if stock_fig is None or stock_metrics is None:
                    st.error(f"Error analyzing {stock_ticker}. This stock may not have enough data or may be illiquid.")
                elif spy_fig is None or spy_metrics is None:
                    st.error("Error analyzing SPY benchmark.")
                else:
                    # Display metrics with error handling
                    st.markdown("---")
                    st.subheader("📊 Key Metrics Comparison")
                    
                    col1, col2, col3, col4 = st.columns(4)

                    # Run technical analysis on the selected stock
                    st.markdown("---")
                    st.subheader("🎯 Trading Signal Analysis")
                    
                    with st.spinner(f"Running technical analysis on {stock_ticker}..."):
                        tech_analysis = analyze_stock_technical(stock_ticker, stock_df)
                    
                    if tech_analysis:
                        # Display signal with color coding
                        signal = tech_analysis['signal']
                        signal_color = {
                            'STRONG_BUY': '🟢',
                            'BUY': '🟢',
                            'HOLD_WATCH_FOR_BUY': '🟡',
                            'HOLD_NEUTRAL': '⚪',
                            'HOLD_WATCH_FOR_SELL': '🟠',
                            'SELL': '🔴',
                            'STRONG_SELL': '🔴',
                            'WAIT_FOR_PULLBACK': '🟡',
                            'WATCH_SUPPORT_BREAK': '🟣'
                        }.get(signal, '⚪')
                        
                        # Main signal display
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Trading Signal",
                                f"{signal_color} {signal}",
                                delta=f"Net Score: {tech_analysis['scores']['net']}"
                            )
                        
                        with col2:
                            st.metric(
                                "Net Score",
                                tech_analysis['scores']['net'],
                                delta=f"Bullish: {tech_analysis['scores']['bullish']} | Bearish: {tech_analysis['scores']['bearish']}"
                            )
                        
                        with col3:
                            st.metric(
                                "Price Position",
                                tech_analysis['technicals']['ma_position'],
                                delta=f"{tech_analysis['technicals']['percent_above_ma200']:.1f}% from 200-MA"
                            )
                        
                        
                        # Detailed breakdown
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                            
                        with col1:
                            st.subheader("📊 Technical Details")

                            stock_data_sorted = stock_df[stock_df['stock'] == stock_ticker].sort_values('date')
                            today_volume = float(stock_df['volume'].iloc[-1])
                            vol_node_ratio = tech_analysis['volume_analysis']['vol_node_ratio']


                            st.write(f"**Today's Price:** ${tech_analysis['current_price']:,.2f}")
                            st.write(f"**Today's Volume:** {today_volume:,.0f} ({vol_node_ratio:.2f}x avg)")

                            st.markdown("---")
                            st.write(f"**50-Day MA:** ${tech_analysis['technicals']['ma_50']:,.2f}")
                            st.write(f"**200-Day MA:** ${tech_analysis['technicals']['ma_200']:,.2f}")
                            st.write(f"**MA Position:** {tech_analysis['technicals']['ma_position']}")
                            st.write(f"**Trend Strength:** {tech_analysis['technicals']['trend_strength']}")

                            st.markdown("---")
                            st.write(f"**Z-Score:** {tech_analysis['distribution']['z_score']:.2f}σ")
                            st.write(f"**Dist Std Devs:** {tech_analysis['volume_analysis']['dist_std_devs']:+.2f}σ from closest dist mean (${tech_analysis['volume_analysis']['closest_dist_mean']:,.0f})")
                            st.write(f"**Vol vs Dist Avg:** {tech_analysis['volume_analysis']['vol_vs_dist_avg']:.2f}x (avg {tech_analysis['volume_analysis']['avg_vol_in_dist']:,.0f} over {tech_analysis['volume_analysis']['days_in_dist']} days)")

                            if tech_analysis['key_levels']['nearest_support']:
                                st.write(f"**Nearest Support:** ${tech_analysis['key_levels']['nearest_support']:,.2f}")
                            if tech_analysis['key_levels']['nearest_resistance']:
                                st.write(f"**Nearest Resistance:** ${tech_analysis['key_levels']['nearest_resistance']:,.2f}")
                        
                        with col2:
                            st.subheader("⚠️ Warning Flags & Recommendation")
                            
                            # Display warnings
                            if tech_analysis['warnings']:
                                for warning in tech_analysis['warnings']:
                                    st.warning(f"⚠️ {warning}")
                            else:
                                st.info("✅ No major warning flags")
                            
                            # Display recommendation
                            st.markdown("---")
                            st.subheader("💡 Action Recommendation")
                            
                            rec = tech_analysis['recommendations']
                            st.markdown("---")
                            st.subheader("💡 Action Recommendation")

                            rec = tech_analysis['recommendations']
                            if rec:
                                st.write(f"**Action:** {rec.get('action', 'N/A')}")
                                if 'rationale' in rec:
                                    st.write(f"**Rationale:** {rec['rationale']}")
                                if 'stop_loss' in rec and rec['stop_loss'] is not None:
                                    st.write(f"**Stop Loss:** ${rec['stop_loss']:,.2f}")
                                if 'take_profit' in rec and rec['take_profit'] is not None:
                                    st.write(f"**Take Profit:** ${rec['take_profit']:,.2f}")
                                if 'ideal_entry' in rec and rec['ideal_entry'] is not None:
                                    st.write(f"**Ideal Entry:** ${rec['ideal_entry']:,.2f}")
                                if 'sell_percentage' in rec and rec['sell_percentage'] is not None:
                                    st.write(f"**Sell:** {rec['sell_percentage']}% of position")
                                if 'alert_price' in rec and rec['alert_price'] is not None:
                                    st.write(f"**Alert Price:** ${rec['alert_price']:,.2f}")
                                if 'critical_level' in rec and rec['critical_level'] is not None:
                                    st.write(f"**Critical Level:** ${rec['critical_level']:,.2f}")
                        
                        # Pattern detection
                        st.markdown("---")
                        st.subheader("🔍 Pattern Detection")
                        
                        col1, col2, col3 = st.columns(3)

                        # ── TSMOM Widget ──────────────────────────────────
                        st.markdown("---")
                        st.subheader("📐 Time Series Momentum (TSMOM)")
                        st.caption("Return vs. own price N trading days ago. All four positive = strong trend confirmation.")
                        if "tsmom" in tech_analysis:
                            render_tsmom_widget(tech_analysis["tsmom"], compact=False)
                        else:
                            st.info("TSMOM data not available")

                    
                    with col1:
                        # Safe access to metrics with defaults
                        current_price = stock_metrics.get('current_price', 0)
                        std_devs = stock_metrics.get('std_devs_from_mean', 0)
                        st.metric(
                            f"{stock_ticker} Current Price",
                            f"${current_price:.2f}" if current_price else "N/A",
                            delta=f"{std_devs:.2f}σ from mean" if std_devs is not None else "N/A"
                        )
                    
                    with col2:
                        peak_price = stock_metrics.get('peak_price', 0)
                        peak_vol = stock_metrics.get('peak_volume_M', 0)
                        st.metric(
                            f"{stock_ticker} Peak Price",
                            f"${peak_price:.0f}" if peak_price else "N/A",
                            delta=f"{peak_vol:.1f}M vol" if peak_vol else "N/A"
                        )
                    
                    with col3:
                        spy_current_price = spy_metrics.get('current_price', 0)
                        spy_std_devs = spy_metrics.get('std_devs_from_mean', 0)
                        st.metric(
                            "SPY Current Price",
                            f"${spy_current_price:.2f}" if spy_current_price else "N/A",
                            delta=f"{spy_std_devs:.2f}σ from mean" if spy_std_devs is not None else "N/A"
                        )
                    
                    with col4:
                        if spy_std_devs != 0 and std_devs is not None:
                            rel_volatility = std_devs / spy_std_devs
                            st.metric(
                                "Relative Volatility",
                                f"{rel_volatility:.2f}x",
                                delta="vs SPY"
                            )
                        else:
                            st.metric(
                                "Relative Volatility",
                                "N/A",
                                delta="Insufficient data"
                            )
                    
                    # Display Selected Stock charts side by side
                    st.markdown("---")
                    st.subheader(f"📊 {stock_ticker} Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{stock_ticker} Distribution Analysis**")
                        if stock_fig:
                            st.pyplot(stock_fig)
                        else:
                            st.error("Could not generate distribution analysis")
                    
                    with col2:
                        st.markdown(f"**{stock_ticker} Price Trend with Volume Profile & Moving Averages**")
                        price_volume_fig = create_price_volume_chart(stock_df, stock_ticker)
                        if price_volume_fig:
                            st.pyplot(price_volume_fig)
                        else:
                            st.error("Could not generate price-volume chart")
                    
                    # Display SPY charts side by side
                    st.markdown("---")
                    st.subheader("📊 SPY (S&P 500) Benchmark Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**SPY Distribution Analysis - Benchmark**")
                        if spy_fig:
                            st.pyplot(spy_fig)
                        else:
                            st.error("Could not generate SPY distribution analysis")
                    
                    with col2:
                        st.markdown("**SPY Price Trend with Volume Profile & Moving Averages - Benchmark**")
                        spy_price_volume_fig = create_price_volume_chart(spy_df, "SPY")
                        if spy_price_volume_fig:
                            st.pyplot(spy_price_volume_fig)
                        else:
                            st.error("Could not generate SPY price-volume chart")

with tab2:
    st.header("🔍 Batch Stock Scanner")
    st.markdown("Analyze multiple stocks and identify opportunities")
    
    # Stock selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_selection = st.multiselect(
            "Select stocks to analyze (or leave empty to analyze all)",
            options=STOCK_SYMBOLS,
            default=[],
            help="Select specific stocks or leave empty to analyze all stocks in the list"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_batch = st.button("🚀 Run Batch Analysis")
    
    if run_batch:
        stocks_to_analyze = stock_selection if stock_selection else STOCK_SYMBOLS
        
        st.info(f"Analyzing {len(stocks_to_analyze)} stocks...")
        progress_bar = st.progress(0)
        
        # Batch analyze
        results = batch_analyze_stocks(stocks_to_analyze, is_hourly=False, progress_bar=progress_bar)
        progress_bar.empty()
        
        if results:
            # Convert to DataFrame
            df_results = pd.DataFrame(results)
            
            # Add SPY relative metrics if SPY is in results
            if 'SPY' in df_results['stock'].values:
                spy_row = df_results[df_results['stock'] == 'SPY'].iloc[0]
                spy_std_dev = spy_row['std_devs_from_mean']
                df_results['relative_std_dev'] = (df_results['std_devs_from_mean'] / spy_std_dev) if spy_std_dev != 0 else 0
            else:
                df_results['relative_std_dev'] = 0

            df_results['relative_vol_ratio'] = df_results['rel_vol']
            
            # Sort by std_devs_from_mean
            df_results = df_results.sort_values('std_devs_from_mean')
            
            # Save to CSV
            csv_filename = f"{DATA_DIR}/stock_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_results.to_csv(csv_filename, index=False)
            
            st.success(f"✅ Analysis complete! Analyzed {len(results)} stocks.")
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                undervalued = len(df_results[df_results['std_devs_from_mean'] < -1])
                st.metric("Undervalued (< -1σ)", undervalued)
            
            with col2:
                overvalued = len(df_results[df_results['std_devs_from_mean'] > 1])
                st.metric("Overvalued (> +1σ)", overvalued)
            
            with col3:
                avg_std_dev = df_results['std_devs_from_mean'].mean()
                st.metric("Avg Std Dev", f"{avg_std_dev:.2f}σ")
            
            with col4:
                total_analyzed = len(df_results)
                st.metric("Total Analyzed", total_analyzed)
            
            # Display top/bottom performers
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔻 Most Undervalued")
                undervalued_stocks = df_results.head(10)[['stock', 'current_price', 'std_devs_from_mean', 'peak_price']]
                st.dataframe(undervalued_stocks)
            
            with col2:
                st.subheader("🔺 Most Overvalued")
                overvalued_stocks = df_results.tail(10)[['stock', 'current_price', 'std_devs_from_mean', 'peak_price']]
                st.dataframe(overvalued_stocks)
            
            # Download results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis Results",
                data=csv,
                file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.error("No results generated. Please check your stock symbols and try again.")

with tab3:
    st.header("📈 Dashboard")
    st.markdown("View and filter previous analysis reports")
    
    # Find existing reports
    report_files = glob.glob(f"{DATA_DIR}/*.csv")
    
    if report_files:
        # Select report
        latest_report = max(report_files)
        selected_report = st.selectbox(
            "Select Report",
            options=sorted(report_files, reverse=True),
            format_func=lambda x: os.path.basename(x)
        )
        
        # Load report
        df = pd.read_csv(selected_report)

        # Merge with portfolio data
        if portfolio_data is not None:
            df = df.merge(portfolio_data, left_on='stock', right_on='Symbol', how='left')
            df['Owned'] = df['Shares'].notna()
            df['Dollar_Impact_10pct'] = (df['Equity'] * 0.10).round(0)
            df['Current_Value'] = df['current_price'] * df['Shares']
            total_live_value = df['Current_Value'].sum()
            df['Position_Size_Pct'] = (df['Current_Value'] / total_live_value * 100).round(2)
            df['delta'] = df['Current_Value'] - df['Average Cost']*df['Shares']
        
        # Display update time
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(selected_report))
        st.info(f"📅 Report Generated: {file_mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate MA relationship if both current_price and MA columns exist
        if 'current_price' in df.columns and 'MA_50' in df.columns and 'MA_200' in df.columns:
            def get_ma_position(row):
                current_price = row['current_price']
                ma_50 = row['MA_50']
                ma_200 = row['MA_200']
                
                if pd.isna(ma_50) or pd.isna(ma_200):
                    return "N/A"
                
                if current_price > ma_50 and current_price > ma_200:
                    return "Above Both MAs"
                elif current_price > ma_50 and current_price < ma_200:
                    return "Above 50MA, Below 200MA"
                elif current_price < ma_50 and current_price > ma_200:
                    return "Below 50MA, Above 200MA"
                elif current_price < ma_50 and current_price < ma_200:
                    return "Below Both MAs"
                else:
                    return "Near MAs"
            
            df['MA_Position'] = df.apply(get_ma_position, axis=1)

        # Run technical analysis on ALL stocks
        st.subheader("🔄 Running Enhanced Technical Analysis on All Stocks...")
        progress_bar = st.progress(0)
        technical_results = []
        
        stocks_to_analyze = df['stock'].unique()
        for i, stock in enumerate(stocks_to_analyze):
            progress_bar.progress((i + 1) / len(stocks_to_analyze))
            
            # Fetch fresh data for technical analysis
            stock_df = fetch_stock_data(stock, days=720)
            if stock_df is not None:
                analysis = analyze_stock_technical(stock, stock_df)
                if analysis:
                    technical_results.append(analysis)
        
        progress_bar.empty()
        
        # Create technical analysis DataFrame and merge with df
        if technical_results:
            tech_df = pd.DataFrame([{
                'stock': result['ticker'],
                'signal': result['signal'],
                'net_score': result['scores']['net'],
                'bullish_score': result['scores']['bullish'],
                'bearish_score': result['scores']['bearish'],
                'ma_position': result['technicals']['ma_position'],
                'percent_above_ma50': result['technicals']['percent_above_ma50'],
                'percent_above_ma200': result['technicals']['percent_above_ma200'],
                'vol_node_ratio':      result['volume_analysis']['vol_node_ratio'],
                'vol_node_percentile': result['volume_analysis']['vol_node_percentile'],
                'pct_from_nearest_node': result['volume_analysis']['pct_from_nearest_node'],
                'z_score': result['distribution']['z_score'],
                'dist_std_devs':   result['volume_analysis']['dist_std_devs'],
                'days_in_dist':    result['volume_analysis']['days_in_dist'],
                'vol_vs_dist_avg': result['volume_analysis']['vol_vs_dist_avg'],
                'warnings': ', '.join(result['warnings']) if result['warnings'] else 'None',
                'recommendation': result['recommendations'].get('action', 'N/A') if result['recommendations'] else 'N/A',
                'tsmom_5d':        result.get('tsmom', {}).get(5,  {}).get('return'),
                'tsmom_10d':       result.get('tsmom', {}).get(10, {}).get('return'),
                'tsmom_20d':       result.get('tsmom', {}).get(20, {}).get('return'),
                'tsmom_60d':       result.get('tsmom', {}).get(60, {}).get('return'),
                'tsmom_126d':      result.get('tsmom', {}).get(126, {}).get('return'),
                'tsmom_252d':      result.get('tsmom', {}).get(252, {}).get('return'),
                'tsmom_alignment': result.get('tsmom', {}).get('alignment', 'N/A'),
            } for result in technical_results])

            # Merge with original data
            enhanced_df = df.merge(tech_df, on='stock', how='left')
            # Replace batch rel_vol with distribution-anchored vol_vs_dist_avg from tech analysis
        if 'vol_vs_dist_avg' in enhanced_df.columns:
            enhanced_df['relative_vol_ratio'] = enhanced_df['vol_vs_dist_avg']
        else:
            enhanced_df = df
            # Add empty technical columns if no analysis results
            tech_columns = ['signal', 'confidence', 'ma_position', 'trend_strength', 'volume_context',
                            'distribution_position', 'z_score', 'bullish_score',
                           'bearish_score', 'net_score', 'warnings', 'recommendation',
                           'tsmom_20d', 'tsmom_60d', 'tsmom_126d', 'tsmom_252d', 'tsmom_alignment']
            for col in tech_columns:
                enhanced_df[col] = 'N/A'
        
        # Display portfolio summary if data available
        if 'Owned' in enhanced_df.columns:
            st.markdown("---")
            st.subheader("📊 Portfolio Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            owned_df = enhanced_df[enhanced_df['Owned'] == True]
            with col1:
                st.metric("Stocks Owned", len(owned_df))
            with col2:
                total_value = owned_df['Equity'].sum()
                st.metric("Total Value", f"${total_value:,.0f}")
            with col3:
                strong_buys = len(owned_df[owned_df['signal'] == 'STRONG_BUY'])
                st.metric("STRONG BUY (Owned)", strong_buys)
            with col4:
                strong_sells = len(owned_df[owned_df['signal'] == 'STRONG_SELL'])
                st.metric("STRONG SELL (Owned)", strong_sells)
            with col5:
                if not owned_df.empty:
                    top = owned_df.nlargest(1, 'Position_Size_Pct').iloc[0]
                    st.metric("Largest Position", f"{top['stock']} ({top['Position_Size_Pct']:.1f}%)")
        
        # Display results for ALL stocks
        st.subheader(f"📊 Results ({len(enhanced_df)} stocks)")
        
        # Interactive Plotly chart for filtered stocks
        if not enhanced_df.empty:
            # Filter stocks based on criteria
            filter_criteria = st.checkbox("Filter: |Rel Std Dev| > 1 AND Rel Vol Ratio > 1 AND Signal != HOLD_NEUTRAL", 
                                         value=True, key="filter_stocks")
            
            if filter_criteria and 'relative_std_dev' in enhanced_df.columns and 'relative_vol_ratio' in enhanced_df.columns:
                scatter_df = enhanced_df[
                    ((enhanced_df['relative_std_dev'].abs() > 1) |
                     (enhanced_df['relative_vol_ratio'] > 1)) &
                    (enhanced_df['signal'] != 'HOLD_NEUTRAL')
                ]
            else:
                scatter_df = enhanced_df
            
            # Only create plot if we have stocks that meet the criteria
            if not scatter_df.empty:
                fig = create_interactive_scatter(
                    df=scatter_df,
                    x_col='relative_std_dev' if 'relative_std_dev' in scatter_df.columns else 'std_devs_from_mean',
                    y_col='relative_vol_ratio' if 'relative_vol_ratio' in scatter_df.columns else 'current_vol',
                    color_col='signal',
                    text_col='stock',
                    hover_data=['delta', 'current_price', 'std_devs_from_mean'],
                    title=f"Stock Distribution Analysis - Colored by Signal ({len(scatter_df)} stocks shown)",
                    height=600
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No stocks meet the filter criteria")
        
        # Display data table with sorting capability
        st.markdown("---")
        st.subheader("📋 Detailed Analysis Results (Click column headers to sort)")
        
        # Define the exact column order with repetitions as specified
        ordered_columns_with_repeats = [
            'stock',
            'Shares',
            'current_price',
            'current_vol',
            'price_change_1d',
            'price_change_1d_pct',
            'Average Cost',
            'Current_Value',
            'Equity',
            'delta',
            'Position_Size_Pct',
            'current_price_2',
            'price_high_52w',
            'price_low_52w',
            'pct_from_high',
            'MA_20',
            'MA_50',
            'MA_200',
            'MA_Position',          # keep — computed from current_price vs MA cols
            'fitted_mean',
            'std_devs_from_mean',
            'peak_volume_M',
            'peak_price',
            'current_price_3',
            'dist_std_devs',
            'vol_vs_dist_avg',
            'days_in_dist',
            'signal',
            'net_score',
            'bullish_score',
            'bearish_score',
            'tsmom_5d',
            'tsmom_10d',
            'tsmom_20d',
            'tsmom_60d',
            'tsmom_126d',
            'tsmom_252d',
            'tsmom_alignment',
            'ma_position',
            'percent_above_ma50',
            'percent_above_ma200',
            'volume_context',
            'z_score',
            'warnings',
            'recommendation',
        ]
        
        # Create display and numeric dataframes
        display_df = pd.DataFrame()
        numeric_df = pd.DataFrame()
        
        # Add each column to both dataframes
        for col in ordered_columns_with_repeats:
            if col in enhanced_df.columns:
                # Original column exists, add it directly to both
                display_df[col] = enhanced_df[col]
                numeric_df[col] = enhanced_df[col]
            elif col.endswith('_2') or col.endswith('_3'):
                # This is a repeated column - extract the base name
                base_col = col[:-2]  # Remove '_2' or '_3'
                if base_col in enhanced_df.columns:
                    display_df[col] = enhanced_df[base_col]
                    numeric_df[col] = enhanced_df[base_col]
        
        # Define the rename mapping with repeats
        rename_map_with_repeats = {
            'stock': 'Stock',
            'Shares': 'Shares Owned',
            'current_price': 'Current Price',
            'price_change_1d': '1D Change ($)',
            'price_change_1d_pct': '1D Change (%)',
            'Average Cost': 'Avg Cost ($)',
            'Current_Value': 'Current Value',
            'Equity': 'Cost Value ($)',
            'delta': 'Delta',
            'Position_Size_Pct': 'Portfolio %',
            'Dollar_Impact_10pct': '$ Impact (10% move)',
            'current_price_2': 'Current Price (2)',
            'current_vol': 'Volume',
            'price_high_52w': '52W High',
            'price_low_52w': '52W Low',
            'pct_from_high': 'pct From High',
            'MA_20': '20-Day MA',
            'MA_50': '50-Day MA',
            'MA_200': '200-Day MA',
            'MA_Position': 'Price vs MAs',
            'fitted_mean': 'Mean',
            'std_devs_from_mean': 'Std Devs',
            'peak_volume_M': 'Peak Volume',
            'peak_price': 'Price @ Peak Vol',
            'current_price_3': 'Current Price (3)',
            'dist_std_devs':   'Dist Std Devs',
            'vol_vs_dist_avg': 'Vol vs Dist Avg',
            'days_in_dist':    'Days in Dist',
            'signal': 'Signal',
            'net_score': 'Net Score',
            'bullish_score': 'Bull Score',
            'bearish_score': 'Bear Score',
            'tsmom_5d':  'TSMOM 5d',
            'tsmom_10d': 'TSMOM 10d',
            'tsmom_20d': 'TSMOM 20d',
            'tsmom_60d': 'TSMOM 60d',
            'tsmom_126d': 'TSMOM 126d',
            'tsmom_252d': 'TSMOM 252d',
            'tsmom_alignment': 'TSMOM Align',
            'ma_position': 'MA Position',
            'percent_above_ma50': '% Above 50MA',
            'percent_above_ma200': '% Above 200MA',
            'volume_context': 'VP Context',
            'z_score': 'Z-Score',
            'warnings': 'Warnings',
            'recommendation': 'Action',
        }
        
        # Apply renaming to both dataframes
        display_df = display_df.rename(columns={k: v for k, v in rename_map_with_repeats.items() if k in display_df.columns})
        numeric_df = numeric_df.rename(columns={k: v for k, v in rename_map_with_repeats.items() if k in numeric_df.columns})
        
        # Define the display_sortable_dataframe function here
        def display_sortable_dataframe(display_df, numeric_df, height=400, key_prefix="df"):
            """
            Display a dataframe with column sorting capability using numeric values for sorting.
            
            Parameters:
            - display_df: DataFrame with formatted values for display
            - numeric_df: DataFrame with original numeric values for sorting
            - height: Height of the dataframe display
            - key_prefix: Unique prefix for session state keys
            
            Returns:
            - None (displays the dataframe in Streamlit)
            """
            if display_df.empty:
                st.warning("No data to display")
                return
            
            # Create sort controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                all_columns = display_df.columns.tolist()
                default_sort = 'Delta' if 'Delta' in all_columns else all_columns[0]
                
                sort_column = st.selectbox(
                    "Sort by:",
                    options=all_columns,
                    index=all_columns.index(default_sort) if default_sort in all_columns else 0,
                    key=f"{key_prefix}_sort_col"
                )
            
            with col2:
                sort_ascending = st.checkbox("Ascending", value=True, key=f"{key_prefix}_sort_asc")
            
            # Sort using the numeric dataframe
            if sort_column in numeric_df.columns:
                # Get the sorting order from numeric data
                sort_values = numeric_df[sort_column].values
                
                # Convert to numeric if possible, but keep original for non-numeric
                try:
                    numeric_sort_values = pd.to_numeric(sort_values, errors='coerce')
                    # If we have at least some numeric values, use them for sorting
                    if not pd.isna(numeric_sort_values).all():
                        sort_indices = np.argsort(numeric_sort_values)
                    else:
                        # Fall back to string sorting
                        sort_indices = np.argsort(sort_values)
                except:
                    sort_indices = np.argsort(sort_values)
                
                if not sort_ascending:
                    sort_indices = sort_indices[::-1]
                
                # Apply the same sort order to display_df
                sorted_display_df = display_df.iloc[sort_indices].reset_index(drop=True)
            else:
                # Fallback to regular sorting on display_df
                sorted_display_df = display_df.sort_values(by=sort_column, ascending=sort_ascending).reset_index(drop=True)
            
            # NOW create a formatted copy for display AFTER sorting
            formatted_display_df = sorted_display_df.copy()
            
            # ===== APPLY FORMATTING AFTER SORTING =====
            # Format columns with comma for thousands
            comma_format_columns = [
                'Current Value',
                'Cost Value ($)',
                'Delta',
                '$ Impact (10% move)',
                'Volume',
                'Peak Volume'
            ]
            
            for col in comma_format_columns:
                if col in formatted_display_df.columns:
                    formatted_display_df[col] = formatted_display_df[col].apply(
                        lambda x: f"{float(x):,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else 
                                 (f"{float(x):,.0f}" if pd.notna(x) and str(x).replace('.','').isdigit() else x)
                    )
            
            # Format columns to nearest round number (integers) without commas
            columns_to_round = [
                'Shares Owned',
                '52W High',
                '52W Low',
                '20-Day MA',
                '50-Day MA',
                '200-Day MA',
                'Mean',
                'Price @ Peak Vol',
                'Current Price',
                'Current Price (2)',
                'Current Price (3)',
                'Avg Cost ($)',
                '1D Change ($)'
            ]
            
            for col in columns_to_round:
                if col in formatted_display_df.columns:
                    formatted_display_df[col] = formatted_display_df[col].apply(
                        lambda x: f"{int(round(float(x)))}" if pd.notna(x) and isinstance(x, (int, float)) else
                                 (f"{int(round(float(x)))}" if pd.notna(x) and str(x).replace('.','').isdigit() else x)
                    )
            
            # Format percentage columns with 1 decimal place
            percentage_columns = ['Portfolio %', 'pct From High', 'Confidence', '1D Change (%)']
            tsmom_pct_columns  = ['TSMOM 5d', 'TSMOM 10d', 'TSMOM 20d', 'TSMOM 60d', 'TSMOM 126d', 'TSMOM 252d']

            for col in percentage_columns:
                if col in formatted_display_df.columns:
                    formatted_display_df[col] = formatted_display_df[col].apply(
                        lambda x: f"{float(x):.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else x
                    )

            for col in tsmom_pct_columns:
                if col in formatted_display_df.columns:
                    # Stored as decimal (0.082), multiply by 100 and show sign
                    formatted_display_df[col] = formatted_display_df[col].apply(
                        lambda x: f"{float(x)*100:+.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else x
                    )
            
            # Format standard deviation columns with 2 decimals
            std_columns = ['Std Devs', 'Rel Std Dev', 'Rel Vol Ratio', 'Z-Score']
            for col in std_columns:
                if col in formatted_display_df.columns:
                    formatted_display_df[col] = formatted_display_df[col].apply(
                        lambda x: f"{float(x):.2f}" if pd.notna(x) and isinstance(x, (int, float)) else
                                 (f"{float(x):.2f}" if pd.notna(x) and str(x).replace('.','').isdigit() else x)
                    )
            # ===== END OF FORMATTING LOGIC =====
            
            # Display the formatted dataframe
            st.data_editor(
                formatted_display_df,
                disabled=True,
                width=2000,
                height=height,
                hide_index=True,
                use_container_width=True
            )
            
            return sorted_display_df
        
        # Call the function with both dataframes
        if not display_df.empty:
            st.write("**Column order with repeats:**")
            sorted_df = display_sortable_dataframe(display_df, numeric_df, key_prefix="daily_dashboard")
        else:
            st.warning("No columns available to display")
        
        # Technical Analysis Summary
        if technical_results:
            st.markdown("---")
            st.subheader("📈 Technical Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                strong_buys = len(enhanced_df[enhanced_df['signal'] == 'STRONG_BUY'])
                buys = len(enhanced_df[enhanced_df['signal'] == 'BUY'])
                st.metric("Bullish Signals", strong_buys + buys)

            with col2:
                strong_sells = len(enhanced_df[enhanced_df['signal'] == 'STRONG_SELL'])
                sells = len(enhanced_df[enhanced_df['signal'] == 'SELL'])
                st.metric("Bearish Signals", strong_sells + sells)

            with col3:
                wait_signals = len(enhanced_df[enhanced_df['signal'] == 'WAIT_FOR_PULLBACK'])
                st.metric("Wait for Pullback", wait_signals)

            with col4:
                full_bull = len(enhanced_df[enhanced_df['tsmom_alignment'] == 'FULL_BULL'])
                st.metric("FULL_BULL TSMOM", full_bull)


        # ── TSMOM Quick-Look Widget ───────────────────────────────────
        st.markdown("---")
        st.subheader("📐 TSMOM Alignment — Stock Quick Look")
        st.caption("Select any stock to see its full TSMOM breakdown.")
        tsmom_stock = st.selectbox(
            "Select stock",
            options=enhanced_df['stock'].tolist(),
            key="tsmom_quicklook"
        )
        if tsmom_stock and technical_results:
            match = next((r for r in technical_results if r['ticker'] == tsmom_stock), None)
            if match and 'tsmom' in match:
                render_tsmom_widget(match['tsmom'], compact=False)
            else:
                st.info("No TSMOM data available for this stock")


        # Download enhanced data
        # Download enhanced data with dashboard columns
        # Ensure the same column order and naming as the dashboard display
        if not enhanced_df.empty:
            # Create a copy of enhanced_df for CSV export
            export_df = enhanced_df.copy()
            
            # Define the exact column order for CSV (same as dashboard display order)
            csv_columns = [
                'stock',
                'Shares',
                'current_price',
                'current_vol',
                'price_change_1d',
                'price_change_1d_pct',
                'Average Cost',
                'Current_Value',
                'Equity',
                'delta',
                'Position_Size_Pct',
                'price_high_52w',
                'price_low_52w',
                'pct_from_high',
                'MA_20',
                'MA_50',
                'MA_200',
                'MA_Position',
                'fitted_mean',
                'std_devs_from_mean',
                'peak_volume_M',
                'peak_price',
                'dist_std_devs',
                'vol_vs_dist_avg',
                'days_in_dist',
                'signal',
                'net_score',
                'bullish_score',
                'bearish_score',
                'tsmom_5d',
                'tsmom_10d',
                'tsmom_20d',
                'tsmom_60d',
                'tsmom_126d',
                'tsmom_252d',
                'tsmom_alignment',
                'ma_position',
                'percent_above_ma50',
                'percent_above_ma200',
                'z_score',
                'warnings',
                'recommendation',
            ]
            
            # Filter to only include columns that actually exist in enhanced_df
            available_columns = [col for col in csv_columns if col in export_df.columns]
            
            # Create the export DataFrame with only the available columns in order
            csv_export_df = export_df[available_columns].copy()
            
            # Apply renaming for CSV (same as dashboard display names)
            rename_map = {
                'stock': 'Stock',
                'Shares': 'Shares Owned',
                'current_price': 'Current Price',
                'current_vol': 'Volume',
                'price_change_1d': '1D Change ($)',
                'price_change_1d_pct': '1D Change (%)',
                'Average Cost': 'Avg Cost ($)',
                'Current_Value': 'Current Value',
                'Equity': 'Cost Value ($)',
                'delta': 'Delta',
                'Position_Size_Pct': 'Portfolio %',
                'price_high_52w': '52W High',
                'price_low_52w': '52W Low',
                'pct_from_high': 'pct From High',
                'MA_20': '20-Day MA',
                'MA_50': '50-Day MA',
                'MA_200': '200-Day MA',
                'MA_Position': 'Price vs MAs',
                'fitted_mean': 'Mean',
                'std_devs_from_mean': 'Std Devs',
                'peak_volume_M': 'Peak Volume',
                'peak_price': 'Price @ Peak Vol',
                'dist_std_devs': 'Dist Std Devs',
                'vol_vs_dist_avg': 'Vol vs Dist Avg',
                'days_in_dist': 'Days in Dist',
                'signal': 'Signal',
                'net_score': 'Net Score',
                'bullish_score': 'Bull Score',
                'bearish_score': 'Bear Score',
                'tsmom_5d': 'TSMOM 5d',
                'tsmom_10d': 'TSMOM 10d',
                'tsmom_20d': 'TSMOM 20d',
                'tsmom_60d': 'TSMOM 60d',
                'tsmom_126d': 'TSMOM 126d',
                'tsmom_252d': 'TSMOM 252d',
                'tsmom_alignment': 'TSMOM Align',
                'ma_position': 'MA Position',
                'percent_above_ma50': '% Above 50MA',
                'percent_above_ma200': '% Above 200MA',
                'z_score': 'Z-Score',
                'warnings': 'Warnings',
                'recommendation': 'Action',
            }
            
            # Rename columns that exist
            csv_export_df = csv_export_df.rename(
                columns={k: v for k, v in rename_map.items() if k in csv_export_df.columns}
            )
            
            # Format TSMOM columns as percentages (they're stored as decimals)
            tsmom_cols = ['TSMOM 5d', 'TSMOM 10d', 'TSMOM 20d', 'TSMOM 60d', 'TSMOM 126d', 'TSMOM 252d']
            for col in tsmom_cols:
                if col in csv_export_df.columns:
                    csv_export_df[col] = csv_export_df[col].apply(
                        lambda x: f"{x*100:.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else x
                    )
            
            # Format percentage columns
            pct_cols = ['Portfolio %', 'pct From High', '1D Change (%)', '% Above 50MA', '% Above 200MA']
            for col in pct_cols:
                if col in csv_export_df.columns:
                    csv_export_df[col] = csv_export_df[col].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else x
                    )
            
            # Format standard deviation columns to 2 decimal places
            std_cols = ['Std Devs', 'Dist Std Devs', 'Z-Score']
            for col in std_cols:
                if col in csv_export_df.columns:
                    csv_export_df[col] = csv_export_df[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
                    )
            
            # Round price columns
            price_cols = ['Current Price', 'Avg Cost ($)', '52W High', '52W Low', '20-Day MA', 
                          '50-Day MA', '200-Day MA', 'Mean', 'Price @ Peak Vol']
            for col in price_cols:
                if col in csv_export_df.columns:
                    csv_export_df[col] = csv_export_df[col].apply(
                        lambda x: f"{int(round(x))}" if pd.notna(x) and isinstance(x, (int, float)) else x
                    )
            
            # Format value columns with commas
            value_cols = ['Current Value', 'Cost Value ($)', 'Delta', 'Volume', 'Peak Volume']
            for col in value_cols:
                if col in csv_export_df.columns:
                    csv_export_df[col] = csv_export_df[col].apply(
                        lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else x
                    )
            
            csv = csv_export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Enhanced Analysis",
                data=csv,
                file_name=f"enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            # Also show a preview of what will be downloaded
            with st.expander("📋 Preview CSV columns"):
                st.write(f"**{len(csv_export_df.columns)} columns will be exported:**")
                st.write(list(csv_export_df.columns))

    else:
        st.warning("No reports found. Please run the Batch Scanner first.")


with tab4:
    st.header("⚙️ Data Management")
    st.markdown("Manage cached stock data and analysis results")
    
    # Data management options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cached Data")
        
        # Clear cache
        if st.button("🗑️ Clear All Cache"):
            st.session_state.spy_data = None
            st.session_state.spy_last_updated = None
            st.success("✅ Cache cleared!")
        
        # Cache info
        if st.session_state.spy_data is not None:
            st.info(f"📅 SPY data last updated: {st.session_state.spy_last_updated}")
            st.metric("Cached SPY Records", len(st.session_state.spy_data))
        else:
            st.warning("No SPY data cached")
    
    with col2:
        st.subheader("📁 Analysis Reports")
        
        # List reports
        report_files = glob.glob(f"{DATA_DIR}/*.csv")
        if report_files:
            st.write(f"Found {len(report_files)} reports")
            
            # Delete old reports
            if st.button("🗑️ Delete Reports > 30 days"):
                cutoff_time = datetime.now() - timedelta(days=30)
                deleted_count = 0
                
                for report in report_files:
                    if datetime.fromtimestamp(os.path.getmtime(report)) < cutoff_time:
                        os.remove(report)
                        deleted_count += 1
                
                st.success(f"✅ Deleted {deleted_count} old reports")
        else:
            st.warning("No analysis reports found")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Stock Distribution Analyzer Pro | Built with Streamlit & Yahoo Finance"
    "</div>",
    unsafe_allow_html=True
)
