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
PORTFOLIO_FILE = 'Robinhood_December_1_-_Sheet1.csv'

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
        portfolio_df['Position_Size_Pct'] = (portfolio_df['Equity'] / total_equity * 100).round(2)
        stock_symbols = portfolio_df['Symbol'].tolist()
        return portfolio_df[['Symbol', 'Shares', 'Equity', 'Position_Size_Pct', 'Average Cost']], stock_symbols
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

def check_trend_momentum(price_data, period=10):
    """
    Check if trend is accelerating or decelerating.
    Useful for detecting exhaustion or building momentum.
    """
    if len(price_data) < period * 2:
        return "UNKNOWN"
    
    recent = price_data[-period:]
    prior = price_data[-period*2:-period]
    
    recent_slope = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0
    prior_slope = (prior[-1] - prior[0]) / prior[0] if prior[0] != 0 else 0
    
    if recent_slope > 0 and prior_slope > 0:
        if recent_slope > prior_slope * 1.3:
            return "ACCELERATING_UP"
        elif recent_slope < prior_slope * 0.7:
            return "DECELERATING_UP"
    elif recent_slope < 0 and prior_slope < 0:
        if recent_slope < prior_slope * 1.3:
            return "ACCELERATING_DOWN"
        elif recent_slope > prior_slope * 0.7:
            return "DECELERATING_DOWN"
    
    return "STEADY"

def get_cached_spy_data():
    """Get SPY data with session-level caching + 1 hour refresh."""
    now = datetime.now()
    
    # Fetch if no data or > 1 hour old
    if (st.session_state.spy_data is None or 
        st.session_state.spy_last_updated is None or
        (now - st.session_state.spy_last_updated).total_seconds() > 3600):
        
        st.session_state.spy_data = fetch_stock_data("SPY", days=365)
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
def fetch_stock_data(ticker, days=365):
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
        
        return df[['stock', 'date', 'price', 'volume']]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data_hourly(ticker, days=30):
    """Fetch hourly stock data from Yahoo Finance for the past N days."""
    try:
        stock = yf.Ticker(ticker)
        # Use period for hourly data as it's more reliable
        df = stock.history(period=f"{days}d", interval="1h")
        
        if df.empty:
            return None
        
        # Prepare data in the format expected by analysis
        df = df.reset_index()
        df['stock'] = ticker
        # Handle different datetime column names
        if 'Datetime' in df.columns:
            df['date'] = df['Datetime']
        elif 'Date' in df.columns:
            df['date'] = df['Date']
        else:
            df['date'] = df.index
        df['price'] = df['Close']
        df['volume'] = df['Volume']
        
        return df[['stock', 'date', 'price', 'volume']]
    except Exception as e:
        st.error(f"Error fetching hourly data for {ticker}: {str(e)}")
        return None

def create_price_volume_chart(df, stock_symbol):
    """Create a price vs date chart with volume profile on secondary x-axis (right side)."""
    if df is None or df.empty:
        return None
    
    # Prepare data
    stock_data = df[df['stock'] == stock_symbol].copy()
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values('date')
    
    # Calculate moving averages
    stock_data['MA_20'] = stock_data['price'].rolling(window=20, min_periods=1).mean()
    stock_data['MA_50'] = stock_data['price'].rolling(window=50, min_periods=1).mean()
    stock_data['MA_200'] = stock_data['price'].rolling(window=200, min_periods=1).mean()
    
    # Convert pandas Series to numpy arrays to avoid the indexing error
    dates = stock_data['date'].values
    prices = stock_data['price'].values
    ma_50 = stock_data['MA_50'].values
    ma_20 = stock_data['MA_20'].values
    ma_200 = stock_data['MA_200'].values
    
    # Create volume histogram by price (for the volume profile)
    # FIX: Handle NaN values before converting to int
    stock_data['price_rounded'] = stock_data['price'].round()
    # Drop any rows with NaN in price_rounded
    stock_data = stock_data.dropna(subset=['price_rounded'])
    # Now safely convert to int
    stock_data['price_rounded'] = stock_data['price_rounded'].astype(int)
    
    volume_by_price = stock_data.groupby('price_rounded')['volume'].sum().sort_index()
    price_levels = volume_by_price.index.values
    volumes = volume_by_price.values
    
    # Create figure with primary axes
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot price and moving averages on primary y-axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    
    # Plot price line
    ax1.plot(dates, prices, color='tab:blue', linewidth=2, label='Price')
    
    # Plot moving averages
    ax1.plot(dates, ma_20, color='green', linewidth=1.5, linestyle='--', label='20-Day MA')
    ax1.plot(dates, ma_50, color='orange', linewidth=1.5, linestyle='--', label='50-Day MA')
    ax1.plot(dates, ma_200, color='red', linewidth=1.5, linestyle='--', label='200-Day MA')
    
    ax1.grid(True, alpha=0.3)
    
    # Create secondary x-axis for volume profile (on the right side)
    ax2 = ax1.twiny()  # Create a twin x-axis
    ax2.set_xlabel('Volume (Millions)', color='tab:green')
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', labelcolor='tab:green')
    
    # Plot volume profile as horizontal bars
    # Use the same y-axis (price) but x-axis is volume
    ax2.barh(price_levels, volumes / 1e6, 
             alpha=0.3, color='tab:green', height=0.8, label='Volume Profile')
    
    # Reverse the volume x-axis so 0 is on the right
    ax2.invert_xaxis()
    
    # Set title
    plt.title(f'{stock_symbol} - Price Trend with Volume Profile & Moving Averages', fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Format x-axis
    plt.xticks(rotation=45)
    fig.tight_layout()
    
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
                df = fetch_stock_data(symbol, days=365)
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

                        
                        # Get the latest moving average values
                        metrics['MA_20'] = stock_data['MA_20'].iloc[-1] if not stock_data['MA_20'].isna().iloc[-1] else None
                        metrics['MA_50'] = stock_data['MA_50'].iloc[-1] if not stock_data['MA_50'].isna().iloc[-1] else None
                        metrics['MA_200'] = stock_data['MA_200'].iloc[-1] if not stock_data['MA_200'].isna().iloc[-1] else None
                    
                    results.append(metrics)
        except Exception as e:
            st.warning(f"Error analyzing {symbol}: {str(e)}")
            continue
    
    return results


def analyze_stock_technical(ticker, df):
    """
    Enhanced comprehensive technical analysis based on the pseudocode framework.
    Incorporates learnings from GOOG, AGX, AMZN, ORCL, CMG, LII, PYPL, etc.
    """
    try:
        if df is None or df.empty:
            return None
            
        # Prepare data
        stock_data = df[df['stock'] == ticker].copy()
        if stock_data.empty:
            return None
            
        stock_data = stock_data.sort_values('date')
        
        # Clean data - remove any rows with NaN or infinite values
        stock_data = stock_data.replace([np.inf, -np.inf], np.nan)
        stock_data = stock_data.dropna(subset=['price', 'volume'])
        
        # Ensure we have enough data
        if len(stock_data) < 50:  # Need at least 50 days for meaningful analysis
            st.warning(f"Not enough data for {ticker} (only {len(stock_data)} days)")
            return None
            
        # Get current data
        current_price = float(stock_data['price'].iloc[-1])
        price_data = stock_data['price'].values.astype(float)
        volume_data = stock_data['volume'].values.astype(float)
        
        # ==================================================================
        # STEP 1: Calculate Key Technical Indicators
        # ==================================================================
        # Calculate moving averages with min_periods to handle NaN at the beginning
        ma_50 = stock_data['price'].rolling(window=50, min_periods=25).mean().iloc[-1]
        ma_200 = stock_data['price'].rolling(window=200, min_periods=100).mean().iloc[-1]
        
        # Handle NaN values in moving averages
        if pd.isna(ma_50) or pd.isna(ma_200):
            ma_50 = current_price if pd.isna(ma_50) else ma_50
            ma_200 = current_price if pd.isna(ma_200) else ma_200
        
        # Calculate volume profile (simplified - group by rounded price)
        # First create rounded prices safely
        stock_data['price_rounded'] = stock_data['price'].round().astype(float)
        stock_data = stock_data.dropna(subset=['price_rounded'])
        stock_data['price_rounded'] = stock_data['price_rounded'].astype(int)
        
        volume_profile = stock_data.groupby('price_rounded')['volume'].sum().sort_index()
        
        # Find high volume nodes (top 20% by volume)
        if not volume_profile.empty:
            high_volume_threshold = volume_profile.quantile(0.80)
            high_volume_nodes = volume_profile[volume_profile >= high_volume_threshold]
        else:
            high_volume_nodes = pd.Series(dtype=float)
        
        # Calculate price distribution (volume-weighted)
        total_volume = volume_data.sum()
        if total_volume > 0 and len(price_data) > 0 and len(volume_data) > 0:
            try:
                weighted_mean = np.average(price_data, weights=volume_data)
                weighted_variance = np.average((price_data - weighted_mean)**2, weights=volume_data)
                std_dev = np.sqrt(weighted_variance) if weighted_variance > 0 else 1.0
            except:
                weighted_mean = np.mean(price_data)
                std_dev = np.std(price_data) if np.std(price_data) > 0 else 1.0
        else:
            weighted_mean = np.mean(price_data) if len(price_data) > 0 else current_price
            std_dev = np.std(price_data) if len(price_data) > 0 and np.std(price_data) > 0 else 1.0
        
        # ==================================================================
        # STEP 2: Determine Price Position Relative to MAs
        # ==================================================================
        percent_above_ma50 = ((current_price - ma_50) / ma_50) * 100 if ma_50 != 0 and not pd.isna(ma_50) else 0
        percent_above_ma200 = ((current_price - ma_200) / ma_200) * 100 if ma_200 != 0 and not pd.isna(ma_200) else 0
        
        if current_price > ma_50 and current_price > ma_200:
            ma_position = "ABOVE_BOTH"
        elif current_price < ma_50 and current_price < ma_200:
            ma_position = "BELOW_BOTH"
        elif current_price > ma_200 and current_price < ma_50:
            ma_position = "BETWEEN"
        else:
            ma_position = "MIXED"
        
        # ==================================================================
        # STEP 3: Check Moving Average Crossovers
        # ==================================================================
        ma_50_series = stock_data['price'].rolling(window=50, min_periods=25).mean()
        ma_200_series = stock_data['price'].rolling(window=200, min_periods=100).mean()
        
        # Remove NaN values for comparison
        ma_50_series = ma_50_series.dropna()
        ma_200_series = ma_200_series.dropna()
        
        if len(ma_50_series) > 0 and len(ma_200_series) > 0:
            if ma_50_series.iloc[-1] > ma_200_series.iloc[-1]:
                crossover_status = "GOLDEN_CROSS"
            elif ma_50_series.iloc[-1] < ma_200_series.iloc[-1]:
                crossover_status = "DEATH_CROSS"
            else:
                crossover_status = "NEUTRAL"
            
            # Check if crossover is recent (within last 20 days)
            if len(ma_50_series) >= 20 and len(ma_200_series) >= 20:
                ma_50_prev = ma_50_series.iloc[-20] if len(ma_50_series) >= 20 else ma_50_series.iloc[0]
                ma_200_prev = ma_200_series.iloc[-20] if len(ma_200_series) >= 20 else ma_200_series.iloc[0]
                
                # Golden cross happened recently
                if ma_50_prev <= ma_200_prev and ma_50_series.iloc[-1] > ma_200_series.iloc[-1]:
                    crossover_status = "GOLDEN_CROSS_RECENT"
                # Death cross happened recently
                elif ma_50_prev >= ma_200_prev and ma_50_series.iloc[-1] < ma_200_series.iloc[-1]:
                    crossover_status = "DEATH_CROSS_RECENT"
        else:
            crossover_status = "NEUTRAL"
        
        # ==================================================================
        # STEP 4: Analyze Volume Profile Position (ENHANCED)
        # ==================================================================
        if not high_volume_nodes.empty:
            # Find nearest high-volume node
            try:
                nearest_node_price = min(high_volume_nodes.index, key=lambda x: abs(float(x) - current_price))
                distance_to_node = abs(current_price - float(nearest_node_price))
                percent_from_node = (distance_to_node / float(nearest_node_price)) * 100 if float(nearest_node_price) != 0 else 0
                
                # ENHANCED: More nuanced volume zone classification
                if percent_from_node < 5:  # Increased from 3 to 5
                    volume_zone = "HIGH_VOLUME_ZONE"
                    volume_context = "AT_VALUE"
                elif percent_from_node > 40:  # New extreme category for GOOG/AGX situations
                    volume_zone = "EXTREMELY_THIN_VOLUME"
                    volume_context = "EXTREMELY_EXTENDED"
                elif percent_from_node > 15:  # Increased from 10 to 15
                    volume_zone = "THIN_VOLUME_ZONE"
                    volume_context = "EXTENDED"
                else:
                    volume_zone = "MODERATE_VOLUME"
                    volume_context = "TRANSITIONAL"
                
                # Determine direction relative to value
                if current_price > float(nearest_node_price) * 1.05:  # More than 5% above
                    volume_direction = "ABOVE_VALUE"
                elif current_price < float(nearest_node_price) * 0.95:  # More than 5% below
                    volume_direction = "BELOW_VALUE"
                else:
                    volume_direction = "AT_VALUE"
            except:
                volume_zone = "UNKNOWN"
                volume_context = "UNKNOWN"
                volume_direction = "UNKNOWN"
                nearest_node_price = None
                percent_from_node = 0
        else:
            volume_zone = "UNKNOWN"
            volume_context = "UNKNOWN"
            volume_direction = "UNKNOWN"
            nearest_node_price = None
            percent_from_node = 0
        
        # ==================================================================
        # STEP 5: Calculate Standard Deviation Position (ENHANCED)
        # ==================================================================
        z_score = (current_price - weighted_mean) / std_dev if std_dev > 0 else 0
        
        # Enhanced classification with more granularity
        if z_score > 10:
            distribution_position = "STATISTICAL_BUBBLE"  # AGX territory
        elif z_score > 5:
            distribution_position = "EXTREME_HIGH"
        elif z_score > 2:
            distribution_position = "HIGH"
        elif z_score > 1:
            distribution_position = "SLIGHTLY_HIGH"
        elif z_score > -1:
            distribution_position = "NORMAL_RANGE"
        elif z_score > -2:
            distribution_position = "SLIGHTLY_LOW"
        elif z_score > -5:
            distribution_position = "LOW"
        elif z_score > -10:
            distribution_position = "EXTREME_LOW"  # CMG/LII territory
        else:
            distribution_position = "STATISTICAL_CRASH"
        
        # ==================================================================
        # STEP 6: Identify Support and Resistance Levels
        # ==================================================================
        support_levels = []
        resistance_levels = []
        
        if not high_volume_nodes.empty:
            for node_price in high_volume_nodes.index:
                node_price_float = float(node_price)
                if node_price_float < current_price * 0.98:  # At least 2% below
                    support_levels.append(node_price_float)
                elif node_price_float > current_price * 1.02:  # At least 2% above
                    resistance_levels.append(node_price_float)
        
        # Sort levels
        support_levels = sorted(support_levels, reverse=True)  # Nearest first
        resistance_levels = sorted(resistance_levels)  # Nearest first
        
        nearest_support = support_levels[0] if support_levels else None
        nearest_resistance = resistance_levels[0] if resistance_levels else None
        
        # Calculate risk/reward
        if nearest_support and nearest_resistance:
            downside_risk = ((current_price - nearest_support) / current_price) * 100
            upside_potential = ((nearest_resistance - current_price) / current_price) * 100
            risk_reward_ratio = upside_potential / downside_risk if downside_risk > 0 else 0
        else:
            risk_reward_ratio = None
            downside_risk = None
            upside_potential = None
        
        # ==================================================================
        # STEP 7: Determine Trend Strength and Direction
        # ==================================================================
        recent_prices = price_data[-20:] if len(price_data) >= 20 else price_data
        if len(recent_prices) > 0:
            recent_high = max(recent_prices)
            recent_low = min(recent_prices)
            
            if current_price > recent_high * 0.98:
                trend_strength = "STRONG_UPTREND"
            elif current_price < recent_low * 1.02:
                trend_strength = "STRONG_DOWNTREND"
            elif ma_50 > ma_200 and current_price > ma_50:
                trend_strength = "MODERATE_UPTREND"
            elif ma_50 < ma_200 and current_price < ma_50:
                trend_strength = "MODERATE_DOWNTREND"
            else:
                trend_strength = "SIDEWAYS"
        else:
            trend_strength = "UNKNOWN"
        
        # ==================================================================
        # STEP 7B: NEW - Additional Pattern Detection
        # ==================================================================
        try:
            is_parabolic, acceleration_rate = detect_parabolic_move(price_data)
        except:
            is_parabolic, acceleration_rate = False, 0
            
        try:
            has_sweep, sweep_direction = detect_liquidity_sweep(price_data, volume_data)
        except:
            has_sweep, sweep_direction = False, None
            
        try:
            momentum_state = check_trend_momentum(price_data)
        except:
            momentum_state = "UNKNOWN"
        
        # ==================================================================
        # STEP 8: ENHANCED DECISION LOGIC - The Core Algorithm
        # ==================================================================
        bullish_score = 0
        bearish_score = 0
        warning_flags = []
        
        # 1. Score based on MA position
        if ma_position == "ABOVE_BOTH":
            bullish_score += 2
        elif ma_position == "BELOW_BOTH":
            bearish_score += 2
        
        # 2. Score based on crossover
        if crossover_status in ["GOLDEN_CROSS", "GOLDEN_CROSS_RECENT"]:
            bullish_score += 2
            if crossover_status == "GOLDEN_CROSS_RECENT":
                warning_flags.append("RECENT_GOLDEN_CROSS")
        elif crossover_status in ["DEATH_CROSS", "DEATH_CROSS_RECENT"]:
            bearish_score += 2
            if crossover_status == "DEATH_CROSS_RECENT":
                warning_flags.append("RECENT_DEATH_CROSS")
        
        # 3. ENHANCED: Score based on volume profile position
        if volume_context == "AT_VALUE":
            if volume_direction == "AT_VALUE":
                bullish_score += 2  # Sweet spot
            elif volume_direction == "BELOW_VALUE":
                bullish_score += 2  # At support
                warning_flags.append("AT_SUPPORT_DECISION_POINT")
            elif volume_direction == "ABOVE_VALUE":
                bearish_score += 1  # Slightly elevated
                
        elif volume_context == "EXTENDED":
            if volume_direction == "ABOVE_VALUE":
                bearish_score += 4  # Increased from 3 - this is critical!
                warning_flags.append("OVEREXTENDED_ABOVE_VALUE")
                warning_flags.append("THIN_VOLUME_RISK")
            elif volume_direction == "BELOW_VALUE":
                bullish_score += 3  # Increased from 2 - oversold opportunity
                warning_flags.append("OVERSOLD_BELOW_VALUE")
                
        elif volume_context == "EXTREMELY_EXTENDED":  # New category for GOOG/AGX
            if volume_direction == "ABOVE_VALUE":
                bearish_score += 6  # Massive penalty
                warning_flags.append("EXTREMELY_OVEREXTENDED_ABOVE_VALUE")
                warning_flags.append("EXTREME_THIN_VOLUME_RISK")
            elif volume_direction == "BELOW_VALUE":
                bullish_score += 4  # Strong contrarian opportunity
                warning_flags.append("EXTREMELY_OVERSOLD_BELOW_VALUE")
        
        # 4. ENHANCED: Score based on distribution position (graduated)
        if z_score > 10:
            bearish_score += 6
            warning_flags.append(f"STATISTICAL_BUBBLE_{z_score:.1f}σ")
        elif z_score > 5:
            bearish_score += 4
            warning_flags.append(f"EXTREME_HIGH_{z_score:.1f}σ")
        elif z_score > 2:
            bearish_score += 2
            warning_flags.append("STATISTICAL_EXTREME_HIGH")
        elif z_score > 1:
            bearish_score += 0  # Normal range, no penalty
            
        elif z_score < -10:
            bullish_score += 6
            warning_flags.append(f"STATISTICAL_CRASH_{z_score:.1f}σ")
        elif z_score < -5:
            bullish_score += 4
            warning_flags.append(f"EXTREME_LOW_{z_score:.1f}σ")
        elif z_score < -2:
            bullish_score += 2
            warning_flags.append("STATISTICAL_EXTREME_LOW")
        elif z_score < -1:
            bullish_score += 0  # Normal range, no bonus
        
        # 5. ENHANCED: Check extension from moving averages (graduated)
        if percent_above_ma200 > 50:  # GOOG territory
            bearish_score += 4
            warning_flags.append("EXTREMELY_EXTENDED_FROM_MAs")
        elif percent_above_ma200 > 30:  # Very extended
            bearish_score += 3
            warning_flags.append("VERY_EXTENDED_FROM_MAs")
        elif percent_above_ma200 > 20 or percent_above_ma50 > 15:  # Moderately extended
            bearish_score += 2
            warning_flags.append("EXTENDED_FROM_MOVING_AVERAGES")
        elif percent_above_ma200 > 12 or percent_above_ma50 > 10:  # Slightly extended
            bearish_score += 1
            warning_flags.append("SLIGHTLY_EXTENDED")
            
        # Oversold conditions
        elif percent_above_ma200 < -30:  # Severely oversold
            bullish_score += 3
            warning_flags.append("SEVERELY_OVERSOLD_VS_MAs")
        elif percent_above_ma200 < -20 or percent_above_ma50 < -15:
            bullish_score += 1
            warning_flags.append("OVERSOLD_VS_MOVING_AVERAGES")
        
        # 6. NEW: Parabolic move detection
        if is_parabolic:
            if acceleration_rate > 0:
                bearish_score += 3
                warning_flags.append(f"PARABOLIC_ACCELERATION_{acceleration_rate:.1%}")
            else:
                bullish_score += 2  # Parabolic decline exhaustion
                warning_flags.append(f"PARABOLIC_DECLINE_{abs(acceleration_rate):.1%}")
        
        # 7. NEW: Liquidity sweep detection
        if has_sweep:
            if sweep_direction == "UPSIDE_SWEEP":
                bearish_score += 2
                warning_flags.append("UPSIDE_LIQUIDITY_SWEEP")
            elif sweep_direction == "DOWNSIDE_SWEEP":
                bullish_score += 2
                warning_flags.append("DOWNSIDE_LIQUIDITY_SWEEP")
        
        # 8. NEW: Momentum state
        if momentum_state == "ACCELERATING_UP" and volume_context in ["EXTENDED", "EXTREMELY_EXTENDED"]:
            bearish_score += 2
            warning_flags.append("ACCELERATING_INTO_THIN_AIR")
        elif momentum_state == "DECELERATING_DOWN" and volume_context == "AT_VALUE":
            bullish_score += 1
            warning_flags.append("POTENTIAL_BOTTOMING_PROCESS")
        
        # 9. NEW: Combined extreme conditions check (CRITICAL)
        extreme_bubble_conditions = sum([
            percent_from_node > 40,
            z_score > 5,
            percent_above_ma200 > 40,
            is_parabolic and acceleration_rate > 0
        ])
        
        if extreme_bubble_conditions >= 3:
            bearish_score += 5  # Massive penalty
            warning_flags.append("MULTIPLE_BUBBLE_CONDITIONS_EXTREME_RISK")
        
        extreme_oversold_conditions = sum([
            percent_from_node > 20 and current_price < nearest_node_price if nearest_node_price else False,
            z_score < -4,
            percent_above_ma200 < -25
        ])
        
        if extreme_oversold_conditions >= 2:
            bullish_score += 4
            warning_flags.append("MULTIPLE_OVERSOLD_CONDITIONS_MEAN_REVERSION")
        
        # ==================================================================
        # STEP 9: Generate Trading Signal (Enhanced thresholds)
        # ==================================================================
        net_score = bullish_score - bearish_score
        
        if net_score >= 6:
            signal = "STRONG_BUY"
        elif net_score >= 4:
            signal = "BUY"
        elif net_score >= 2:
            signal = "HOLD_WATCH_FOR_BUY"
        elif net_score >= 1:
            signal = "HOLD_NEUTRAL"
        elif net_score <= -6:
            signal = "STRONG_SELL"
        elif net_score <= -4:
            signal = "SELL"
        elif net_score <= -2:
            signal = "HOLD_WATCH_FOR_SELL"
        elif net_score <= -1:
            signal = "HOLD_NEUTRAL"
        else:
            signal = "HOLD_NEUTRAL"
        
        # Override logic for specific dangerous situations
        if "MULTIPLE_BUBBLE_CONDITIONS_EXTREME_RISK" in warning_flags:
            signal = "STRONG_SELL"
            
        if "OVEREXTENDED_ABOVE_VALUE" in warning_flags and "THIN_VOLUME_RISK" in warning_flags:
            if net_score > 0:
                signal = "WAIT_FOR_PULLBACK"
        
        if "AT_SUPPORT_DECISION_POINT" in warning_flags and bearish_score > bullish_score:
            signal = "WATCH_SUPPORT_BREAK"
        
        # ==================================================================
        # STEP 10: Generate Entry and Exit Recommendations
        # ==================================================================
        recommendations = {}
        
        if signal in ["STRONG_BUY", "BUY", "HOLD_WATCH_FOR_BUY"]:
            if volume_context in ["EXTENDED", "EXTREMELY_EXTENDED"] and volume_direction == "ABOVE_VALUE":
                recommendations = {
                    "action": "WAIT",
                    "ideal_entry": nearest_support,
                    "rationale": "Wait for pullback to high-volume support zone",
                    "alert_price": nearest_support
                }
            else:
                stop_loss = nearest_support * 0.95 if nearest_support else current_price * 0.90
                recommendations = {
                    "action": "BUY" if signal in ["STRONG_BUY", "BUY"] else "WATCH",
                    "entry_price": current_price,
                    "stop_loss": round(stop_loss, 2) if stop_loss else None,
                    "take_profit": nearest_resistance,
                    "position_size": "NORMAL" if signal == "STRONG_BUY" else "SMALL"
                }
        
        elif signal in ["STRONG_SELL", "SELL", "HOLD_WATCH_FOR_SELL"]:
            if volume_context == "AT_VALUE" and volume_direction in ["AT_VALUE", "BELOW_VALUE"]:
                stop_loss = nearest_support * 0.98 if nearest_support else current_price * 0.95
                recommendations = {
                    "action": "HOLD_AND_WATCH",
                    "stop_loss": round(stop_loss, 2) if stop_loss else None,
                    "rationale": "At support - give it a chance to bounce, but use tight stop"
                }
            else:
                trailing_stop = current_price * 0.93
                recommendations = {
                    "action": "REDUCE_OR_EXIT" if signal == "STRONG_SELL" else "TRIM_POSITION",
                    "sell_percentage": 75 if signal == "STRONG_SELL" else 50,
                    "trailing_stop": round(trailing_stop, 2),
                    "rationale": "Downtrend confirmed or overextended, protect capital"
                }
        
        elif signal == "WAIT_FOR_PULLBACK":
            recommendations = {
                "action": "WAIT",
                "watch_levels": support_levels[:3] if support_levels else [],
                "buy_trigger": "Price reaches high-volume support zone",
                "alert_price": nearest_support,
                "rationale": "Extended from value - let it come back to you"
            }
        
        elif signal == "WATCH_SUPPORT_BREAK":
            recommendations = {
                "action": "WATCH_CLOSELY",
                "critical_level": nearest_support,
                "buy_trigger": f"Strong bounce from {nearest_support}",
                "sell_trigger": f"Break below {nearest_support}",
                "rationale": "At critical decision point - wait for direction"
            }
        
        else:  # HOLD_NEUTRAL
            stop_loss = nearest_support * 0.97 if nearest_support else current_price * 0.92
            recommendations = {
                "action": "HOLD",
                "stop_loss": round(stop_loss, 2) if stop_loss else None,
                "take_profit": nearest_resistance,
                "rationale": "No clear signal - maintain position with stops"
            }
        
        # ==================================================================
        # STEP 11: Return Comprehensive Analysis
        # ==================================================================
        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "signal": signal,
            "confidence": min(abs(net_score) / 10.0, 1.0),
            
            "technicals": {
                "ma_50": round(ma_50, 2) if not pd.isna(ma_50) else None,
                "ma_200": round(ma_200, 2) if not pd.isna(ma_200) else None,
                "ma_position": ma_position,
                "crossover_status": crossover_status,
                "trend_strength": trend_strength,
                "percent_above_ma50": round(percent_above_ma50, 2),
                "percent_above_ma200": round(percent_above_ma200, 2)
            },
            
            "volume_analysis": {
                "volume_zone": volume_zone,
                "volume_context": volume_context,
                "volume_direction": volume_direction,
                "nearest_node": nearest_node_price,
                "distance_from_value": round(percent_from_node, 2) if percent_from_node else 0
            },
            
            "distribution": {
                "position": distribution_position,
                "z_score": round(z_score, 2),
                "mean_price": round(weighted_mean, 2),
                "std_dev": round(std_dev, 2)
            },
            
            "pattern_detection": {
                "is_parabolic": is_parabolic,
                "acceleration_rate": round(acceleration_rate, 4) if is_parabolic else None,
                "has_liquidity_sweep": has_sweep,
                "sweep_direction": sweep_direction,
                "momentum_state": momentum_state
            },
            
            "key_levels": {
                "support": [round(s, 2) for s in support_levels[:5]] if support_levels else [],
                "resistance": [round(r, 2) for r in resistance_levels[:5]] if resistance_levels else [],
                "nearest_support": round(nearest_support, 2) if nearest_support else None,
                "nearest_resistance": round(nearest_resistance, 2) if nearest_resistance else None
            },
            
            "risk_reward": {
                "downside_risk": round(downside_risk, 2) if downside_risk else None,
                "upside_potential": round(upside_potential, 2) if upside_potential else None,
                "ratio": round(risk_reward_ratio, 2) if risk_reward_ratio else None
            },
            
            "scores": {
                "bullish": bullish_score,
                "bearish": bearish_score,
                "net": net_score
            },
            
            "warnings": warning_flags,
            "recommendations": recommendations
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
            stock_df = fetch_stock_data(stock_ticker, days=365)
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
                                delta=f"Confidence: {tech_analysis['confidence']:.0%}"
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
                        
                        with col4:
                            st.metric(
                                "Volume Context",
                                tech_analysis['volume_analysis']['volume_context'],
                                delta=f"{tech_analysis['volume_analysis']['volume_direction']}"
                            )
                        
                        # Detailed breakdown
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Technical Details")
                            
                            st.write(f"**50-Day MA:** ${tech_analysis['technicals']['ma_50']:,.2f}")
                            st.write(f"**200-Day MA:** ${tech_analysis['technicals']['ma_200']:,.2f}")
                            st.write(f"**Crossover Status:** {tech_analysis['technicals']['crossover_status']}")
                            st.write(f"**Trend Strength:** {tech_analysis['technicals']['trend_strength']}")
                            
                            st.write(f"**Z-Score:** {tech_analysis['distribution']['z_score']:.2f}σ")
                            st.write(f"**Distribution Position:** {tech_analysis['distribution']['position']}")
                            
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
                        
                        with col1:
                            if tech_analysis['pattern_detection']['is_parabolic']:
                                rate = tech_analysis['pattern_detection']['acceleration_rate']
                                st.warning(f"⚠️ Parabolic Move Detected ({rate:.1%})")
                            else:
                                st.info("✅ No parabolic move")
                        
                        with col2:
                            if tech_analysis['pattern_detection']['has_liquidity_sweep']:
                                direction = tech_analysis['pattern_detection']['sweep_direction']
                                st.warning(f"⚠️ Liquidity Sweep: {direction}")
                            else:
                                st.info("✅ No liquidity sweep")
                        
                        with col3:
                            momentum = tech_analysis['pattern_detection']['momentum_state']
                            st.info(f"Momentum: {momentum}")
                    
                    else:
                        st.error(f"Could not perform technical analysis on {stock_ticker}")
                    
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
                spy_vol_to_peak = spy_row['current_vol']/spy_row['peak_volume_M']
                if spy_std_dev != 0:
                    df_results['relative_std_dev'] = df_results['std_devs_from_mean'] / spy_std_dev
                    df_results['vol_to_peak'] = df_results['current_vol']/df_results['peak_volume_M']
                    df_results['relative_vol_ratio'] = df_results['vol_to_peak']/spy_vol_to_peak 
                else:
                    df_results['relative_std_dev'] = 0
                    df_results['vol_to_peak'] = df_results['current_vol']/df_results['peak_volume_M']
                    df_results['relative_vol_ratio'] = df_results['vol_to_peak']/spy_vol_to_peak 
            else:
                df_results['relative_std_dev'] = 0
                df_results['vol_to_peak'] = df_results['current_vol']/df_results['peak_volume_M']
                df_results['relative_vol_ratio'] = df_results['vol_to_peak']/1.0
            
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
            stock_df = fetch_stock_data(stock, days=365)
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
                'confidence': result['confidence'],
                'ma_position': result['technicals']['ma_position'],
                'trend_strength': result['technicals']['trend_strength'],
                'volume_context': result['volume_analysis']['volume_context'],
                'volume_direction': result['volume_analysis']['volume_direction'],
                'distribution_position': result['distribution']['position'],
                'z_score': result['distribution']['z_score'],
                'bullish_score': result['scores']['bullish'],
                'bearish_score': result['scores']['bearish'],
                'net_score': result['scores']['net'],
                'warnings': ', '.join(result['warnings']) if result['warnings'] else 'None',
                'recommendation': result['recommendations'].get('action', 'N/A') if result['recommendations'] else 'N/A'
            } for result in technical_results])
            
            # Merge with original data
            enhanced_df = df.merge(tech_df, on='stock', how='left')
        else:
            enhanced_df = df
            # Add empty technical columns if no analysis results
            tech_columns = ['signal', 'confidence', 'ma_position', 'trend_strength', 'volume_context', 
                           'volume_direction', 'distribution_position', 'z_score', 'bullish_score', 
                           'bearish_score', 'net_score', 'warnings', 'recommendation']
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
            'stock',                    # Stock symbol
            'Shares',                    # Shares Owned
            'current_price',              # Current Price
            'Average Cost',               # Avg Cost ($)
            'Current_Value',              # Current Value
            'Equity',                     # Cost Value ($)
            'delta',                      # Delta
            'Position_Size_Pct',          # Portfolio %
            'Dollar_Impact_10pct',        # $ Impact (10% move)
            'current_price_2',            # Current Price (repeat)
            'current_vol',                 # Volume
            'price_high_52w',              # 52W High
            'price_low_52w',               # 52W Low
            'pct_from_high',               # pct From High
            'MA_20',                       # 20-Day MA
            'MA_50',                       # 50-Day MA
            'MA_200',                      # 200-Day MA
            'MA_Position',                  # Price vs MAs
            'fitted_mean',                  # Mean
            'std_devs_from_mean',           # Std Devs
            'peak_volume_M',                # Peak Volume
            'peak_price',                   # Price @ Peak Vol
            'current_price_3',              # Current Price (repeat again)
            'relative_std_dev',              # Rel Std Dev
            'relative_vol_ratio',            # Rel Vol Ratio
            'signal',                        # Signal
            'confidence',                    # Confidence
            'ma_position',                   # MA Position
            'trend_strength',                # Trend Strength
            'volume_context',                 # Volume Context
            'volume_direction',               # Volume Direction
            'distribution_position',          # Distribution Position
            'z_score',                        # Z-Score
            'bullish_score',                  # Bullish Score
            'bearish_score',                  # Bearish Score
            'net_score',                      # Net Score
            'warnings',                       # Warnings
            'recommendation'                  # Recommendation
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
            'relative_std_dev': 'Rel Std Dev',
            'relative_vol_ratio': 'Rel Vol Ratio',
            'signal': 'Signal',
            'confidence': 'Confidence',
            'ma_position': 'MA Position',
            'trend_strength': 'Trend Strength',
            'volume_context': 'Volume Context',
            'volume_direction': 'Volume Direction',
            'distribution_position': 'Distribution Position',
            'z_score': 'Z-Score',
            'bullish_score': 'Bullish Score',
            'bearish_score': 'Bearish Score',
            'net_score': 'Net Score',
            'warnings': 'Warnings',
            'recommendation': 'Recommendation'
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
                'Avg Cost ($)'
            ]
            
            for col in columns_to_round:
                if col in formatted_display_df.columns:
                    formatted_display_df[col] = formatted_display_df[col].apply(
                        lambda x: f"{int(round(float(x)))}" if pd.notna(x) and isinstance(x, (int, float)) else
                                 (f"{int(round(float(x)))}" if pd.notna(x) and str(x).replace('.','').isdigit() else x)
                    )
            
            # Format percentage columns with 1 decimal place
            percentage_columns = ['Portfolio %', 'pct From High', 'Confidence']
            for col in percentage_columns:
                if col in formatted_display_df.columns:
                    formatted_display_df[col] = formatted_display_df[col].apply(
                        lambda x: f"{float(x):.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else
                                 (f"{float(x):.1f}%" if pd.notna(x) and str(x).replace('.','').isdigit() else x)
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
            st.subheader("📈 Enhanced Technical Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                strong_buys = len(enhanced_df[enhanced_df['signal'] == 'STRONG_BUY'])
                buys = len(enhanced_df[enhanced_df['signal'] == 'BUY'])
                st.metric("Bullish Signals", f"{strong_buys + buys}")
            
            with col2:
                strong_sells = len(enhanced_df[enhanced_df['signal'] == 'STRONG_SELL'])
                sells = len(enhanced_df[enhanced_df['signal'] == 'SELL'])
                st.metric("Bearish Signals", f"{strong_sells + sells}")
            
            with col3:
                wait_signals = len(enhanced_df[enhanced_df['signal'].str.contains('WAIT', na=False)])
                st.metric("Wait Signals", wait_signals)
            
            with col4:
                avg_confidence = enhanced_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            
            # Display pattern detection summary
            if technical_results:
                parabolic_count = sum(1 for r in technical_results if r['pattern_detection']['is_parabolic'])
                sweep_count = sum(1 for r in technical_results if r['pattern_detection']['has_liquidity_sweep'])
                
                st.markdown("---")
                st.subheader("🔍 Pattern Detection Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Parabolic Moves", parabolic_count)
                with col2:
                    st.metric("Liquidity Sweeps", sweep_count)
                with col3:
                    accelerating = sum(1 for r in technical_results if r['pattern_detection']['momentum_state'] == 'ACCELERATING_UP')
                    st.metric("Accelerating Trends", accelerating)
                with col4:
                    extreme_bubble = sum(1 for r in technical_results if "MULTIPLE_BUBBLE_CONDITIONS" in r['warnings'])
                    st.metric("Extreme Bubble Conditions", extreme_bubble)
        
        # Download enhanced data
        csv = enhanced_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Enhanced Analysis",
            data=csv,
            file_name=f"enhanced_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
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

# HOURLY TABS (Same structure as daily but with hourly data)
with tab5:
    st.header("⏰ Single Stock Hourly Analysis")
    st.markdown("Analyze hourly price distributions for short-term insights")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_ticker_hourly = st.text_input(
            "Enter Stock Ticker (Hourly)",
            value="AAPL",
            help="Enter any valid stock ticker (e.g., AAPL, TSLA, MSFT)",
            key="hourly_ticker"
        ).upper()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button_hourly = st.button("🔍 Analyze Stock (Hourly)")
    
    if analyze_button_hourly and stock_ticker_hourly:
        with st.spinner(f"Fetching hourly data for {stock_ticker_hourly}..."):
            stock_df_hourly = fetch_stock_data_hourly(stock_ticker_hourly, days=30)
            spy_df_hourly = get_cached_spy_data_hourly()
            
            if stock_df_hourly is None:
                st.error(f"Could not fetch hourly data for {stock_ticker_hourly}. Please check the ticker symbol.")
            elif spy_df_hourly is None:
                st.error("Could not fetch SPY hourly data for benchmark comparison.")
            else:
                # Analyze both stocks
                with st.spinner(f"Analyzing {stock_ticker_hourly} (hourly)..."):
                    stock_fig_hourly, stock_metrics_hourly = analyze_distributions_with_valleys(stock_df_hourly, stock_ticker_hourly)
                
                with st.spinner("Analyzing SPY hourly benchmark..."):
                    spy_fig_hourly, spy_metrics_hourly = analyze_distributions_with_valleys(spy_df_hourly, "SPY")
                
                if stock_fig_hourly is None or spy_fig_hourly is None:
                    st.error("Error generating hourly analysis. Please try again.")
                else:
                    # Display metrics
                    st.markdown("---")
                    st.subheader("📊 Key Metrics Comparison (Hourly)")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            f"{stock_ticker_hourly} Current Price",
                            f"${stock_metrics_hourly['current_price']:.2f}",
                            delta=f"{stock_metrics_hourly['std_devs_from_mean']:.2f}σ from mean"
                        )
                    
                    with col2:
                        st.metric(
                            f"{stock_ticker_hourly} Peak Price",
                            f"${stock_metrics_hourly['peak_price']:.0f}",
                            delta=f"{stock_metrics_hourly['peak_volume_M']:.1f}M vol"
                        )
                    
                    with col3:
                        st.metric(
                            "SPY Current Price",
                            f"${spy_metrics_hourly['current_price']:.2f}",
                            delta=f"{spy_metrics_hourly['std_devs_from_mean']:.2f}σ from mean"
                        )
                    
                    with col4:
                        if spy_metrics_hourly['std_devs_from_mean'] != 0:
                            rel_volatility = stock_metrics_hourly['std_devs_from_mean'] / spy_metrics_hourly['std_devs_from_mean']
                            st.metric(
                                "Relative Volatility",
                                f"{rel_volatility:.2f}x",
                                delta="vs SPY"
                            )
                    
                    # Display charts
                    st.markdown("---")
                    st.subheader(f"📊 {stock_ticker_hourly} Hourly Distribution Analysis")
                    st.pyplot(stock_fig_hourly)
                    
                    # NEW: Add price-volume chart for hourly data
                    st.markdown("---")
                    st.subheader(f"📈 {stock_ticker_hourly} Hourly Price Trend with Volume")
                    price_volume_fig_hourly = create_price_volume_chart(stock_df_hourly, stock_ticker_hourly)
                    if price_volume_fig_hourly:
                        st.pyplot(price_volume_fig_hourly)
                    else:
                        st.error("Could not generate hourly price-volume chart")
                    
                    st.markdown("---")
                    st.subheader("📊 SPY (S&P 500) Hourly Distribution Analysis - Benchmark")
                    st.pyplot(spy_fig_hourly)

with tab6:
    st.header("🔍 Hourly Batch Stock Scanner")
    st.markdown("Analyze multiple stocks using hourly data for short-term opportunities")
    
    # Stock selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_selection_hourly = st.multiselect(
            "Select stocks to analyze (or leave empty to analyze all)",
            options=STOCK_SYMBOLS,
            default=[],
            help="Select specific stocks or leave empty to analyze all stocks in the list",
            key="hourly_multiselect"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_batch_hourly = st.button("🚀 Run Hourly Batch Analysis")
    
    if run_batch_hourly:
        stocks_to_analyze_hourly = stock_selection_hourly if stock_selection_hourly else STOCK_SYMBOLS
        
        st.info(f"Analyzing {len(stocks_to_analyze_hourly)} stocks with hourly data...")
        progress_bar_hourly = st.progress(0)
        
        # Batch analyze
        results_hourly = batch_analyze_stocks(stocks_to_analyze_hourly, is_hourly=True, progress_bar=progress_bar_hourly)
        progress_bar_hourly.empty()
        
        if results_hourly:
            # Convert to DataFrame
            df_results_hourly = pd.DataFrame(results_hourly)
            
            # Add SPY relative metrics if SPY is in results
            if 'SPY' in df_results_hourly['stock'].values:
                spy_row_hourly = df_results_hourly[df_results_hourly['stock'] == 'SPY'].iloc[0]
                spy_std_dev_hourly = spy_row_hourly['std_devs_from_mean']
                spy_vol_to_peak_hourly = spy_row_hourly['current_vol']/spy_row_hourly['peak_volume_M']
                if spy_std_dev_hourly != 0:
                    df_results_hourly['relative_std_dev'] = df_results_hourly['std_devs_from_mean'] / spy_std_dev_hourly
                    df_results_hourly['vol_to_peak'] = df_results_hourly['current_vol']/df_results_hourly['peak_volume_M']
                    df_results_hourly['relative_vol_ratio'] = df_results_hourly['vol_to_peak']/spy_vol_to_peak_hourly 
                else:
                    df_results_hourly['relative_std_dev'] = 0
                    df_results_hourly['vol_to_peak'] = df_results_hourly['current_vol']/df_results_hourly['peak_volume_M']
                    df_results_hourly['relative_vol_ratio'] = df_results_hourly['vol_to_peak']/spy_vol_to_peak_hourly 
            else:
                df_results_hourly['relative_std_dev'] = 0
                df_results_hourly['vol_to_peak'] = df_results_hourly['current_vol']/df_results_hourly['peak_volume_M']
                df_results_hourly['relative_vol_ratio'] = df_results_hourly['vol_to_peak']/1.0
            
            # Sort by std_devs_from_mean
            df_results_hourly = df_results_hourly.sort_values('std_devs_from_mean')
            
            # Save to CSV
            csv_filename_hourly = f"{DATA_DIR_HOURLY}/stock_report_hourly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_results_hourly.to_csv(csv_filename_hourly, index=False)
            
            st.success(f"✅ Hourly analysis complete! Analyzed {len(results_hourly)} stocks.")
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                undervalued_hourly = len(df_results_hourly[df_results_hourly['std_devs_from_mean'] < -1])
                st.metric("Undervalued (< -1σ)", undervalued_hourly)
            
            with col2:
                overvalued_hourly = len(df_results_hourly[df_results_hourly['std_devs_from_mean'] > 1])
                st.metric("Overvalued (> +1σ)", overvalued_hourly)
            
            with col3:
                avg_std_dev_hourly = df_results_hourly['std_devs_from_mean'].mean()
                st.metric("Avg Std Dev", f"{avg_std_dev_hourly:.2f}σ")
            
            with col4:
                total_analyzed_hourly = len(df_results_hourly)
                st.metric("Total Analyzed", total_analyzed_hourly)
            
            # Display top/bottom performers
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔻 Most Undervalued (Hourly)")
                undervalued_stocks_hourly = df_results_hourly.head(10)[['stock', 'current_price', 'std_devs_from_mean', 'peak_price']]
                st.dataframe(undervalued_stocks_hourly)
            
            with col2:
                st.subheader("🔺 Most Overvalued (Hourly)")
                overvalued_stocks_hourly = df_results_hourly.tail(10)[['stock', 'current_price', 'std_devs_from_mean', 'peak_price']]
                st.dataframe(overvalued_stocks_hourly)
            
            # Download results
            csv_hourly = df_results_hourly.to_csv(index=False)
            st.download_button(
                label="📥 Download Hourly Analysis Results",
                data=csv_hourly,
                file_name=f"stock_analysis_hourly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.error("No hourly results generated. Please check your stock symbols and try again.")

with tab7:
    st.header("📈 Hourly Dashboard")
    st.markdown("View and filter previous hourly analysis reports")
    
    # Find existing reports
    report_files_hourly = glob.glob(f"{DATA_DIR_HOURLY}/*.csv")
    
    if report_files_hourly:
        # Select report
        latest_report_hourly = max(report_files_hourly)
        selected_report_hourly = st.selectbox(
            "Select Hourly Report",
            options=sorted(report_files_hourly, reverse=True),
            format_func=lambda x: os.path.basename(x),
            key="hourly_report_select"
        )
        
        # Load report
        df_hourly = pd.read_csv(selected_report_hourly)
        
        # Display update time
        file_mod_time_hourly = datetime.fromtimestamp(os.path.getmtime(selected_report_hourly))
        st.info(f"📅 Hourly Report Generated: {file_mod_time_hourly.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Filter options
        st.subheader("🎯 Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_std_hourly = st.slider("Min Std Devs", -3.0, 3.0, -3.0, 0.1, key="hourly_min_std")
            max_std_hourly = st.slider("Max Std Devs", -3.0, 3.0, 3.0, 0.1, key="hourly_max_std")
        
        with col2:
            selected_stocks_hourly = st.multiselect(
                "Filter Stocks",
                options=df_hourly['stock'].unique(),
                default=[],
                key="hourly_stock_filter"
            )
        
        with col3:
            sort_by_hourly = st.selectbox(
                "Sort By",
                options=['std_devs_from_mean', 'current_price', 'peak_price', 'relative_std_dev'],
                format_func=lambda x: x.replace('_', ' ').title(),
                key="hourly_sort"
            )
            sort_ascending_hourly = st.checkbox("Ascending", value=True, key="hourly_ascending")
        
        # Apply filters
        filtered_df_hourly = df_hourly[
            (df_hourly['std_devs_from_mean'] >= min_std_hourly) & 
            (df_hourly['std_devs_from_mean'] <= max_std_hourly)
        ]
        
        if selected_stocks_hourly:
            filtered_df_hourly = filtered_df_hourly[filtered_df_hourly['stock'].isin(selected_stocks_hourly)]
        
        filtered_df_hourly = filtered_df_hourly.sort_values(sort_by_hourly, ascending=sort_ascending_hourly)
        
        # Display metrics
        st.markdown("---")
        st.subheader("📊 Summary (Hourly)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered Stocks", len(filtered_df_hourly))
        
        with col2:
            avg_std_hourly = filtered_df_hourly['std_devs_from_mean'].mean()
            st.metric("Avg Std Dev", f"{avg_std_hourly:.2f}σ")
        
        with col3:
            st.metric("Min Std Dev", f"{filtered_df_hourly['std_devs_from_mean'].min():.2f}σ")
        
        with col4:
            st.metric("Max Std Dev", f"{filtered_df_hourly['std_devs_from_mean'].max():.2f}σ")
        
        # Display table with sorting capability
        st.markdown("---")
        st.subheader("📋 Hourly Analysis Results (Click column headers to sort)")
        
        display_cols_hourly = ['stock', 'current_price', 'std_devs_from_mean', 'peak_price', 'peak_volume_M', 
                              'fitted_std', 'relative_std_dev', 'vol_to_peak', 'relative_vol_ratio']
        available_cols_hourly = [col for col in display_cols_hourly if col in filtered_df_hourly.columns]
        
        display_df_hourly = filtered_df_hourly[available_cols_hourly].copy()
        
        # Rename columns for better display
        rename_map_hourly = {
            'stock': 'Stock',
            'current_price': 'Current Price ($)',
            'std_devs_from_mean': 'Std Devs (σ)',
            'peak_price': 'Peak Price ($)',
            'peak_volume_M': 'Peak Vol (M)',
            'fitted_std': 'Fitted Std ($)',
            'relative_std_dev': 'Rel Std Dev',
            'vol_to_peak': 'Vol to Peak',
            'relative_vol_ratio': 'Rel Vol Ratio'
        }
        
        display_df_hourly = display_df_hourly.rename(columns={k: v for k, v in rename_map_hourly.items() if k in display_df_hourly.columns})


        
        # Display sortable dataframe
        sorted_hourly_df = display_sortable_dataframe(display_df_hourly, key_prefix="hourly_dashboard")
        
        # Download filtered results
        csv_hourly_filtered = filtered_df_hourly.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Hourly Results",
            data=csv_hourly_filtered,
            file_name=f"filtered_analysis_hourly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No hourly analysis reports found. Run a batch analysis first.")

with tab8:
    st.header("⚙️ Hourly Data Management")
    st.markdown("Manage cached hourly stock data and analysis results")
    
    # Data management options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cached Hourly Data")
        
        # Clear cache
        if st.button("🗑️ Clear All Hourly Cache"):
            st.session_state.spy_data_hourly = None
            st.session_state.spy_last_updated_hourly = None
            st.success("✅ Hourly cache cleared!")
        
        # Cache info
        if st.session_state.spy_data_hourly is not None:
            st.info(f"📅 SPY hourly data last updated: {st.session_state.spy_last_updated_hourly}")
            st.metric("Cached SPY Hourly Records", len(st.session_state.spy_data_hourly))
        else:
            st.warning("No SPY hourly data cached")
    
    with col2:
        st.subheader("📁 Hourly Analysis Reports")
        
        # List reports
        report_files_hourly = glob.glob(f"{DATA_DIR_HOURLY}/*.csv")
        if report_files_hourly:
            st.write(f"Found {len(report_files_hourly)} hourly reports")
            
            # Delete old reports
            if st.button("🗑️ Delete Hourly Reports > 7 days"):
                cutoff_time_hourly = datetime.now() - timedelta(days=7)
                deleted_count_hourly = 0
                
                for report in report_files_hourly:
                    if datetime.fromtimestamp(os.path.getmtime(report)) < cutoff_time_hourly:
                        os.remove(report)
                        deleted_count_hourly += 1
                
                st.success(f"✅ Deleted {deleted_count_hourly} old hourly reports")
        else:
            st.warning("No hourly analysis reports found")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Stock Distribution Analyzer Pro | Built with Streamlit & Yahoo Finance"
    "</div>",
    unsafe_allow_html=True
)
