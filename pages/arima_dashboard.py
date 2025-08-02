# ---
# Imports
# ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import io
warnings.filterwarnings('ignore')

# ---
# Page Configuration
# ---
st.set_page_config(
    page_title="ARIMA Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ARIMA Time Series Forecasting Dashboard")
st.markdown("Upload your dataset, select a channel, and generate ARIMA forecasts with validation metrics.")

# ---
# Data Processing Functions
# ---
@st.cache_data
def load_and_prepare_data(uploaded_file, channel_name):
    """Load daily data and filter for selected channel"""
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Try different encoding and parsing options
        try:
            df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False, encoding='latin-1')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8-sig')
        
        # Debug: Print column info
        st.write(f"**Debug Info:** Dataset has {len(df)} rows and columns: {list(df.columns)}")
        st.write(f"**Available channels:** {sorted(df['channel_display_name'].unique())}")
        st.write(f"**Sample dates from file:** {df['date'].head().tolist()}")
        
        # Convert date column to datetime - handle multiple formats
        original_count = len(df)
        
        if df['date'].dtype == 'object':
            # Try MM/DD/YYYY format first
            df_test = df.copy()
            df_test['date'] = pd.to_datetime(df_test['date'], format='%m/%d/%Y', errors='coerce')
            valid_mm_dd_yyyy = df_test['date'].notna().sum()
            
            # Try YYYYMMDD format
            df_test2 = df.copy()
            df_test2['date'] = pd.to_datetime(df_test2['date'], format='%Y%m%d', errors='coerce')
            valid_yyyymmdd = df_test2['date'].notna().sum()
            
            # Use the format that gives more valid dates
            if valid_mm_dd_yyyy > valid_yyyymmdd:
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
                st.write(f"**Using MM/DD/YYYY format:** {valid_mm_dd_yyyy}/{original_count} valid dates")
            else:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                st.write(f"**Using YYYYMMDD format:** {valid_yyyymmdd}/{original_count} valid dates")
        else:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            valid_dates = df['date'].notna().sum()
            st.write(f"**Date conversion:** {valid_dates}/{original_count} valid dates")
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Filter for selected channel
        channel_df = df[df['channel_display_name'] == channel_name].copy()
        
        if channel_df.empty:
            return None
        
        st.write(f"**Found {len(channel_df)} records for channel '{channel_name}'**")
        
        # Group by date and sum views for each day
        daily_views = channel_df.groupby('date')['views'].sum().reset_index()
        daily_views = daily_views.sort_values('date')
        
        # Check if we have valid dates
        if daily_views.empty or daily_views['date'].isna().all():
            st.error("No valid dates found in the data")
            return None
            
        # Create complete date range to ensure no missing dates
        start_date = daily_views['date'].min()
        end_date = daily_views['date'].max()
        
        if pd.isna(start_date) or pd.isna(end_date):
            st.error("Invalid date range detected")
            return None
            
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        daily_views = daily_views.set_index('date').reindex(full_date_range, fill_value=0)
        daily_views.index.name = 'date'
        daily_views = daily_views.reset_index()
        
        return daily_views
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.write("**Error details:**", e)
        return None

def check_stationarity(ts_data, alpha=0.05):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(ts_data.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= alpha
    }

def find_best_arima(train_data, max_p=5, max_d=2, max_q=5):
    """Find best ARIMA parameters using AIC - searches within the specified ranges"""
    print(f"Finding best ARIMA parameters with ranges: p(0-{max_p}), d(0-{max_d}), q(0-{max_q})")
    
    best_aic = float('inf')
    best_params = None
    best_model = None
    
    progress_bar = st.progress(0)
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    current_combination = 0
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except:
                    continue
                
                current_combination += 1
                progress_bar.progress(current_combination / total_combinations)
    
    progress_bar.empty()
    
    print(f"Best ARIMA parameters: {best_params}")
    print(f"Best AIC: {best_aic:.2f}")
    
    return best_model, best_params

def build_arima_model(train_data, params):
    """Build ARIMA model with given parameters"""
    model = ARIMA(train_data, order=params)
    fitted_model = model.fit()
    return fitted_model

def calculate_metrics(actual, predicted):
    """Calculate validation metrics"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return rmse, mae, mape

def make_predictions(model, n_periods=7):
    """Generate forecasts for next n periods"""
    forecast = model.forecast(steps=n_periods)
    conf_int = model.get_forecast(steps=n_periods).conf_int()
    
    return forecast, conf_int

# ---
# Visualization Functions
# ---
def plot_series_analysis(data, channel_name):
    """Plot original series, ACF, and PACF for analysis"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original series
    axes[0].plot(data['date'], data['views'])
    axes[0].set_title(f'{channel_name} Daily Views')
    axes[0].set_ylabel('Views')
    axes[0].tick_params(axis='x', rotation=45)
    
    # ACF plot
    plot_acf(data['views'].dropna(), ax=axes[1], lags=30)
    axes[1].set_title('Autocorrelation Function (ACF)')
    
    # PACF plot
    plot_pacf(data['views'].dropna(), ax=axes[2], lags=30)
    axes[2].set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    return fig

def plot_predictions(train_data, test_data, predictions, conf_int, test_dates, channel_name):
    """Plot actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot training data (last 30 days for context)
    train_plot_data = train_data[-30:]
    train_dates = test_dates.iloc[0] - pd.Timedelta(days=len(train_plot_data))
    train_plot_dates = pd.date_range(start=train_dates, periods=len(train_plot_data))
    
    ax.plot(train_plot_dates, train_plot_data, label='Training Data (Last 30 days)', 
            color='blue', alpha=0.7)
    
    # Plot actual test data
    ax.plot(test_dates, test_data, label='Actual', color='red', 
            marker='o', linewidth=2, markersize=6)
    
    # Plot predictions
    ax.plot(test_dates, predictions, label='Predicted', color='green', 
            marker='s', linewidth=2, markersize=6)
    
    # Plot confidence intervals
    ax.fill_between(test_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                    color='green', alpha=0.2, label='95% Confidence Interval')
    
    ax.set_title(f'{channel_name} Viewership: Actual vs Predicted ({len(test_data)} Days)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Views', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ---
# Streamlit App Layout
# ---
def main():
    # Sidebar for inputs
    st.sidebar.header("📊 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload your daily viewership dataset (CSV format)"
    )
    
    if uploaded_file is not None:
        # Load data to get channel list
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Try different encodings for channel list
            try:
                df_sample = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df_sample = pd.read_csv(uploaded_file, low_memory=False, encoding='latin-1')
            except:
                uploaded_file.seek(0)
                df_sample = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8-sig')
            
            # Show basic file info
            st.sidebar.info(f"Loaded {len(df_sample):,} rows")
            
            # Get unique channels
            if 'channel_display_name' in df_sample.columns:
                channels = sorted(df_sample['channel_display_name'].unique())
                st.sidebar.success(f"Found {len(channels)} channels")
            else:
                st.sidebar.error("Column 'channel_display_name' not found!")
                st.sidebar.write("Available columns:", list(df_sample.columns))
                return
            
            # Channel selection
            selected_channel = st.sidebar.selectbox(
                "Select Channel",
                channels,
                help="Choose which channel to analyze"
            )
            
            # Model parameters
            st.sidebar.subheader("🔧 Model Parameters")
            test_days = st.sidebar.slider("Test Period (days)", 3, 14, 7)
            max_p = st.sidebar.slider("Max AR terms (p)", 1, 8, 5)
            max_d = st.sidebar.slider("Max Differencing (d)", 0, 3, 2)
            max_q = st.sidebar.slider("Max MA terms (q)", 1, 8, 5)
            
            # Run analysis button
            if st.sidebar.button("🚀 Run ARIMA Analysis", type="primary"):
                with st.spinner("Processing data and building model..."):
                    # Load and prepare data
                    daily_data = load_and_prepare_data(uploaded_file, selected_channel)
                    
                    if daily_data is None:
                        st.error(f"No data found for channel: {selected_channel}")
                        return
                    
                    # Display data info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Days", len(daily_data))
                    with col2:
                        st.metric("Date Range", f"{daily_data['date'].min().strftime('%Y-%m-%d')} to {daily_data['date'].max().strftime('%Y-%m-%d')}")
                    with col3:
                        st.metric("Total Views", f"{daily_data['views'].sum():,}")
                    
                    # Split data
                    train_data = daily_data['views'][:-test_days]
                    test_data = daily_data['views'][-test_days:]
                    test_dates = daily_data['date'][-test_days:]
                    
                    # Main content area
                    tab1, tab2, tab3 = st.tabs(["📈 Predictions", "📊 Analysis", "📋 Detailed Results"])
                    
                    with tab1:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Check stationarity
                            stationarity_results = check_stationarity(train_data)
                            
                            # Find best ARIMA parameters
                            st.info("Finding optimal ARIMA parameters...")
                            model, best_params = find_best_arima(train_data, max_p, max_d, max_q)
                            best_aic = model.aic
                            
                            # Make predictions
                            predictions, conf_int = make_predictions(model, n_periods=test_days)
                            
                            # Calculate metrics
                            rmse, mae, mape = calculate_metrics(test_data, predictions)
                            
                            # Plot predictions
                            fig_pred = plot_predictions(train_data, test_data, predictions, conf_int, test_dates, selected_channel)
                            st.pyplot(fig_pred)
                        
                        with col2:
                            st.subheader("🎯 Model Performance")
                            
                            # Model info
                            st.info(f"**Best ARIMA Parameters:** {best_params}")
                            st.info(f"**AIC Score:** {best_aic:.2f}")
                            
                            # Stationarity test
                            if stationarity_results['is_stationary']:
                                st.success("✅ Series is stationary")
                            else:
                                st.warning("⚠️ Series is non-stationary")
                            
                            st.metric("ADF p-value", f"{stationarity_results['p_value']:.6f}")
                            
                            # Validation metrics
                            st.subheader("📊 Validation Metrics")
                            st.metric("RMSE", f"{rmse:.2f}")
                            st.metric("MAE", f"{mae:.2f}")
                            st.metric("MAPE", f"{mape:.2f}%")
                            
                            # Model quality indicator
                            if mape < 10:
                                st.success("🎯 Excellent forecast accuracy!")
                            elif mape < 20:
                                st.info("👍 Good forecast accuracy")
                            elif mape < 30:
                                st.warning("⚠️ Moderate forecast accuracy")
                            else:
                                st.error("❌ Poor forecast accuracy")
                    
                    with tab2:
                        st.subheader("📊 Time Series Analysis")
                        fig_analysis = plot_series_analysis(daily_data, selected_channel)
                        st.pyplot(fig_analysis)
                        
                        # Additional statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📈 Data Statistics")
                            st.write(daily_data['views'].describe())
                        
                        with col2:
                            st.subheader("🔍 Stationarity Details")
                            st.write(f"**ADF Statistic:** {stationarity_results['adf_statistic']:.6f}")
                            st.write(f"**p-value:** {stationarity_results['p_value']:.6f}")
                            st.write("**Critical Values:**")
                            for key, value in stationarity_results['critical_values'].items():
                                st.write(f"  {key}: {value:.3f}")
                    
                    with tab3:
                        st.subheader("📋 Detailed Prediction Comparison")
                        
                        # Create detailed comparison dataframe
                        comparison_df = pd.DataFrame({
                            'Date': test_dates.dt.strftime('%Y-%m-%d'),
                            'Actual': test_data.values,
                            'Predicted': predictions.values,
                            'Difference': test_data.values - predictions.values,
                            'Absolute Error': np.abs(test_data.values - predictions.values),
                            'Percentage Error': np.abs((test_data.values - predictions.values) / test_data.values) * 100
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Download results
                        csv = comparison_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"arima_results_{selected_channel.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                        # Model summary
                        st.subheader("🔧 Model Summary")
                        st.text(str(model.summary()))
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show sample data format
        st.subheader("📄 Expected Data Format")
        sample_data = pd.DataFrame({
            'date': ['1/1/2024', '1/2/2024', '1/3/2024'],
            'channel_display_name': ['Shark Tank Global', 'Shark Tank Global', 'Shark Tank Global'],
            'views': [10000, 12000, 9500]
        })
        st.dataframe(sample_data)
        st.caption("Your CSV should contain columns: 'date' (MM/DD/YYYY format), 'channel_display_name', and 'views'")

if __name__ == "__main__":
    main() 