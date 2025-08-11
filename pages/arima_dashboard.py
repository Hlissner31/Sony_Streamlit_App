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
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

def find_best_sarima(train_data, max_p=3, max_d=1, max_q=3, max_P=1, max_D=1, max_Q=1, s=7):
    """Find best SARIMA parameters using AIC with a seasonal period s."""
    print(
        f"Finding best SARIMA parameters with ranges: p(0-{max_p}), d(0-{max_d}), q(0-{max_q}); "
        f"P(0-{max_P}), D(0-{max_D}), Q(0-{max_Q}); s={s}"
    )

    best_aic = float('inf')
    best_params = None
    best_seasonal = None
    best_model = None

    progress_bar = st.progress(0)
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1) * (max_P + 1) * (max_D + 1) * (max_Q + 1)
    current_combination = 0

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            try:
                                model = SARIMAX(
                                    train_data,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                fitted_model = model.fit(disp=False)
                                aic = fitted_model.aic

                                if aic < best_aic:
                                    best_aic = aic
                                    best_params = (p, d, q)
                                    best_seasonal = (P, D, Q)
                                    best_model = fitted_model
                            except Exception:
                                pass
                            finally:
                                current_combination += 1
                                progress_bar.progress(current_combination / total_combinations)

    progress_bar.empty()

    print(f"Best SARIMA order: {best_params}, seasonal: {best_seasonal} x {s}")
    print(f"Best AIC: {best_aic:.2f}")

    return best_model, best_params, best_seasonal, s

def build_arima_model(train_data, params):
    """Build ARIMA model with given parameters"""
    model = ARIMA(train_data, order=params)
    fitted_model = model.fit()
    return fitted_model

def build_sarima_model(train_data, params, seasonal_params, s=7):
    """Build SARIMA model with given parameters and seasonal order."""
    model = SARIMAX(
        train_data,
        order=params,
        seasonal_order=(seasonal_params[0], seasonal_params[1], seasonal_params[2], s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
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
# Momentum (Day-over-Day % Change) Utilities
# ---
def compute_momentum_percentages(previous_value, forecast_values):
    """Compute day-over-day percent change for each step relative to the prior day's value.

    The first step compares against previous_value (last known actual). Subsequent steps compare
    to the immediately prior forecast value.
    """
    pct_changes = []
    last_val = float(previous_value)
    for value in list(forecast_values):
        value_float = float(value)
        if last_val == 0:
            pct = np.nan
        else:
            pct = (value_float - last_val) / last_val * 100.0
        pct_changes.append(pct)
        last_val = value_float
    return pd.Series(pct_changes, index=getattr(forecast_values, 'index', None))

def format_momentum(pct):
    """Return a human-friendly momentum string with arrow and percent formatting."""
    if pd.isna(pct):
        return "—"
    arrow = "↑" if pct >= 0 else "↓"
    return f"{arrow} {abs(pct):.2f}%"

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
# Future Forecast Visualization
# ---
def plot_future_forecast(history_dates, history_values, forecast_dates, forecast_values, conf_int, channel_name):
    """Plot recent history and future forecast values with confidence intervals"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot recent history
    ax.plot(history_dates, history_values, label='History', color='blue', linewidth=2)

    # Plot forecast
    ax.plot(forecast_dates, forecast_values, label='Forecast', color='green', marker='s', linewidth=2, markersize=6)

    # Plot confidence intervals
    ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                    color='green', alpha=0.2, label='95% Confidence Interval')

    ax.set_title(f'{channel_name} Future Viewership Forecast', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Views', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

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
            use_seasonality = st.sidebar.toggle("Use SARIMA (weekly seasonality)", value=True, help="Enable seasonal components with period s=7")
            if use_seasonality:
                st.sidebar.caption("Seasonal search ranges (small, fast): P,Q∈{0,1}, D∈{0,1}, s=7")
            
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
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Predictions", "📊 Analysis", "📋 Detailed Results", "🔮 Future Forecast", "ℹ️ Glossary"])
                    
                    with tab1:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Check stationarity
                            stationarity_results = check_stationarity(train_data)
                            
                            # Find best parameters (ARIMA or SARIMA)
                            if use_seasonality:
                                st.info("Finding optimal SARIMA parameters (weekly seasonality)...")
                                model, best_params, best_seasonal, seasonal_period = find_best_sarima(
                                    train_data, max_p=max_p, max_d=max_d, max_q=max_q,
                                    max_P=1, max_D=1, max_Q=1, s=7
                                )
                                best_aic = model.aic
                            else:
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
                            if use_seasonality:
                                st.info(f"**Best SARIMA Parameters:** order={best_params}, seasonal={(best_seasonal[0], best_seasonal[1], best_seasonal[2], seasonal_period)}")
                                st.info(f"**AIC Score:** {best_aic:.2f}")
                            else:
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
                        # Momentum: percent change day-over-day for actual and predicted
                        actual_prev = daily_data['views'].iloc[-len(test_data)-1]
                        pred_momentum = compute_momentum_percentages(actual_prev, predictions)
                        # For actual momentum, compare each actual to the previous actual (first uses actual_prev)
                        actual_series = pd.Series(test_data.values, index=test_dates)
                        actual_momentum = compute_momentum_percentages(actual_prev, actual_series)

                        comparison_df = pd.DataFrame({
                            'Day': test_dates.dt.day_name(),
                            'Date': test_dates.dt.strftime('%Y-%m-%d'),
                            'Actual': test_data.values,
                            'Predicted': predictions.values,
                            'Difference': test_data.values - predictions.values,
                            'Absolute Error': np.abs(test_data.values - predictions.values),
                            'Percentage Error': np.abs((test_data.values - predictions.values) / test_data.values) * 100,
                            'Actual DoD %': [format_momentum(x) for x in actual_momentum.values],
                            'Predicted DoD %': [format_momentum(x) for x in pred_momentum.values]
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

                    with tab4:
                        st.subheader("🔮 Forecast Future Values")
                        st.caption("Out-of-sample forecast using the same horizon as the Test Period.")

                        # Use the same horizon as selected Test Period
                        forecast_days = test_days

                        # Refit model on full data using the best params
                        if use_seasonality:
                            full_model = build_sarima_model(daily_data['views'], best_params, best_seasonal, s=7)
                        else:
                            full_model = build_arima_model(daily_data['views'], best_params)
                        future_forecast, future_conf_int = make_predictions(full_model, n_periods=forecast_days)

                        # Build future date range
                        last_date = daily_data['date'].max()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')

                        # Plot: show recent history (last 60 days) + forecast
                        history_window = 60 if len(daily_data) > 60 else len(daily_data)
                        history_dates = daily_data['date'][-history_window:]
                        history_values = daily_data['views'][-history_window:]

                        fig_future = plot_future_forecast(history_dates, history_values, future_dates, future_forecast, future_conf_int, selected_channel)
                        st.pyplot(fig_future)

                        # Tabular results
                        st.subheader("📅 Forecast Table")
                        future_df = pd.DataFrame({
                            'Day': future_dates.day_name(),
                            'Date': future_dates.strftime('%Y-%m-%d'),
                            'Forecast': future_forecast.values,
                            'Lower CI': future_conf_int.iloc[:, 0].values,
                            'Upper CI': future_conf_int.iloc[:, 1].values
                        })
                        # Momentum for forecast vs last actual, then sequentially
                        last_actual = daily_data['views'].iloc[-1]
                        future_momentum = compute_momentum_percentages(last_actual, future_forecast)
                        future_df['DoD %'] = [format_momentum(x) for x in future_momentum.values]
                        st.dataframe(future_df, use_container_width=True)

                        # Download button
                        forecast_csv = future_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast as CSV",
                            data=forecast_csv,
                            file_name=f"arima_future_forecast_{selected_channel.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )

                    with tab5:
                        st.subheader("ℹ️ Glossary of Terms")
                        st.markdown(
                            """
                            - **RMSE (Root Mean Squared Error)**: Square root of the average squared differences between actual and predicted values. Penalizes large errors.
                            - **MAE (Mean Absolute Error)**: Average of absolute differences between actual and predicted values. Easy to interpret in the original units.
                            - **MAPE (Mean Absolute Percentage Error)**: Average absolute percentage error relative to actuals. Lower is better; sensitive when actuals are near zero.
                            - **AIC (Akaike Information Criterion)**: Model selection score balancing fit and complexity. Lower AIC indicates a better model within the compared set.
                            - **Stationarity**: A stationary series has constant mean/variance over time. Many models (like ARIMA) assume stationarity.
                            - **ADF Test (Augmented Dickey-Fuller)**: Statistical test for stationarity. A small p-value (≤ 0.05) suggests the series is stationary.
                            - **ARIMA(p, d, q)**: Time series model with:
                              - **p (AR)**: Autoregressive lags — dependence on prior values.
                              - **d (I)**: Differencing order — number of times data is differenced to remove trend.
                              - **q (MA)**: Moving average lags — dependence on past forecast errors.
                            - **SARIMA(p, d, q)(P, D, Q)<sub>s</sub>**: ARIMA with a seasonal component.
                              - **P, D, Q**: Seasonal AR, differencing, and MA orders.
                              - **s (seasonal period)**: Number of time steps per season; here **s = 7** captures day‑of‑week effects.
                              - Example: `(p,d,q)=(1,1,1)` and seasonal `(P,D,Q,s)=(1,0,1,7)` models weekly patterns in daily data.
                            - **ACF (Autocorrelation Function)**: Correlation of a series with its own past values; helps choose MA order (q) and diagnose residual autocorrelation.
                            - **PACF (Partial ACF)**: Correlation of a series with its past values while controlling for intermediate lags; helps choose AR order (p).
                            - **Confidence Interval (95%)**: Range that likely contains the true value 95% of the time under model assumptions.
                            - **Test Period (days)**: The held-out last N days used to validate predictions.
                            - **DoD Momentum (%)**: Day-over-day percent change. First day compares to the last prior value, then each day compares to the previous day.
                            """
                        )
        
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