import streamlit as st
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data_group = f['df']
        data = data_group['block0_values'][:]
        timestamps = data_group['axis1'][:]
    
    timestamps = pd.to_datetime(timestamps, unit="ns")
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = "timestamp"
    return df

# ---------------- Create Dataset ----------------
def create_dataset(series, seq_length=12):
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])
    return np.array(X), np.array(y)

# ---------------- Main App ----------------
st.title("🚦 AI-Driven Traffic Flow Optimization")
st.write("Using **XGBoost & LSTM** to predict traffic flow from METR-LA dataset.")

try:
    # Load data
    df = load_data("METR-LA.h5")
    st.success("✅ Data loaded successfully!")
    
    # Display basic info
    st.write("📊 **Dataset Info:**")
    st.write(f"- Shape: {df.shape}")
    st.write(f"- Date range: {df.index.min()} to {df.index.max()}")
    st.write(f"- Number of sensors: {df.shape[1]}")
    
    # ---------------- User Inputs ----------------
    st.sidebar.header("⚙️ Configuration")
    sensor_id = st.sidebar.selectbox("Select Sensor ID", range(df.shape[1]), index=0)
    sequence_length = st.sidebar.slider("Sequence Length (minutes)", 6, 24, 12)
    
    # ---------------- Data Preparation ----------------
    traffic_series = df.iloc[:, sensor_id].values
    
    # Handle NaN values
    traffic_series = pd.Series(traffic_series).fillna(method='ffill').fillna(method='bfill').values
    
    # Scale data
    scaler = StandardScaler()
    traffic_scaled = scaler.fit_transform(traffic_series.reshape(-1, 1))
    
    # Create dataset
    X, y = create_dataset(traffic_scaled, seq_length=sequence_length)
    
    if len(X) > 0:
        # Train-test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        st.write(f"📈 **Training samples:** {len(X_train)}")
        st.write(f"🧪 **Testing samples:** {len(X_test)}")
        
        # ---------------- XGBoost Model ----------------
        with st.spinner("🔄 Training XGBoost model..."):
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
            xgb_model.fit(X_train_flat, y_train.ravel())
            
            y_pred_xgb = xgb_model.predict(X_test_flat)
            mse_xgb = mean_squared_error(y_test, y_pred_xgb.reshape(-1, 1))
        
        # ---------------- LSTM Model ----------------
        with st.spinner("🔄 Training LSTM model..."):
            lstm_model = Sequential([
                LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            
            # Train with progress bar
            progress_bar = st.progress(0)
            for epoch in range(10):
                lstm_model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)
                progress_bar.progress((epoch + 1) / 10)
            
            y_pred_lstm = lstm_model.predict(X_test)
            mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        
        # ---------------- Results ----------------
        col1, col2 = st.columns(2)
        with col1:
            st.metric("XGBoost MSE", f"{mse_xgb:.4f}")
        with col2:
            st.metric("LSTM MSE", f"{mse_lstm:.4f}")
        
        # ---------------- Visualization ----------------
        st.subheader("📈 Traffic Flow Prediction")
        
        # Create sample visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Show first 200 predictions
        sample_size = min(200, len(y_test))
        ax.plot(y_test[:sample_size], label="Actual", linewidth=2, color='blue')
        ax.plot(y_pred_xgb[:sample_size], label="XGBoost", alpha=0.7, color='red')
        ax.plot(y_pred_lstm[:sample_size], label="LSTM", alpha=0.7, color='green')
        
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Traffic Speed (Normalized)")
        ax.set_title(f"Traffic Flow Prediction - Sensor {sensor_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # ---------------- Data Preview ----------------
        st.subheader("📊 Data Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Traffic Data:**")
            recent_data = pd.DataFrame({
                'Timestamp': df.index[-100:],
                'Speed': df.iloc[-100:, sensor_id]
            })
            st.dataframe(recent_data.tail(10))
        
        with col2:
            st.write("**Statistics:**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Std', 'Min', 'Max'],
                'Value': [
                    traffic_series.mean(),
                    traffic_series.std(),
                    traffic_series.min(),
                    traffic_series.max()
                ]
            })
            st.dataframe(stats_df)
            
    else:
        st.error("❌ Not enough data to create sequences. Try reducing sequence length.")

except FileNotFoundError:
    st.error("❌ METR-LA.h5 file not found. Please ensure the file is in the same directory.")
    st.info("📁 Current directory contents:")
    import os
    st.write(os.listdir('.'))
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
