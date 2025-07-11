import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import io

# Set page config
st.set_page_config(page_title="Grameenphone Stock Prediction", layout="wide")

# Title and description
st.title("📈 Grameenphone Stock Price Prediction")
st.markdown("""
This application predicts Grameenphone stock prices using a RandomForest model.
Explore historical data, view predictions, and analyze model performance.
""")

# Load data and models
@st.cache_data
def load_data_and_models():
    try:
        df = pd.read_csv('models/processed_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        rf_model = joblib.load('models/rf_model.pkl')
        scaler_X = joblib.load('models/scaler_X.pkl')
        scaler_y = joblib.load('models/scaler_y.pkl')
        return df, rf_model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        return None, None, None, None

df, rf_model, scaler_X, scaler_y = load_data_and_models()
if df is None:
    st.stop()

# Feature columns
feature_cols = ['Lag1', 'Lag2', 'Lag3', 'Lag1_Vol', 'Lag1_Return',
                'Lag1_Open', 'Lag1_High', 'Lag1_Low', 'Lag1_Change',
                'RollingMean7', 'RollingStd7', 'RollingMean14', 'RollingStd14',
                'Volatility']

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select Option", [
    "Historical Data",
    "Model Predictions",
    "Feature Importance",
    "Predict Future Price",
    "Model Performance",
    "Trading Insights"
])

# Historical Data Visualization
if options == "Historical Data":
    st.header("Historical Stock Data")
    date_range = st.slider(
        "Select Date Range",
        min_value=df['Date'].min().to_pydatetime(),
        max_value=df['Date'].max().to_pydatetime(),
        value=(df['Date'].min().to_pydatetime(), df['Date'].max().to_pydatetime()),
        format="YYYY-MM-DD"
    )
    filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
    
    st.subheader("Price Trend")
    fig = px.line(filtered_df, x='Date', y='Price', title='Grameenphone Stock Price')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Volume Trend")
    fig = px.line(filtered_df, x='Date', y='Vol.', title='Trading Volume')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Download Historical Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="grameenphone_historical_data.csv",
        mime="text/csv"
    )

# Model Predictions
elif options == "Model Predictions":
    st.header("Model Predictions")
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)
    test_df = df.iloc[split_index:].copy()
    
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler_X.transform(X_test)
    y_test = test_df['Price'].values
    predictions_scaled = rf_model.predict(X_test_scaled).reshape(-1, 1)
    predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
    
    pred_df = pd.DataFrame({
        'Date': test_df['Date'],
        'Actual Price': y_test,
        'Predicted Price': predictions
    })
    
    st.subheader("Actual vs Predicted Prices")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Actual Price'], mode='lines+markers', name='Actual Price'))
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Price'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
    fig.update_layout(title='Actual vs Predicted Stock Prices', xaxis_title='Date', yaxis_title='Price (BDT)')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Download Predictions")
    csv = pred_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="grameenphone_predictions.csv",
        mime="text/csv"
    )

# Feature Importance
elif options == "Feature Importance":
    st.header("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title='Feature Importance in RandomForest Model')
    st.plotly_chart(fig, use_container_width=True)

# Predict Future Price
elif options == "Predict Future Price":
    st.header("Predict Future Stock Price")
    st.markdown("Enter a date to predict the stock price. The model uses the last available data to generate features.")
    
    max_date = df['Date'].max()
    future_date = st.date_input(
        "Select Prediction Date",
        min_value=max_date + timedelta(days=1),
        max_value=max_date + timedelta(days=30)
    )
    
    if st.button("Predict"):
        # Get the last row to generate features
        last_row = df.iloc[-1].copy()
        last_date = last_row['Date']
        days_diff = (future_date - last_date.date()).days
        
        if days_diff <= 0:
            st.error("Please select a future date.")
        else:
            # Simplified feature generation for one-step prediction
            features = np.array([[
                last_row['Price'],  # Lag1
                last_row['Lag1'],   # Lag2
                last_row['Lag2'],   # Lag3
                last_row['Vol.'],   # Lag1_Vol
                last_row['Change %'],  # Lag1_Return
                last_row['Open'],   # Lag1_Open
                last_row['High'],   # Lag1_High
                last_row['Low'],    # Lag1_Low
                last_row['Change %'],  # Lag1_Change
                df['Price'].tail(7).mean(),  # RollingMean7
                df['Price'].tail(7).std(),   # RollingStd7
                df['Price'].tail(14).mean(), # RollingMean14
                df['Price'].tail(14).std(),  # RollingStd14
                last_row['High'] - last_row['Low']  # Volatility
            ]])
            features_scaled = scaler_X.transform(features)
            pred_scaled = rf_model.predict(features_scaled).reshape(-1, 1)
            pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
            
            st.success(f"Predicted Stock Price for {future_date}: {pred_price:.2f} BDT")

# Model Performance
elif options == "Model Performance":
    st.header("Model Performance Metrics")
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)
    test_df = df.iloc[split_index:].copy()
    
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler_X.transform(X_test)
    y_test = test_df['Price'].values
    predictions_scaled = rf_model.predict(X_test_scaled).reshape(-1, 1)
    predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
    
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        accuracy = (1 - mape) * 100
        return rmse, mae, mape, r2, accuracy
    
    rmse, mae, mape, r2, accuracy = calculate_metrics(y_test, predictions)
    
    st.subheader("Performance Metrics")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MAPE: {mape*100:.2f}%")
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"Accuracy (1 - MAPE): {accuracy:.2f}%")

# Trading Insights
elif options == "Trading Insights":
    st.header("Trading Insights")
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)
    test_df = df.iloc[split_index:].copy()
    
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler_X.transform(X_test)
    predictions_scaled = rf_model.predict(X_test_scaled).reshape(-1, 1)
    predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
    
    threshold = 0.01  # 1% change threshold
    signals = np.where(predictions[1:] > predictions[:-1] * (1 + threshold), 1,
                       np.where(predictions[1:] < predictions[:-1] * (1 - threshold), -1, 0))
    returns = np.diff(test_df['Price'].values) * signals
    transaction_cost = 0.5  # 0.5 BDT per trade
    total_profit = np.sum(returns) - np.sum(np.abs(signals) > 0) * transaction_cost
    
    st.subheader("Simulated Trading Performance")
    st.write(f"Total Profit (with 1% threshold and 0.5 BDT transaction cost): {total_profit:.2f} BDT")
    
    signal_df = pd.DataFrame({
        'Date': test_df['Date'].iloc[1:],
        'Signal': signals,
        'Return': returns
    })
    signal_df['Signal'] = signal_df['Signal'].map({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    
    st.subheader("Trading Signals")
    st.dataframe(signal_df)
    
    st.subheader("Download Trading Signals")
    csv = signal_df.to_csv(index=False)
    st.download_button(
        label="Download Signals CSV",
        data=csv,
        file_name="grameenphone_trading_signals.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit | Data Source: Grameenphone Stock Price History")