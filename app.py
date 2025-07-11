import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Streamlit app configuration
st.set_page_config(page_title="Grameenphone Stock Price Forecasting", layout="wide")
st.title('Grameenphone Stock Price Forecasting (Hybrid LSTM+SARIMA)')
st.write('Upload historical stock price data (CSV) and select forecast horizon.')

# Load saved models and scalers
try:
    lstm_model = load_model('hybrid_lstm_model.h5')
    scaler_X = joblib.load('scaler_X.joblib')
    scaler_y = joblib.load('scaler_y.joblib')
    sarima_params = joblib.load('sarima_params.joblib')
except FileNotFoundError as e:
    st.error(f"Error: Missing model or scaler file. Ensure 'hybrid_lstm_model.h5', 'scaler_X.joblib', 'scaler_y.joblib', and 'sarima_params.joblib' are in the directory. {e}")
    st.stop()

# File upload
uploaded_file = st.file_uploader('Upload CSV (Date, Price, Open, High, Low, Vol., Change %)', type='csv')

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%Y')
        if df['Date'].isna().any():
            st.error("Error: Invalid date values in 'Date' column. Please check the CSV.")
            st.stop()
        df = df.sort_values('Date').reset_index(drop=True)

        # Preprocess data (same as training)
        df['Vol.'] = df['Vol.'].apply(lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else 
                                      float(x.replace('M', '')) * 1000000 if isinstance(x, str) and 'M' in x else float(x))
        df['Change %'] = df['Change %'].str.replace('%', '', regex=False).astype(float) / 100
        df = df.fillna(method='ffill')

        # Feature engineering
        df['Lag1'] = df['Price'].shift(1)
        df['Lag2'] = df['Price'].shift(2)
        df['Lag3'] = df['Price'].shift(3)
        df['Lag1_Vol'] = df['Vol.'].shift(1)
        df['Lag1_Return'] = df['Change %'].shift(1)
        df['Lag1_Open'] = df['Open'].shift(1)
        df['Lag1_High'] = df['High'].shift(1)
        df['Lag1_Low'] = df['Low'].shift(1)
        df['Lag1_Change'] = df['Change %'].shift(1)
        df['RollingMean7'] = df['Price'].rolling(window=7).mean()
        df['RollingStd7'] = df['Price'].rolling(window=7).std()
        df['RollingMean14'] = df['Price'].rolling(window=14).mean()
        df['RollingStd14'] = df['Price'].rolling(window=14).std()
        df['RollingStd30'] = df['Price'].rolling(window=30).std()
        df['Volatility'] = df['High'] - df['Low']
        df = df.dropna().reset_index(drop=True)

        # Load selected features
        feature_cols = sarima_params.get('selected_features', [
            'Lag1', 'Lag1_Vol', 'Lag1_Return', 'RollingMean7', 'RollingStd7', 'RollingMean14', 'RollingStd14', 'Volatility'
        ])

        # Verify required columns
        required_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Error: CSV must contain {required_columns}")
            st.stop()

        # Forecast function
        def forecast_hybrid(model, scaler_X, scaler_y, feature_cols, df, time_steps=10, sarima_params=sarima_params, forecast_horizon=30):
            X_all = df[feature_cols].values
            if len(X_all) < time_steps:
                st.error(f"Error: Insufficient data. CSV must have at least {time_steps} rows after preprocessing.")
                return None, None
            X_all_scaled = scaler_X.transform(X_all)
            last_sequence = X_all_scaled[-time_steps:].reshape(1, time_steps, len(feature_cols))
            last_lstm_pred_scaled = model.predict(last_sequence, verbose=0)
            last_lstm_pred = scaler_y.inverse_transform(last_lstm_pred_scaled)[0, 0]
            last_actual = df['Price'].values[-1]
            residual_history = [last_actual - last_lstm_pred] * 7  # Placeholder; ideally load residuals
            predictions = []
            future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
            w_lstm = sarima_params.get('w_lstm', 0.5)
            w_sarima = sarima_params.get('w_sarima', 0.5)

            for _ in range(forecast_horizon):
                lstm_pred_scaled = model.predict(last_sequence, verbose=0)
                lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)[0, 0]
                try:
                    sarima_model = SARIMAX(residual_history, order=sarima_params['order'], seasonal_order=sarima_params['seasonal_order']).fit(disp=False)
                    residual_pred = sarima_model.forecast(steps=1)[0]
                except:
                    residual_pred = 0
                hybrid_pred = w_lstm * lstm_pred + w_sarima * residual_pred
                predictions.append(hybrid_pred)

                new_features = []
                for col in feature_cols:
                    if col == 'Lag1':
                        new_features.append(hybrid_pred)
                    elif col == 'Lag2':
                        new_features.append(df['Price'].values[-1])
                    elif col == 'Lag3':
                        new_features.append(df['Price'].values[-2])
                    elif col == 'Lag1_Vol':
                        new_features.append(df['Vol.'].values[-1])
                    elif col == 'Lag1_Return':
                        new_features.append(df['Change %'].values[-1])
                    elif col == 'Lag1_Open':
                        new_features.append(df['Open'].values[-1])
                    elif col == 'Lag1_High':
                        new_features.append(df['High'].values[-1])
                    elif col == 'Lag1_Low':
                        new_features.append(df['Low'].values[-1])
                    elif col == 'Lag1_Change':
                        new_features.append(df['Change %'].values[-1])
                    elif col == 'RollingMean7':
                        new_features.append(np.mean(np.append(df['Price'].values[-6:], hybrid_pred)))
                    elif col == 'RollingStd7':
                        new_features.append(np.std(np.append(df['Price'].values[-6:], hybrid_pred)))
                    elif col == 'RollingMean14':
                        new_features.append(np.mean(np.append(df['Price'].values[-13:], hybrid_pred)))
                    elif col == 'RollingStd14':
                        new_features.append(np.std(np.append(df['Price'].values[-13:], hybrid_pred)))
                    elif col == 'RollingStd30':
                        new_features.append(np.std(np.append(df['Price'].values[-29:], hybrid_pred)))
                    elif col == 'Volatility':
                        new_features.append(df['High'].values[-1] - df['Low'].values[-1])
                new_features_scaled = scaler_X.transform([new_features])[0]
                last_sequence = np.append(last_sequence[:, 1:, :], [[new_features_scaled]], axis=1)
                residual_history.append(hybrid_pred - lstm_pred)

            return np.array(predictions), future_dates

        # Generate predictions
        horizon = st.slider('Select Forecast Horizon (Days)', 1, 60, 30)
        predictions, future_dates = forecast_hybrid(lstm_model, scaler_X, scaler_y, feature_cols, df, forecast_horizon=horizon)
        
        if predictions is not None:
            # Display results
            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price (BDT)': predictions.round(2)})
            st.write('**Predicted Prices**')
            st.dataframe(pred_df, use_container_width=True)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            ax.plot(df['Date'].iloc[-100:], df['Price'].iloc[-100:], label='Historical Prices', color='blue', linewidth=2)
            ax.plot(future_dates, predictions, label='Predicted Prices', color='orange', linestyle='--', linewidth=2)
            ax.set_title('Grameenphone Stock Price Forecast', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price (BDT)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Option to download predictions
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="hybrid_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
else:
    st.info("Please upload a CSV file to generate predictions.")