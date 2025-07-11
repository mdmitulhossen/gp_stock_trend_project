import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib
from datetime import datetime, timedelta
import io

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

def show_features(selected_feature):
    df, rf_model, scaler_X, scaler_y = load_data_and_models()
    if df is None:
        return

    feature_cols = ['Lag1', 'Lag2', 'Lag3', 'Lag1_Vol', 'Lag1_Return',
                    'Lag1_Open', 'Lag1_High', 'Lag1_Low', 'Lag1_Change',
                    'RollingMean7', 'RollingStd7', 'RollingMean14', 'RollingStd14',
                    'Volatility']

    if selected_feature == "Historical Data":
        st.header("📊 Historical Stock Data")
       
        # Date Range
        date_range = st.slider(
            "Select Date Range",
            min_value=df['Date'].min().to_pydatetime(),
            max_value=df['Date'].max().to_pydatetime(),
            value=(df['Date'].min().to_pydatetime(), df['Date'].max().to_pydatetime()),
            format="YYYY-MM-DD"
        )

        filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]

        # ---------- Shared Layout ----------
        custom_layout = dict(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # ---------- 1️⃣ Candlestick Chart ----------
        st.subheader("📈 Candlestick Price Chart")
        candlestick_fig = go.Figure(
            data=[go.Candlestick(
                x=filtered_df['Date'],
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Price'],
                name="Price",
                increasing_line_color='#00FFAA',  # neon green
                decreasing_line_color='#FF4B91'   # pink red
            )]
        )
        candlestick_fig.update_layout(
            title="Grameenphone Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (BDT)",
            **custom_layout
        )
        st.plotly_chart(candlestick_fig, use_container_width=True)

        # ---------- 2️⃣ Volume Bar Chart ----------
        st.subheader("📊 Trading Volume (Bar Chart)")
        volume_fig = px.bar(
            filtered_df,
            x='Date',
            y='Vol.',
            title='Grameenphone Trading Volume',
            color='Vol.',
            color_continuous_scale=[(0, "#FF4B91"), (0.5, "#00FFAA"), (1, "#00FFAA")]
        )
        volume_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Volume",
            coloraxis_showscale=False,
            **custom_layout
        )
        st.plotly_chart(volume_fig, use_container_width=True)

        # ---------- 3️⃣ Moving Average Line Chart ----------
        st.subheader("📉 7-Day Moving Average")
        filtered_df['7MA'] = filtered_df['Price'].rolling(window=7).mean()

        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Price'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#00FFAA', width=2)
        ))
        ma_fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['7MA'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='#FF4B91', dash='dash', width=2)
        ))
        ma_fig.update_layout(
            title="Price vs 7-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Price (BDT)",
            **custom_layout
        )
        st.plotly_chart(ma_fig, use_container_width=True)

        # ----- Data filtering part for search -----
        search_date = st.text_input(
            "🔍 Search by Date (YYYY-MM-DD)",
            placeholder="Enter date like 2021-07-26"
            
        )

        if search_date:
            try:
                # Parse to datetime to ensure correct format
                search_dt = pd.to_datetime(search_date).date()
                filtered_table_df = filtered_df[filtered_df['Date'].dt.date == search_dt]
                if filtered_table_df.empty:
                    st.warning("No data found for this date.")
            except Exception as e:
                st.error("Invalid date format. Please use YYYY-MM-DD.")
                filtered_table_df = filtered_df
        else:
            filtered_table_df = filtered_df

        # ----- Table -----
        st.subheader("📋 Filtered Data Table")
        st.dataframe(
            filtered_table_df.style.set_properties(**{
                'background-color': '#1e1e1e',
                'color': 'white',
                'border-color': 'white'
            }),
            use_container_width=True,
            hide_index=True
        )

        # ---------- CSV Download ----------
        st.subheader("⬇️ Download Historical Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            type='primary',
            file_name="grameenphone_historical_data.csv",
            mime="text/csv"
        )

    elif selected_feature == "Model Predictions":
                # ---------- Data preparation ----------
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
        pred_df['Error'] = pred_df['Actual Price'] - pred_df['Predicted Price']

        # ---------- Shared Layout ----------
        custom_layout = dict(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # ---------- Page Title ----------
        st.header("🤖 Model Predictions")

        # ---------- 1️⃣ Actual vs Predicted Prices ----------
        st.subheader("Actual vs Predicted Prices")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_df['Date'], y=pred_df['Actual Price'],
            mode='lines+markers', name='Actual Price',
            line=dict(color='#00FFAA', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=pred_df['Date'], y=pred_df['Predicted Price'],
            mode='lines', name='Predicted Price',
            line=dict(color='#FF4B91', dash='dash', width=2)
        ))
        fig.update_layout(
            title='Actual vs Predicted Stock Prices',
            xaxis_title='Date',
            yaxis_title='Price (BDT)',
            **custom_layout
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------- 2️⃣ Prediction Error Histogram ----------
        st.subheader("Prediction Error Distribution")
        error_fig = px.histogram(
            pred_df, x='Error',
            nbins=30,
            title='Distribution of Prediction Errors',
            color_discrete_sequence=['#00FFAA']
        )
        error_fig.update_layout(**custom_layout)
        st.plotly_chart(error_fig, use_container_width=True)

        # ---------- 3️⃣ Prediction Residuals over Time ----------
        st.subheader("Prediction Residuals Over Time")
        residual_fig = go.Figure()
        residual_fig.add_trace(go.Scatter(
            x=pred_df['Date'],
            y=pred_df['Error'],
            mode='lines+markers',
            line=dict(color='#FF4B91', width=2),
            name='Residual (Actual - Predicted)'
        ))
        residual_fig.update_layout(
            title='Residual Plot',
            xaxis_title='Date',
            yaxis_title='Residual (BDT)',
            **custom_layout
        )
        st.plotly_chart(residual_fig, use_container_width=True)

        # ---------- 4️⃣ Data Table & Download ----------
        st.subheader("Prediction Data Table")
        st.dataframe(
            pred_df.style.set_properties(**{
                'background-color': '#1e1e1e',
                'color': 'white',
                'border-color': 'white'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("⬇️ Download Predictions")
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            type='primary',
            file_name="grameenphone_predictions.csv",
            mime="text/csv"
        )

    elif selected_feature == "Feature Importance":
            # ---------- Feature Importance Calculation ----------
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # ---------- Shared Layout ----------
        custom_layout = dict(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # ---------- Page Title ----------
        st.header("🧠 Model Feature Analysis")

        # ---------- 1️⃣ Feature Importance Bar Chart ----------
        st.subheader("Feature Importance (Horizontal Bar)")
        fig_bar = px.bar(
            feature_importance,
            x='Importance', y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=[(0, "#FF4B91"), (0.5, "#00FFAA"), (1, "#00FFAA")],
            title='Feature Importance in RandomForest Model'
        )
        fig_bar.update_layout(**custom_layout)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ---------- 2️⃣ Feature Contribution Pie Chart ----------
        st.subheader("🥧 Feature Contribution Pie")
        fig_pie = px.pie(
            feature_importance,
            names='Feature',
            values='Importance',
            color_discrete_sequence=['#FF4B91', '#00FFAA', '#FF4B91', '#00FFAA']
        )
        fig_pie.update_layout(**custom_layout)
        st.plotly_chart(fig_pie, use_container_width=True)

        # ---------- 3️⃣ Top Features Highlight Cards ----------
        st.subheader("🌟 Top Features Driving The Model")
        col1, col2, col3 = st.columns(3)
        top_features = feature_importance.head(3).reset_index(drop=True)

        with col1:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #00FFAA; padding: 20px; border-radius: 10px;">
                    <h3 style="color:#00FFAA;">{top_features.loc[0,'Feature']}</h3>
                    <p style="color:white;">Importance: {top_features.loc[0,'Importance']:.4f}</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #FF4B91; padding: 20px; border-radius: 10px;">
                    <h3 style="color:#FF4B91;">{top_features.loc[1,'Feature']}</h3>
                    <p style="color:white;">Importance: {top_features.loc[1,'Importance']:.4f}</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #00FFAA; padding: 20px; border-radius: 10px;">
                    <h3 style="color:#00FFAA;">{top_features.loc[2,'Feature']}</h3>
                    <p style="color:white;">Importance: {top_features.loc[2,'Importance']:.4f}</p>
                </div>
            """, unsafe_allow_html=True)

        # ---------- 4️⃣ Feature Importance Table ----------
        st.subheader("Full Feature Importance Table")
        st.dataframe(
            feature_importance.style.set_properties(**{
                'background-color': '#1e1e1e',
                'color': 'white',
                'border-color': 'white'
            }),
            use_container_width=True,
            hide_index=True
        )

    elif selected_feature == "Predict Future Price":
        # ---------- 1️⃣ Future Prediction ----------
        st.header("🔮 Predict Future Stock Price")
        st.markdown("""
            <p style='color:#adb5bd;'>Enter a date to predict the stock price. 
            The model uses the last available data to generate features.</p>
        """, unsafe_allow_html=True)

        max_date = df['Date'].max()
        future_date = st.date_input(
            "Select Prediction Date",
            min_value=max_date + timedelta(days=1),
            max_value=max_date + timedelta(days=30)
        )

        if st.button("Predict",type='primary'):
            last_row = df.iloc[-1].copy()
            last_date = last_row['Date']
            days_diff = (future_date - last_date.date()).days
            
            if days_diff <= 0:
                st.error("🚫 Please select a future date.")
            else:
                features = np.array([[ 
                    last_row['Price'], last_row['Lag1'], last_row['Lag2'],
                    last_row['Vol.'], last_row['Change %'],
                    last_row['Open'], last_row['High'], last_row['Low'],
                    last_row['Change %'],
                    df['Price'].tail(7).mean(), df['Price'].tail(7).std(),
                    df['Price'].tail(14).mean(), df['Price'].tail(14).std(),
                    last_row['High'] - last_row['Low']
                ]])
                features_scaled = scaler_X.transform(features)
                pred_scaled = rf_model.predict(features_scaled).reshape(-1, 1)
                pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
                
                st.success(f"🎯 Predicted Stock Price for {future_date}: {pred_price:.2f} BDT")

                # ---------- 2️⃣ Feature Snapshot ----------
                st.subheader("🧩 Features Used for Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding:15px; border-left:5px solid #00FFAA; border-radius:8px;">
                            <p style="color:white;">Last Price: <b>{last_row['Price']:.2f}</b></p>
                            <p style="color:white;">Lag1: <b>{last_row['Lag1']:.2f}</b></p>
                            <p style="color:white;">Lag2: <b>{last_row['Lag2']:.2f}</b></p>
                            <p style="color:white;">Volume: <b>{last_row['Vol.']:.2f}</b></p>
                            <p style="color:white;">Change %: <b>{last_row['Change %']:.4f}</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding:15px; border-left:5px solid #FF4B91; border-radius:8px;">
                            <p style="color:white;">7-day Mean: <b>{df['Price'].tail(7).mean():.2f}</b></p>
                            <p style="color:white;">7-day Std: <b>{df['Price'].tail(7).std():.2f}</b></p>
                            <p style="color:white;">14-day Mean: <b>{df['Price'].tail(14).mean():.2f}</b></p>
                            <p style="color:white;">14-day Std: <b>{df['Price'].tail(14).std():.2f}</b></p>
                            <p style="color:white;">High-Low Spread: <b>{last_row['High'] - last_row['Low']:.2f}</b></p>
                        </div>
                    """, unsafe_allow_html=True)
    elif selected_feature == "Model Performance":
        # ---------- Data Split & Prediction ----------
        split_ratio = 0.8
        split_index = int(len(df) * split_ratio)
        test_df = df.iloc[split_index:].copy()

        X_test = test_df[feature_cols].values
        X_test_scaled = scaler_X.transform(X_test)
        y_test = test_df['Price'].values
        predictions_scaled = rf_model.predict(X_test_scaled).reshape(-1, 1)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

        # ---------- Metrics Calculation ----------
        def calculate_metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            accuracy = (1 - mape) * 100
            return rmse, mae, mape, r2, accuracy

        rmse, mae, mape, r2, accuracy = calculate_metrics(y_test, predictions)

        # ---------- Shared Neon Layout ----------
        custom_layout = dict(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # ---------- Page Header ----------
        st.header("🚀 Model Performance Metrics")

        # ---------- 1️⃣ Neon Stat Cards ----------
        st.subheader("Key Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #00FFAA; padding: 15px; border-radius: 10px;">
                    <h3 style="color:#00FFAA;">RMSE</h3>
                    <p style="color:white; font-size:20px;"><b>{rmse:.2f}</b></p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #FF4B91; padding: 15px; border-radius: 10px;">
                    <h3 style="color:#FF4B91;">MAE</h3>
                    <p style="color:white; font-size:20px;"><b>{mae:.2f}</b></p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #00FFAA; padding: 15px; border-radius: 10px;">
                    <h3 style="color:#00FFAA;">MAPE</h3>
                    <p style="color:white; font-size:20px;"><b>{mape*100:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #FF4B91; padding: 15px; border-radius: 10px;">
                    <h3 style="color:#FF4B91;">R²</h3>
                    <p style="color:white; font-size:20px;"><b>{r2:.4f}</b></p>
                </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #00FFAA; padding: 15px; border-radius: 10px;">
                    <h3 style="color:#00FFAA;">Accuracy</h3>
                    <p style="color:white; font-size:20px;"><b>{accuracy:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

        # ---------- 2️⃣ Performance Summary Bar Chart ----------
        st.subheader("Performance Summary Chart")
        summary_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'MAPE%', 'R²', 'Accuracy%'],
            'Value': [rmse, mae, mape*100, r2*100, accuracy]
        })
        summary_fig = px.bar(
            summary_df,
            x='Metric', y='Value',
            color='Value',
            color_continuous_scale=['#FF4B91', '#00FFAA'],
            text_auto='.2f'
        )
        summary_fig.update_layout(**custom_layout)
        st.plotly_chart(summary_fig, use_container_width=True)

        # ---------- 3️⃣ Residual Plot ----------
        st.subheader("Residuals Over Time")
        residuals = y_test - predictions
        residual_fig = go.Figure()
        residual_fig.add_trace(go.Scatter(
            x=test_df['Date'],
            y=residuals,
            mode='lines+markers',
            line=dict(color='#FF4B91', width=2),
            name='Residuals'
        ))
        residual_fig.update_layout(
            title='Prediction Residuals (Actual - Predicted)',
            xaxis_title='Date',
            yaxis_title='Residual (BDT)',
            **custom_layout
        )
        st.plotly_chart(residual_fig, use_container_width=True)

        # ---------- 4️⃣ Actual vs Predicted Scatter ----------
        # st.subheader("📈 Actual vs Predicted Scatter Plot")
        # scatter_fig = px.scatter(
        #     x=y_test,
        #     y=predictions,
        #     labels={'x':'Actual Price', 'y':'Predicted Price'},
        #     trendline='ols',
        #     color_discrete_sequence=['#00FFAA']
        # )
        # scatter_fig.update_layout(
        #     title="Actual vs Predicted Price with Trendline",
        #     **custom_layout
        # )
        # st.plotly_chart(scatter_fig, use_container_width=True)

    elif selected_feature == "Trading Insights":
        # ---------- Trading Simulation ----------
        split_ratio = 0.8
        split_index = int(len(df) * split_ratio)
        test_df = df.iloc[split_index:].copy()

        X_test = test_df[feature_cols].values
        X_test_scaled = scaler_X.transform(X_test)
        predictions_scaled = rf_model.predict(X_test_scaled).reshape(-1, 1)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

        threshold = 0.01
        signals = np.where(predictions[1:] > predictions[:-1] * (1 + threshold), 1,
                        np.where(predictions[1:] < predictions[:-1] * (1 - threshold), -1, 0))
        returns = np.diff(test_df['Price'].values) * signals
        transaction_cost = 0.5
        total_profit = np.sum(returns) - np.sum(np.abs(signals) > 0) * transaction_cost

        signal_df = pd.DataFrame({
            'Date': test_df['Date'].iloc[1:],
            'Signal': signals,
            'Return': returns
        })
        signal_df['Signal_Label'] = signal_df['Signal'].map({1: 'Buy', -1: 'Sell', 0: 'Hold'})
        signal_df['Cumulative_Profit'] = signal_df['Return'].cumsum() - (np.abs(signal_df['Signal']) > 0).cumsum() * transaction_cost

        # ---------- Shared Neon Layout ----------
        custom_layout = dict(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            # yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # ---------- Page Header ----------
        st.header("💹 Trading Insights")

        # ---------- 1️⃣ Simulated Trading Performance Card ----------
        st.subheader("Trading Summary")
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border-left: 5px solid #00FFAA; 
                        padding: 15px; border-radius: 10px;">
                <h3 style="color:#00FFAA;">Total Simulated Profit</h3>
                <p style="color:white; font-size:22px;"><b>{total_profit:.2f} BDT</b></p>
                <p style="color:#adb5bd;">(using 1% threshold & 0.5 BDT transaction cost)</p>
            </div>
        """, unsafe_allow_html=True)

        # ---------- 2️⃣ Trading Signal Timeline Chart ----------
        st.subheader("Trading Signal Timeline")
        signal_colors = {'Buy': '#00FFAA', 'Sell': '#FF4B91', 'Hold': '#888'}
        timeline_fig = go.Figure()
        for label, color in signal_colors.items():
            subset = signal_df[signal_df['Signal_Label'] == label]
            timeline_fig.add_trace(go.Scatter(
                x=subset['Date'], y=subset['Signal'],
                mode='markers', name=label, marker=dict(color=color, size=8),
                hovertemplate=f"Signal: {label}<extra></extra>"
            ))
        timeline_fig.update_layout(
            title="Buy / Sell / Hold Signals Over Time",
            yaxis=dict(title='Signal', tickvals=[-1, 0, 1], ticktext=['Sell', 'Hold', 'Buy']),
            xaxis_title='Date',
            **custom_layout
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        # ---------- 3️⃣ Cumulative Profit Curve ----------
        st.subheader("Cumulative Profit Over Time")
        profit_fig = go.Figure()
        profit_fig.add_trace(go.Scatter(
            x=signal_df['Date'],
            y=signal_df['Cumulative_Profit'],
            mode='lines+markers',
            line=dict(color='#00FFAA', width=3),
            name='Cumulative Profit'
        ))
        profit_fig.update_layout(
            title="Cumulative Profit From Strategy",
            xaxis_title="Date",
            yaxis_title="Profit (BDT)",
            **custom_layout
        )
        st.plotly_chart(profit_fig, use_container_width=True)

        # ---------- 4️⃣ Trading Signals Table & Download ----------
        st.subheader("Detailed Trading Signals")
        signal_df_display = signal_df[['Date', 'Signal_Label', 'Return', 'Cumulative_Profit']].rename(columns={
            'Signal_Label': 'Signal',
            'Return': 'Instant Return',
            'Cumulative_Profit': 'Cumulative Profit'
        })
        st.dataframe(
            signal_df_display.style.set_properties(**{
                'background-color': '#1e1e1e',
                'color': 'white',
                'border-color': 'white'
            }),
            use_container_width=True,
            hide_index=True
        )
        csv = signal_df_display.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Signals CSV",
            data=csv,
            type='primary',
            file_name="grameenphone_trading_signals.csv",
            mime="text/csv"
        )