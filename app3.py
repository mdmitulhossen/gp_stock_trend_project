import streamlit as st
from home import show_home
from about_gp import show_about_gp
from features import show_features
from analysis_with_ai import show_analysis_with_ai
import os

# Set page configuration
st.set_page_config(page_title="GP Stock Prediction", page_icon=":chart_with_upwards_trend:", layout="wide")

# Load CSS
css_file_path = os.path.join(os.path.dirname(__file__), "styles.css")
try:
    with open(css_file_path, "r") as css_file:
        css = css_file.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Error: styles.css not found.")
    st.stop()

# Define navigation pages
pages = [
    st.Page(show_home, title="Home", url_path="home"),
    st.Page(show_about_gp, title="About GP", url_path="about_gp"),
    st.Page(show_analysis_with_ai, title="Analysis with AI", url_path="analysis_with_ai"),
    
    # Features - with 📂 prefix to visually group
    st.Page(lambda: show_features("Historical Data"), title="📂Historical Data", url_path="historical_data"),
    st.Page(lambda: show_features("Model Predictions"), title="📂 Model Predictions", url_path="model_predictions"),
    st.Page(lambda: show_features("Feature Importance"), title="📂 Features: Feature Importance", url_path="feature_importance"),
    st.Page(lambda: show_features("Predict Future Price"), title="📂 Features: Predict Future Price", url_path="predict_future_price"),
    st.Page(lambda: show_features("Model Performance"), title="📂 Features: Model Performance", url_path="model_performance"),
    st.Page(lambda: show_features("Trading Insights"), title="📂 Features: Trading Insights", url_path="trading_insights"),
]

# Setup navigation
pg = st.navigation(pages, position="top", expanded=True)

# Run selected page
pg.run()
