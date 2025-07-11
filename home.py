import streamlit as st
from features import load_data_and_models
from charts.price_trend_chart import show_price_trend_chart

def show_home():
    df, rf_model, scaler_X, scaler_y = load_data_and_models()
    # Hero Section
    st.markdown("""
    <div style="padding: 40px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h1 style="color: #fff;">Step into Bangladesh’s thriving stock market </br> with Grameenphone Lt. (GP)</h1>
        <p style="font-size: 18px; color: #adb5bd;">Unlock cutting-edge AI-driven predictions for Grameenphone stock prices with our advanced interactive platform. Dive deep into historical trends, gain unparalleled </br> model insights, and make informed investment decisions with confidence.</p>
        <button style="background-color: #6ee7c1; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">Explore</button>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stock Market Sections
    col1, col2 = st.columns(2)

    with col1:
            st.image("https://plus.unsplash.com/premium_photo-1661725727190-0ef02515deb1?q=80&w=1171&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")

    with col2:
           st.markdown("""
                <div>
                     <h2>Empowering You to Invest in Grameenphone with Confidence</h2>
                     <p style='color: #adb5bd;'>We bring data-driven insights, AI predictions, and powerful tools together to help you make smarter, more profitable decisions in the Bangladesh stock market.>
                      <h2>Why Choose Us?</h2>
                     <p class='black-overlay' style='color: #adb5bd;'>Built specifically for the Bangladesh stock market with a focus on GP.</p>
                     <p class='black-overlay' style='color: #adb5bd;'>Designed for both new and experienced investors.</p>
                     <p class='black-overlay' style='color: #adb5bd;'>Combines AI forecasts + human-understandable explanations.</p>
                </div>
             """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Using columns for layout
    col1, col2,col3 = st.columns(3)

    with col1:
            st.markdown("""
                <div>
                     <h2>Unlock the Full Potential of GP Investments</h2>
                     <p class='black-overlay'>Detailed Historical Data Analysis</p>
                     <p class='black-overlay'>AI-Powered Price Predictions</p>
                     <p class='black-overlay'>Model Insights & Performance Metrics</p>
                </div>
            """, unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1629963918958-1b62cfe3fe92?q=80&w=1172&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
    with col3:
            st.markdown("""
                <div>
                     <h2>Empowering You to Invest in Grameenphone with Confidence</h2>
                     <p style='color: #adb5bd;'>We bring data-driven insights, AI predictions, and powerful tools together to help you make smarter, more profitable decisions in the Bangladesh stock market.</p>
                     <p style='color: #adb5bd;'>Whether you're a seasoned investor or just starting, our platform equips you with all the tools to understand, predict, and grow with GP.</p>
                </div>
            """, unsafe_allow_html=True)

     # Finally chart
    st.markdown("<hr>", unsafe_allow_html=True)
    show_price_trend_chart(df) 
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div>
        <p style="color: #adb5bd;;">💹 Empowering smarter GP investments with data, insights & AI.Developed By @Mitul, Faisol, Ratri</p>
    </div>
    """, unsafe_allow_html=True)