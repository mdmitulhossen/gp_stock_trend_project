import streamlit as st
from features import load_data_and_models
from charts.candlestick_chart import show_candlestick_chart
from components.financial_cards import show_financial_snapshot
def show_about_gp():
    df, rf_model, scaler_X, scaler_y = load_data_and_models()
    
    st.markdown("""
        <div style="padding: 40px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h1 style="color: #fff;">Grameenphone Lt. (GP)</h1>
            <p style="font-size: 18px; color: #adb5bd;">Grameenphone (GP) is the leading telecommunications operator in Bangladesh, serving over 80 million subscribers nationwide. As a publicly listed company on the Dhaka Stock Exchange, Grameenphone has established a robust reputation for sustained profitability, regular dividend distributions, and maintaining a dominant position in the country's telecom market.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stock Market Sections
    col1, col2 = st.columns(2)

    with col1:
            st.image("https://i.ibb.co/bMmtk8b9/Chat-GPT-Image-Jul-11-2025-03-34-51-PM.png")

    with col2:
           st.markdown("""
                <div>
                     <h2>Why GP Stock is a Top Pick</h2>
                     <p style='color: #adb5bd;'>Stable Cash Flow: Telecom is a utility-like sector. Strong Financials: Healthy revenue & EBITDA margins. Market Leader: ~45% subscriber share. Consistent Dividends: Attractive to income-focused investors</p>
                      <h3>Key Facts of GP</h3>
                     <p class='black-overlay' style='color: #adb5bd;'>Founded: 1997,Listed: 2009 on DSE</p>
                     <p class='black-overlay' style='color: #adb5bd;'>Market Cap: Largest in telecom sector</p>
                     <p class='black-overlay' style='color: #adb5bd;'>Avg Dividend Yield: ~6%,Parent: Telenor Group (Norway)</p>
                     <h3>The Road Ahead for GP</h3>
                      <p class='black-overlay' style='color: #adb5bd;'>Investing in 4G expansion & potential 5G.</p>
                     <p class='black-overlay' style='color: #adb5bd;'>Growing data revenue,Strategic partnerships.</p>
                </div>
             """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="padding: 40px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h1 style="color: #fff;">GP’s Stock Performance Over the Years</h1>
            <p style="font-size: 18px; color: #adb5bd;">Showcase a multi-year price chart (DSE data) highlighting major events, like spectrum auctions, regulatory changes, or dividend declarations.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    show_financial_snapshot()

    st.markdown("<br>", unsafe_allow_html=True)
    
    show_candlestick_chart(df) 
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div>
            <p style="color: #adb5bd;;">💹 Empowering smarter GP investments with data, insights & AI.Developed By @Mitul, Faisol, Ratri</p>
        </div>
    """, unsafe_allow_html=True)