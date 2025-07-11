import streamlit as st

def show_financial_snapshot():
    st.markdown("""
    <style>
    .stat-card-container {
        display: flex;
        justify-content: space-around;
        gap: 20px;
        flex-wrap: wrap;
    }
    .stat-card {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 15px;
        width: 220px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,255,170,0.2);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,255,170,0.4);
    }
    .stat-icon {
        font-size: 40px;
        margin-bottom: 10px;
    }
    .stat-title {
        color: #00FFAA;
        font-size: 18px;
        margin-bottom: 5px;
    }
    .stat-value {
        color: #FFFFFF;
        font-size: 22px;
        font-weight: bold;
    }
    </style>

    <div>
        <h2 style="color: #FFFFFF; text-align: center;">5️⃣ Financial Health</h2>
        <p style="color: #adb5bd; text-align: center;">💹 GP by the Numbers</p>
    </div>

    <div class="stat-card-container">
        <div class="stat-card">
            <div class="stat-icon">💼</div>
            <div class="stat-title">Revenue</div>
            <div class="stat-value">৳157.46 billion</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📊</div>
            <div class="stat-title">EBITDA Margin</div>
            <div class="stat-value">~42.4%</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">💰</div>
            <div class="stat-title">Net Profit</div>
            <div class="stat-value">৳29.27 billion</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🏆</div>
            <div class="stat-title">Dividend Payout</div>
            <div class="stat-value">~131%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
