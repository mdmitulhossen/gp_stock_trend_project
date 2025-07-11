import streamlit as st
import plotly.graph_objects as go

def show_candlestick_chart(df):
    st.subheader("Grameenphone Candlestick & Volume")

    fig = go.Figure()

    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Price'],  # Assuming Price = Close
        increasing_line_color='#00FFAA',  # Neon green for up
        decreasing_line_color='#FF4B91',  # Pink-red for down
        name='Price'
    ))

    # Volume bar trace
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Vol.'],
        marker_color='rgba(0,255,170,0.3)',
        name='Volume',
        yaxis='y2'
    ))

    # Layout
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        hovermode='x unified',
        title=dict(
            text='Grameenphone Candlestick with Volume',
            font=dict(size=20),
            x=0.5
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True, gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='Price (BDT)',
            showgrid=True, gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
