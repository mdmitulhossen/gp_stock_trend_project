import streamlit as st
import plotly.graph_objects as go

def show_price_trend_chart(filtered_df):
    st.subheader("Grameenphone Stock Price Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Price'],
        mode='lines+markers',
        line=dict(color='#00FFAA', width=2),
        marker=dict(size=4, color='#00FFAA'),
        name='Price'
    ))

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        title=dict(
            text='Grameenphone Stock Price',
            font=dict(size=20),
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)
