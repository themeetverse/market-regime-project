
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import plotly.graph_objects as go
from datetime import datetime
st.set_page_config(layout="wide", page_title="Market Regime Dashboard")

# Correct Windows LOCAL project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Models and data stored inside folders
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")


# Load models
kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.joblib"))
scaler = joblib.load(os.path.join(PROJECT_DIR, "models_scaler.joblib"))
clf = joblib.load(os.path.join(MODELS_DIR, "xgb_regime_clf.joblib"))

# Utility: load data CSV (created earlier)
DATA_CSV = os.path.join(PROJECT_DIR, "data_full_with_regimes.csv")
df_all = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)

st.title("Market Regime Detection & Next-Day Forecast (Indian Stocks)")

# Sidebar
st.sidebar.header("Controls")
tickers = sorted(df_all['ticker'].unique())
ticker = st.sidebar.selectbox("Select stock", tickers, index=0)
end_date_default = df_all.index.max().date()
start_date = st.sidebar.date_input("Start date", value=(end_date_default.replace(year=end_date_default.year-2)))
end_date = st.sidebar.date_input("End date", value=end_date_default)
n_clusters = len(kmeans.cluster_centers_)

# Filter data
df = df_all[df_all['ticker']==ticker].sort_index()
df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]

# Features list for UI
feat_cols = ["return","vol_5","vol_21","ma_10","ma_21","ma_50","mom_10","volume_change","rsi_14","bb_width"]

# Last available day prediction
if not df.empty:
    last_row = df.iloc[[-1]]
    X_last = last_row[feat_cols].values
    Xs = scaler.transform(X_last)
    pred_regime = int(clf.predict(Xs)[0])
    proba = clf.predict_proba(Xs)[0]

    st.subheader(f"{ticker} — Last date: {last_row.index[0].date()}")
    col1, col2 = st.columns([3,1])
    with col1:
        # Price chart with regimes overlay
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', hoverinfo='x+y'))

        # Add small volume subplot by adding bar traces (scaled)
        vol_trace = go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', opacity=0.3)
        fig.add_trace(vol_trace)

        # Shade regimes
        colors = {0: 'rgba(0,200,0,0.10)', 1:'rgba(0,0,200,0.10)', 2:'rgba(200,0,0,0.10)', 3:'rgba(200,150,0,0.10)'}
        # For each contiguous block of regimes, add vrect
        regimes = df['regime'].values
        idx = df.index
        start_i = 0
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                fig.add_vrect(x0=idx[start_i], x1=idx[i-1], fillcolor=colors.get(int(regimes[i-1]), 'rgba(100,100,100,0.1)'), opacity=0.2, line_width=0)
                start_i = i
        # final segment
        if len(regimes)>0:
            fig.add_vrect(x0=idx[start_i], x1=idx[-1], fillcolor=colors.get(int(regimes[-1]), 'rgba(100,100,100,0.1)'), opacity=0.2, line_width=0)

        fig.update_layout(xaxis_rangeslider_visible=False, height=600,
                          yaxis=dict(domain=[0.2,1]), yaxis2=dict(domain=[0,0.18], anchor='x'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Next-day prediction")
        st.write(f"**Predicted regime:** {pred_regime}")
        for i,p in enumerate(proba):
            st.write(f"Regime {i}: {p:.2%}")
        st.markdown("---")
        st.write("Feature snapshot (last row)")
        st.table(last_row[feat_cols].T)

    st.markdown("---")
    st.subheader("Regime distribution (selected range)")
    st.bar_chart(df['regime'].value_counts().sort_index())

    st.subheader("Recent rows")
    st.dataframe(df.tail(50))
else:
    st.warning("No data for the selected date range/ticker.")

st.markdown("---")
st.info("Models saved in Google Drive under market_regime_project/models. To deploy publicly, upload the project folder to GitHub and deploy via Streamlit Cloud.")
