📈 Market Regime Detection (ML Project)

Detect different market regimes (Bull, Bear, Sideways, High-Volatility, Low-Volatility) using unsupervised machine learning on historical stock price data.

📌 1. Overview

This project uses KMeans Clustering to identify hidden patterns in market behavior based on:

Daily returns

Volatility

Momentum

Trend indicators

Price range

The final output is a regime timeline showing how the market shifts between phases.

📌 2. Features Used
Feature	Description
Returns	Daily % price change
Volatility (10D/30D)	Rolling standard deviation
Momentum (10D)	Short-term trend
MA-20	20-day moving average
Range%	(High − Low) / Close
📌 3. ML Approach
Algorithm: KMeans

Best for unsupervised clustering

Simple, scalable, and explainable

Works well on normalized financial features

Why not others?

DBSCAN → fails on trending time-series

Agglomerative → slow & less stable

HMM/LSTM → complex, needs labels

📌 4. Workflow

Download stock data (via yfinance)

Clean & engineer features

Scale features with StandardScaler

Cluster using KMeans (k = 4/5 regimes)

Plot price vs. regime timeline

📌 5. Output

Colored scatter plot showing price vs regime

Each color = a detected regime

Helps identify:

Bull phases

Bear phases

Sideways zones

High-volatility periods

📌 6. Requirements
yfinance
pandas
numpy
scikit-learn
matplotlib

📌 7. Conclusion

This project reveals hidden market structures using unsupervised ML.
Useful for risk management, strategy design, and market understanding.

<img width="1872" height="997" alt="image" src="https://github.com/user-attachments/assets/afa74654-4731-4240-8399-f12ddd0d1136" />
<img width="1864" height="760" alt="image" src="https://github.com/user-attachments/assets/a4f4fe33-1146-49f0-8dd8-8fbf70a72a53" />
<img width="1886" height="977" alt="image" src="https://github.com/user-attachments/assets/3b6a3e50-8652-42ec-8063-b9a969006a80" />


