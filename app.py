import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf

st.set_page_config(layout="wide")
st.title("📈 Stock Price Analysis & Prediction (AAPL)")

@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def clean_stock_data(df):
    numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    imputer = SimpleImputer(strategy='median')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    df.drop_duplicates(inplace=True)

    def remove_outliers(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        return column[(column >= Q1 - 1.5 * IQR) & (column <= Q3 + 1.5 * IQR)]

    for col in numerical_columns:
        df[col] = remove_outliers(df[col])

    return df

def engineer_features(df):
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Daily_Volatility'] = df['Daily_Return'].rolling(window=7).std()

    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = calculate_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

@st.cache_data
def preprocess_data():
    df = engineer_features(clean_stock_data(load_stock_data('AAPL', '2018-01-01', '2023-01-01')))
    
    # Fix datetime index here
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        # Create datetime index if it doesn't exist
        start_date = pd.to_datetime('2018-01-01')
        df.index = pd.date_range(start=start_date, periods=len(df), freq='B')
    
    return df

with st.spinner("Loading and preprocessing data..."):
    df = preprocess_data()

st.subheader("1. 📊 Data Preview")
st.dataframe(df.tail(), use_container_width=True)

st.subheader("2. 📈 Data Visualization")
fig, ax = plt.subplots(figsize=(14,6))
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['MA7'], label='MA7')
plt.plot(df.index, df['MA30'], label='MA30')
plt.title("Close Price with Moving Averages")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("3. 🤖 Linear vs Random Forest Regression")
features = ['Open', 'High', 'Low', 'Volume', 'MA7', 'MA30', 'Daily_Return']
target = 'Close'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)

lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

def show_model_results(model, name):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"**{name}**")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    plt.plot(y_test.values[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title(f"{name} - Actual vs Predicted")
    plt.xlabel('Time Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

show_model_results(lr_model, "Linear Regression")
show_model_results(rf_model, "Random Forest Regressor")

# Additional Random Forest detailed plot
st.subheader("🌲 Random Forest Detailed Analysis")
rf_pred = rf_model.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

st.write(f"**Random Forest Detailed Metrics:**")
st.write(f"RMSE: {rf_rmse:.2f}")
st.write(f"R² Score: {rf_r2:.2f}")

fig, ax = plt.subplots(figsize=(12,6))
plt.plot(y_test.values[:100], label='Actual Price', linewidth=2)
plt.plot(rf_pred[:100], label='Predicted Price', linewidth=2)
plt.title('Apple Stock Price Prediction using Random Forest')
plt.xlabel('Time Index')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
st.pyplot(fig)


st.subheader("4. 🔮 LSTM Future 30-Day Forecast")

if st.button("🔮 Predict"):
    scaler_lstm = MinMaxScaler()
    scaled_close = scaler_lstm.fit_transform(df[['Close']])

    SEQ_LENGTH = 60
    X_lstm, y_lstm = [], []
    for i in range(len(scaled_close) - SEQ_LENGTH):
        X_lstm.append(scaled_close[i:i+SEQ_LENGTH])
        y_lstm.append(scaled_close[i+SEQ_LENGTH])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    split = int(len(X_lstm)*0.8)
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training LSTM model..."):
        model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm), verbose=0)

    last_seq = scaled_close[-SEQ_LENGTH:]
    future_preds_scaled = []
    current_seq = last_seq.reshape(1, SEQ_LENGTH, 1)
    for _ in range(30):
        pred = model.predict(current_seq, verbose=0)[0,0]
        future_preds_scaled.append(pred)
        current_seq = np.append(current_seq[:,1:,:], [[[pred]]], axis=1)

    future_preds = scaler_lstm.inverse_transform(np.array(future_preds_scaled).reshape(-1,1)).flatten()

    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    fig, ax = plt.subplots(figsize=(14,6))
    plt.plot(df.index[-100:], df['Close'].values[-100:], label='Historical', linewidth=2)
    plt.plot(future_dates, future_preds, label='LSTM Forecast', color='orange', linewidth=2)
    plt.title('📅 Next 30-Day Forecast with LSTM')
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    avg_price = np.mean(future_preds)
    st.markdown(f"**📌 Average predicted Close Price for next 30 days: ${avg_price:.2f}**")

    st.subheader("📅 Daily Predictions")
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': [f"${price:.2f}" for price in future_preds]
    })
    st.dataframe(predictions_df, use_container_width=True)
else:
    st.info("Click the **Predict** button to generate the 30-day LSTM forecast.")
