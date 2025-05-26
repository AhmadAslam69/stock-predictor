# Assignment 2 code


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import yfinance as yf

def load_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)

        print("Dataset Loaded Successfully!")
        print("\nDataset Information:")
        print(df.info())

        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_stock_data(df):
    cleaned_df = df.copy()

    numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    print("\n--- Missing Value Analysis ---")
    print(cleaned_df[numerical_columns].isnull().sum())

    imputer = SimpleImputer(strategy='median')
    cleaned_df[numerical_columns] = imputer.fit_transform(cleaned_df[numerical_columns])

    duplicates_count = cleaned_df.duplicated().sum()
    cleaned_df.drop_duplicates(inplace=True)
    print(f"\nDuplicates Removed: {duplicates_count}")

    def remove_outliers(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return column[(column >= lower_bound) & (column <= upper_bound)]

    print("\n--- Outlier Removal ---")
    for col in numerical_columns:
        original_count = len(cleaned_df)
        cleaned_df[col] = remove_outliers(cleaned_df[col])
        outliers_removed = original_count - len(cleaned_df)
        print(f"{col}: {outliers_removed} outliers removed")

    return cleaned_df

def engineer_features(df):
    df_engineered = df.copy()

    df_engineered['MA7'] = df_engineered['Close'].rolling(window=7).mean()
    df_engineered['MA30'] = df_engineered['Close'].rolling(window=30).mean()

    df_engineered['Daily_Return'] = df_engineered['Close'].pct_change()

    df_engineered['Daily_Volatility'] = df_engineered['Daily_Return'].rolling(window=7).std()

    def calculate_rsi(data, periods=14):
        delta = data.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df_engineered['RSI'] = calculate_rsi(df_engineered['Close'])

    df_engineered.dropna(inplace=True)

    return df_engineered

def explore_data(df):
    plt.figure(figsize=(15, 10))

    # Descriptive Statisticsp
    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    # Correlation Heatmap
    plt.subplot(2, 2, 1)
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')

    # Closing Price Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['Close'], kde=True)
    plt.title('Distribution of Closing Prices')

    # Daily Returns Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['Daily_Return'], kde=True)
    plt.title('Distribution of Daily Returns')

    # Time Series Plot
    plt.subplot(2, 2, 4)
    plt.plot(df['Date'], df['Close'], label='Closing Price')
    plt.plot(df['Date'], df['MA7'], label='7-Day Moving Average', alpha=0.7)
    plt.plot(df['Date'], df['MA30'], label='30-Day Moving Average', alpha=0.7)
    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def main(ticker='AAPL', start_date='2022-01-01', end_date='2023-12-31'):
    # 1. Load Data
    df = load_stock_data(ticker, start_date, end_date)

    if df is None:
        return None

    # 2. Clean Data
    cleaned_df = clean_stock_data(df)

    # 3. Engineer Features
    engineered_df = engineer_features(cleaned_df)

    # 4. Explore Data
    explore_data(engineered_df)

    return engineered_df

# Run the main function
if __name__ == "__main__":
    processed_data = main()
    if processed_data is not None:
        print("\nData processing completed successfully!")





# assignment 3 code

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


df = processed_data.copy()

# Prepare features and target
X = df.drop(['Date', 'Close'], axis=1)
y = df['Close']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)

# Train models
lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"Model: {type(model).__name__}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}\n")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.title(f"{type(model).__name__} - Actual vs Predicted Stock Prices")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Evaluate models
lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test)
rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Start from the cleaned, feature-engineered data
df = engineer_features(clean_stock_data(load_stock_data('AAPL', '2018-01-01', '2023-01-01')))
df = df.dropna()  # ensure no NaNs from moving averages or returns

# Define features and target
features = ['Open', 'High', 'Low', 'Volume', 'MA7', 'MA30', 'Daily_Return']
target = 'Close'

X = df[features]
y = df[target]

# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now you can safely use GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
                      scoring='neg_mean_squared_error', n_jobs=-1)

rf_grid.fit(X_train_scaled, y_train)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Ensure features are engineered
df = engineer_features(clean_stock_data(load_stock_data('AAPL', '2018-01-01', '2023-01-01')))

# Drop rows with NaN values created by moving averages
df = df.dropna()

# Features and target
features = ['Open', 'High', 'Low', 'Volume', 'MA7', 'MA30', 'Daily_Return']
target = 'Close'

X = df[features]
y = df[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

r2 = r2_score(y_test, y_pred)

print(f"Random Forest RMSE: {rmse:.2f}")
print(f"Random Forest R² Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100], label='Actual Price', linewidth=2)     # smooth line
plt.plot(y_pred[:100], label='Predicted Price', linewidth=2)         # smooth line
plt.title(' Apple Stock Price Prediction using Random Forest')
plt.xlabel('Time Index')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.show()




#  LSTM Model for future Stock Price Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
# 1. Load and prepare data (using your cleaned df)
df = engineer_features(clean_stock_data(load_stock_data('AAPL', '2018-01-01', '2023-01-01')))
df = df.dropna()
close_prices = df['Close'].values.reshape(-1, 1)

# 2. Scale data to [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)

# 3. Create sequences for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_close, SEQ_LENGTH)

# 4. Split train-test
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Fix index to datetime for plotting and future date generation
if not pd.api.types.is_datetime64_any_dtype(df.index):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        start_date = pd.to_datetime('2018-01-01')
        df.index = pd.date_range(start=start_date, periods=len(df), freq='B')

# 5. Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train model
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# 7. Predict next 30 days recursively
last_sequence = scaled_close[-SEQ_LENGTH:]
future_preds_scaled = []

current_seq = last_sequence.reshape(1, SEQ_LENGTH, 1)
for _ in range(30):
    pred = model.predict(current_seq)[0,0]
    future_preds_scaled.append(pred)
    current_seq = np.append(current_seq[:,1:,:], [[[pred]]], axis=1)

# 8. Inverse scale the predictions
future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1,1)).flatten()

# 9. Create future dates index safely
# Ensure the index is datetime first
if not pd.api.types.is_datetime64_any_dtype(df.index):
    df.index = pd.to_datetime(df.index, errors='coerce')
# Drop any rows where datetime conversion failed
df = df[~df.index.isna()]
# Now extract last date safely
last_date = df.index.max()
# Create future dates
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# 10. Plot future predictions
plt.figure(figsize=(12,6))
plt.plot(df.index[-100:], df['Close'].values[-100:], label='Historical Close')
plt.plot(future_dates, future_preds, label='Predicted Next 30 Days', color='orange')
plt.title('AAPL Stock Close Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Print predicted prices day by day
for i, price in enumerate(future_preds, 1):
    print(f"Predicted price for day {i}: ${price:.2f}")

avg_price = sum(future_preds) / len(future_preds)
print(f"\nAverage predicted price for next 30 days: ${avg_price:.2f}")
