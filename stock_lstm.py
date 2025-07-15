# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")

st.title("📊 Stock Price Prediction using LSTM")
st.markdown("Upload your CSV file with stock data (Date, Open, High, Low, Close, Volume).")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your stock CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.sort_values('Date', inplace=True)

        st.success("✅ File uploaded and loaded successfully!")
        st.dataframe(df.head())

        # Use only closing prices
        data = df.filter(['Close'])
        dataset = data.values

        # Scale
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        # Train/Test split
        training_data_len = int(len(dataset) * 0.8)
        train_data = scaled_data[0:training_data_len]
        
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Train
        with st.spinner("⏳ Training LSTM model..."):
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=0)
        st.success("✅ Model trained!")

        # Test Data
        test_data = scaled_data[training_data_len - 60:]
        x_test, y_test = [], dataset[training_data_len:]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Plotting
        fig, ax = plt.subplots()
        ax.plot(y_test, label='Actual Stock Price', color='blue')
        ax.plot(predictions, label='Predicted Stock Price', color='red')
        ax.set_title("Stock Price Prediction using LSTM")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Predict tomorrow
        last_60_days = scaled_data[-60:]
        next_input = np.reshape(last_60_days, (1, 60, 1))
        next_price_scaled = model.predict(next_input)
        next_price = scaler.inverse_transform(next_price_scaled)

        st.subheader("🔮 Predicted Closing Price for Tomorrow:")
        st.success(f"₹ {next_price[0][0]:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")
