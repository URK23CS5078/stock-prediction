import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io

st.set_page_config(page_title="Stock Price Prediction using LSTM", layout="centered")
st.title("📈 Stock Price Prediction using LSTM")
st.subheader("Upload your stock CSV file and see future trend prediction!")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Fixing date format
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df.sort_values('Date')
        df = df[['Date', 'Close']].dropna()

        st.success("✅ File uploaded and cleaned successfully!")
        st.subheader("📊 Preview of Data")
        st.dataframe(df.head())

        # Scaling
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(np.array(df['Close']).reshape(-1, 1))

        # Prepare training data
        def create_dataset(data, time_step=100):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 100
        X, y = create_dataset(df_scaled, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=5, batch_size=64, verbose=0)

        # Prediction
        predicted = model.predict(X)
        predicted_prices = scaler.inverse_transform(predicted)
        actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

        # Plotting
        st.subheader("📈 Actual vs Predicted Prices")
        fig, ax = plt.subplots()
        ax.plot(actual_prices, label='Actual Price', color='blue')
        ax.plot(predicted_prices, label='Predicted Price', color='red')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.set_title("Stock Price Prediction using LSTM")
        ax.legend()
        st.pyplot(fig)

        # Download button
        st.download_button(
            label="📥 Download Prediction Results",
            data=pd.DataFrame({
                "Actual Price": actual_prices.flatten(),
                "Predicted Price": predicted_prices.flatten()
            }).to_csv(index=False).encode('utf-8'),
            file_name="predicted_stock_prices.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
