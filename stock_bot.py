import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import plotly.graph_objs as go
import streamlit as st
from datetime import timedelta

# Streamlit UI
st.title("📈 Stock Price Prediction")
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA)", value="TSLA")
start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
prediction_days = st.slider("How many days into the future to predict?", min_value=1, max_value=30, value=5)
show_full_price = st.checkbox("Show Full Historical Price Line", value=True)

if st.button("Predict"):
    # Extend training data to include more history before start_date
    history_buffer = timedelta(days=365 * 5)  # 5 years
    training_start_date = pd.Timestamp(start_date) - history_buffer

    df = yf.download(ticker, start=training_start_date, end=end_date)[['Close']]

    if df.empty:
        st.error("No data found for the given inputs.")
    else:
        # Preprocessing
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        def create_sequences(data, seq_length):
            X, y, dates = [], [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
                dates.append(df.index[i])
            return np.array(X), np.array(y), dates

        SEQ_LEN = 60
        X, y, dates = create_sequences(scaled_data, SEQ_LEN)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        dates_test = dates[split:]

        # Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        predicted = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted)
        real_prices = scaler.inverse_transform(y_test)

        # Predict future values beyond the end_date
        future_inputs = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        future_preds = []
        future_dates = []
        last_date = df.index[-1]
        for i in range(prediction_days):
            next_pred = model.predict(future_inputs)
            future_preds.append(next_pred[0][0])
            next_pred_reshaped = next_pred.reshape(1, 1, 1)
            future_inputs = np.concatenate((future_inputs[:, 1:, :], next_pred_reshaped), axis=1)
            future_dates.append(last_date + timedelta(days=i+1))

        # Filter predictions from selected start_date onward
        visible_indices = [i for i, date in enumerate(dates_test) if date >= pd.Timestamp(start_date)]
        visible_dates = [dates_test[i] for i in visible_indices]
        visible_real = real_prices[visible_indices]
        visible_pred = predicted_prices[visible_indices]

        # Error Metrics
        mae = mean_absolute_error(visible_real, visible_pred)
        rmse = np.sqrt(mean_squared_error(visible_real, visible_pred))
        mape = np.mean(np.abs((visible_real - visible_pred) / visible_real)) * 100

        # Full real price line
        df['Close_Scaled'] = scaler.inverse_transform(scaled_data)

        # Plotly Chart
        fig = go.Figure()
        if show_full_price:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close_Scaled'], mode='lines', name='Full Price (Actual)', line=dict(color='gray', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=visible_dates, y=visible_real.flatten(), mode='lines', name='Real Price'))
        fig.add_trace(go.Scatter(x=visible_dates, y=visible_pred.flatten(), mode='lines', name='Predicted Price'))

        # Confidence Band (using RMSE as margin)
        upper_bound = visible_pred.flatten() + rmse
        lower_bound = visible_pred.flatten() - rmse
        fig.add_trace(go.Scatter(
            x=visible_dates + visible_dates[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name='Confidence Band'))

        # Future predictions
        fig.add_trace(go.Scatter(x=future_dates, y=scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten(), mode='lines+markers', name='Future Prediction', line=dict(color='orange', dash='dash')))

        fig.update_layout(title=f"Predicted vs Real Stock Prices for {ticker}",
                          xaxis=dict(title='Date (YYYY-MM)', tickformat='%Y-%m', tickangle=-45, tickmode='auto'),
                          yaxis_title='Price',
                          width=1000,
                          height=500)

        st.plotly_chart(fig, use_container_width=True)

        # Display error metrics
        st.subheader("📊 Model Performance Metrics")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")

        st.success("Prediction complete!")
