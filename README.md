# 📈 Stock Price Prediction App

This is a Streamlit-based app that predicts stock prices using LSTM (Long Short-Term Memory) neural networks. Users can select a stock ticker, choose a date range, and forecast future prices for up to 30 days. Results are visualized with interactive Plotly charts.

---

## 🔍 Features

- LSTM-based stock price prediction
- Uses yfinance for real-time historical stock data
- Predicts future prices beyond user-selected range
- Toggle for displaying full historical price line
- Adjustable forecast range (1–30 days)
- Interactive Plotly charts
- Built with TensorFlow, Streamlit, and Plotly

---

## 🚀 Installation

To run this project locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/mmiller5604/Stock-Price-Prediction-App.git
   cd Stock-Price-Prediction-App
   ```
2. (Optional but recommended) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\\Scripts\\activate
```
3. Install dependencies

```bash
pip install -r requirements.txt
```
4. Run the app

```bash
streamlit run stock_app.py
```

This will open the app in your browser at http://localhost:8501/.

🧠 How It Works
Downloads stock data using yfinance

Scales the data using MinMaxScaler

Trains an LSTM model with 60-day time windows

Predicts stock price for user-defined future days

Visualizes predictions with Plotly line charts

📁 Project Structure
``` bash
Stock-Price-Prediction-App/
├── stock_app.py           # Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
