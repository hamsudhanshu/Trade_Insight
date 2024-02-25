
I have built a Streamlit web application for stock price prediction using a LSTM. Here's a brief description for your project:

Trade Insight: Stock Price Prediction

Trade Insight is a web application designed to provide insights into stock price movements, specifically focusing on predicting future stock prices based on historical data. Here's what the application offers:

Data Input: Users can input the stock symbol (e.g., AAPL for Apple Inc.) and select the desired start and end dates to retrieve historical stock price data.

Data Visualization: The application displays the retrieved stock price data, along with descriptive statistics such as mean, standard deviation, etc. Users can visualize the opening price trend over time through interactive charts.

Moving Averages: Users can observe the opening price trend overlaid with 100-day and 200-day moving averages, providing additional insights into the stock's long-term performance.

Prediction: The application utilizes a pre-trained LSTM model to predict future stock prices. It takes the last 100 days of historical data as input and predicts the next day's opening price.

Prediction Visualization: Users can compare the predicted stock prices with the actual prices on an interactive chart, allowing for an assessment of the model's performance.

Evaluation: The application calculates and displays the root mean squared error (RMSE) as a measure of the prediction model's accuracy.
