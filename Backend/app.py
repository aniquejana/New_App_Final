from flask import Flask, request, jsonify
import pandas as pd
import io
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import math
from flask_cors import CORS
from sklearn.metrics import mean_squared_error

# Import the CORS extension
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the CSV data from the request
    csv_data = request.data.decode('utf-8')
    print("Received CSV data:", csv_data)  # Add this line to print the received CSV data

    # Convert the CSV data to a pandas DataFrame
    df = pd.read_csv(io.StringIO(csv_data))

    # Preprocess the data
    scaler = MinMaxScaler()
    df['Price (INR)'] = scaler.fit_transform(df['Price (INR)'].values.reshape(-1, 1))

    # Split the data into train and test sets
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]

    # Prepare the data for LSTM
    look_back = 24
    X_train, y_train = [], []
    for i in range(look_back, len(train_data)):
        X_train.append(train_data['Price (INR)'].values[i - look_back:i])
        y_train.append(train_data['Price (INR)'].values[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the input data for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    for epoch in range(100):
        model.fit(X_train, y_train, epochs=1, batch_size=32)

    # Make predictions on the test data
    X_test = []
    for i in range(look_back, len(test_data)):
        X_test.append(test_data['Price (INR)'].values[i - look_back:i])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_prices = model.predict(X_test)

    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Calculate the root mean squared error
    y_test = test_data['Price (INR)'].values[look_back:]
    rmse = math.sqrt(mean_squared_error(y_test, predicted_prices))

    # Calculate the prediction accuracy percentage
    accuracy = 1 - (rmse / np.mean(y_test))
    accuracy_percentage = accuracy * 100

    # Predict the future price range
    future_prices = model.predict(X_test[-1].reshape(1, look_back, 1))
    future_price = scaler.inverse_transform(future_prices)[0][0]
    future_price_range = (future_price * 0.95, future_price * 1.05)

    # Return the prediction result
    prediction_result = {
        'future_price_range': future_price_range
    }
    
    # Convert float32 values to float
    future_price_range = prediction_result['future_price_range']
    future_price_range = (float(future_price_range[0]), float(future_price_range[1]))

    # Update the prediction result with the converted values
    prediction_result['future_price_range'] = future_price_range

    # Return the prediction result
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run()
