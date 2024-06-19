from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess data (assuming data is small and can be stored in-memory)
data = pd.read_excel('Smartfarm data_Feb_Winter.xlsx')
data = data.dropna()

# Prepare the model (example model, adjust according to your notebook)
model = LinearRegression()
X = data[['A1. Temp. (°C)', 'A1. Humidity (%)']].values
y = data['Some_Target_Column'].values  # Replace with actual target column
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temp = data['temp']
    humidity = data['humidity']
    features = np.array([[temp, humidity]])
    
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

@app.route('/')
def home():
    return "Welcome to the Sensor Data Prediction API!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

