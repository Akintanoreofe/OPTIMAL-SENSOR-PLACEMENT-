from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Generate synthetic data
np.random.seed(0)
n_samples = 100
n_features = 3

X = np.random.rand(n_samples, n_features)
coefficients = np.array([1.5, -2.0, 3.0])  # Random coefficients for the true model
y = X @ coefficients + np.random.randn(n_samples) * 0.5  # Add some noise

# Train the model
model = LinearRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data.get(f'feature{i}') for i in range(1, n_features + 1)])
    
    if None in features:
        return jsonify({'error': 'Missing one or more feature values'}), 400

    try:
        features = features.astype(float).reshape(1, -1)
    except ValueError:
        return jsonify({'error': 'Invalid feature values'}), 400

    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

@app.route('/')
def home():
    return "Welcome to the Random Multi-Regression Prediction API!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
