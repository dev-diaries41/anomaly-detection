from flask import Flask, request, jsonify
from predict import predict_anomaly  # Import the predict_anomaly function from predict.py

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json.get('input_data')
        
        if input_data is None:
            return jsonify({'error': 'No input data provided.'}), 400
        
        if not isinstance(input_data, list) or not all(isinstance(i, (int, float)) for i in input_data):
            return jsonify({'error': 'Input data must be a list of numerical values.'}), 400
        
        is_anomaly = predict_anomaly(input_data)
        
        return jsonify({'is_anomaly': is_anomaly.tolist()})  # Convert numpy array to list for JSON response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
