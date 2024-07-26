from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('/Users/safwan/Onlineretail/demand_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        lag1 = float(request.form['Lag1'])
        lag7 = float(request.form['Lag7'])
        day_of_week = int(request.form['DayOfWeek'])
        month = int(request.form['Month'])

        input_features = pd.DataFrame({
            'Lag1': [lag1],
            'Lag7': [lag7],
            'DayOfWeek': [day_of_week],
            'Month': [month]
        })

        prediction = model.predict(input_features)[0]

        return jsonify({'Prediction': prediction})

    except Exception as e:
        return jsonify({'Error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
