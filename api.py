from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("stress_dataset.csv")

X = data[['heart_rate', 'breathing_rate']]
y = data['stress_level']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Yoga suggestion
def suggest_yoga(stress):
    if stress == "High":
        return "Deep Breathing"
    elif stress == "Medium":
        return "Anulom Vilom"
    else:
        return "Meditation"

# API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    hr = data['heart_rate']
    br = data['breathing_rate']

    input_data = pd.DataFrame([[hr, br]],
                              columns=['heart_rate', 'breathing_rate'])

    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    return jsonify({
        "stress": prediction,
        "suggestion": suggest_yoga(prediction),
        "probability": probs.tolist()
    })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001, debug=True)