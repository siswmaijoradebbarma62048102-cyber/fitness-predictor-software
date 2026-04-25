import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("stress_dataset.csv")

# Features (input)
X = data[['heart_rate', 'breathing_rate']]

# Target (output)
y = data['stress_level']

# Split data (train + test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
# Example prediction
import pandas as pd

sample = pd.DataFrame([[95, 22]], columns=['heart_rate', 'breathing_rate'])
prediction = model.predict(sample)
print("Predicted Stress Level:", prediction[0])

# Yoga suggestion
def suggest_yoga(stress):
    if stress == "High":
        return "Deep Breathing"
    elif stress == "Medium":
        return "Anulom Vilom"
    else:
        return "Meditation"

print("Suggested Activity:", suggest_yoga(prediction[0]))