import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Stress Detector", page_icon="🧠", layout="centered")

# ------------------ TITLE ------------------
st.title("🧠 AI Stress Detection System")
st.markdown("### Detect your stress level & get personalized yoga suggestions")

# ------------------ LOAD DATA ------------------
data = pd.read_csv("stress_dataset.csv")

X = data[['heart_rate', 'breathing_rate']]
y = data['stress_level']

# ------------------ TRAIN MODEL ------------------
model = DecisionTreeClassifier()
model.fit(X, y)

# ------------------ SESSION STATE ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ INPUT SECTION ------------------
st.markdown("## 📥 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    heart_rate = st.slider("❤️ Heart Rate", 60, 140, 80)

with col2:
    breathing_rate = st.slider("🌬️ Breathing Rate", 10, 30, 18)

# ------------------ BUTTON ------------------
if st.button("🚀 Predict Stress"):

    # Create input
    input_data = pd.DataFrame([[heart_rate, breathing_rate]],
                              columns=['heart_rate', 'breathing_rate'])

    # Prediction
    prediction = model.predict(input_data)[0]

    # ------------------ YOGA SUGGESTION ------------------
    def suggest_yoga(stress):
        if stress == "High":
            return "🧘 Deep Breathing (5 min slow inhale-exhale)"
        elif stress == "Medium":
            return "🌿 Anulom Vilom (alternate nostril breathing)"
        else:
            return "🧠 Meditation (focus on breath)"

    suggestion = suggest_yoga(prediction)

    # ------------------ RESULT DISPLAY ------------------
    st.markdown("## 📊 Results")

    if prediction == "High":
        st.error(f"Stress Level: {prediction}")
    elif prediction == "Medium":
        st.warning(f"Stress Level: {prediction}")
    else:
        st.success(f"Stress Level: {prediction}")

    st.info(f"Suggested Activity: {suggestion}")

    # ------------------ PROBABILITY ------------------
    probs = model.predict_proba(input_data)[0]

    st.markdown("## 🔬 Stress Probability (Quantum-Inspired)")

    st.progress(float(probs[0]))
    st.write(f"Low: {probs[0]:.2f}")

    st.progress(float(probs[1]))
    st.write(f"Medium: {probs[1]:.2f}")

    st.progress(float(probs[2]))
    st.write(f"High: {probs[2]:.2f}")

    # ------------------ BAR GRAPH ------------------
    st.markdown("## 📊 Stress Probability Graph")

    labels = ["Low", "Medium", "High"]
    values = probs

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Probability")
    ax.set_title("Stress Level Distribution")

    st.pyplot(fig)

    # ------------------ SAVE HISTORY ------------------
    st.session_state.history.append({
        "heart_rate": heart_rate,
        "breathing_rate": breathing_rate,
        "stress": prediction
    })

# ------------------ HISTORY GRAPH ------------------
if st.session_state.history:

    st.markdown("## 📈 Stress History")

    history_df = pd.DataFrame(st.session_state.history)

    # Convert stress to numeric
    stress_map = {"Low": 1, "Medium": 2, "High": 3}
    history_df["stress_value"] = history_df["stress"].map(stress_map)

    fig2, ax2 = plt.subplots()
    ax2.plot(history_df["stress_value"], marker='o')
    ax2.set_title("Stress Trend Over Time")
    ax2.set_ylabel("Stress Level (1=Low, 3=High)")
    ax2.set_xlabel("Time Step")

    st.pyplot(fig2)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("AI-based Stress Detection System | Mini Project")