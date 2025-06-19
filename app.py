import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# --- Page Settings ---
st.set_page_config(page_title="CGPA Predictor", layout="centered")
st.title("🎓 CGPA Predictor with Grade Converter, What-If & AI Prediction")

# --- Theme Toggle ---
theme = st.radio("🎨 Choose Theme:", ["Light", "Dark"], horizontal=True)
if theme == "Dark":
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            background-color: #0E1117;
            color: white;
        }
        .stButton>button {
            background-color: #333;
            color: white;
        }
        .stSelectbox div, .stNumberInput div {
            color: black !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Grade to GPA mappings ---
letter_to_gpa = {
    "A (Excellent)": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B (Good)": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C (Average)": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D (Poor)": 1.0,
    "F* (Failure)": 0.0
}

def percent_to_gpa(percent):
    if percent >= 93: return 4.0
    elif percent >= 90: return 3.7
    elif percent >= 87: return 3.3
    elif percent >= 83: return 3.0
    elif percent >= 80: return 2.7
    elif percent >= 77: return 2.3
    elif percent >= 73: return 2.0
    elif percent >= 70: return 1.7
    elif percent >= 67: return 1.3
    elif percent >= 60: return 1.0
    else: return 0.0

# --- Input Section ---
st.markdown("## 📘 Semester Data Entry")
num_semesters = st.number_input("How many semesters completed?", min_value=1, max_value=12, step=1, key="sem_count")

credits_list = []
gpa_list = []

for i in range(1, num_semesters + 1):
    st.markdown(f"### Semester {i}")
    col1, col2 = st.columns(2)

    with col1:
        credits = st.number_input("Credits", min_value=1, step=1, key=f"credits_{i}")

    with col2:
        grade_input_type = st.selectbox(
            "Grade Type",
            options=["GPA (0.0-4.0)", "Letter Grade (A, B+, etc.)", "Percentage (%)"],
            key=f"input_type_{i}"
        )

    if grade_input_type == "GPA (0.0-4.0)":
        gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.01, key=f"gpa_{i}")
    elif grade_input_type == "Letter Grade (A, B+, etc.)":
        letter = st.selectbox("Letter Grade", options=list(letter_to_gpa.keys()), key=f"letter_{i}")
        gpa = letter_to_gpa[letter]
        st.info(f"Converted GPA: **{gpa:.2f}**")
    else:
        percent = st.number_input("Percentage", min_value=0, max_value=100, step=1, key=f"percent_{i}")
        gpa = percent_to_gpa(percent)
        st.info(f"Converted GPA: **{gpa:.2f}**")

    credits_list.append(credits)
    gpa_list.append(gpa)

# --- CGPA Calculation ---
st.markdown("## 🎯 Current CGPA Summary")
total_credits = sum(credits_list)
weighted_sum = sum([c * g for c, g in zip(credits_list, gpa_list)])

if total_credits > 0:
    current_cgpa = weighted_sum / total_credits
    st.success(f"🎯 Your current CGPA is: **{current_cgpa:.2f}**")

# --- What-If Simulator ---
st.markdown("## 📊 What-If GPA Simulator")
simulate = st.checkbox("Add planned future semester?", key="simulate_check")

if simulate:
    future_credits = st.number_input("Planned credits for next semester", min_value=1, step=1, key="future_credits")
    future_gpa = st.number_input("Planned GPA for next semester", min_value=0.0, max_value=4.0, step=0.01, key="future_gpa")

    new_total = total_credits + future_credits
    new_weighted = weighted_sum + (future_credits * future_gpa)
    new_cgpa = new_weighted / new_total

    st.info(f"📌 Your projected CGPA will be: **{new_cgpa:.2f}**")

# --- AI Prediction ---
st.markdown("## 🤖 AI Prediction of Future GPA")

if len(gpa_list) >= 2:
    X = np.array(range(1, num_semesters + 1)).reshape(-1, 1)
    y = np.array(gpa_list)

    model = LinearRegression()
    model.fit(X, y)

    next_semester = num_semesters + 1
    predicted_gpa = model.predict(np.array([[next_semester]]))[0]
    predicted_gpa = max(0.0, min(predicted_gpa, 4.0))

    st.success(f"📈 Predicted GPA for Semester {next_semester}: **{predicted_gpa:.2f}**")

    predicted_credits = st.number_input("Credits for predicted semester", min_value=1, value=15, step=1, key="ai_credits")
    future_weighted = weighted_sum + predicted_credits * predicted_gpa
    future_total = total_credits + predicted_credits
    future_cgpa = future_weighted / future_total

    st.info(f"📘 Predicted CGPA after Semester {next_semester}: **{future_cgpa:.2f}**")

    st.markdown("## 📉 GPA Trend Visualization")
    semesters = list(range(1, num_semesters + 1)) + [next_semester]
    gpas = list(gpa_list) + [predicted_gpa]

    plt.figure(figsize=(8, 4))
    plt.plot(semesters, gpas, marker="o", linestyle="-", color="blue")
    plt.title("GPA Trend (with Prediction)")
    plt.xlabel("Semester")
    plt.ylabel("GPA")
    plt.ylim(0, 4)
    plt.grid(True)
    st.pyplot(plt)
else:
    st.warning("⚠️ Add at least 2 semesters to enable AI prediction.")

# --- 📊 Performance Analytics & GPA Heatmap ---

st.markdown("## 📊 Performance Analytics")

if num_semesters > 0:
    max_gpa = max(gpa_list)
    min_gpa = min(gpa_list)
    improvement = gpa_list[-1] - gpa_list[0] if len(gpa_list) > 1 else 0

    st.write(f"🏆 Highest Semester GPA: **{max_gpa:.2f}** (Semester {gpa_list.index(max_gpa) + 1})")
    st.write(f"🔻 Lowest Semester GPA: **{min_gpa:.2f}** (Semester {gpa_list.index(min_gpa) + 1})")
    st.write(f"📈 Overall GPA Improvement from Semester 1 to Last: **{improvement:.2f}**")

    # GPA Heatmap Visualization
    st.markdown("## 📌 GPA Heatmap")

    # Create a 1-row matrix of GPA values for heatmap (reshape needed for seaborn)
    gpa_matrix = np.array(gpa_list).reshape(1, -1)

    plt.figure(figsize=(max(8, num_semesters), 1.5))
    sns.heatmap(
        gpa_matrix,
        annot=True,
        cmap="YlGnBu",
        cbar=True,
        xticklabels=[f"S{i}" for i in range(1, num_semesters + 1)],
        yticklabels=[],
        vmin=0,
        vmax=4,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 14},
    )
    plt.title("GPA Heatmap per Semester")
    plt.yticks([])
    st.pyplot(plt)
