import streamlit as st

from burnout.config import LABELS, RANGES
from burnout.predict import load_model, predict_risk, recommendations

st.set_page_config(page_title="Student Stress Predictor", page_icon="🔥")


@st.cache_resource
def get_model():
    return load_model()


def slider(label, key, default, step=1):
    lo, hi = RANGES[key]
    return st.slider(label, lo, hi, default, step)


st.title("🔥 Student Stress / Burnout Risk Predictor")
st.caption(
    "Proyek ML portofolio. Dilatih pada University Student Stress Dataset (Bangladesh). "
    "Hasil adalah estimasi, bukan diagnosis."
)

with st.sidebar:
    st.header("Kebiasaanmu")
    inputs = {
        "Study_Hours": slider("Jam belajar per hari", "Study_Hours", 6),
        "Class_Attendance": slider("Kehadiran kuliah (%)", "Class_Attendance", 75),
        "Sleep_Hours": slider("Jam tidur per hari", "Sleep_Hours", 5),
        "Assignment_Load": slider("Beban tugas (1 ringan - 9 berat)", "Assignment_Load", 7),
        "Exam_Frequency": slider("Frekuensi ujian (1 jarang - 9 sering)", "Exam_Frequency", 5),
        "Screen_Time": slider("Total waktu layar per hari (jam)", "Screen_Time", 8),
        "Social_Media_Use": slider("Media sosial per hari (jam)", "Social_Media_Use", 3),
        "Physical_Exercise": st.radio("Berolahraga?", ["No", "Yes"], index=0, horizontal=True),
    }

try:
    probs = predict_risk(inputs, get_model())
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

top = max(probs, key=probs.get)
icons = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
st.subheader(f"Risiko: {icons[top]} {top.upper()}")
for label in LABELS:
    c1, c2 = st.columns([1, 4])
    c1.write(f"**{label.upper()}**")
    c2.progress(probs[label], text=f"{probs[label]:.0%}")

st.subheader("Saran")
for tip in recommendations(inputs):
    st.write(f"- {tip}")
