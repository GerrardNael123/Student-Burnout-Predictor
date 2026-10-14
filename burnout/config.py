from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "university_student_stress_dataset.csv"
MODEL_PATH = ROOT / "models" / "stress_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
CONFUSION_PATH = ROOT / "reports" / "confusion_matrix.png"

# Fitur "kebiasaan" yang dipakai (nama kolom persis seperti di dataset).
# Sengaja TIDAK dipakai: Anxiety_Level, Family_Support, Peer_Pressure (penilaian
# psikologis yang sudah terlalu dekat dengan hasil), serta demografi/ekonomi.
NUMERIC_FEATURES = [
    "Study_Hours",        # jam belajar per hari (0-9)
    "Class_Attendance",   # kehadiran kuliah, % (40-99)
    "Sleep_Hours",        # jam tidur per hari (4-9)
    "Assignment_Load",    # beban tugas, skala 1-9
    "Exam_Frequency",     # frekuensi ujian, skala 1-9
    "Screen_Time",        # total jam layar per hari (1-11)
    "Social_Media_Use",   # jam media sosial per hari (0-7)
]
CATEGORICAL_FEATURES = ["Physical_Exercise"]  # "No" / "Yes"
EXERCISE_LEVELS = ["No", "Yes"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "Stress_Level"
LABELS = ["Low", "Medium", "High"]
RANDOM_STATE = 42

# Rentang nilai di data latih (dipakai untuk slider di app)
RANGES = {
    "Study_Hours": (0, 9),
    "Class_Attendance": (40, 99),
    "Sleep_Hours": (4, 9),
    "Assignment_Load": (1, 9),
    "Exam_Frequency": (1, 9),
    "Screen_Time": (1, 11),
    "Social_Media_Use": (0, 7),
}
