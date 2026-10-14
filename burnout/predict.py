import joblib
import pandas as pd

from .config import FEATURES, LABELS, MODEL_PATH


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model belum ada di {path}. Jalankan dulu: python -m burnout.train"
        )
    return joblib.load(path)


def predict_risk(inputs: dict, model=None) -> dict:
    """Return probabilitas {'Low': p, 'Medium': p, 'High': p}; total = 1."""
    model = model or load_model()
    df = pd.DataFrame([inputs])[FEATURES]
    proba = dict(zip(model.classes_, model.predict_proba(df)[0]))
    return {label: float(proba[label]) for label in LABELS}


def recommendations(inputs: dict) -> list[str]:
    """Saran berbasis aturan sederhana (bukan saran medis, bukan output model)."""
    tips = []
    if inputs["Sleep_Hours"] < 6:
        tips.append("Tidur kurang dari 6 jam. Usahakan 7-8 jam.")
    if inputs["Screen_Time"] >= 8:
        tips.append("Waktu layar sangat tinggi. Coba jeda layar tiap 1-2 jam dan batasi malam hari.")
    if inputs["Social_Media_Use"] >= 5:
        tips.append("Media sosial 5 jam/hari atau lebih. Coba batasi dengan timer aplikasi.")
    if inputs["Assignment_Load"] >= 7 or inputs["Exam_Frequency"] >= 7:
        tips.append("Beban tugas/ujian tinggi. Buat prioritas dan pecah jadi bagian kecil.")
    if inputs["Physical_Exercise"] == "No":
        tips.append("Belum berolahraga. Jalan kaki 20-30 menit beberapa kali seminggu sudah membantu.")
    if inputs["Study_Hours"] >= 8:
        tips.append("Jam belajar sangat panjang. Beri jeda istirahat rutin.")
    return tips or ["Kebiasaanmu tampak seimbang. Pertahankan!"]
