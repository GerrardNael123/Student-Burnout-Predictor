import pandas as pd

from .config import FEATURES, RAW_PATH, TARGET


def load_dataset(path=RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {path}.\n"
            "Unduh 'University Student Stress Dataset' (Mendeley Data, DOI 10.17632/rc5htd5dfr.1), "
            "lalu salin Cleaned_Data/university_student_stress_dataset.csv ke folder data/raw/."
        )
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di CSV: {sorted(missing)}")
    return df[FEATURES + [TARGET]]
