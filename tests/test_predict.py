"""Tes memakai data tiruan berskema sama dengan dataset, agar tes tidak butuh CSV asli."""
import joblib
import numpy as np
import pandas as pd
import pytest

from burnout import config as C
from burnout.predict import predict_risk
from burnout.train import run

CALM = dict(Study_Hours=4, Class_Attendance=90, Sleep_Hours=9, Assignment_Load=1,
            Exam_Frequency=1, Screen_Time=1, Social_Media_Use=0, Physical_Exercise="Yes")
STRESSED = dict(Study_Hours=6, Class_Attendance=75, Sleep_Hours=4, Assignment_Load=9,
                Exam_Frequency=9, Screen_Time=11, Social_Media_Use=7, Physical_Exercise="No")


def fake_df(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Study_Hours": rng.integers(0, 10, n), "Class_Attendance": rng.integers(40, 100, n),
        "Sleep_Hours": rng.integers(4, 10, n), "Assignment_Load": rng.integers(1, 10, n),
        "Exam_Frequency": rng.integers(1, 10, n), "Screen_Time": rng.integers(1, 12, n),
        "Social_Media_Use": rng.integers(0, 8, n),
        "Physical_Exercise": rng.choice(["No", "Yes"], n),
    })
    score = (df.Assignment_Load + df.Exam_Frequency + df.Screen_Time + df.Social_Media_Use
             - df.Sleep_Hours - 2 * (df.Physical_Exercise == "Yes") + rng.normal(0, 3, n))
    df[C.TARGET] = np.where(score <= 7, "Low", np.where(score <= 17, "Medium", "High"))
    return df


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("out")
    run(df=fake_df(), model_path=d / "m.joblib", metrics_path=d / "m.json",
        confusion_path=d / "cm.png", verbose=False)
    return joblib.load(d / "m.joblib")


def test_probabilities_sum_to_one(model):
    assert sum(predict_risk(STRESSED, model).values()) == pytest.approx(1.0)


def test_calm_is_low(model):
    p = predict_risk(CALM, model)
    assert max(p, key=p.get) == "Low"


def test_stressed_is_high(model):
    p = predict_risk(STRESSED, model)
    assert max(p, key=p.get) == "High"
