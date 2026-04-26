from __future__ import annotations

from enum import Enum

import numpy as np

from src.rag.retrieve import model as _model


class QueryClass(Enum):
    OFF_TOPIC = "off_topic"
    UNKNOWN_MEDICAL = "unknown_medical"


_MEDICAL_ANCHORS = [
    "diabetes blood sugar insulin glucose A1C",
    "heart disease cardiovascular cholesterol blood pressure",
    "cancer screening mammogram colonoscopy biopsy",
    "vaccination immunization booster dose schedule",
    "stroke symptoms treatment prevention",
    "obesity BMI weight management diet",
    "hypertension antihypertensive medication",
    "HIV AIDS antiretroviral treatment prevention",
    "asthma respiratory lung inhaler",
    "mental health depression anxiety medication",
    "kidney renal disease chronic failure",
    "CDC guidelines clinical recommendation prevention",
]

_ANCHOR_MATRIX = np.stack([
    _model.encode(phrase, normalize_embeddings=True)
    for phrase in _MEDICAL_ANCHORS
])  # shape: (12, 384)


# empirically validated against test_classifier.py; see scripts/test_classifier.py for cases
_MEDICAL_THRESHOLD = 0.55


def classify_query(question: str) -> QueryClass:
    q_vec = _model.encode(question, normalize_embeddings=True)
    max_sim = float((_ANCHOR_MATRIX @ q_vec).max())
    return QueryClass.UNKNOWN_MEDICAL if max_sim >= _MEDICAL_THRESHOLD else QueryClass.OFF_TOPIC
