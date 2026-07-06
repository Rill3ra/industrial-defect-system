from dataclasses import dataclass
import numpy as np


@dataclass
class Prediction:
    final_label: int
    decision: str
    classifier_prob: float
    patchcore_score: float
    heatmap: np.ndarray
    bbox: list | None