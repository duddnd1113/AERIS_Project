"""
V4 전처리
- log1p 변환: clip 없이 전체 농도 범위 보존
- 역변환: expm1
"""
import numpy as np


class RoadPMTransformer:
    """log1p 변환기 — fitting 불필요."""

    def fit(self, y: np.ndarray) -> "RoadPMTransformer":
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(np.asarray(y, dtype=float))

    def inverse_transform(self, y_log: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(np.asarray(y_log, dtype=float)), 0.0, None)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def summary(self):
        print("  변환: log1p(y)   역변환: expm1(y_log)")
        print("  clip 없음 — 전체 농도 범위 보존 (0 ~ max)")
