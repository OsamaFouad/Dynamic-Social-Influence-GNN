"""
Baseline models for influence prediction.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LastWindowBaseline:
    """
    Predicts future influence based on current-window incoming interaction count.
    Uses feature index 7 (total_in_window) as the influence proxy score.
    """

    SCORE_FEATURE_IDX = 7  # total_in_window

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.SCORE_FEATURE_IDX]

    def predict(self, X: np.ndarray, threshold_percentile: float = 90.0) -> np.ndarray:
        scores = self.predict_scores(X)
        if scores.max() == 0:
            return np.zeros(len(scores), dtype=int)
        thresh = np.percentile(scores[scores > 0], threshold_percentile)
        return (scores >= thresh).astype(int)


class SKLearnBaseline:
    """Thin wrapper around sklearn classifiers."""

    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                **kwargs,
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                n_jobs=-1,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.scaler = StandardScaler()
        self.model_type = model_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
