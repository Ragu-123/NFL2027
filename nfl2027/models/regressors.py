"""Tabular, ensemble, and hybrid regression models for NFL translation tasks."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_model(
    model_type: str = "ridge", alpha: float = 10.0, random_state: int = 42
) -> Pipeline:
    """Build standardized linear baseline model (M0)."""
    if model_type == "ridge":
        reg = Ridge(alpha=alpha, random_state=random_state)
    elif model_type == "elastic_net":
        reg = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=random_state)
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")

    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])


def build_kinematic_gbdt(
    n_estimators: int = 45,
    max_depth: int = 2,
    learning_rate: float = 0.06,
    subsample: float = 0.8,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    """Build regularized gradient boosting regressor for kinematic translation."""
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
    )


class HybridTrajectoryRegressor(BaseEstimator, RegressorMixin):
    """Hybrid regressor combining tabular physical covariates with latent trajectory embeddings."""

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        scale_inputs: bool = True,
    ):
        """Initialize HybridTrajectoryRegressor."""
        self.base_estimator = base_estimator
        self.scale_inputs = scale_inputs
        self.scaler = StandardScaler() if scale_inputs else None
        self.fitted_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridTrajectoryRegressor":
        """Fit hybrid regressor on concatenated features."""
        if self.scale_inputs:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        if self.base_estimator is None:
            self.fitted_model = Ridge(alpha=10.0)
        else:
            from sklearn.base import clone
            self.fitted_model = clone(self.base_estimator)

        self.fitted_model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target."""
        if self.scale_inputs:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        return self.fitted_model.predict(X_scaled)


def get_dl_get_off_models() -> Dict[str, Tuple[List[str], BaseEstimator]]:
    """Return feature sets and model estimators for Task 1: DL Get-Off."""
    trad_cols = [
        "combine_weight",
        "combine_height",
        "ten_yd_split",
        "forty",
        "vertical",
        "broad_jump",
    ]
    naive_cols = trad_cols + ["all_peak_speed", "all_peak_accel"]
    m2_cols = [
        "ten_yd_split",
        "forty_accel_05",
        "forty_jerk_05",
        "forty_power_05",
        "dl_peak_an",
        "combine_weight",
        "broad_jump",
        "vertical",
    ]

    m0 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    m1 = GradientBoostingRegressor(
        n_estimators=40, max_depth=2, learning_rate=0.08, subsample=0.8, random_state=42
    )
    m2 = GradientBoostingRegressor(
        n_estimators=45, max_depth=2, learning_rate=0.06, subsample=0.8, random_state=42
    )

    return {
        "M0": (trad_cols, m0),
        "M1": (naive_cols, m1),
        "M2": (m2_cols, m2),
    }


def get_pressure_rate_models() -> Dict[str, Tuple[List[str], BaseEstimator]]:
    """Return feature sets and model estimators for Task 2: Pass Rush Pressure Rate."""
    trad_cols = [
        "combine_weight",
        "combine_height",
        "ten_yd_split",
        "forty",
        "vertical",
        "broad_jump",
    ]
    naive_cols = trad_cols + ["all_peak_speed", "all_peak_accel"]
    m2_cols = trad_cols + ["forty_jerk_05", "forty_power_05", "dl_peak_an"]

    m0 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    m1 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    m2 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=8.0))])

    return {
        "M0": (trad_cols, m0),
        "M1": (naive_cols, m1),
        "M2": (m2_cols, m2),
    }


def get_wr_cushion_models() -> Dict[str, Tuple[List[str], BaseEstimator]]:
    """Return feature sets and model estimators for Task 3: WR Cushion Respect."""
    trad_cols = [
        "combine_weight",
        "combine_height",
        "ten_yd_split",
        "forty",
        "vertical",
        "broad_jump",
    ]
    naive_cols = trad_cols + ["all_peak_speed", "all_peak_accel"]
    m2_cols = [
        "forty_accel_05",
        "forty_jerk_05",
        "all_peak_speed",
        "vertical",
        "combine_height",
        "arm_length",
    ]

    m0 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=30.0))])
    m1 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=30.0))])
    m2 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=15.0))])

    return {
        "M0": (trad_cols, m0),
        "M1": (naive_cols, m1),
        "M2": (m2_cols, m2),
    }
