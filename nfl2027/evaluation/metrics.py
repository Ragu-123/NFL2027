"""Statistical evaluation metrics, out-of-fold cross-validation, and predictive benchmarks."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict


def evaluate_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    cv: Optional[KFold] = None,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute out-of-fold cross-validation and compute comprehensive regression statistics.

    Args:
        X: Feature matrix of shape (N, D)
        y: Target array of shape (N,)
        model: Scikit-learn compatible estimator or Pipeline
        cv: Optional KFold instance. If None, KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        n_splits: Number of CV splits (default: 5)
        random_seed: Random state for KFold shuffle

    Returns:
        Dictionary containing:
        - 'predictions': Out-of-fold predicted values (N,)
        - 'r2': R^2 coefficient of determination
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'pearson_r': Pearson correlation coefficient
        - 'pearson_p': Pearson p-value
        - 'spearman_rho': Spearman rank correlation
        - 'spearman_p': Spearman p-value
    """
    if cv is None:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    preds = cross_val_predict(model, X, y, cv=cv)

    r2 = float(r2_score(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))

    # Guard against zero-variance edge cases in small arrays
    if np.std(preds) > 1e-8 and np.std(y) > 1e-8:
        r_val, p_val = pearsonr(y, preds)
        rho_val, rho_p = spearmanr(y, preds)
    else:
        r_val, p_val = 0.0, 1.0
        rho_val, rho_p = 0.0, 1.0

    return {
        "predictions": preds,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": float(r_val),
        "pearson_p": float(p_val),
        "spearman_rho": float(rho_val),
        "spearman_p": float(rho_p),
    }


def run_benchmark_suite(
    tasks_dict: Dict[str, Tuple[pd.DataFrame, str, Dict[str, Tuple[List[str], Any]]]],
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run full multi-model benchmark across multiple translation tasks.

    Args:
        tasks_dict: Dict mapping task short names to tuples:
                    (df_task, target_col, model_dict)
                    where model_dict has keys like 'M0', 'M1', 'M2'
                    and values (feature_columns, estimator)
        n_splits: Number of cross-validation folds
        random_seed: Seed for KFold

    Returns:
        List of benchmark summary dictionaries with M0, M1, M2 metrics and gains.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    results = []

    for task_name, (df_task, target_col, model_dict) in tasks_dict.items():
        if len(df_task) < 10:
            continue

        y = df_task[target_col].values.astype(np.float64)
        task_res = {
            "Task": task_name,
            "ShortName": task_name,
            "N": len(y),
            "actual": y,
        }

        eval_records = {}
        for m_key in ["M0", "M1", "M2"]:
            if m_key in model_dict:
                cols, model = model_dict[m_key]
                # Impute missing values with median
                X = df_task[cols].fillna(df_task[cols].median()).values
                eval_out = evaluate_pipeline(X, y, model, cv=kf)
                eval_records[m_key] = eval_out
                task_res[f"{m_key}_R2"] = eval_out["r2"]
                task_res[f"{m_key}_RMSE"] = eval_out["rmse"]
                task_res[f"{m_key}_r"] = eval_out["pearson_r"]
                task_res[f"pred_{m_key.lower()}"] = eval_out["predictions"]

        # Information gains
        if "M0" in eval_records and "M2" in eval_records:
            r2_m0 = eval_records["M0"]["r2"]
            r2_m2 = eval_records["M2"]["r2"]
            rmse_m0 = eval_records["M0"]["rmse"]
            rmse_m2 = eval_records["M2"]["rmse"]

            task_res["Delta_R2"] = r2_m2 - r2_m0
            task_res["RMSE_Reduction_Pct"] = (
                ((rmse_m0 - rmse_m2) / max(rmse_m0, 1e-6)) * 100.0
            )

        results.append(task_res)

    return results
