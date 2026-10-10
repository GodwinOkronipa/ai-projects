import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_model(model_name: str, random_state: int = 42, **kwargs):
    """
    Factory function to instantiate classification models with tuned hyperparameters.
    """
    model_name = model_name.lower()

    if model_name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=kwargs.get("C", 0.5),
                max_iter=kwargs.get("max_iter", 2000),
                random_state=random_state,
                solver="lbfgs"
            ))
        ])

    elif model_name in ["lgbm", "lightgbm"]:
        if not HAS_LGBM:
            raise ImportError("lightgbm is not installed.")
        return LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            learning_rate=kwargs.get("learning_rate", 0.03),
            num_leaves=kwargs.get("num_leaves", 15),
            max_depth=kwargs.get("max_depth", 4),
            min_child_samples=kwargs.get("min_child_samples", 30),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            reg_alpha=kwargs.get("reg_alpha", 0.5),
            reg_lambda=kwargs.get("reg_lambda", 2.0),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    elif model_name in ["xgb", "xgboost"]:
        if not HAS_XGB:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            learning_rate=kwargs.get("learning_rate", 0.03),
            max_depth=kwargs.get("max_depth", 4),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            reg_alpha=kwargs.get("reg_alpha", 0.5),
            reg_lambda=kwargs.get("reg_lambda", 2.0),
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )

    elif model_name in ["histgb", "hist"]:
        return HistGradientBoostingClassifier(
            max_iter=kwargs.get("max_iter", 150),
            learning_rate=kwargs.get("learning_rate", 0.03),
            max_depth=kwargs.get("max_depth", 4),
            min_samples_leaf=kwargs.get("min_samples_leaf", 30),
            l2_regularization=kwargs.get("l2_regularization", 1.0),
            random_state=random_state
        )

    elif model_name in ["extratrees", "et"]:
        return ExtraTreesClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            min_samples_leaf=kwargs.get("min_samples_leaf", 20),
            random_state=random_state,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")
