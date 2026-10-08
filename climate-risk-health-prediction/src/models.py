import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional gradient boosting imports
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

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_model(model_name: str, random_state: int = 42, **kwargs):
    """
    Factory function to instantiate classification models with tuned defaults.
    """
    model_name = model_name.lower()

    if model_name == "logistic":
        # Logistic regression baseline with scaling
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=kwargs.get("C", 0.5),
                max_iter=kwargs.get("max_iter", 2000),
                random_state=random_state,
                class_weight=kwargs.get("class_weight", "balanced"),
                solver="lbfgs"
            ))
        ])

    elif model_name in ["lgbm", "lightgbm"]:
        if not HAS_LGBM:
            raise ImportError("lightgbm is not installed.")
        return LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 400),
            learning_rate=kwargs.get("learning_rate", 0.03),
            num_leaves=kwargs.get("num_leaves", 31),
            max_depth=kwargs.get("max_depth", 6),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            reg_alpha=kwargs.get("reg_alpha", 0.1),
            reg_lambda=kwargs.get("reg_lambda", 1.0),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    elif model_name in ["xgb", "xgboost"]:
        if not HAS_XGB:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 400),
            learning_rate=kwargs.get("learning_rate", 0.03),
            max_depth=kwargs.get("max_depth", 5),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            reg_alpha=kwargs.get("reg_alpha", 0.1),
            reg_lambda=kwargs.get("reg_lambda", 1.0),
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )

    elif model_name in ["catboost", "cb"]:
        if not HAS_CATBOOST:
            raise ImportError("catboost is not installed.")
        return CatBoostClassifier(
            iterations=kwargs.get("iterations", 500),
            learning_rate=kwargs.get("learning_rate", 0.03),
            depth=kwargs.get("depth", 6),
            l2_leaf_reg=kwargs.get("l2_leaf_reg", 3.0),
            random_seed=random_state,
            verbose=0
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")
