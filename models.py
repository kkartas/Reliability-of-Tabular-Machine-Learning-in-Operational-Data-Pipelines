from typing import Dict

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_models(seed: int) -> Dict[str, object]:
    return {
        "logreg": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_depth=None,
            learning_rate=0.1,
            max_iter=200,
            random_state=seed,
            loss="log_loss",
        ),
    }
