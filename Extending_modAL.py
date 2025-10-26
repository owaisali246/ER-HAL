from cte import N_INSTANCES
import numpy as np
from sklearn.base import BaseEstimator
from modAL.utils.data import modALinput
from modAL.density import information_density
from modAL.uncertainty import classifier_uncertainty
from modAL.utils.combination import make_linear_combination

def classifier_density(classifier: BaseEstimator, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    return information_density(X, 'l1')

def update_omega_values(e):
    return make_linear_combination(classifier_uncertainty, classifier_density, weights=[1 - e, e])

def _as_int(n):
    try:
        return int(np.asarray(n).item())
    except Exception:
        return int(n)

def custom_query_strategy(classifier, X, omega_argument=None):
    linear_combination = update_omega_values(omega_argument)
    utilities = linear_combination(classifier, X)

    utilities = np.asarray(utilities, dtype=float).ravel()
    utilities = np.nan_to_num(utilities, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)

    n = _as_int(N_INSTANCES)
    n = max(1, min(n, utilities.shape[0]))

    top = np.argpartition(utilities, -n)[-n:]
    top = top[np.argsort(utilities[top])[::-1]]
    query_idx = np.asarray(top, dtype=np.intp)

    return query_idx
