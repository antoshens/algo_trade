import numpy as np
from scipy.optimize import minimize


def find_max_sharpe_mocked(
    er: np.ndarray, cov: np.ndarray, rf: float = 0.04
) -> np.ndarray:
    n = len(er)
    init_weights = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def neg_sharpe(w):
        port_ret = np.dot(w, er)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if port_vol < 1e-8:
            return 0.0

        return -(port_ret - rf) / port_vol

    res = minimize(
        neg_sharpe, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )

    return res.x if res.success else init_weights
