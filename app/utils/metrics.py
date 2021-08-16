import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


class Product:
    def __init__(self, close):
        self.close = close

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change().fillna(0)

    def cumulative_returns(self):
        return self.returns().add(1).cumprod().sub(1)

    def annualized_volatility(self, periods_in_year: int = 252):
        return self.volatility() * np.sqrt(periods_in_year)

    def annualized_returns(self, periods_in_year: int = 252):
        cum_rets = self.cumulative_returns()
        total_rets = cum_rets.tail(1)
        return (1 + total_rets) ** (periods_in_year / cum_rets.shape[0]) - 1

    def total_returns(self) -> pd.DataFrame:
        return self.cumulative_returns().tail(1)

    def volatility(self) -> pd.DataFrame:
        return self.returns().std()

    def sharpe(self) -> pd.DataFrame:
        risk_free_rate = 0.03
        excess_return = self.annualized_returns() - risk_free_rate
        return excess_return / self.annualized_volatility()

    def drawdown(self):
        wealth_index = self.returns().add(1).cumprod()
        previous_peaks = wealth_index.cummax()
        return ((wealth_index - previous_peaks) / previous_peaks).fillna(0)

    def skewness(self):
        rets = self.returns()
        demeaned_rets = rets - rets.mean()
        exp = (demeaned_rets ** 3).mean()
        sigma_r = rets.std(ddof=0)
        return exp / sigma_r ** 3

    def kurtosis(self):
        rets = self.returns()
        demeaned_rets = rets - rets.mean()
        exp = (demeaned_rets ** 4).mean()
        sigma_r = rets.std(ddof=0)
        return exp / sigma_r ** 4

    def is_normal(self, level: float = 0.01):
        rets = self.returns()
        statistic, pvalue = scipy.stats.jarque_bera(rets)
        return pvalue > level

    def semi_deviation(self):
        rets = self.returns()
        is_negative = rets < 0
        return rets[is_negative].std(ddof=0)

    def var_historic(self, level: float = 5):
        rets = self.returns()
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.var_historic, level=level)
        elif isinstance(rets, pd.Series):
            return -np.percentile(rets, level)    # - because var already reference a loss
        else:
            raise TypeError("Expected DataFrame or Series")

    def c_var_historic(self, level=5):
        rets = self.returns()
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.c_var_historic, level=level)
        elif isinstance(rets, pd.Series):
            is_beyond = rets <= -self.var_historic(level=level)  # returns that are less than historic var
            return -rets[is_beyond].mean()
        else:
            raise TypeError("Expected DataFrame or Series")

    def var_gaussian(self, level=5):
        rets = self.returns()
        z_score = norm.ppf(level/100)
        return -(rets.mean() + z_score * rets.std(ddof=0))

    def var_cornish_fisher(self, level=5):
        rets = self.returns()
        z = norm.ppf(level/100)
        s = self.skewness()
        k = self.kurtosis()
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
        return -(rets.mean() + z * rets.std(ddof=0))

    def correlation(self):
        return self.returns().corr()


class Portfolio(Product):
    def __init__(self, close, weights):
        super().__init__(close)
        self.weights = weights
        self.product_rets = Product(close).returns()

    def returns(self):
        product_rets = self.product_rets
        return (product_rets * self.weights).sum(axis=1)

    def volatility(self):
        cov_mat = self.product_rets.cov()
        return (self.weights.T @ cov_mat @ self.weights) ** 0.5

    @staticmethod
    def portfolio_volatility_ef(cov_mat, weights):
        return (weights.T @ cov_mat @ weights) ** 0.5

    @staticmethod
    def portfolio_return_ef(weights, rets):
        return weights.T @ rets

    def optimal_weights(self, n_points, er, cov):
        target_rets = np.linspace(er.min(), er.max(), n_points)
        weights = [self.minimize_vol(target_ret, er, cov) for target_ret in target_rets]
        return weights

    def minimize_vol(self, target_ret, er, cov):
        n_assets = er.shape[0]
        init_guess = np.repeat(1 / n_assets, n_assets)
        bounds = ((0.0, 1.0),) * n_assets

        is_target_return = {
            "type": "eq",
            "args": (er,),
            "fun": lambda weights, er: target_ret - self.portfolio_return_ef(weights, er)
        }
        full_allocated = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        }

        results = minimize(self.portfolio_volatility_ef, init_guess,
                           args=(cov,), method="SLSQP",
                           options={"disp": False},
                           constraints=(is_target_return, full_allocated),
                           bounds=bounds)
        return results.x

    def efficient_frontier(self, n_points=25):
        er = self.annualized_returns()
        cov = self.product_rets.cov()
        weights = self.optimal_weights(n_points, er, cov)
        rets = [self.portfolio_return_ef(w, er) for w in weights]
        vols = [self.portfolio_volatility_ef(w, cov) for w in weights]
        return pd.DataFrame({"Returns": rets, "Volatility": vols})