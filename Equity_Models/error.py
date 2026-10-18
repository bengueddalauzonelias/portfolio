import numpy as np
import pandas as pd
from Black_Scholes import BlackScholes
from ImpliedVolatilitySurface import ImpliedVolatilitySurface


class CalibrationLoss:
    """
    Loss functions to evaluate / calibrate an option pricing model against the market implied volatility surface.

    The market quotes are the (expiry, strike) points of an ImpliedVolatilitySurface. For each quote, the market
    price, Black vega and implied volatility are computed once with the BlackScholes model (OTM convention :
    puts below the forward, calls above).

    The model must expose price_vanillas(strikes, maturity, isCall) -> np.ndarray (e.g. HestonModel),
    or be a callable with the same signature.

    Available losses (see LOSSES) :
    - Price space      : 'price_mse', 'price_rmse', 'price_mae', 'price_max'
    - Relative price   : 'rel_price_mse', 'rel_price_rmse', 'mape', 'log_price_mse'
    - Vega weighted    : 'vega_mse'      -> (P_model - P_mkt)^2 / Vega           (1/Vega weights)
                         'vega_iv_mse'   -> ((P_model - P_mkt) / Vega)^2         (1st order IV error, no inversion)
                         'vega_iv_rmse', 'vega_iv_huber'
    - Implied vol      : 'iv_mse', 'iv_rmse', 'iv_mae', 'iv_max'                (exact Black inversion)

    Each loss is aggregated either quote by quote (by_expiry=False) or first per expiry then across expiries
    (by_expiry=True), so that each expiry weighs the same whatever its number of quotes.
    """

    LOSSES = ('price_mse', 'price_rmse', 'price_mae', 'price_max',
              'rel_price_mse', 'rel_price_rmse', 'mape', 'log_price_mse',
              'vega_mse', 'vega_iv_mse', 'vega_iv_rmse', 'vega_iv_huber',
              'iv_mse', 'iv_rmse', 'iv_mae', 'iv_max')

    def __init__(self, implied_vol_surface, bs_model, weights=None, bid_ask_vol_spreads=None,
                 by_expiry=True, failure_penalty=10.0, min_vega=1e-8):
        """
        Init method.
        - implied_vol_surface : market ImpliedVolatilitySurface (set_implied_vol_surface already called)
        - bs_model            : BlackScholes model sharing the same forward curve
        - weights             : optional {expiry: list of weights} aligned with implied_vol_surface.strikes
        - bid_ask_vol_spreads : optional {expiry: list of (ask vol - bid vol)}, gives weights 1 / spread^2
        - by_expiry           : aggregation mode (see class docstring)
        - failure_penalty     : value returned when the model produces non-finite prices / failed inversions
        - min_vega            : floor on vegas, avoids divisions by zero for deep OTM short-dated quotes
        """
        # Sanity checks
        if implied_vol_surface is None or not isinstance(implied_vol_surface, ImpliedVolatilitySurface):
            raise ValueError("implied_vol_surface must be an object of the ImpliedVolatilitySurface class.")
        if not hasattr(implied_vol_surface, 'implied_vols') or not implied_vol_surface.implied_vols:
            raise RuntimeError("Implied volatility surface not initialized. Call set_implied_vol_surface first.")
        if bs_model is None or not isinstance(bs_model, BlackScholes):
            raise ValueError("bs_model must be an object of the BlackScholes class.")
        if weights is not None and bid_ask_vol_spreads is not None:
            raise ValueError("Give either weights or bid_ask_vol_spreads, not both.")
        # Attributes
        self.iv_surface = implied_vol_surface
        self.bs_model = bs_model
        self.by_expiry = by_expiry
        self.failure_penalty = failure_penalty
        self.set_market_quotes(weights, bid_ask_vol_spreads, min_vega)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def set_market_quotes(self, weights=None, bid_ask_vol_spreads=None, min_vega=1e-8):
        """
        Setter method : stores, per expiry, the strikes, option types, market prices, vegas, vols and weights.
        """
        self.quotes = []
        for exp in self.iv_surface.expiries:
            strikes = np.array(self.iv_surface.strikes[exp], dtype=float)
            vols = np.array(self.iv_surface.implied_vols[exp], dtype=float)
            isCall = strikes > self.iv_surface.forward_price(exp)
            prices = np.array([self.bs_model.bs_price_vanilla(exp, K, v, bool(c)) for K, v, c in zip(strikes, vols, isCall)])
            vegas = np.array([self.bs_model.bs_vega(exp, K, v) for K, v in zip(strikes, vols)])
            if bid_ask_vol_spreads is not None:
                w = 1.0 / np.maximum(np.array(bid_ask_vol_spreads[exp], dtype=float), 1e-6) ** 2
            elif weights is not None:
                w = np.array(weights[exp], dtype=float)
            else:
                w = np.ones_like(strikes)
            if len(w) != len(strikes):
                raise ValueError(f"Weights at expiry {exp} must have one value per quoted strike.")
            self.quotes.append({"expiry": exp, "strikes": strikes, "isCall": isCall, "prices": prices,
                                "vegas": np.maximum(vegas, min_vega), "vols": vols, "weights": w})
        self.n_quotes = sum(len(q["strikes"]) for q in self.quotes)

    # ------------------------------------------------------------------
    # Model prices and residuals
    # ------------------------------------------------------------------
    @staticmethod
    def _pricer(model):
        if hasattr(model, 'price_vanillas'):
            return model.price_vanillas
        if callable(model):
            return model
        raise TypeError("model must expose price_vanillas(strikes, maturity, isCall) or be such a callable.")

    def model_prices(self, model):
        """
        Returns the list (one array per expiry) of model prices on the market quotes.
        """
        pricer = self._pricer(model)
        return [np.asarray(pricer(q["strikes"], q["expiry"], q["isCall"]), dtype=float) for q in self.quotes]

    def model_implied_vols(self, model, model_prices=None):
        """
        Returns the list (one array per expiry) of model implied volatilities (NaN where the inversion fails).
        """
        if model_prices is None:
            model_prices = self.model_prices(model)
        vols = []
        for q, prices in zip(self.quotes, model_prices):
            vols_exp = []
            for K, p, c in zip(q["strikes"], prices, q["isCall"]):
                v = None
                if np.isfinite(p) and p >= 0:
                    v = self.iv_surface.get_implied_vol(q["expiry"], K, float(p), bool(c), verbose=False)
                vols_exp.append(np.nan if v is None else v)
            vols.append(np.array(vols_exp))
        return vols

    def residuals(self, model, kind='vega_iv', model_prices=None):
        """
        Returns the list (one array per expiry) of residuals :
        - 'price'     : P_model - P_mkt
        - 'rel_price' : (P_model - P_mkt) / P_mkt
        - 'log_price' : ln(P_model) - ln(P_mkt)
        - 'vega_iv'   : (P_model - P_mkt) / Vega_mkt       (first order implied vol error)
        - 'iv'        : sigma_model - sigma_mkt            (exact inversion)
        """
        if model_prices is None:
            model_prices = self.model_prices(model)
        if kind == 'price':
            return [p - q["prices"] for q, p in zip(self.quotes, model_prices)]
        if kind == 'rel_price':
            return [(p - q["prices"]) / q["prices"] for q, p in zip(self.quotes, model_prices)]
        if kind == 'log_price':
            with np.errstate(divide='ignore', invalid='ignore'):
                return [np.log(p) - np.log(q["prices"]) for q, p in zip(self.quotes, model_prices)]
        if kind == 'vega_iv':
            return [(p - q["prices"]) / q["vegas"] for q, p in zip(self.quotes, model_prices)]
        if kind == 'iv':
            vols = self.model_implied_vols(model, model_prices)
            return [v - q["vols"] for q, v in zip(self.quotes, vols)]
        raise ValueError(f"Unknown residual kind '{kind}'.")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(self, values, reduce='mean'):
        """
        Weighted aggregation of per-quote values (list of arrays, one per expiry).
        reduce = 'mean' (weighted mean) or 'max'. Non-finite values return the failure penalty.
        """
        if any(not np.all(np.isfinite(v)) for v in values):
            return self.failure_penalty
        if reduce == 'max':
            return float(max(np.max(v) for v in values))
        weights = [q["weights"] for q in self.quotes]
        if self.by_expiry:
            return float(np.mean([np.sum(w * v) / np.sum(w) for w, v in zip(weights, values)]))
        w_all, v_all = np.concatenate(weights), np.concatenate(values)
        return float(np.sum(w_all * v_all) / np.sum(w_all))

    @staticmethod
    def huber(x, delta):
        """
        Huber function : quadratic below delta, linear above (robust to outlier quotes).
        """
        a = np.abs(x)
        return np.where(a <= delta, 0.5 * x**2, delta * (a - 0.5 * delta))

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def price_mse(self, model, model_prices=None):
        """ Mean squared error on prices. Overweights expensive (long-dated, ATM) options. """
        return self._aggregate([r**2 for r in self.residuals(model, 'price', model_prices)])

    def price_rmse(self, model, model_prices=None):
        """ Root mean squared error on prices (same unit as prices). """
        return np.sqrt(self.price_mse(model, model_prices))

    def price_mae(self, model, model_prices=None):
        """ Mean absolute error on prices. """
        return self._aggregate([np.abs(r) for r in self.residuals(model, 'price', model_prices)])

    def price_max(self, model, model_prices=None):
        """ Largest absolute price error. """
        return self._aggregate([np.abs(r) for r in self.residuals(model, 'price', model_prices)], reduce='max')

    def rel_price_mse(self, model, model_prices=None):
        """ Mean squared relative price error. Overweights cheap (deep OTM, short-dated) options. """
        return self._aggregate([r**2 for r in self.residuals(model, 'rel_price', model_prices)])

    def rel_price_rmse(self, model, model_prices=None):
        """ Root mean squared relative price error. """
        return np.sqrt(self.rel_price_mse(model, model_prices))

    def mape(self, model, model_prices=None):
        """ Mean absolute percentage error on prices (in %). """
        return 100 * self._aggregate([np.abs(r) for r in self.residuals(model, 'rel_price', model_prices)])

    def log_price_mse(self, model, model_prices=None):
        """ Mean squared error on log prices. """
        return self._aggregate([r**2 for r in self.residuals(model, 'log_price', model_prices)])

    def vega_mse(self, model, model_prices=None):
        """ Vega-weighted price loss : (P_model - P_mkt)^2 / Vega. Compromise between price and IV losses. """
        res = self.residuals(model, 'price', model_prices)
        return self._aggregate([r**2 / q["vegas"] for q, r in zip(self.quotes, res)])

    def vega_iv_mse(self, model, model_prices=None):
        """ ((P_model - P_mkt) / Vega)^2 : first order approximation of the IV MSE, without any inversion. """
        return self._aggregate([r**2 for r in self.residuals(model, 'vega_iv', model_prices)])

    def vega_iv_rmse(self, model, model_prices=None):
        """ Square root of vega_iv_mse (in vol units, e.g. 0.004 = 0.4 vol point). """
        return np.sqrt(self.vega_iv_mse(model, model_prices))

    def vega_iv_huber(self, model, model_prices=None, delta=0.01):
        """ Huber loss on the first order IV error (delta in vol units, default 1 vol point). """
        return self._aggregate([self.huber(r, delta) for r in self.residuals(model, 'vega_iv', model_prices)])

    def iv_mse(self, model, model_prices=None):
        """ Mean squared implied volatility error (exact inversion, failed inversions -> failure penalty). """
        return self._aggregate([r**2 for r in self.residuals(model, 'iv', model_prices)])

    def iv_rmse(self, model, model_prices=None):
        """ Root mean squared implied volatility error. """
        return np.sqrt(self.iv_mse(model, model_prices))

    def iv_mae(self, model, model_prices=None):
        """ Mean absolute implied volatility error. """
        return self._aggregate([np.abs(r) for r in self.residuals(model, 'iv', model_prices)])

    def iv_max(self, model, model_prices=None):
        """ Largest absolute implied volatility error. """
        return self._aggregate([np.abs(r) for r in self.residuals(model, 'iv', model_prices)], reduce='max')

    def loss(self, model, name='vega_iv_mse', model_prices=None, **kwargs):
        """
        Evaluates the loss called name (see LOSSES).
        """
        if name not in self.LOSSES:
            raise ValueError(f"Unknown loss '{name}'. Available losses : {self.LOSSES}")
        return getattr(self, name)(model, model_prices=model_prices, **kwargs)

    # ------------------------------------------------------------------
    # Penalties (Heston parameters)
    # ------------------------------------------------------------------
    @staticmethod
    def feller_penalty(model, mu=1.0):
        """
        Soft Feller condition : mu * max(xi^2 - 2 kappa theta, 0)^2.
        mu must be scaled to the loss : IV losses are ~1e-5 (vol units squared), so mu ~ 1e-4 - 1e-3,
        whereas price losses are ~1e2 - 1e3.
        """
        return mu * max(model.xi**2 - 2 * model.kappa * model.theta, 0.0) ** 2

    @staticmethod
    def tikhonov_penalty(params, params_ref, lam=1e-3, scale=None):
        """
        Tikhonov regularization toward a reference parameter set (e.g. the previous day's calibration) :
        lam * || (params - params_ref) / scale ||^2.
        """
        params, params_ref = np.asarray(params, dtype=float), np.asarray(params_ref, dtype=float)
        scale = np.ones_like(params) if scale is None else np.asarray(scale, dtype=float)
        return lam * float(np.sum(((params - params_ref) / scale) ** 2))

    # ------------------------------------------------------------------
    # Calibration objective
    # ------------------------------------------------------------------
    def objective(self, model, name='vega_iv_mse', param_names=('v0', 'theta', 'kappa', 'xi', 'rho'),
                  feller_mu=None, tikhonov=None, **kwargs):
        """
        Returns f(params) -> loss, ready for scipy.optimize.minimize.
        params are written into the model attributes listed in param_names before each evaluation.
        - feller_mu : if given, adds feller_penalty(model, feller_mu)
        - tikhonov  : if given, dict(params_ref=..., lam=..., scale=...) passed to tikhonov_penalty
        """
        def f(params):
            for name_, value in zip(param_names, params):
                setattr(model, name_, value)
            value = self.loss(model, name, **kwargs)
            if feller_mu is not None:
                value += self.feller_penalty(model, feller_mu)
            if tikhonov is not None:
                value += self.tikhonov_penalty(params, **tikhonov)
            return value
        return f

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report(self, model):
        """
        Returns (summary, by_expiry) DataFrames with the main error metrics. IV errors are in vol points.
        """
        prices = self.model_prices(model)
        price_res = self.residuals(model, 'price', prices)
        rel_res = self.residuals(model, 'rel_price', prices)
        iv_res = self.residuals(model, 'iv', prices)
        vega_iv_res = self.residuals(model, 'vega_iv', prices)

        rows = []
        for q, rp, rr, ri, rv in zip(self.quotes, price_res, rel_res, iv_res, vega_iv_res):
            rows.append({"Expiry": q["expiry"], "Quotes": len(q["strikes"]),
                         "Price RMSE": np.sqrt(np.mean(rp**2)), "Price max": np.max(np.abs(rp)),
                         "MAPE (%)": 100 * np.mean(np.abs(rr)),
                         "IV RMSE (vol pts)": 100 * np.sqrt(np.nanmean(ri**2)),
                         "IV max (vol pts)": 100 * np.nanmax(np.abs(ri)),
                         "IV bias (vol pts)": 100 * np.nanmean(ri),
                         "Vega-IV RMSE (vol pts)": 100 * np.sqrt(np.mean(rv**2)),
                         "Failed inversions": int(np.sum(np.isnan(ri)))})
        by_expiry = pd.DataFrame(rows).set_index("Expiry")

        all_p, all_r = np.concatenate(price_res), np.concatenate(rel_res)
        all_i, all_v = np.concatenate(iv_res), np.concatenate(vega_iv_res)
        summary = pd.Series({
            "Quotes": self.n_quotes,
            "Price RMSE": np.sqrt(np.mean(all_p**2)),
            "Price MAE": np.mean(np.abs(all_p)),
            "Price max": np.max(np.abs(all_p)),
            "MAPE (%)": 100 * np.mean(np.abs(all_r)),
            "IV RMSE (vol pts)": 100 * np.sqrt(np.nanmean(all_i**2)),
            "IV MAE (vol pts)": 100 * np.nanmean(np.abs(all_i)),
            "IV max (vol pts)": 100 * np.nanmax(np.abs(all_i)),
            "Vega-IV RMSE (vol pts)": 100 * np.sqrt(np.mean(all_v**2)),
            "Failed inversions": int(np.sum(np.isnan(all_i))),
        }, name="Global (quote by quote)")
        return summary, by_expiry
