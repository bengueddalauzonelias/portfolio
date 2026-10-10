import time
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import qmc
from error import CalibrationLoss


# This script is used to search the global minimum of a model calibration.
# It takes as input one model (exposing a PARAMS dict {name: (lower, upper)}) and a CalibrationLoss from error.py.
# Available searches : Latin Hypercube multi-start (LHS + local L-BFGS-B) and Differential Evolution,
# both with an optional Tikhonov penalty toward a reference parameter set.
# All optimizers work on parameters rescaled to [0, 1] (u = (x - lower) / (upper - lower)),
# so that parameters of different magnitudes (v0 ~ 0.02, kappa ~ 2) are treated alike.


class Calibrator:
    """
    Model-agnostic calibrator.
    """

    def __init__(self, model, loss, loss_name='vega_iv_mse', bounds=None, fixed=None, tikhonov=None, penalties=None):
        """
        Init method.
        - model     : object whose calibrated parameters are attributes, with a PARAMS dict {name: (lower, upper)}
        - loss      : CalibrationLoss object (error.py)
        - loss_name : one of CalibrationLoss.LOSSES
        - bounds    : optional {name: (lower, upper)} overriding model.PARAMS
        - fixed     : optional {name: value}, parameters kept constant during the calibration
        - tikhonov  : optional dict(params_ref={name: value}, lam=float), adds
                      lam * sum_i ((x_i - x_ref_i) / (upper_i - lower_i))^2 over the free parameters
        - penalties : optional list of callables model -> float added to the loss (e.g. a soft Feller condition)
        """
        # Sanity checks
        if not isinstance(loss, CalibrationLoss):
            raise ValueError("loss must be an object of the CalibrationLoss class.")
        if loss_name not in CalibrationLoss.LOSSES:
            raise ValueError(f"Unknown loss '{loss_name}'. Available losses : {CalibrationLoss.LOSSES}")
        all_bounds = dict(getattr(model, 'PARAMS', {}))
        all_bounds.update(bounds or {})
        if not all_bounds:
            raise ValueError("No parameter bounds : define model.PARAMS or pass bounds.")
        self.fixed = dict(fixed or {})
        for name in self.fixed:
            if name not in all_bounds:
                raise ValueError(f"Fixed parameter '{name}' is not a model parameter.")
        # Attributes
        self.model = model
        self.loss = loss
        self.loss_name = loss_name
        self.names = [n for n in all_bounds if n not in self.fixed]
        self.lower = np.array([all_bounds[n][0] for n in self.names], dtype=float)
        self.upper = np.array([all_bounds[n][1] for n in self.names], dtype=float)
        self.penalties = list(penalties or [])
        self.tikhonov = None
        if tikhonov is not None:
            ref = tikhonov["params_ref"]
            self.tikhonov = {"u_ref": self._to_unit(np.array([ref[n] for n in self.names], dtype=float)),
                             "lam": float(tikhonov["lam"])}
        self.runs = []
        self.n_evals = 0

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def _to_unit(self, x):
        return (np.asarray(x, dtype=float) - self.lower) / (self.upper - self.lower)

    def _from_unit(self, u):
        return self.lower + np.clip(u, 0.0, 1.0) * (self.upper - self.lower)

    def _set_params(self, x):
        for name, value in self.fixed.items():
            setattr(self.model, name, value)
        for name, value in zip(self.names, x):
            setattr(self.model, name, float(value))

    def to_dict(self, x):
        """ Full parameter dictionary (free + fixed) from a free parameter vector x. """
        params = dict(zip(self.names, map(float, x)))
        params.update(self.fixed)
        return params

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def fit_loss(self, x):
        """ Loss without penalties, at the free parameter vector x (original scale). """
        self._set_params(x)
        return self.loss.loss(self.model, self.loss_name)

    def penalty(self, u):
        """ Tikhonov + user penalties at the unit vector u (model parameters must already be set). """
        value = 0.0
        if self.tikhonov is not None:
            value += self.tikhonov["lam"] * float(np.sum((u - self.tikhonov["u_ref"]) ** 2))
        for pen in self.penalties:
            value += pen(self.model)
        return value

    def objective(self, u):
        """ Objective minimised by every search, as a function of the unit vector u. """
        self.n_evals += 1
        value = self.fit_loss(self._from_unit(u)) + self.penalty(np.clip(u, 0.0, 1.0))
        return value if np.isfinite(value) else self.loss.failure_penalty

    # ------------------------------------------------------------------
    # Searches
    # ------------------------------------------------------------------
    def local(self, x0, maxiter=200, ftol=1e-10, label="local"):
        """
        Local L-BFGS-B minimisation from x0 (original scale : dict or vector of the free parameters).
        """
        if isinstance(x0, dict):
            x0 = [x0[n] for n in self.names]
        start, n0 = time.time(), self.n_evals
        u0 = np.clip(self._to_unit(x0), 0.0, 1.0)
        res = opt.minimize(self.objective, u0, method='L-BFGS-B', bounds=[(0.0, 1.0)] * len(self.names),
                           options={'ftol': ftol, 'maxiter': maxiter})
        return self._record(label, res.x, res.success, self.n_evals - n0, time.time() - start, start_x=x0)

    def latin_hypercube(self, n_samples=200, n_starts=10, seed=0, maxiter=200):
        """
        Multi-start search :
        1. draws n_samples points with a Latin Hypercube over the bounds and evaluates the objective on each ;
        2. runs a local L-BFGS-B from the n_starts best points ;
        3. returns the best result (all runs are kept in self.runs).
        """
        start, n0 = time.time(), self.n_evals
        samples = qmc.LatinHypercube(d=len(self.names), seed=seed).random(n_samples)
        values = np.array([self.objective(u) for u in samples])
        self.screening = pd.DataFrame(self._from_unit(samples), columns=self.names).assign(objective=values)
        best = [self.local(self._from_unit(samples[i]), maxiter=maxiter, label=f"lhs start {k + 1}")
                for k, i in enumerate(np.argsort(values)[:n_starts])]
        best_run = min(best, key=lambda r: r["objective"])
        print(f"Latin Hypercube : {n_samples} samples + {n_starts} local runs, {self.n_evals - n0} evaluations, "
              f"{time.time() - start:.1f}s")
        return self._finish(best_run)

    def differential_evolution(self, popsize=15, maxiter=100, tol=1e-8, seed=0, polish=True):
        """
        Differential Evolution over the bounds (population initialised with a Latin Hypercube),
        followed by a local L-BFGS-B polish.
        """
        start, n0 = time.time(), self.n_evals
        res = opt.differential_evolution(self.objective, bounds=[(0.0, 1.0)] * len(self.names), popsize=popsize,
                                         maxiter=maxiter, tol=tol, seed=seed, polish=polish, init='latinhypercube')
        run = self._record("differential evolution", res.x, res.success, self.n_evals - n0, time.time() - start)
        print(f"Differential Evolution : {res.nit} generations, {self.n_evals - n0} evaluations, "
              f"{time.time() - start:.1f}s")
        return self._finish(run)

    def run(self, method='lhs', **kwargs):
        """
        Single entry point : method = 'lhs' (Latin Hypercube multi-start), 'de' (Differential Evolution)
        or 'local' (kwargs must then contain x0).
        """
        if method == 'lhs':
            return self.latin_hypercube(**kwargs)
        if method == 'de':
            return self.differential_evolution(**kwargs)
        if method == 'local':
            return self._finish(self.local(**kwargs))
        raise ValueError("method must be 'lhs', 'de' or 'local'.")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _record(self, label, u, success, n_evals, elapsed, start_x=None):
        x = self._from_unit(u)
        fit = self.fit_loss(x)
        run = {"run": label, **self.to_dict(x), "loss": fit, "objective": fit + self.penalty(np.clip(u, 0.0, 1.0)),
               "evaluations": n_evals, "time (s)": elapsed, "success": bool(success)}
        if start_x is not None:
            run["start"] = np.round(np.asarray(start_x, dtype=float), 4).tolist()
        self.runs.append(run)
        return run

    def _finish(self, run):
        """ Sets the best parameters into the model and returns the run. """
        self._set_params([run[n] for n in self.names])
        return run

    def summary(self):
        """ One row per optimisation run, sorted by objective. """
        return pd.DataFrame(self.runs).sort_values("objective").reset_index(drop=True)

    def distinct_minima(self, tol=0.02):
        """
        Groups the runs whose unit-scaled parameters are within tol of each other.
        Returns one row per distinct minimum with the number of runs that reached it.
        """
        runs = sorted(self.runs, key=lambda r: r["objective"])
        groups = []
        for r in runs:
            u = self._to_unit([r[n] for n in self.names])
            for g in groups:
                if np.max(np.abs(u - g["u"])) < tol:
                    g["count"] += 1
                    break
            else:
                groups.append({"u": u, "count": 1, "run": r})
        rows = [{**{n: g["run"][n] for n in self.names}, "objective": g["run"]["objective"], "runs reaching it": g["count"]}
                for g in groups]
        return pd.DataFrame(rows)
