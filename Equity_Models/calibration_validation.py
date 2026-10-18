# This script provides the out-of-sample and stability tests of the Heston calibrations (Model/heston_test.ipynb,
# section 11). Every test reruns the whole calibration procedure of section 10 on modified data :
#   - out-of-sample : one expiry left out (leave-one-expiry-out), or one strike out of two left out ;
#   - stability     : market implied volatilities perturbed by an i.i.d. Gaussian noise.
# The calibrations are independent : they run in parallel with joblib.
import copy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from Heston import HestonModel, TimeDependentHestonModel
from optimizer import Calibrator

FULL = ("theta", "xi", "rho")
# Name -> number of free parameters, in the order of the calibration procedure
FAMILY = {"Heston": 5, "theta(t), global": 15, "(theta, xi, rho)(t), bootstrap": 40,
          "(theta, xi, rho)(t), global": 41, "(theta, xi, rho)(t), global + smoothness": 41}


# ------------------------------------------------------------------
# Modified calibration data
# ------------------------------------------------------------------
def subset_loss(loss, keep):
    """ Copy of a CalibrationLoss restricted to the expiries whose index is in keep. """
    out = copy.copy(loss)
    out.quotes = [q for i, q in enumerate(loss.quotes) if i in set(keep)]
    out.n_quotes = sum(len(q["strikes"]) for q in out.quotes)
    return out


def strike_split_loss(loss, parity):
    """ Copy of a CalibrationLoss where the quotes of index parity, parity + 2, ... (per expiry) are kept for the
    calibration (weight 1) and the others are left out (weight 0). """
    out = copy.copy(loss)
    out.quotes = []
    for q in loss.quotes:
        q = dict(q)
        q["weights"] = (np.arange(len(q["strikes"])) % 2 == parity).astype(float)
        out.quotes.append(q)
    return out


def perturbed_loss(loss, bs_model, vol_noise, seed):
    """ Copy of a CalibrationLoss whose implied volatilities are shifted by an i.i.d. N(0, vol_noise^2) noise (prices
    recomputed, vegas kept). """
    rng = np.random.default_rng(seed)
    out = copy.copy(loss)
    out.quotes = []
    for q in loss.quotes:
        q = dict(q)
        q["vols"] = q["vols"] + vol_noise * rng.standard_normal(len(q["vols"]))
        q["prices"] = np.array([bs_model.bs_price_vanilla(q["expiry"], K, v, bool(c))
                                for K, v, c in zip(q["strikes"], q["vols"], q["isCall"])])
        out.quotes.append(q)
    return out


# ------------------------------------------------------------------
# Calibration procedure (section 10)
# ------------------------------------------------------------------
def calibrate_family(loss, heston, smoothness=1e-4, global_maxiter=300, smooth_maxiter=150):
    """
    Reruns the calibration procedure of section 10 on loss :
    constant Heston (local run from heston), theta(t) bootstrap + global, (theta, xi, rho)(t) bootstrap, + global,
    + global with smoothness. The intervals end at the expiries of loss.
    Returns {name: parameters} (constant : dict ; time-dependent : dict with arrays and times).
    """
    times = [q["expiry"] for q in loss.quotes]
    constant = copy.copy(heston)
    Calibrator(constant, loss, loss_name='vega_iv_mse').run('local', x0=heston.params)

    theta = TimeDependentHestonModel.from_heston(constant, times)
    theta.calibrate_bootstrap(loss, free=("theta",), verbose=False)
    theta.calibrate_global(loss, free=("theta",), maxiter=global_maxiter, verbose=False)
    boot = TimeDependentHestonModel.from_heston(constant, times)
    boot.calibrate_bootstrap(loss, verbose=False)
    full = copy.copy(boot)
    full.calibrate_global(loss, maxiter=global_maxiter, verbose=False)
    smooth = copy.copy(full)
    smooth.calibrate_global(loss, smoothness=smoothness, maxiter=smooth_maxiter, verbose=False)

    out = {"Heston": dict(constant.params)}
    for name, model in zip(list(FAMILY)[1:], [theta, boot, full, smooth]):
        out[name] = {**model.params, "times": np.array(times)}
    return out


def build_model(params, heston):
    """ Model from the output of calibrate_family. """
    if "times" not in params:
        model = copy.copy(heston)
        for name, value in params.items():
            setattr(model, name, value)
        return model
    return TimeDependentHestonModel(heston.zc_rate, heston.forward_curve, params["times"], v0=params["v0"],
                                    kappa=params["kappa"], theta=params["theta"], xi=params["xi"], rho=params["rho"])


def run_parallel(jobs, n_jobs=-1, verbose=0, **kwargs):
    """ Runs calibrate_family(loss, heston, **kwargs) for each (key, loss, heston) of jobs. Returns {key: result}. """
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(calibrate_family)(loss, heston, **kwargs)
                                                       for _, loss, heston in jobs)
    return {key: res for (key, _, _), res in zip(jobs, results)}


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------
def iv_errors(model, loss):
    """ Model minus market implied volatilities (vol units), one array per expiry of loss. """
    vols = loss.model_implied_vols(model)
    return [v - q["vols"] for v, q in zip(vols, loss.quotes)]


def rmse(errors):
    """ RMSE in vol points of a list of arrays of vol errors (NaN ignored). """
    e = np.concatenate([np.ravel(x) for x in errors])
    return 100 * np.sqrt(np.nanmean(e**2))
