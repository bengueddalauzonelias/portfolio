import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import savgol_filter



class ForwardCurve:
    """
    The implied forward curve class.
    """
    def __init__(self, S0, zc_rate, listed_expiries, market_strikes, call_prices, put_prices):
        """
        Init method : It sets the implied forwards, borrow + dividend yields, and plot them both.
        """
        # Sanity checks
        if S0 <= 0: raise ValueError("S0 (spot price) must be positive.")
        if zc_rate is None or not callable(zc_rate):
            raise ValueError("zc_rate must be a callable function.")
        if not listed_expiries or len(listed_expiries) == 0:
            raise ValueError("listed_expiries cannot be empty.")
        if not all(exp > 0 for exp in listed_expiries):
            raise ValueError("All expiries must be positive.")
        if len(market_strikes) != len(call_prices) or len(market_strikes) != len(put_prices):
            raise ValueError("market_strikes, call_prices, and put_prices must have the same length.")
        if not all(exp in market_strikes for exp in listed_expiries):
            raise ValueError("All listed_expiries must exist in market_strikes dictionary.")
        if any(len(market_strikes[exp]) == 0 for exp in listed_expiries):
            raise ValueError("No empty strike lists allowed.")
        
        # Attributes
        self.S0 = S0
        self.zc_rate = zc_rate
        self.listed_expiries = listed_expiries
        self.strikes = market_strikes
        self.call_prices = call_prices
        self.put_prices = put_prices
        self.set_implied_forwards()
        self.set_implied_borrow_and_divs()
        self.plot_implied_borrow_and_divs()



    
    def implied_forward(self, exp):
        """
        returns the implied forward using the Call / Put parity.
        The decreasing function K => Call(K) - Put(K) is linearly interpolated between the market strikes,
        so its root is computed exactly on the segment where it changes sign.
        If there is no sign change, the root is linearly extrapolated from the first / last segment
        (consistent with the parity : Call(K) - Put(K) = DF * (F - K) is linear in K).
        """
        # Sanity checks
        if exp not in self.listed_expiries: raise ValueError(f"Expiry {exp} not found in listed_expiries.")
        if len(self.strikes[exp]) < 2: raise ValueError(f"At least 2 strikes are needed for expiry {exp}.")

        # Call - Put differences on the (sorted) strike grid
        strikes = np.asarray(self.strikes[exp], dtype=float)
        order = np.argsort(strikes)
        strikes = strikes[order]
        if np.any(np.diff(strikes) == 0): raise ValueError(f"Duplicate strikes for expiry {exp}.")
        diff = (np.asarray(self.call_prices[exp], dtype=float) - np.asarray(self.put_prices[exp], dtype=float))[order]

        # Segment where Call - Put changes sign
        sign_change = np.where(diff[:-1] * diff[1:] <= 0)[0]
        if len(sign_change) > 0:
            i = sign_change[0]
        elif diff[0] < 0:
            i = 0                   # Forward below the lowest strike : extrapolate from the first segment
        else:
            i = len(strikes) - 2    # Forward above the highest strike : extrapolate from the last segment

        # Root of the linear interpolation on [K_i, K_i+1]
        K_lo, K_hi = strikes[i], strikes[i + 1]
        d_lo, d_hi = diff[i], diff[i + 1]
        if d_lo == d_hi:
            raise ValueError(f"Call-Put diff is flat on [{K_lo}, {K_hi}] for expiry {exp}: cannot imply the forward.")
        return K_lo - d_lo * (K_hi - K_lo) / (d_hi - d_lo)

    def set_implied_forwards(self):
        """
        Setter method : Stores the market implied forwards in a dictionary.
        """
        implied_forwards = dict()
        for exp in self.listed_expiries: implied_forwards[exp] = self.implied_forward(exp)
        self.implied_forwards = implied_forwards

    def set_implied_borrow_and_divs(self):
        """
        Setter method of implied borrow plus dividends, assuming they are piecewise constant between the market expiries.
        """
        implied_borrow_and_divs = dict()
        cumul = 0.
        exp_prev = 0
        for exp in self.listed_expiries:
            dt = exp - exp_prev
            if dt <= 0: raise ValueError("The given listed expiries are not strictly increasing. Check input data.")
            implied_borrow_and_divs[exp] = np.log(self.S0 * np.exp(self.zc_rate(exp) * exp - cumul) / self.implied_forwards[exp]) / (exp - exp_prev)
            cumul += implied_borrow_and_divs[exp] * (exp - exp_prev)
            exp_prev = exp
        # Implied Borrow + Dividend Yields
        self.implied_borrow_and_divs = implied_borrow_and_divs
        # Borrow + Dividend function : Piece-Wise Interpolation / Flat Extrapolation
        times = [0] + self.listed_expiries
        values = list(implied_borrow_and_divs.values()) + [list(implied_borrow_and_divs.values())[-1]]
        self.borrow_div_func = interp1d(times, values, kind='previous', bounds_error=False, fill_value=(values[0], values[-1]))
        
    def calc_forward(self, exp):
        """
        Computes the forward up to exp, given the borrow_div_func method.
        """
        if exp < 0: raise ValueError("Expiry (exp) must be non-negative.")
        if exp == 0: return self.S0
        expiries = [0] + self.listed_expiries + [exp]
        expiries = sorted(list(set(expiries)))
        expiries = expiries[:expiries.index(exp) + 1]
        cumul = 0
        for i in range(len(expiries) - 1):
            cumul += (expiries[i + 1] - expiries[i]) * self.borrow_div_func(expiries[i])
        return self.S0 * np.exp(self.zc_rate(exp) * exp - cumul)

    def fwd_curve(self, apply_smoothness=True, dt= 1e-4, n_interv=101, order=3):
        """
        Computes the forward curve function, using the calc_forward method.
        A smoothness is applied if apply_smoothness is True.
        """
        if not apply_smoothness : 
            return (lambda exps : [self.calc_forward(exp) for exp in exps])
        grid_t = list(np.linspace(0, self.listed_expiries[-1], int(1/dt) + 1, endpoint=True)) +  self.listed_expiries
        grid_t = sorted(list(set(grid_t)))
        calc_fwd_func = lambda exps : [self.calc_forward(exp) for exp in exps]
        smooth_fwds = savgol_filter(calc_fwd_func(grid_t), n_interv, order)
        return interp1d(grid_t, smooth_fwds, extrapolate=True)
    
    def plot_implied_borrow_and_divs(self, n_interv=1000):
        """
        Implied data plotter.
        """
        print("Implied forward prices :")
        for exp in self.listed_expiries:
            print(f"Expiry {exp:.4f} : The implied forward is {self.implied_forwards[exp]:.4f}")
        print("\nAnd its corresponding \"Borrow + Dividend\" curve :")
        plt.title("Implied Borrow & Dividend Yields")
        plt.ylabel("Borrow + Dividend Yield (%)")
        plt.xlabel("Expiry (years)")
        x_axis = np.linspace(0, self.listed_expiries[-1] + 0.1, n_interv)
        plt.plot(x_axis, 100 * self.borrow_div_func(x_axis))
        plt.show()