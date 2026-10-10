# This script defines the callable SOFR range accrual note of the paper (Table 1) and its exact schedule :
# - quarterly coupon dates, modified following on the US government securities calendar ;
# - every calendar day d of (T_{j-1}, T_j] accrues (R_d + s) 1{L <= R_d <= U} / 360, where R_d is the SOFR of the
#   last business day <= d, frozen at the fixing of the cut-off date (5 business days before T_j) after it ;
# - issuer call at par on each coupon date from the end of year 1 (maturity excluded), notice 10 business days.
# The note only describes its contract : the coupon of each period is a weighted sum over its distinct fixing dates
# (weight = number of calendar days that use the fixing). Pricers (range_accrual_pricer.py) read these schedules.
import numpy as np
import pandas as pd


class CallableRangeAccrual:
    """
    Callable SOFR range accrual note (investor side : long the note).
    """
    kind = "callable_range_accrual"

    def __init__(self, issue_date, calendar, maturity_years=3, notional=10_000_000.0, spread=0.0125, lower=0.03,
                 upper=0.0425, frequency=4, non_call_years=1, notice_days=10, cutoff_days=5, callable=True):
        """
        Init method.
        - issue_date     : start of the first period (T_0)
        - calendar       : USGovCalendar (usd_market.py)
        - maturity_years : maturity in years
        - notional       : notional N (USD)
        - spread         : spread s added to SOFR on the days in range (0.0125 = 1.25%)
        - lower, upper   : range [L, U], bounds included (decimal)
        - frequency      : coupons per year
        - non_call_years : first call on the coupon date at the end of this period
        - notice_days    : call notice in business days before the call date
        - cutoff_days    : rate cut-off in business days before the end of each period
        - callable       : False gives the same note without the issuer call
        """
        # Sanity checks
        if not 0 <= lower < upper: raise ValueError("Need 0 <= lower < upper.")
        if notional <= 0: raise ValueError("notional must be positive.")
        if not 0 <= non_call_years < maturity_years: raise ValueError("Need 0 <= non_call_years < maturity_years.")
        # Attributes
        self.issue_date = pd.Timestamp(issue_date)
        self.calendar = calendar
        self.maturity_years = maturity_years
        self.notional = float(notional)
        self.spread = float(spread)
        self.lower = float(lower)
        self.upper = float(upper)
        self.frequency = int(frequency)
        self.non_call_years = non_call_years
        self.notice_days = int(notice_days)
        self.cutoff_days = int(cutoff_days)
        self.callable = bool(callable)
        self._build()

    def _build(self):
        months = 12 // self.frequency
        n = int(round(self.maturity_years * self.frequency))
        unadjusted = [self.issue_date + pd.DateOffset(months=months * k) for k in range(1, n + 1)]
        self.coupon_dates = pd.DatetimeIndex([self.calendar.modified_following(d) for d in unadjusted])
        starts = pd.DatetimeIndex([self.issue_date]).append(self.coupon_dates[:-1])
        self.periods = []
        for j, (start, end) in enumerate(zip(starts, self.coupon_dates), start=1):
            cutoff = self.calendar.add_business_days(end, -self.cutoff_days)
            days = pd.date_range(start + pd.Timedelta(days=1), end, freq="D")            # (T_{j-1}, T_j]
            fixing = pd.DatetimeIndex([self.calendar.previous_business_day(d) for d in days])
            fixing = fixing.where(fixing <= cutoff, cutoff)
            dates, weights = np.unique(fixing.values, return_counts=True)
            self.periods.append({"number": j, "start": start, "end": end, "cutoff": cutoff, "days": len(days),
                                 "fixing_dates": pd.DatetimeIndex(dates), "weights": weights.astype(float)})
        first_call = int(round(self.non_call_years * self.frequency))
        self.call_periods = list(range(first_call, n)) if self.callable else []    # coupon numbers j with a call
        self.notice_dates = {j: self.calendar.add_business_days(self.coupon_dates[j - 1], -self.notice_days)
                             for j in self.call_periods}

    @property
    def maturity_date(self):
        return self.coupon_dates[-1]

    def with_terms(self, **kwargs):
        """ Copy of the note with some terms changed (e.g. with_terms(callable=False, upper=0.045)). """
        terms = dict(issue_date=self.issue_date, calendar=self.calendar, maturity_years=self.maturity_years,
                     notional=self.notional, spread=self.spread, lower=self.lower, upper=self.upper,
                     frequency=self.frequency, non_call_years=self.non_call_years, notice_days=self.notice_days,
                     cutoff_days=self.cutoff_days, callable=self.callable)
        return CallableRangeAccrual(**{**terms, **kwargs})

    def accrual(self, rate):
        """ Daily accrual (R + s) 1{L <= R <= U}, annualised (vectorised in the rate). """
        rate = np.asarray(rate, dtype=float)
        return np.where((rate >= self.lower) & (rate <= self.upper), rate + self.spread, 0.0)

    def realized_coupons(self, fixings):
        """
        Coupons (per unit notional) of the periods fully fixed in `fixings` (Series of SOFR indexed by date) :
        sum_d accrual(R_d) / 360. Also returns the share of days in range.
        """
        rows = []
        for p in self.periods:
            if p["fixing_dates"][-1] > fixings.index[-1]:
                break
            rates = fixings.reindex(p["fixing_dates"]).values
            if np.isnan(rates).any():
                raise ValueError(f"Missing SOFR fixings in period {p['number']}.")
            in_range = (rates >= self.lower) & (rates <= self.upper)
            rows.append({"period": p["number"], "start": p["start"].date(), "end": p["end"].date(),
                         "days": p["days"], "days in range": int(np.sum(p["weights"] * in_range)),
                         "coupon (% of N)": 100 * np.sum(p["weights"] * self.accrual(rates)) / 360,
                         "coupon rate (% p.a.)": 100 * np.sum(p["weights"] * self.accrual(rates)) / p["days"]})
        return pd.DataFrame(rows)

    def schedule_frame(self):
        """ One row per coupon period : dates, cut-off, number of days and fixings, call and notice dates. """
        return pd.DataFrame([{"period": p["number"], "start": p["start"].date(), "end (payment)": p["end"].date(),
                              "cut-off": p["cutoff"].date(), "calendar days": p["days"],
                              "distinct fixings": len(p["fixing_dates"]),
                              "call date": p["number"] in self.call_periods,
                              "notice date": self.notice_dates[p["number"]].date()
                              if p["number"] in self.call_periods else None} for p in self.periods]).set_index("period")

    def __repr__(self):
        call = (f"callable from year {self.non_call_years:g} ({len(self.call_periods)} dates)" if self.callable
                else "non-callable")
        return (f"CallableRangeAccrual({self.issue_date.date()} -> {self.maturity_date.date()}, SOFR + "
                f"{1e4 * self.spread:.0f}bp in [{100 * self.lower:.2f}%, {100 * self.upper:.2f}%], {call})")
