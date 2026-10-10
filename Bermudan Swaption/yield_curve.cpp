#include <vector>
#include <iostream>
#include <algorithm>


class linear_yieldcurve {

private:

    std::vector<double> maturities;
    std::vector<double> rates;

public:
    // Constructeur pour initialiser la courbe des taux.
    yieldcurve(const std::vector<double>& mats, const std::vector<double>& rts) : maturities(mats), rates(rts) {}

    // Methodology in order to get the rate for all t.

    double getrate(double t) const{

        // We do not manage the extrapolation of the yield curve

        if (t <= maturities.front()) return rates.front();
        if (t >= maturities.back()) return rates.back();

        // find the intervalle

        auto it = std::lower_bound(maturities.begin(), maturities.end(), t);
        int idx = std::distance(maturities.begin(), it);

        int idx1 = idx - 1;
        int idx2 = idx;

        double t1 = maturities[idx1];
        double t2 = maturities[idx2];
        double r1 = rates[idx1];
        double r2 = rates[idx2];

        return r1 + (t - t1) / (t2 - t1) * (r2 - r1);

    }
    double getDiscountFactor(double t) const {
        double r = getrate(t);
        return std::exp(-r * t);
    }

};


class 