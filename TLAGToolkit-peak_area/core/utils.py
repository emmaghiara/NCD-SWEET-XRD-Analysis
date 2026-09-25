# from pybaselines.polynomial import modpoly, imodpoly
from pybaselines import polynomial as p
from pybaselines import spline as sp
# from BaselineRemoval import BaselineRemoval

def spline_50_2_lam10(data):
    return sp.irsqr(data=data, lam=10, quantile=0.05, num_knots=50, spline_degree=2, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def zero(data):
    return p.imodpoly(data=data, poly_order=2, max_iter=500, tol=1e-6)[0] * 0

BASELINES_FUNCTIONS = {
    "None" : lambda x: x,
    "zero_line" : zero,
    "spline_50_2_lam10" : spline_50_2_lam10
}