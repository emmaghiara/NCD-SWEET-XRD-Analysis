# from pybaselines.polynomial import modpoly, imodpoly
from pybaselines import polynomial as p
from pybaselines import spline as sp
# from BaselineRemoval import BaselineRemoval
  
def imodpoly_2(data):
    return p.imodpoly(data=data, poly_order=2, max_iter=500, tol=1e-6)[0]

def spline_30_2_lam10(data):
    return sp.irsqr(data=data, lam=10, quantile=0.05, num_knots=30, spline_degree=2, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def spline_30_3_lam100(data):
    return sp.irsqr(data=data, lam=100, quantile=0.05, num_knots=30, spline_degree=3, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def spline_30_3_lam10(data):
    return sp.irsqr(data=data, lam=10, quantile=0.05, num_knots=30, spline_degree=3, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def spline_30_3(data):
    return sp.irsqr(data=data, quantile=0.05, num_knots=30, spline_degree=3, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def spline_50_2_lam100(data):
    return sp.irsqr(data=data, lam=100, quantile=0.05, num_knots=50, spline_degree=2, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def spline_50_2_lam10(data):
    return sp.irsqr(data=data, lam=10, quantile=0.05, num_knots=50, spline_degree=2, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def spline_50_2_lam50(data):
    return sp.irsqr(data=data, lam=50, quantile=0.05, num_knots=50, spline_degree=2, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def splinemm_50_2_lam10(data):
    return sp.mixture_model(data, lam=10.0, p=0.01, num_knots=50, spline_degree=2, diff_order=3, max_iter=50, tol=0.001, weights=None, symmetric=False, num_bins=None)[0]

def spline_100_2_lam10(data):
    return sp.irsqr(data=data, lam=10, quantile=0.05, num_knots=100, spline_degree=2, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def spline_50_1_lam10(data):
    return sp.irsqr(data=data, lam=10, quantile=0.05, num_knots=50, spline_degree=1, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]

def zero(data):
    return p.imodpoly(data=data, poly_order=2, max_iter=500, tol=1e-6)[0] * 0

# def imodpoly_5(data):
#     return imodpoly(data=data, poly_order=5, max_iter=500, tol=1e-6)[0]

BASELINES_FUNCTIONS = {
    "None" : lambda x: x,
    "imodpoly_2" : imodpoly_2,
    "zero_line" : zero,
    "spline_30_2_lam10" : spline_30_2_lam10,
    "spline_30_3" : spline_30_3,
    "spline_30_3_lam10" : spline_30_3_lam10,
    "spline_30_3_lam100" : spline_30_3_lam100,
    "spline_50_2_lam100" : spline_50_2_lam100,
    "spline_50_2_lam10" : spline_50_2_lam10,
    "splinemm_50_2_lam10" : splinemm_50_2_lam10,
    "spline_50_1_lam10" : spline_50_1_lam10,
    "spline_100_2_lam10" : spline_100_2_lam10,
    "spline_50_2_lam50" : spline_50_2_lam50
}