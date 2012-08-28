#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This snippet illustrates how lbfgs and leastsq work.
Useful for my own comprehension
"""


from scipy.optimize import *
from numpy import *
from matplotlib.pyplot import *



def simple_example():
    def f(x):   # The rosenbrock function
        return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
        
    out = fmin_l_bfgs_b(f, [2, 2], fprime=None, approx_grad=True)
    print(out)

from scipy.optimize.minpack import leastsq
def least_example():
    print('least example')
    def f(x, b0, b1):
        return b0*sin(b1*x)


    def res(params, xdata, ydata, function):
        return function(xdata, *params) - ydata

    x = linspace(20,40, 100)
    y = f(x,2.4,1.14)
    args = (x,y,f)
    out = leastsq(res, [2.2, 1.2], args=args)
    print(out)


def more_complex_example():
    print('complex example')
    def f(x, b0, b1):
        return b0*sin(b1*x)

    def res(params, xdata, ydata, function):
        return function(xdata, *params) - ydata

    def res_sum(params, xdata, ydata, function):
        return sum(  (function(xdata, *params) - ydata)**2 )

    x = linspace(20,40, 100)
    y = f(x,2.4,1.14)
    args = (x,y,f)
    out = fmin_l_bfgs_b(res_sum, [2.3, 1.1], args=args, fprime=None, approx_grad=True)

    print(out)


if __name__ == '__main__':
    simple_example()
    least_example()
    more_complex_example()
