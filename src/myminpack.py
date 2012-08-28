import warnings

"""
Code adapted from scipy.optimize
"""

from numpy import isscalar, sum

#quick and dirty imports
from scipy.optimize.minpack import leastsq
from scipy.optimize.minpack import *
from scipy.optimize.minpack import _general_function, _weighted_general_function
from scipy.optimize import fmin_l_bfgs_b






def sum_residual(params, xdata, ydata, function):
    """
    Return the sum of residuals
    """
    evaluation = function(xdata, *params) - ydata
    return sum( evaluation**2 )


# Use L-BFGS
#FIXME docstring
def curve_fit2(f, xdata, ydata, p0=None, sigma=None, **kw):
    """
    Use non-linear least squares to fit a function, f, to data.

    Assumes ``ydata = f(xdata, *params) + eps``

    Parameters
    ----------
    f : callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : An N-length sequence or an (k,N)-shaped array
        for functions with k predictors.
        The independent variable where the data is measured.
    ydata : N-length sequence
        The dependent data --- nominally f(xdata, ...)
    p0 : None, scalar, or M-length sequence
        Initial guess for the parameters.  If None, then the initial
        values will all be 1 (if the number of parameters for the function
        can be determined using introspection, otherwise a ValueError
        is raised).
    sigma : None or N-length sequence
        If not None, it represents the standard-deviation of ydata.
        This vector, if given, will be used as weights in the
        least-squares problem.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared error
        of ``f(xdata, *popt) - ydata`` is minimized
    pcov : 2d array
        The estimated covariance of popt.  The diagonals provide the variance
        of the parameter estimate.

    See Also
    --------
    leastsq

    Notes
    -----
    The algorithm uses the Levenburg-Marquardt algorithm through `leastsq`.
    Additional keyword arguments are passed directly to that algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import curve_fit
    >>> def func(x, a, b, c):
    ...     return a*np.exp(-b*x) + c

    >>> x = np.linspace(0,4,50)
    >>> y = func(x, 2.5, 1.3, 0.5)
    >>> yn = y + 0.2*np.random.normal(size=len(x))

    >>> popt, pcov = curve_fit(func, x, yn)

    """
    if p0 is None:
        # determine number of parameters by inspecting the function
        import inspect
        args, varargs, varkw, defaults = inspect.getargspec(f)
        if len(args) < 2:
            msg = "Unable to determine number of fit parameters."
            raise ValueError(msg)
        if 'self' in args:
            p0 = [1.0] * (len(args)-2)
        else:
            p0 = [1.0] * (len(args)-1)

    if isscalar(p0):
        p0 = array([p0])

    args = (xdata, ydata, f)
    if sigma is None:
        func = _general_function
    else:
        func = _weighted_general_function
        args += (1.0/asarray(sigma),)

    # Remove full_output from kw, otherwise we're passing it in twice.
    return_full = kw.pop('full_output', False)
    #res = leastsq(func, p0, args=args, full_output=1, **kw)
    # fprime None & approx_grad True -> no derivative provided
    res = fmin_l_bfgs_b( sum_residual, p0, args=args, fprime=None, approx_grad=True, factr=100 , pgtol=1e-9, iprint=1) #, **kw)
    #(popt, pcov, infodict, errmsg, ier) = res

    #if ier not in [1,2,3,4]:
    #    msg = "Optimal parameters not found: " + errmsg
    #    raise RuntimeError(msg)

    #if (len(ydata) > len(p0)) and pcov is not None:
    #    s_sq = (func(popt, *args)**2).sum()/(len(ydata)-len(p0))
    #    pcov = pcov * s_sq
    #else:
    #    pcov = inf

    #if return_full:
    #    return popt, pcov, infodict, errmsg, ier
    #else:
    #    return popt, pcov
    return res
