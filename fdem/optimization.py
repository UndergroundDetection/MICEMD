# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:17:35 2020

@author: Liu Wen
"""


def steepest_descent(fun, grad, x0, iterations, tol):
    """


    Parameters
    ----------
    fun : function
        Objective function.
    grad : function
        Gradient function of objective function.
    x0 : numpy.array, size=9
        Initial value of the parameters to be estimated.
    iterations : int
        Maximum iterations of optimization algorithms.
    tol : float
        Tolerance of optimization algorithms.

    Returns
    -------
    xk : numpy.array, size=9
        Parameters wstimated by optimization algorithms.
    fval : float
        Objective function value at xk.
    grad_val : float
        Gradient value of objective function at xk.
    grad_log : numpy.array
        The record of gradient of objective function of each iteration.
    """

    xk = None
    fval = None
    grad_val = None
    grad_log = None

    return xk, fval, grad_val, grad_log


def BFGS(fun, grad, x0, iterations, tol):

    xk = None
    fval = None
    grad_val = None
    grad_log = None

    return xk, fval, grad_val, grad_log


def conjugate_gradient(fun, grad, x0, iterations, tol):

    xk = None
    fval = None
    grad_val = None
    grad_log = None

    return xk, fval, grad_val, grad_log


def LM(fun, grad, x0, iterations, tol):

    xk = None
    fval = None
    grad_val = None
    grad_log = None

    return xk, fval, grad_val, grad_log


def step_length1():
    pass


def step_length2():
    pass
