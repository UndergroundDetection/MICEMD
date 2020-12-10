# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:17:35 2020

@author: Liu Wen
"""

import numpy as np
from numpy import asarray,Inf,isinf
from scipy.optimize.optimize import _line_search_wolfe12,_LineSearchError

def Steepest_descent(fun, grad, x0, iterations, tol):
    """
    Minimization of scalar function of one or more variables using the
    steepest descent algorithm.

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

    fval = None
    grad_val = None
    x_log = []
    y_log = []
    grad_log = []

    x0 = asarray(x0).flatten()
    iterations = len(x0) * 200
    old_fval = fun(x0)
    gfk = grad(x0)
    k = 0

    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    x_log = np.append(x_log, xk.T)
    y_log = np.append(y_log, fun(xk))
    grad_log = np.append(grad_log, np.linalg.norm(xk - x_log[-1:]))
    gnorm = np.amax(np.abs(gfk))

    while (gnorm > tol) and (k < iterations):
        pk = -gfk
        try:
            alpha, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(fun, grad, xk, pk, gfk, old_fval, old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            break
        xk = xk + alpha * pk
        k += 1
        grad_log = np.append(grad_log, np.linalg.norm(xk - x_log[-1:]))
        x_log = np.append(x_log, xk.T)
        y_log = np.append(y_log, fun(xk))
        if (gnorm <= tol):
            break
    fval = old_fval
    grad_val = grad_log[-1]

    return xk, fval, grad_val, x_log, y_log, grad_log

def BFGS(fun, grad, x0, iterations, tol):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

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

    fval = None
    grad_val = None
    x_log = []
    y_log = []
    grad_log = []

    x0 = asarray(x0).flatten()
    iterations = len(x0) * 200
    old_fval = fun(x0)
    gfk = grad(x0)
    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    x_log = np.append(x_log, xk.T)
    y_log = np.append(y_log, fun(xk))
    grad_log = np.append(grad_log, np.linalg.norm(xk - x_log[-1:]))
    gnorm = np.amax(np.abs(gfk))

    while (gnorm > tol) and (k < iterations):
        pk = -np.dot(Hk, gfk)
        try:
            alpha, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(fun, grad, xk, pk, gfk, old_fval, old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            break
        x1 = xk + alpha * pk
        sk = x1 - xk
        xk = x1
        if gfkp1 is None:
            gfkp1 = grad(x1)
        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        gnorm = np.amax(np.abs(gfk))
        grad_log = np.append(grad_log, np.linalg.norm(xk - x_log[-1:]))
        x_log = np.append(x_log, xk.T)
        y_log = np.append(y_log, fun(xk))
        if (gnorm <= tol):
            break
        if not np.isfinite(old_fval):
            break
        try:
            rhok = 1.0 / (np.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
        if isinf(rhok):
            rhok = 1000.0
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])
    fval = old_fval
    grad_val = grad_log[-1]

    return xk, fval, grad_val, x_log, y_log, grad_log



def conjugate_gradient(fun, grad, x0, iterations, tol):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.

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

    fval = None
    grad_val = None
    x_log = []
    y_log = []
    grad_log = []

    x0 = asarray(x0).flatten()
    iterations = len(x0) * 200
    old_fval = fun(x0)
    gfk = grad(x0)

    k = 0
    xk = x0
    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    pk = -gfk
    x_log = np.append(x_log, xk.T)
    y_log = np.append(y_log, fun(xk))
    grad_log = np.append(grad_log, np.linalg.norm(xk - x_log[-1:]))
    gnorm = np.amax(np.abs(gfk))

    sigma_3 = 0.01

    while (gnorm > tol) and (k < iterations):
        deltak = np.dot(gfk, gfk)

        cached_step = [None]

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = grad(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = np.amax(np.abs(gfkp1))
            return (alpha, xkp1, pkp1, gfkp1, gnorm)

        def descent_condition(alpha, xkp1, fp1, gfkp1):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            #
            # See Gilbert & Nocedal, "Global convergence properties of
            # conjugate gradient methods for optimization",
            # SIAM J. Optimization 2, 21 (1992).
            cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step

            # Accept step if it leads to convergence.
            if gnorm <= tol:
                return True

            # Accept step if sufficient descent condition applies.
            return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)

        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(fun, grad, xk, pk, gfk, old_fval,
                                     old_old_fval, c2=0.4, amin=1e-100, amax=1e100,
                                     extra_condition=descent_condition)
        except _LineSearchError:
            break

        # Reuse already computed results if possible
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)
        k += 1
        grad_log = np.append(grad_log, np.linalg.norm(xk - x_log[-1:]))
        x_log = np.append(x_log, xk.T)
        y_log = np.append(y_log, fun(xk))

    fval = old_fval
    grad_val = grad_log[-1]

    return xk, fval, grad_val, x_log, y_log, grad_log


def LM(fun, grad, jacobian, x0, iterations, tol):
    """
    Minimization of scalar function of one or more variables using the
    Levenberg-Marquardt algorithm.

    Parameters
    ----------
    fun : function
        Objective function.
    grad : function
        Gradient function of objective function.
    jacobian :function
        function of objective function.
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

    fval = None  # y的最小值
    grad_val = None  # 梯度的最后一次下降的值
    x_log = []  # x的迭代值的数组，n*9，9个参数
    y_log = []  # y的迭代值的数组，一维
    grad_log = []  # 梯度下降的迭代值的数组

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    iterations = len(x0) * 200

    k = 1
    xk = x0

    updateJ = 1
    lamda = 0.01
    old_fval = fun(x0)
    gfk = grad(x0)

    gnorm = np.amax(np.abs(gfk))

    J = [None]
    H = [None]

    while (gnorm > tol) and (k < iterations):
        if updateJ == 1:
            x_log = np.append(x_log, xk.T)
            yk = fun(xk)
            y_log = np.append(y_log, yk)
            J = jacobian(x0)
            H = np.dot(J.T, J)

        H_lm = H + (lamda * np.eye(9))
        gfk = grad(xk)
        pk = - np.linalg.inv(H_lm).dot(gfk)
        pk = pk.A.reshape(1,-1)[0] #二维变一维
        xk1 = xk + pk
        fval = fun(xk1)
        if fval < old_fval:
            lamda = lamda / 10
            xk = xk1
            old_fval = fval
            updateJ = 1
        else:
            updateJ = 0
            lamda = lamda * 10
        gnorm = np.amax(np.abs(gfk))
        k = k+1
        grad_log = np.append(grad_log, np.linalg.norm(xk - x_log[-1:]))

    fval = old_fval
    grad_val = grad_log[-1]

    return xk, fval, grad_val, x_log, y_log, grad_log
