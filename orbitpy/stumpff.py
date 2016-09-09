#! /usr/local/bin/python
# -*- coding: utf-8 -*-
__author__     = 'Zach Dischner'
__copyright__  = "CYGNSS - Southwest Research Institute"
__credits__    = ["NA"]
__license__    = "NA"
__version__    = "0.0.0"
__maintainer__ = "Zach Dischner"
__email__      = "dischnerz@boulder.swri.edu"
__status__     = "Dev"
__doc__        ="""
File name: sumpff.py
Authors:  
Created:  Aug/06/2016
Modified: Aug/06/2016

Description:
    Provide classic stumpff calculation functions

References: 
    Bate, Mueller, White

Note:
    c1: Not implemented
    c2: get_C(), get_C_safe()
    c3: get_S(), get_S_safe()
    c4: not implemented

"""
##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import numpy as np
from numpy import math,cos,cosh,sin,sinh,pi,arccos,sqrt
from . import utils
from numba import jit, njit, float64, autojit

##############################################################################
#                            Main Functions
#----------*----------*----------*----------*----------*----------*----------*
# Bate, Mueller, White S and C functions: 196
@jit(nopython=True)
def get_C(z):
    """C function, with catches for near z zero behavior
    Safety warning stuff disabled so we can @jit
    """
    if z > 0.1:
        C = (1-np.cos(z**0.5))/z
    elif z < -0.1:
        C = (1-np.cosh( (-z)**0.5) )/z
    else:
        #Precompute   2!     4!          6!           8!            10!              12!                 14!                     16!
        C = (1.0/2.0) - z/24.0 + z**2/720.0 - z**3/40320.0+ z**4/3628800.0 - z**5/479001600.0 + z**6/87178291200.0 - z**6/20922789888000.0
    return C

def get_C_safe(z):
    """C function, with catches for near z zero behavior
    Safety warning stuff disabled so we can @jit
    """
   
    def c_series():
        return (1.0/(factorial(2.0))) - z/factorial(4.0) + z**2/factorial(6.0) - z**3/factorial(8.0) + z**4/factorial(10.0) - z**5/factorial(12.0)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            if z > 0.1:
                C = (1-np.cos(z**0.5))/z
            elif z < -0.1:
                C = (1-np.cosh( (-z)**0.5) )/z
            else:
                C = c_series()
            ## This helps for near zero cases
            if (C == 0) or abs(C) > 100000:
                raise(RuntimeWarning)
        except RuntimeWarning:
            C = c_series()
            utils.printRed("Runtime warning computing true C(z) where z is {}, returning power series expansion ==> {}".format(z,C))
    return C

def get_Cprime(z):
    """C' function, with catches for near z zero behavior"""

    if abs(z) >= 1:
        Cprime = 1.0/(2.0*z) * (1.0 - get_S(z)*z - 2*get_C(z))
    else:
        Cprime = -1.0/factorial(4.0) + (2.0*z)/factorial(6) - (3.0*z**2)/factorial(8.0) + (4.0*z**3)/factorial(10.0) - (5.0*z**4)/factorial(12.0) + (6.0*z**5)/factorial(14.0) - (7.0*z**6)/factorial(16.0) + (8.0*z**7)/factorial(18.0)
    return Cprime

@jit(nopython=True) 
def get_S(z):
    """Optimized c3/S computation function"""
    if z > 0.1:
        S = (z**0.5 - sin(z**0.5)) / (z**(3.0/2.0)) 
    elif z < -0.1:
        S = (sinh((-z)**0.5) - (-z)**0.5)/ ((-z)**(3.0/2.0))
    else:
        #Precompute   3!     5!          7!              9!              11!                 13!                 15!                     17!
        S = (1.0/6.0) - z/120.0 + z**2/5040.0 - z**3/362880.0 + z**4/39916800.0 - z**5/6227020800.0 + z**6/1307674368000.0 - z**7/355687428096000.0

    return S

def get_S_safe(z):
    """S function, with catches for znear zero behavior"""

    def s_series():
        return (1.0/(factorial(3.0))) - z/factorial(5.0) + z**2/factorial(7.0) - z**3/factorial(9.0) + z**4/factorial(11.0) - z**5/factorial(13.0) + z**6/factorial(15.0) - z**7/factorial(17.0)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            if z > 0.1:
                S = (z**0.5 - sin(z**0.5)) / (z**(3.0/2.0)) 
            elif z < -0.1:
                S = (sinh((-z)**0.5) - (-z)**0.5)/ ((-z)**(3.0/2.0))
            else:
                S = s_series()
            ## This helps for near zero cases
            if (S == 0) or abs(S) > 100000:
                raise(RuntimeWarning)
        except RuntimeWarning:
            S = s_series()
            utils.printRed("Runtime warning computing true S(z) where z is {}, returning power series expansion ==> {}".format(z,S))

    return S

def get_Sprime(z):
    """S' function, with catches for z near zero behavior"""
    if abs(z) >= 1:
        Sprime = (get_C(z) - 3*get_S(z))/(2*z)
    else:
        Sprime = -1.0/factorial(5.0) + (2.0*z)/factorial(7.0) - (3.0*z**2)/factorial(9.0) + (4.0*z**3)/factorial(11.0) - (5.0*z**4)/factorial(13.0) + (6.0*z**5)/factorial(15.0) - (7.0*z**6)/factorial(17.0)
    return Sprime


    ###### Define so you can have cross-literature reference compatibility
    c2 = get_C
    c3 = get_S