#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 19:07:32 2025

@author: brad
"""

from bradLib import errorPropagate
from sympy import *

rho, XMag, radius, S, chi, Xx, Xy, N, B, B0, LL0 = symbols("rho X r_0 S chi X_x X_y N B B_0 L_L_0")

XMagEq = Eq(XMag, sqrt(Xx**2 + Xy**2))
XMagCalc = errorPropagate.multipleEquations([XMagEq])

chiEq = Eq(chi, -acos((Xy)/(XMag)))
chiCalc = errorPropagate.multipleEquations([chiEq])


rhoEq = Eq(rho, sin(XMag / radius) - XMag * S / radius)
BEq = Eq(B, asin(sin(B0)*cos(rho) + cos(B0)*sin(rho)*cos(chi)))
LL0Eq = Eq(LL0, asin(sin(rho) * sin(chi) * (1/cos(B))))



calc = errorPropagate.multipleEquations([rhoEq, BEq, LL0Eq])
calc.evalEquations()