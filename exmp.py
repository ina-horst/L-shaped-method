# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:29:52 2019

@author: Ina
"""

from Stoch_LP import Stoch_LP
import numpy as np
from pysmps import smps_loader as smps

###############################################################################
# LandS ORINGINAL
###############################################################################

electric = Stoch_LP.from_smps("path\\to\\electric\\LandS\\set")
electric.solve()

###############################################################################
# LandS MODIFICATION
###############################################################################

LandS = smps.load_2stage_problem("path\\to\\electric\\LandS\\set")

h = []
p = []

begin = 0
end = 4
N = 5
for i in range(N):
    for j in range(N):
        for k in range(N):
            h.append(np.array([0,0,0,0,(end - begin) * i / N, (end - begin) * j / N, (end - begin) * k / N]))
            p.append(1 / N**3)

stoch = Stoch_LP(c = LandS["c"], A = LandS["A"], b = LandS["b"], q = [LandS["q"][0]] * N**3, h = h, T = [LandS["T"][0]] * N**3, W = LandS["W"], p = p)
stoch.solve()

###############################################################################
# GENERAL APPLICATION TO SMPS FILES CONTAINING 2STAGE STOCHASTIC LIN PROGS
###############################################################################

stoch = Stoch_LP.from_smps("path/to/your/smps")
stoch.solve()
