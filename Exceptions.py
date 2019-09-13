# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:10:27 2019

@author: Ina
"""

class InfeasibleProblem(Exception):
    pass
class UnboundedProblem(Exception):
    pass
class NoPivotException(Exception):
    def __init__(self, B, N, tableau, i, col):
        self.B = B
        self.N = N
        self.tableau = tableau
        self.i = i
        self.col = col
    pass