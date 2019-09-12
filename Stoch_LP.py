# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:43:48 2019
This is an implementation of a two stage stochastic linear program with fixed 
recourse solvable due to the van Slyke and Wets algorithm. 

@author: Ina
"""
import numpy as np
import math

from pysmps import smps_loader as smps

from Exceptions import InfeasibleProblem, UnboundedProblem
from LP import LP
    
    
class Stoch_LP(LP):
    
    DEBUG = False
    
    def __init__(self, c, A, b, q, h, T, W, p, eps = 1e-8):
        self.n = c.size
        super().__init__(c = np.concatenate((
                c,
                np.array([0, 0])
            )), A_eq = np.concatenate((
                A,
                np.zeros((A.shape[0],2))
            ), axis = 1),
            b_eq = b,
            eps = eps)
        self.K = len(q)
        if len(h) != self.K or len(T) != self.K or len(p) != self.K:
            raise Exception("random variables must have the same number of cases")
        self.q = q
        self.h = h
        self.T = T
        self.W = W
        self.p = p
    
    def from_smps(path, eps = 1e-8):
        d = smps.load_2stage_problem(path)
        return Stoch_LP(d["c"], d["A"], d["b"], d["q"], d["h"], d["T"], d["W"], d["p"], eps)
    
    def __add_constraint_sigma(self, D, d):
        if Stoch_LP.DEBUG:
            print("Adding sigma-constraint")
            print("D = sigma^t T:", D)
            print("d = sigma^t h:", d)
        self.solved = False
        D_mod = np.concatenate((
                    D,
                    np.matrix([0,0])
                ), axis = 1)
        if self.A_geq.shape[1] == 0:
            self.A_geq = D_mod
            self.b_geq = np.ravel(np.asarray(d))
            return
        self.A_geq = np.concatenate((
            self.A_geq,
            D_mod
        ), axis = 0)
        self.b_geq = np.concatenate((
            self.b_geq,
            np.ravel(np.asarray(d))
        ))
    
    def __add_constraint_pi(self, E, e):
        if Stoch_LP.DEBUG:
            print("Adding pi-constraint")
            print("E = sigma^t T:", E)
            print("e = sigma^t h:", e)
        self.solved = False
        self.c[-2] = 1
        self.c[-1] = -1
        E_mod = np.concatenate((
                    E,
                    np.matrix([1,-1])
                ), axis = 1)
        if self.A_geq.shape[1] == 0:
            self.A_geq = E_mod
            self.b_geq = np.ravel(np.asarray(e))
            return
        self.A_geq = np.concatenate((
            self.A_geq,
            E_mod
        ), axis = 0)
        self.b_geq = np.concatenate((
            self.b_geq,
            np.ravel(np.asarray(e))
        ))
    
    def __is_second_stage_problem_feasible(self, x, NB, is_constraint_shifted):
        ones  = np.concatenate((
                np.zeros(self.W.shape[1]),
                np.repeat(1, 2*self.W.shape[0])
            ))
        for k in range(0, self.K):
            lp = LP(c=ones, A_eq = NB, b_eq=
                    is_constraint_shifted * self.h[k] - np.ravel(np.asarray(
                            self.T[k].dot(x)
                        )),
                    calculate_multipliers = True)
            lp.solve()
            if lp.optimal_value > self.eps:
                return lp.multipliers, k
        return None, -1
                       
        
    def __solve_second_stage_problem(self, x, k, is_constraint_shifted):
        lp = LP(c = self.q[k], A_eq = self.W, b_eq = 
                is_constraint_shifted * self.h[k] - np.ravel(np.asarray(
                        self.T[k].dot(x)
                    )), 
                calculate_multipliers = True)
        sol = lp.get_solution()
        if not sol["bounded"]:
            raise UnboundedProblem()
        return lp.optimal_value, lp.solution, lp.multipliers

    
    def solve(self):
        s = 0
        r = 0
        WII = np.concatenate((
            self.W,
            np.identity(self.W.shape[0]),
            -np.identity(self.W.shape[0])
            ), axis=1)
        
        # function to unpack the optimal values of the LP
        def unzip(solution, first_run):
            if solution["bounded"]:
                if first_run == 0:
                    solution["x"][-2] = 0
                    solution["x"][-1] = math.inf
                return solution["x"][0:-2], solution["x"][-2] - solution["x"][-1], solution["bounded"]
            if first_run == 0:
                solution["x"]["p"][-2] = 0
                solution["x"]["p"][-1] = math.inf
            return {"p": solution["x"]["p"][0:-2], "d": solution["x"]["d"][0:-2]}, solution["x"]["p"][-2] - solution["x"]["p"][-1], solution["bounded"]
        
        optimal = False
        while not optimal:
            feasible = False
            while not feasible:
                super().solve()
                self.solved = True
                x, theta, bounded = unzip(super().get_solution(), self.c[-1])
                if Stoch_LP.DEBUG:
                    print("Current optimal solution:")
                    print({"x": x, "theta": theta})
                if bounded:
                    if Stoch_LP.DEBUG:
                        print("Therefore the problem is bounded!")
                    sigma, k = self.__is_second_stage_problem_feasible(x, WII, bounded)
                    if Stoch_LP.DEBUG:
                        print({"sigma": sigma, "k": k})
                    if k < 0: # return k = -1 if no such k exists
                        feasible = True
                    else:
                        self.__add_constraint_sigma(sigma.T.dot(self.T[k]),
                                                    sigma.T.dot(self.h[k]))
                        r = r+1
                else: #unbounded
                    if Stoch_LP.DEBUG:
                        print("Therefore the problem is unbounded!")
                    #x["p"], x["d"]
                    sigma_d, k_d = self.__is_second_stage_problem_feasible(x["d"], WII, bounded)
                    if Stoch_LP.DEBUG:
                        print({"sigma_d": sigma_d, "k_d": k_d})
                    if k_d < 0: #dann alle w' = 0
                        sigma_p, k_p = self.__is_second_stage_problem_feasible(x["p"], WII, True)
                        if Stoch_LP.DEBUG:
                            print({"sigma_p": sigma_p, "k_p": k_p})
                        if k_p < 0: #dann alle w = 0
                            # gehe zu 6.2.3. 
                            E = np.zeros((1, len(self.c)-1))
                            e = 0
                            direction = np.matrix(self.c).dot(x["d"])
                            for k in range(0, self.K):
                                w_curr, y_curr, pi_curr = self.__solve_second_stage_problem(x["d"], k, False)
                                E = E + self.p[k] * pi_curr.T.dot(self.T[k])
                                e = e + self.p[k] * pi_curr.T.dot(self.h[k])
                                direction += self.p[k] * w_curr
                            if direction < 0:  # Umgang mit numerischen Ungenauigkeiten?
                                self.solution = x
                                return
                                #"y": [...,
                                #    {"p":, "d":},
                                #    ...]} # Ausgangs-LP ist unbeschränkt
                                #raise Exception("Input Problem is unbounded.")
                            self.__add_constraint_pi(E, e)
                            s = s+1
                        else: # es ex. k sodass w > 0
                            self.__add_constraint_sigma(sigma_p.T.dot(self.T[k_p]),
                                                        sigma_p.T.dot(self.h[k_p]))
                            r = r+1
                    else: #es ex. k mit w' > 0
                        self.__add_constraint_sigma(sigma_d.T.dot(self.T[k_d]),
                                                    sigma_d.T.dot(self.h[k_d]))
                        r = r+1
                        
            E = np.zeros((1, self.n))
            e = 0
            y = []
            for k in range(0, self.K):     
                w, y_curr, pi = self.__solve_second_stage_problem(x, k, True)
                E = E + self.p[k] * pi.T.dot(self.T[k])
                e = e + self.p[k] * pi.T.dot(self.h[k])
                y.append(y_curr.T)
                #print(y)
            if e - E.dot(x)[0,0] <= theta + self.eps:
                optimal = True
                self.solution = {"x": x.T, "y": y}
                self.optimal_value = np.ravel(np.asarray(np.matrix(self.c[0:-2]).dot(x))) + theta
                print("STOCHASTIC OPTIMIZATION FINISHED!")
                print("Solution:", self.solution)
                print("Optimal Value:", self.optimal_value)
                print("Added r =", r, " many constraints of type sigma and s =", s, "many of type pi.")
            else:
                self.__add_constraint_pi(E, e)
                s = s+1
        
        
        
        
        
        
        
        
        