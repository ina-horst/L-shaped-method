# -*- coding: utf-8 -*-
"""
This is an implementation of a classical linear program solvable due to the two
-phase simplex algorithm. We assume in the simplex algorithm that the box 
restrictions are given as 0 <= x < +inf.
  

@author: Ina
"""

import numpy as np

from Exceptions import InfeasibleProblem, NoPivotException

class LP:
    """
    An instance of LP offers the following:
        - Fields:
            c (np.array): Vector of coefficients.
            A_eq (np.matrix): The matrix of equality constraints (row-wise)
            b_eq (np.array): The vector of equality constraint values
            A_geq (np.matrix): The matrix of inequality constraints (row-wise
                  greater or equal)
            b_geq (np.array): The vector of inequality constraint values
            solved (bool): Whether the Problem has been solved or not
            solution (dict {"p": (np.matrix), "d": (np.matrix)} | np.matrix):
                Has a value if solved. It is a dictionary iff the problem is
                unbounded. All np.matrices occuring in this variable are
                column-vectors, i.e. of shape (x,1).
            optimal_value (numeric): Value of the objective function in the
                optimum. Obviously solved needs to be true in order for this
                to have a meaningful value.
            multipliers (np.matrix): The optimal simplex multipliers of this
                problem. In order for this to have a meaningful value
                solved needs to be true and calculate_multipliers needs to be
                true. It is a column vector, i.e. of dimension (x,1).
                
            DEBUG (static, final, bool): If true we print the tableau and other
                information to current steps in the algorithm.
        - Functions:
            get_solution:
                - Arguments:
                    self
                - Result:
                    If the Problem is unsolved (solved = False) yet it is
                    getting solved (solve() is getting called).
                    In either case it returns the dictionary
                        {"x": (dict {"p": (np.matrix), "d": (np.matrix)} |
                        np.matrix), "bounded": (bool)}
                    where x is the solution or a ray of unbounded decrease
                    and bounded is a bool indicating whether x is a solution
                    or a ray.
                - Calls and Exceptions:
                    solve()
            solve:
                - Arguments:
                    self
                - Result:
                    Uses 2-Phase Simplex Algorithm to solve the given linear
                    program. Sets the fields solution, optimal_value,
                    multipliers if possible or chosen to.
                - Exceptions:
                    InfeasibleProblem: If the Problem to solve is infeasible
            
    """
    
    DEBUG = False
    
    def __init__(self, c, A_eq = None, b_eq = None,
                 A_geq = None, b_geq = None, calculate_multipliers = False, eps
                 = 1e-8):
        self.c      = c
        self.A_eq   = A_eq if A_eq is not None else np.matrix([])
        self.b_eq   = b_eq if b_eq is not None else np.array([])
        self.A_geq  = A_geq if A_geq is not None else np.matrix([])
        self.b_geq  = b_geq if b_geq is not None else np.array([])
        self.solved = False
        self.calculate_multipliers = calculate_multipliers
        self.eps    = eps
        if LP.DEBUG:
            print("min", "\t", c, "*", "x")
            print("wrt", "\t", "A_eq * x = b_eq")
            print("\t", "A_geq * x >= b_geq, where")
            print("A_eq = ")
            print(A_eq)
            print("b_eq = ")
            print(b_eq)
            print("A_geq = ")
            print(A_geq)
            print("b_geq = ")
            print(b_geq)
        
    
    def get_solution(self):
        if not self.solved:
            self.solve()
        return {"x": self.solution, "bounded": not type(self.solution) is dict}
    
    def solve(self):
        iterations = 0
        # local function
        def base_vector(i, dim):
            v = np.zeros(dim)
            v[i] = 1
            return np.matrix(v).T
        
        b_eq, b_geq = np.copy(self.b_eq), np.copy(self.b_geq)
        A_eq, A_geq = np.copy(self.A_eq), np.copy(self.A_geq)
        c = np.copy(self.c)
                        
        # if A_eq or A_geq is empty, set m_1 resp. m_2 to 0 (default row count 
        # is 1)
        m_1, m_2 = b_eq.size, b_geq.size # number of respective constraints
        m = m_1 + m_2 # number of all constraints 
        n = c.size # number of variables before inserting slack or artificial variables
        
        # generate tableau
        if A_eq.shape[1] != A_geq.shape[1]:
            if A_eq.shape[1] == 0:
                tableau = A_geq
            elif A_geq.shape[1] == 0:
                tableau = A_eq
            else:
                raise Exception("Invalid Argument: A_eq and A_geq must be of same column size or one needs to be empty!")
        else: # A_eq.shape[1] == A_geq.shape[1]:
            tableau = np.concatenate((
                    A_eq,
                    A_geq
                    ), axis=0)
        swapped = [] # list of indices of equality constraints whose rhs values 
        # were negative; these are used for calculating the signs of the multipliers
        a = 0 # number of artificial variables inserted for geq constraints
        # insert artificial variables if there are no geq constraints        
        if A_geq.shape[1] == 0:
            for i in [i for i in range(0, m_1) if b_eq[i] < 0]: 
                    tableau[i,:] = -tableau[i,:]
                    b_eq[i] = -b_eq[i]
                    swapped.append(i)
            tableau = np.concatenate((
                    tableau,
                    np.identity(m_1),
                    np.matrix(b_eq).T
                    ),axis=1)  # done. Go to phase 1.
            B = list(range(n, n + m_1)) # current basis variable indices
            A = B.copy() # Set of artificial variable indices
            
        # insert slack variables for geq constraints where necessary
        else: # A_geq.shape[1] >= 1: (therefore m_2 >= 1)
            if m_1 == 0:
                tableau = np.concatenate((
                        tableau,
                        - np.identity(m_2)
                        ), axis = 1)
            else: 
                tableau = np.concatenate((
                        tableau,
                        np.concatenate((
                                np.zeros(shape = (m_1,m_2)),
                                - np.identity(m_2)
                                ), axis=0)
                        ), axis=1)
                for i in [i for i in range(0, m_1) if b_eq[i] < 0]: 
                    tableau[i,:] = -tableau[i,:]
                    b_eq[i] = -b_eq[i]
                    swapped.append(i)
                # insert artificial variables for eq constraints
                tableau = np.concatenate((
                    tableau,
                    np.concatenate((
                        np.identity(m_1),
                        np.zeros((m_2, m_1))
                    ), axis=0)
                ), axis=1)
            B = list(range(n + m_2, n + m_2 + m_1))
            A = list(range(n + m_2, n + m_2 + m_1))
            # insert artificial variables for geq constraints where necessary
            for i in range(0,m_2):
                if b_geq[i] > 0:
                    tableau = np.concatenate((
                        tableau,
                        base_vector(m_1 + i, m_1 + m_2)
                    ), axis=1)
                    B.append(n + m + a)
                    A.append(n + m + a)
                    a = a + 1
                else: # b_geq[i] < 0
                    tableau[m_1 + i,:] = -tableau[m_1 + i,:]
                    b_geq[i] = -b_geq[i]
                    swapped.append(i)
                    B.append(n + i) # slack variable is now a standard basis
                    # vector, therefore can be put into basis
            tableau = np.concatenate((
                tableau,
                np.concatenate((
                    np.matrix(b_eq).T,
                    np.matrix(b_geq).T
                ), axis=0)
            ), axis=1)
       
        '''
        Tableau is a matrix of the form:
                  n          m_2       m_1         a       1
            +-----------+----------+----------+--------+------+
        +/- |           |          |          |        |      |
        +/- |   A_eq    |     0    |     I    |    0   | b_eq |
        +/- |           |          |          |        |      |
            +-----------+----------+----------+--------+------+
        +/- |           |          |          |        |      |
        +/- |   A_geq   |    -I    |     0    |  (e_i) | b_geq|
        +/- |           |          |          |        |      |
            +-----------+----------+----------+--------+------+    
                                    
                                    ---------A---------
                        - - - - - - ---------B---------
        '''
                
        # Preparation phase 1
        initial_b = tableau[:,-1]
             
        c_curr = np.matrix(np.zeros(n + m + a))
        c_curr[:,A] = np.repeat(1, m_1 + a)
        z = - c_curr[:,B].dot(tableau[:, n + m + a])
        N = [i for i in range(0, n + m + a) if i not in B]
        # c_N - c_B*B^-1*N
        c_curr[:,N] = - c_curr[:,B].dot(tableau[:,N])
        c_curr[:,B] = np.zeros(m)
        if LP.DEBUG:
            print(tableau.shape[1])
            print(c.shape)
            print(m+a+1)
        tableau = np.concatenate((# append actual objective function to tableau
            tableau,
            np.concatenate((
                np.matrix(c),
                np.zeros((1, m + a + 1))
            ), axis=1)
        ), axis = 0)
            
        tableau = np.concatenate((# append auxiliary objective function to tabl
            tableau,
            np.concatenate((
                c_curr,
                np.matrix(z)
            ), axis=1)
        ), axis=0)
        
        initial_B = B.copy()
        
        if LP.DEBUG:
            print({"B": B, "N": N, "A": A, "m_1": m_1, "m_2": m_2, "n": n, "a": a})
            print("Tableau:")
            print(tableau)
        
        '''
        local function phase
        Parameters:
            B (list): basis variable indices
            N (list): nonbasis variable indices
            A (list): artificial variable indices
            c_curr (np.matrix of shape (1,*)): current objective function coeff
            tableau (np.matrix): tableau as defined above
            n (numeric): column-count of first block
            m (numeric): column-count of second block
            rows (numeric): rows for pivot search
            a (numeric): column-count of third block
            phase (numeric): phase 1 or 2
        '''
        
        def phase(B, N, A, c_curr, tableau, n, m, rows, a, phase, iterations):
            if LP.DEBUG:
                print("Entering phase", phase, "with parameters:\n",
                      {"n": n, "m": m, "rows": rows, "a": a})
            bland = False
            degenerated = 0
            # while there is a negative coefficient in c_curr    
            while np.shape(c_curr[:,N][c_curr[:,N] < - self.eps])[1] > 0:
                iterations = iterations + 1
                z = tableau[rows + 2 - phase, n + m + a] # for checking cycle
                if LP.DEBUG:
                    print("Initial Tableau:")
                    print(tableau)
                
                if bland:
                    index = np.argmax(c_curr[:,N] < - self.eps)
                    i = N[index]
                else:
                    index = np.argmin(c_curr[:,N]) # entering variable
                    i = N[index]
                col = tableau[0:rows,i] # i-th column without c value
                if LP.DEBUG:
                    print("Entering Variable:", i)
                    print("Index in non-basic Variable set:", index)
                    print(col)
                pos = [i for i in range(0,rows) if col[i] > self.eps]
                if LP.DEBUG:
                    print(col.T)
                    print("Positive value basic variable indices:", pos)
                    print("Quotients:", tableau[0:rows,-1][pos,:] / col[pos,:])
                if len(pos) == 0:
                    raise NoPivotException(B, N, tableau, i, col)
                j = pos[np.argmin(tableau[0:rows,-1][pos,:] / col[pos,:])] # leaving variable
                if LP.DEBUG:
                    print("Leaving Variable:", B[j])
                    print("Index in basic variable set:", j)
                tableau[j,:] = tableau[j,:] / tableau[j,i]
                for k in range(0, rows + 3 - phase):
                    if k == j:
                        continue
                    tableau[k,:] = tableau[k,:] - tableau[k,i] * tableau[j,:]
                entering = B[j]
                B[j] = i
                N[index] = entering
                if entering in A:
                    N.remove(entering)
                if LP.DEBUG:
                    print("Updated basis:", B)
                    print("Updated non-basis:", N)
                if abs(z - tableau[rows + 2 - phase, n + m + a]) < self.eps: # checking cycle
                    degenerated = degenerated + 1
                    if degenerated >= 10:
                        bland = True
                        if LP.DEBUG:
                            print("Caught in cycle; using BLANDS RULE now.")
                c_curr = tableau[rows + 2 - phase, 0:(n + m + a)]
                if LP.DEBUG:
                    print("Next c:", c_curr)
            if LP.DEBUG:
                print("Finished Tableau:")
                print(tableau)
            return B, N, tableau, iterations
        #Phase 1
        try:
            B, N, tableau, iterations = phase(B, N, A, c_curr, tableau,
                                  n, m, rows=m, a = a, phase = 1, iterations = iterations)
        except (NoPivotException) as e:
            raise Exception("No Pivot could been found. Break...")
        if tableau[m + 1, n + m + a] < - self.eps:
            if LP.DEBUG:
                print("Infeasible, Tableau:")
                print(tableau)
                print("z:", tableau[m + 1, n + m + a])
            raise InfeasibleProblem()
        
        AnB = [i for i in A if i in B]
        
        if len(AnB) == 1:
            if LP.DEBUG:
                print("There is one artificial variable in the basis variables.")
                print(B, N, A)
            index = B.index(AnB[0])
            if LP.DEBUG:
                print(index)
            if tableau[index,-1] < self.eps:
                if LP.DEBUG:
                    print(tableau[index,:])
                row = np.ravel(np.asarray(tableau[index,0:-1]))
                if np.nonzero(row[N])[0].size > 0:
                    j = np.nonzero(row[N])[0][0]
                    tableau[index,:] = tableau[index,:] / tableau[index, N[j]]
                    for k in range(0, tableau.shape[0]):
                        if k == index:
                            continue
                        tableau[k,:] = tableau[k,:] - tableau[k,N[j]] * tableau[index,:]
                    B[index] = N[j]
                    del N[j]
                else:
                    raise Exception("The constraints are linearly dependent. No base could be chosen.")
            else:
                raise Exception("Phase 1 exited without a feasible solution! Exiting.")
            if LP.DEBUG:
                print(B, N)
                print(tableau)
        elif len(AnB) > 1:
            raise Exception("There are multiple artificial variables in the basis variables.")
           
        # PHASE 2:
        tableau = tableau[0:(m + 1),:]
        # slice out artificial variable columns:
        if not self.calculate_multipliers:
            tableau = np.delete(tableau, slice(n+m_2, n+m+a), axis = 1)
        
        if LP.DEBUG:
            print("BASIS AFTER PHASE 1:", B)
        try:
            B, N, tableau, iterations = phase(B,
                  [i for i in range(0,n+m_2) if i not in B],
                  [],
                  tableau[m, 0:(n+m_2)],
                  tableau,
                  n,
                  m = m_2,
                  rows=m,
                  a = 0,
                  phase = 2,
                  iterations = iterations)
        except (NoPivotException) as e:
            x_p = np.zeros((len(e.B) + len(e.N), 1))
            x_d = np.zeros((len(e.B) + len(e.N), 1))
            x_p[e.B,:] = e.tableau[0:len(B), -1]
            x_d[e.B,:] = - e.col
            x_d[e.i,0] = 1
            self.solution = {"p": x_p, "d": x_d}
            return
        x = np.zeros((1,n + m_2))
        x[:, B] = tableau[0:m, -1].T
        self.solution = x[:, 0:n].T
        self.optimal_value = - tableau[-1,-1]
        if self.calculate_multipliers:
            self.multipliers = tableau[m, initial_B].T
            not_swapped = [i for i in range(0,m) if i not in swapped]
            self.multipliers[not_swapped,:] = - self.multipliers[not_swapped,:]
            if LP.DEBUG:
                print("ASSERT MULTIPLIERS: pi^t b = ", self.multipliers.T.dot(initial_b))
        if LP.DEBUG:
            print("FINISHED!")
            print("Solution:", self.solution.T)
            print("Optimal Value:", self.optimal_value)
        return iterations
            
