# An   Implementation   of   the   L-shaped   Method   using   the   Two-Phase-Simplex-Algorithm
Implementation of an algorithm for two-stage stochastic linear programs with fixed recourse according to van Slyke and Wets [1].

### Information
##### `Stoch_LP`
The `Stoch_LP` class resembles a two-stage stochastic linear program with fixed recourse. The documentation on which fields and functions it offers is contained in the file as comment.
##### `LP`
Same applies to the `LP` class; the fields and functions are documented inside the code.
##### `exmp.py`
The file states exemplary how to apply the algorithm to given data using the `electric` (`LandS`) set originated in [2]; can be downloaded from [Test-Problem Collection for Stochastic Linear Programming](https://www4.uwsp.edu/math/afelt/slptestset/download.html). Inspired by the idea from [The Empirical Behavior of Sampling Methods for Stochastic Programming](http://pages.cs.wisc.edu/~swright/stochastic/sampling/), [3], to modify the `LandS` set to have more stochastic cases, `exmp.py` generates a similar (but smaller) sample and executes the algorithm on it.

### Depends on
- numpy:  https://pypi.org/project/numpy/
- pysmps: https://pypi.org/project/pysmps/

### Citation
[1] Slyke, R. M. V.; Wets, R.: L-Shaped Linear Programs with Application to Optimal Control and Stochastic Programming. In: SIAM Journal on Applied Mathematics 17 (1969), p. 638–663.<br>
[2] Louveaux, F.V.; Smeers, Y.: Optimal investments for electricity generation: A stochastic model and a test-problem. In: Wets, R. J-B., Ermoliev, Y. (editors): Numerical Techniques for Stochastic Optimization. Berlin, Heidelberg, NewYork: Springer-Verlag, 1988, Kapitel 24, p. 445–453<br>
[3] Linderoth, J. T.; Shapiro, A.; Wright, S. J.: The Empirical Behavior of Sampling Methods for Stochastic Programming / Computer Science Department, University of Wisconsin-Madison. 2002. – Optimization Technical Report 02-01
