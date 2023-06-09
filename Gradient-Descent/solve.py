from scipy.optimize import minimize
import numpy as np
from gradient_descent import GradientDescent


# Example
epsilon = 0.01
x_start = np.array([0, 0])

def f(X):
    x1, x2 = X
    return 7*x1**2 + 2*x1*x2 + 5*x2**2 + x1 - 10*x2

def dydx1(X):
    x1, x2 = X
    return 14*x1 + 2*x2 + 1

def dydx2(X):
    x1, x2 = X
    return 2*x1 + 10*x2 - 10


gd = GradientDescent(f, dydx1, dydx2, epsilon, x_start)
min_f, coordinates = gd.solve()
print("\n\n## My solution ##")
print("F =", min_f)
print("X =", coordinates)


print("\n\n## Scipy solution ##")
print(minimize(f, x_start))
