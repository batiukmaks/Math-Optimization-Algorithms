from scipy.optimize import minimize_scalar
from golden_section_search import GoldenSectionSearch, plot_function_on_interval
import numpy as np


# Example
a = -5
b = -4
epsilon = 0.02
def f(x):
    return x * np.sin(x) + 2 * np.cos(x)


solver = GoldenSectionSearch(f)
x, f_x = solver.search(a, b, epsilon)
print('# My Program #')
print(f"Minimum is at x = {x:.4f} with f(x) = {f_x:.4f}")


print('\n# scipy.optimize.minimize_scalar #')
res = minimize_scalar(f, bounds=(a, b), method="bounded")
print(f"Minimum is at x = {res.x} with f(x) = {res.fun}")
