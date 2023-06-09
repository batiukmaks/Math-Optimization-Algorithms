from scipy.optimize import minimize, LinearConstraint
import numpy as np
from conditional_gradient import ConditionalGradient


# Example
epsilon = 0.00001
x_start = np.array([2, 3])
def f(X):
    return -3*X[0]**2 + 54*X[1]**2 + 10*X[0] + 15*X[1]
# Domain of a function in canonical form
domain_canonical_A = np.array([
    [2, 3, 1, 0],
    [2, 1, 0, 1]
])
domain_canonical_b = np.array([13, 10])


print("## My program ##")
cg = ConditionalGradient(f, domain_canonical_A,
                         domain_canonical_b, epsilon, x_start)
result, X = cg.solve()
print(f"Result: {result:.20f}, X: {[f'{val:.20f}' for val in X]}")


print("\n\n## scipy.optimize ##")
constraint = LinearConstraint(
    domain_canonical_A[:, :2], 0, ub=domain_canonical_b)
res = minimize(fun=f, x0=x_start, constraints=constraint,
               tol=epsilon, bounds=((0, None), (0, None)))
print(f"Result: {res.fun:.20f}, X: {[f'{val:.20f}' for val in res.x]}")
