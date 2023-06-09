import numpy as np
from primal import PrimalSimplexAlgorithm
from dual import DualSimplexAlgorithm

# Example
A = np.array([
    [2, 3, 1, 0, 0], 
    [2, 1, 0, 1, 0], 
    [-1, -1, 0, 0, 1]
    ], dtype=np.float32)
c = np.array([1, 2, 0, 0, 0])
b = np.array([8, 5, -1])


print('## My Program: Dual ##')
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
solver_dual = DualSimplexAlgorithm(c, A, b)
c, A, b, iterations, is_result_optimal, is_result_undefined = solver_dual.solve()
print(f'Dual iterations: {iterations}')

# primal simplex method
if is_result_optimal:
    primal_method = PrimalSimplexAlgorithm(c, A, b)
    function_result, x, new_c, new_A, new_b, primal_iterations, is_result_optimal, is_primal_result_undefined = primal_method.solve()
    print(f"Primal iterations: {primal_iterations}")
    iterations += primal_iterations
    is_result_undefined |= is_primal_result_undefined

if is_result_undefined:
    print('Result undefined.')
else:
    print(f'Iterations needed = {iterations}')
    print(f'F = {function_result}')
    print(f'x = {x}')
print()


print("## scipy.optimize.linprog(method='simplex') ##")
from scipy import optimize
print(optimize.linprog(c=c, A_eq=A, b_eq=b, method='simplex'))
