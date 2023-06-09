import numpy as np
from primal_and_artificial_variables import simplex_algorithm

# Example
A = np.array([
    [5, -2, 1, 0, 0], 
    [-1, 2, 0, 1, 0], 
    [1, 1, 0, 0, -1]
    ], dtype=np.float32)
c = np.array([-3, 6, 0, 0, 0])
b = np.array([4, 4, 4])


print('## My Program ##')
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
function_result, x, new_c, new_A, new_b, iterations, is_result_optimal, is_result_undefined = simplex_algorithm(c, A, b)
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