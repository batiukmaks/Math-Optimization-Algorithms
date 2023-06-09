# A - Constraint matrix
# c - Coefficients of the objective function
# b - "Plan"
# f -> min

import numpy as np


def simplex_algorithm(init_coefficients, restrictions, plan, max_iter=10):
    is_result_optimal, is_result_undefined = False, False
    function_result = 0
    x = []
    iterations = -1
    c, A, b = init_coefficients, restrictions, plan

    for iteration in range(max_iter):
        basis_indexes, A, c, b = get_basis(A, c, b)

        reduced_cost_coefficients = get_reduced_cost_coefficients(basis_indexes, c, A) # delta-row
        k = pivot_column_index(reduced_cost_coefficients, A)
        is_result_optimal, is_result_undefined = analyze_result(reduced_cost_coefficients, k, A, basis_indexes, init_coefficients)

    
        if is_result_undefined:
            break
        if is_result_optimal:
            x = np.array([x for i, x in sorted(list(zip(basis_indexes, b)) + list(zip(set(range(init_coefficients.shape[0])) - set(basis_indexes), [0]*init_coefficients.shape[0])))], dtype=np.float32)
            function_result = get_function_result(init_coefficients, x)
            iterations = iteration
            break

        l, ratios = key_row(k, b, A)
        A, b, basis_indexes = update_tableau(A, b, basis_indexes, l, k)
    return function_result, x, c, A, b, iterations, is_result_optimal, is_result_undefined


def get_basis(A, c, b):
    basis_indexes = np.full(shape=A.shape[0], fill_value=-1, dtype=np.int64)
    for index, column in enumerate(A.T):
        if all(val in [0, 1] for val in column) and (pos := np.where(column==1))[0].size == 1:
            basis_indexes[pos] = index

    A, c, basis_indexes = add_artificial_variables(A, c, basis_indexes)
    return basis_indexes, A, c, b


def add_artificial_variables(A, c, basis_indexes):
    for index in (no_identity_row := np.where(basis_indexes==-1)[0]):
        c = np.append(c, [np.inf])
        col = np.array([[1 if i == index else 0 for i in range(A.shape[0])]]).T
        A = np.append(A, col, axis=1)
        basis_indexes[index] = c.shape[0] - 1
    return A, c, basis_indexes


def get_reduced_cost_coefficients(basis_indexes, coefficients, A):
    basis_coef = get_basis_coefficients(basis_indexes, coefficients)
    reduced_cost_coefficients = np.array([basis_coef @ A[:,j] - coefficients[j] if j not in basis_indexes else 0 for j in range(A.shape[1])])
    return reduced_cost_coefficients


def get_basis_coefficients(indexes, values):
    return [values[index] for index in indexes]


def pivot_column_index(reduced_cost_coefficients, A):
    return np.argmax(reduced_cost_coefficients)


def analyze_result(reduced_cost_coefficients, pivot_column_index, A, basis_indexes, init_coefficients):
    is_undefined = False
    is_optimal = reduced_cost_coefficients[pivot_column_index] <= 0
    is_undefined = np.count_nonzero(A[:, pivot_column_index] > 0) == 0 # no positive numbers in the pivot column
    is_undefined |= is_optimal and is_artificial_variable_in_basis(basis_indexes=basis_indexes, initial_variables_number=len(init_coefficients)) # The artificial variable is in the optimal result
    return is_optimal, is_undefined


def is_artificial_variable_in_basis(basis_indexes, initial_variables_number):
    return np.count_nonzero(basis_indexes >= initial_variables_number) > 0


def get_function_result(c, x):
    return c @ x


def key_row(key_col, b, A):
    """ Returns key row index """
    ratios = b / A[:, key_col]
    minimum_greater_that_zero = min(ratio for ratio in ratios if ratio > 0)
    key_row = np.where(ratios == minimum_greater_that_zero)[0][0]
    return key_row, ratios


def update_tableau(A, b, basis, l, k):
    new_A, new_b, new_basis = np.copy(A), np.copy(b), np.copy(basis)

    # Calculate the plan and the matrix
    new_b = b[:] - b[l] * A[:, k] / A[l, k]
    for j in range(A.shape[1]):
        new_A[:, j] = A[:, j] - A[l, j] * A[:, k] / A[l, k]

    # Change pivot row
    new_basis[l] = k
    new_A[l, :] = A[l, :] / A[l, k]
    new_b[l] = b[l] / A[l, k]

    # Set zeros in the pivot column
    new_A[:, k] = 0
    new_A[l, k] = 1
    return new_A, new_b, new_basis




# # My task
# A = np.array([
#     [5, -2, 1, 0, 0], 
#     [-1, 2, 0, 1, 0], 
#     [1, 1, 0, 0, -1]
#     ], dtype=np.float32)
# c = np.array([-3, 6, 0, 0, 0])
# b = np.array([4, 4, 4])


# # My from Lab 1 
# A = np.array([[2, 1, 1, 0, 0, 0], [2, 3, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]], dtype=np.float32)
# c = -np.array([1, 1, 0, 0, 0, 0])
# b = np.array([2400, 4800, 800, 1000])


# # Example from the lecture
# A = np.array([
#     [2, 3, 1, 0, 0], 
#     [4, 1, 0, 1, 0], 
#     [6, 7, 0, 0, 1]
# ], dtype=np.float32)
# c = -np.array([16, 6, 0, 0, 0])
# b = np.array([180, 240, 426])


# # The artificial variable is in the optimal result. Result is undefined
# A = np.array([
#     [5, -2, 1, 0, 0], 
#     [-1, 3, 0, 1, 0], 
#     [1, 1, 0, 0, -1]
#     ], dtype=np.float32)
# c = np.array([3, -6, 0, 0, 0])
# b = np.array([4, 4, 4])


# print('## My Program ##')
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# function_result, x, new_c, new_A, new_b, iterations, is_result_optimal, is_result_undefined = simplex_algorithm(c, A, b)
# if is_result_undefined:
#     print('Result undefined.')
# else:
#     print(f'Iterations needed = {iterations}')
#     print(f'F = {function_result}')
#     print(f'x = {x}')
# print()


# print("## scipy.optimize.linprog(method='simplex') ##")
# from scipy import optimize
# print(optimize.linprog(c=c, A_eq=A, b_eq=b, method='simplex'))
