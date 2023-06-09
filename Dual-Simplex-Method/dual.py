# A - Constraint matrix
# c - Coefficients of the objective function
# b - "Plan"
# f -> min

import numpy as np


class DualSimplexAlgorithm:
    def __init__(self, coefficients, restrictions, plan, max_iter=30):
        self.coefficients = coefficients
        self.restrictions = restrictions 
        self.plan = plan 
        self.max_iter = max_iter

    def solve(self):
        return self.dual_simplex_algorithm()

    def dual_simplex_algorithm(self):
        is_result_optimal, is_result_undefined = False, False
        c, A, b = self.coefficients, self.restrictions, self.plan
        iterations = 0

        for iteration in range(self.max_iter):
            basis_indexes, A, c, b = self.get_basis(A, c, b)
            l, is_result_optimal, is_result_undefined = self.pivot_row_index(basis_indexes, b, A)

            if is_result_optimal:
                is_result_undefined = False
                iterations = iteration
                break
            if is_result_undefined:
                iterations = iteration
                break

            reduced_cost_coefficients = self.get_reduced_cost_coefficients(basis_indexes, c, A) # delta-row
            k, ratios = self.pivot_column_index(reduced_cost_coefficients, l, A)
            A, b, basis_indexes = self.update_tableau(A, b, basis_indexes, l, k)
        return c, A, b, iterations, is_result_optimal, is_result_undefined

    
    def get_basis(self, A, c, b):
        basis_indexes = np.full(shape=A.shape[0], fill_value=-1, dtype=np.int64)
        for index, column in enumerate(A.T):
            if all(val in [0, 1] for val in column) and (pos := np.where(column==1))[0].size == 1:
                basis_indexes[pos] = index
        return basis_indexes, A, c, b
    

    def pivot_row_index(self, basis_indexes, b, A):
        l = np.argmin(b)
        is_plan_nonnegative = b[l] >= 0
        is_any_negative_values = any(val < 0 for val in A[l])
        is_only_nonnegative_values = not is_any_negative_values
        return l, is_plan_nonnegative, is_only_nonnegative_values
    

    def get_basis_coefficients(self, indexes, values):
        return [values[index] for index in indexes]
    

    def get_reduced_cost_coefficients(self, basis_indexes, coefficients, A):
        basis_coef = self.get_basis_coefficients(basis_indexes, coefficients)
        reduced_cost_coefficients = np.array([basis_coef @ A[:,j] - coefficients[j] if j not in basis_indexes else 0 for j in range(A.shape[1])])
        return reduced_cost_coefficients


    def pivot_column_index(self, reduced_cost_coefficients, pivot_row, A):
        ratios = np.array([abs(delta/a) if a < 0 else np.inf for delta, a in zip(reduced_cost_coefficients, A[pivot_row])])
        pivot_column = np.argmin(ratios)
        return pivot_column, ratios


    def update_tableau(self, A, b, basis, l, k):
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

