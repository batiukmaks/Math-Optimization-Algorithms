import numpy as np

class PrimalSimplexAlgorithm:
    def __init__(self, coefficients, restrictions, plan, max_iter=30):
        self.coefficients = coefficients
        self.restrictions = restrictions 
        self.plan = plan 
        self.max_iter = max_iter

    def solve(self):
        return self.simplex_algorithm()

    def simplex_algorithm(self):
        is_result_optimal, is_result_undefined = False, False
        function_result = 0
        x = []
        iterations = -1
        c, A, b = self.coefficients, self.restrictions, self.plan

        for iteration in range(self.max_iter):
            basis_indexes, A, c, b = self.get_basis(A, c, b)

            reduced_cost_coefficients = self.get_reduced_cost_coefficients(basis_indexes, c, A) # delta-row
            k = self.pivot_column_index(reduced_cost_coefficients, A)
            is_result_optimal, is_result_undefined = self.analyze_result(reduced_cost_coefficients, k, A, basis_indexes)

        
            if is_result_undefined:
                break
            if is_result_optimal:
                x = np.array([x for i, x in sorted(list(zip(basis_indexes, b)) + list(zip(set(range(self.coefficients.shape[0])) - set(basis_indexes), [0]*self.coefficients.shape[0])))])
                function_result = self.get_function_result(self.coefficients, x)
                iterations = iteration
                break

            l, ratios = self.key_row(k, b, A)
            A, b, basis_indexes = self.update_tableau(A, b, basis_indexes, l, k)
        return function_result, x, c, A, b, iterations, is_result_optimal, is_result_undefined


    def get_basis(self, A, c, b):
        basis_indexes = np.full(shape=A.shape[0], fill_value=-1, dtype=np.int64)
        for index, column in enumerate(A.T):
            if all(val in [0, 1] for val in column) and (pos := np.where(column==1))[0].size == 1:
                basis_indexes[pos] = index

        return basis_indexes, A, c, b


    def add_artificial_variables(self, A, c, basis_indexes):
        for index in (no_identity_row := np.where(basis_indexes==-1)[0]):
            c = np.append(c, [np.inf])
            col = np.array([[1 if i == index else 0 for i in range(A.shape[0])]]).T
            A = np.append(A, col, axis=1)
            basis_indexes[index] = c.shape[0] - 1
        return A, c, basis_indexes


    def get_reduced_cost_coefficients(self, basis_indexes, coefficients, A):
        basis_coef = self.get_basis_coefficients(basis_indexes, coefficients)
        reduced_cost_coefficients = np.array([basis_coef @ A[:,j] - coefficients[j] if j not in basis_indexes else 0 for j in range(A.shape[1])])
        return reduced_cost_coefficients


    def get_basis_coefficients(self, indexes, values):
        return [values[index] for index in indexes]


    def pivot_column_index(self, reduced_cost_coefficients, A):
        return np.argmax(reduced_cost_coefficients)


    def analyze_result(self, reduced_cost_coefficients, pivot_column_index, A, basis_indexes):
        is_undefined = False
        is_optimal = reduced_cost_coefficients[pivot_column_index] <= 0
        is_undefined = np.count_nonzero(A[:, pivot_column_index] > 0) == 0 # no positive numbers in the pivot column
        return is_optimal, is_undefined


    def is_artificial_variable_in_basis(self, basis_indexes, initial_variables_number):
        return np.count_nonzero(basis_indexes >= initial_variables_number) > 0


    def get_function_result(self, c, x):
        return c @ x


    def key_row(self, key_col, b, A):
        """ Returns key row index """
        ratios = b / A[:, key_col]
        minimum_greater_that_zero = min(ratio for ratio in ratios if ratio > 0)
        key_row = np.where(ratios == minimum_greater_that_zero)[0][0]
        return key_row, ratios


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
