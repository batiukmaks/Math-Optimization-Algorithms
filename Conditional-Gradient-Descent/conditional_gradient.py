import numpy as np
from golden_section_search import GoldenSectionSearch
from primal_and_artificial_variables import simplex_algorithm


class ConditionalGradient:
    def __init__(self, f, domain_canonical_A, domain_canonical_b, epsilon, x_start=np.array([0, 0])):
        self.f = f
        self.A = domain_canonical_A
        self.b = domain_canonical_b
        self.epsilon = epsilon
        self.x_start = x_start

    def solve(self):
        X_curr = self.x_start
        X_prev = X_curr + 2*self.epsilon

        itr = 0
        while np.abs(self.f(X_prev) - self.f(X_curr)) > self.epsilon:
            itr += 1
            X_prev = X_curr

            # coefficients (a, b) of the f0(x): self.gradient(X_curr) @ (X - X_curr)
            grad = self.gradient(X_curr)
            c = np.concatenate(
                (grad, np.array([0] * (self.A.shape[1] - grad.shape[0]))))
            function_result, x, new_c, new_A, new_b, iterations, is_result_optimal, is_result_undefined = simplex_algorithm(
                c, self.A, self.b)
            Y_curr = x[:len(self.x_start)] # Next point where the best result is
            H_curr = Y_curr - X_curr # Direction from X to Y
            X_curr = X_curr + self.get_beta(X_curr) * H_curr # Next X approximation
            print(
                f"{itr} | F = {self.f(X_curr)} | X = {X_curr} | grad = {self.gradient(X_curr)}\n")
        return self.f(X_curr), X_curr

    def get_beta(self, x):
        gss = GoldenSectionSearch()

        def next_X_approximation(beta):
            return self.f(x - beta * self.gradient(x))

        beta, *other = gss.search(next_X_approximation, 0, 1, self.epsilon)
        return beta

    def gradient(self, X):
        run = 1e-5
        grad = []
        for i, x in enumerate(X):
            X_plus_run = X.copy()
            X_plus_run[i] += run
            derivative = (self.f(X_plus_run) - self.f(X)) / run
            grad.append(derivative)
        return np.array(grad)
