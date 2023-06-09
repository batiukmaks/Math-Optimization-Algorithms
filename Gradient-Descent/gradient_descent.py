import numpy as np
from golden_section_search import GoldenSectionSearch


class GradientDescent:
    def __init__(self, f, dydx1, dydx2, epsilon, x_start=np.array([0, 0])):
        self.f = f
        self.dydx1 = dydx1
        self.dydx2 = dydx2
        self.epsilon = epsilon
        self.x_start = x_start

    def solve(self):
        x_curr = self.x_start
        x_prev = x_curr + 2*self.epsilon

        itr = 0
        while np.abs(self.f(x_prev) - self.f(x_curr)) > self.epsilon:
            itr += 1
            x_prev = x_curr
            x_curr = x_prev - self.get_beta(x_prev) * self.get_gradient(x_prev)

            print(
                f"{itr} | F = {self.f(x_curr)} | X = {x_curr} | grad = {self.get_gradient(x_curr)}")

        return self.f(x_curr), x_curr

    def get_gradient(self, x):
        return np.array([self.dydx1(x), self.dydx2(x)])

    def get_beta(self, x):
        gss = GoldenSectionSearch()

        def next_X_approximation(beta):
            return self.f(x - beta * self.get_gradient(x))

        beta, *other = gss.search(next_X_approximation, 0, 10, self.epsilon)
        return beta
