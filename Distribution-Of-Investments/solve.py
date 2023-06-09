from distribution_of_investments import Distributor, Company, Investment
import numpy as np

# Example
budget = 8
companies = [
    Company([2, 4, 3, 2, 5],
            [0.5, 0.8, 1.1, 0.6, 1.3]),
    Company([1, 3, 1, 5, 3],
            [0.3, 0.6, 0.4, 1.5, 0.8]),
    Company([1, 2, 1, 5, 4],
            [0.3, 0.4, 0.4, 1.5, 1])
]

print("## My program ##")
solver = Distributor(budget, companies)
investments = solver.solve()
print(*investments, sep='\n')
print(f"Total profit: {sum(investment.profit for investment in investments if investment.profit)}")


print("\n\n## pulp ##")
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary, PULP_CBC_CMD
def check():
    profits = [company.profits for company in companies]
    costs = [company.costs for company in companies]
    investments = budget

    model = LpProblem("Investment_Optimization", LpMaximize)

    num_companies = len(profits)
    num_projects = len(profits[0])
    x = [[LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat=LpBinary) for j in range(num_projects)] for i in
        range(num_companies)]
    model += sum(profits[i][j] * x[i][j] for i in range(num_companies) for j in range(num_projects))
    model += sum(costs[i][j] * x[i][j] for i in range(num_companies) for j in range(num_projects)) <= investments

    for i in range(num_companies):
        model += sum(x[i][j] for j in range(num_projects)) == 1

    for j in range(num_projects):
        model += sum(costs[i][j] * x[i][j] for i in range(num_companies)) <= investments

    model.solve(PULP_CBC_CMD(msg=0))

    print('Profit:', model.objective.value())
    print("Optimal Strategy:")
    for i in range(num_companies):
        for j in range(num_projects):
            if x[i][j].value() > 0:
                print(f"Company {i + 1}. Project {j + 1}")

check()