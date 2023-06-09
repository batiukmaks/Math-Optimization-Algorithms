import numpy as np


class State:
    def __init__(self, profits, costs):
        self.profits = profits
        self.costs = costs


class Company:
    def __init__(self, costs, profits):
        self.costs = costs
        self.profits = profits


class Investment:
    def __init__(self, company, project, cost, profit):
        self.company = company
        self.project = project
        self.cost = cost
        self.profit = profit

    def __str__(self):
        return f"Company {self.company}, project {self.project}, cost {self.cost}, profit {self.profit}"


class Distributor:
    def __init__(self, budget, companies):
        self.budget = budget
        self.companies = companies
        self.states = []
        self.max_cost = max(cost for company in companies for cost in company.costs) # for the table width
    
    
    def solve(self):
        for company in reversed(self.companies):
            D = np.zeros((self.budget+1, self.max_cost+1))
            prev_state = self.states[-1] if self.states else State([0]*(self.budget + 1), [0]*(self.budget + 1))

            # Fill the income table
            for x in range(self.budget + 1):
                for k in range(self.max_cost + 1):
                    if x < k:
                        D[x][k] = 0
                    else:
                        D[x][k] = max([profit for cost, profit in zip(company.costs, company.profits) if cost == k] + [0]) + prev_state.profits[x-k]
            
            # Select the best investments for every budget
            state = State([0]*(self.budget + 1), [0]*(self.budget + 1))
            for i, row in enumerate(D):
                cost = np.argmax(row)
                profit = row[cost]
                state.profits[i] = profit
                state.costs[i] = cost
            self.states.append(state)
        

        investments = []
        budget = self.budget
        for i, state in enumerate(reversed(self.states)):
            # Use try except to avoid errors when we do not select any project from the company
            try:
                project_cost = state.costs[budget]

                projects = list(enumerate(zip(self.companies[i].costs, self.companies[i].profits)))
                possible_projects = list(filter(lambda project: project[1][0] == project_cost, list(projects)))
                project = max(possible_projects, key=lambda project: project[1][1])[0]

                investment = Investment(i + 1, project + 1, project_cost, self.companies[i].profits[project])
                investments.append(investment)

                budget -= int(investment.cost)
            except:
                pass
        return investments
