"""
A - supply
B - demand
C - cost
"""

import numpy as np

# Example
A = np.array([30, 50, 20], dtype=np.float32)
B = np.array([15, 15, 40, 30], dtype=np.float32)
C = np.array([
    [1, 8, 2, 3],
    [4, 7, 5, 1],
    [5, 3, 4, 4]
], dtype=np.float32)


class TransprotrationProblemSolver:
    def __init__(self, supply, demand, cost):
        self.supply = supply
        self.demand = demand
        self.cost = cost
    

    def solve(self):
        X = []
        minimum_cost_indices, X = self.minimum_cost_method()
        print(X)
        result, X = self.optimize(minimum_cost_indices, X)
        return result, X


    def minimum_cost_method(self):
        # X[i][j] = None --> not chosen, not in the basis
        X = np.full((len(self.supply), len(self.demand)), None)

        # a[i] or b[j] or C[i][j] = np.inf --> row/column/value is excluded from consideration in the following steps (supply and/or demand are filled)
        a = np.copy(self.supply)
        b = np.copy(self.demand)
        C = np.copy(self.cost)

        def has_unused_supply():
            return any(val != np.inf for val in a)
        
        def delete_row(i):
            C[i] = np.inf
            a[i] = np.inf

        def delete_column(j):
            C[:,j] = np.inf
            b[j] = np.inf

        while has_unused_supply():
            i, j = np.unravel_index(C.argmin(), C.shape)
            amount = np.min([a[i], b[j]])
            a[i] -= amount
            b[j] -= amount
            X[i][j] = amount

            if a[i] < b[j]:
                delete_row(i)
            elif a[i] > b[j]:
                delete_column(j)
            elif a[i] == b[j] and any(not val in [0, np.inf] for val in a):
                # delete row and set the minimum element in this column to zero, then delete the column
                delete_row(i)
                i = np.argmin(C[:, j])
                X[i, j] = 0
                delete_column(j)
            else:
                # last element
                C[i,j] = np.inf
                a[i] = np.inf
                b[j] = np.inf
        route_indices = list(zip(*np.where(X != None)))
        return route_indices, X


    def optimize(self, minimum_cost_indices, X):
        for _ in range(100):
            u, v = self.get_potentials(minimum_cost_indices)
            new_i, new_j, ratios, is_result_optimal = self.get_new_basis_variable(u, v)
            if is_result_optimal:
                break
            cycle = self.get_cycle((new_i, new_j), minimum_cost_indices)
            X, minimum_cost_indices = self.update_tableau(X, cycle, minimum_cost_indices)
        result = np.sum([X[i][j]*self.cost[i][j] for i, j in minimum_cost_indices])
        return result, X


    def get_potentials(self, route_indices):
        m, n = self.cost.shape
        u = np.full(m, None)
        v = np.full(n, None)
        u[0] = 0

        for _ in range(m+n):
            for i, j in route_indices:
                if not u[i] is None:
                    v[j] = self.cost[i][j] - u[i]
                elif not v[j] is None:
                    u[i] = self.cost[i][j] - v[j]
        return u, v


    def get_new_basis_variable(self, u, v):
        m, n = self.cost.shape
        ratios = np.array([[u[i] + v[j] - self.cost[i][j] for j in range(n)] for i in range(m)])
        i, j = np.unravel_index(ratios.argmax(), ratios.shape)
        is_result_optimal = ratios[i][j] <= 0
        return i, j, ratios, is_result_optimal


    def get_cycle(self, starting_point, x_indices):
        vertices = x_indices + [starting_point]
        neighbors = [vertex for vertex in vertices if (vertex[0] == starting_point[0] or vertex[1] == starting_point[1]) and vertex != starting_point]
        for neighbor in neighbors:
            is_visited = {key: False for key in vertices}
            cycle = self.dfs(is_visited, vertices, neighbor, starting_point, reach_parent_along_row=neighbor[0]==starting_point[0])
            if cycle:
                return cycle

    def dfs(self, is_visited_from_parent, vertices, current_vertex, cycle_start, reach_parent_along_row):
        if is_visited_from_parent[current_vertex]:
            return []
        is_visited_from_parent[current_vertex] = True
        if current_vertex == cycle_start:
            return [current_vertex]
        if reach_parent_along_row:
            neighbors = [vertex for vertex in vertices if vertex[1] == current_vertex[1]]
        else:
            neighbors = [vertex for vertex in vertices if vertex[0] == current_vertex[0]]
        for neighbor in neighbors:
            cycle_path = self.dfs(is_visited_from_parent, vertices, neighbor, cycle_start, reach_parent_along_row=not reach_parent_along_row)
            if cycle_path:
                return cycle_path + [current_vertex]
        return []


    def update_tableau(self, X, cycle, minimum_cost_indices):
        m, n = X.shape
        new_i, new_j = cycle[0]
        
        # Select the cell to remove from the basis
        minimum_basis_index = np.argmin([[X[i][j] if (i,j) in cycle[1::2] else np.inf for j in range(n)] for i in range(m)])
        out_i, out_j = np.unravel_index(minimum_basis_index, X.shape)

        # Update basis values
        X[new_i][new_j] = 0
        amount = X[out_i][out_j]
        for index, (i, j) in enumerate(cycle):
            if index % 2 == 0:
                X[i][j] += amount
            else:
                X[i][j] -= amount
        X[out_i][out_j] = None
        minimum_cost_indices[minimum_cost_indices.index((out_i, out_j))] = (new_i, new_j)
        return X, minimum_cost_indices


import pulp
def show_right_answer(supply, demand, cost):
    # Create the LP problem
    transport_lp = pulp.LpProblem('Transportation', pulp.LpMinimize)

    # Define the decision variables
    rows = len(supply)
    cols = len(demand)

    variables = pulp.LpVariable.dicts("Transport", ((i, j) for i in range(rows) for j in range(cols)), lowBound=0, cat='Continuous')

    # Define the objective function
    transport_lp += pulp.lpSum([variables[i, j] * cost[i][j] for i in range(rows) for j in range(cols)])

    # Define the supply constraints
    for i in range(rows):
        transport_lp += pulp.lpSum([variables[i, j] for j in range(cols)]) <= supply[i]

    # Define the demand constraints
    for j in range(cols):
        transport_lp += pulp.lpSum([variables[i, j] for i in range(rows)]) == demand[j]

    # Solve the LP problem
    status = transport_lp.solve(pulp.PULP_CBC_CMD(msg=0))

    # Print the results
    print("Status: ", pulp.LpStatus[status])
    print("Minimum Cost: ", pulp.value(transport_lp.objective))
    tableau = np.array([v.varValue for v in transport_lp.variables()]).reshape(C.shape)
    print(f'Tableau: \n{tableau}')


print('## My Solution ##')
solver = TransprotrationProblemSolver(supply=A, demand=B, cost=C)
result, X = solver.solve()
# X = np.array([[0 if val is None else val for val in row] for row in X])
print(f'Minimum Cost: {result} \nAmount of every shipment:\n{X}')


print('\n\n## Check result with `pulp` ##')
show_right_answer(supply=A, demand=B, cost=C)
