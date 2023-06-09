import numpy as np
from travelling_salesman_problem import BranchAndBoundAlgorithm

# Example
table = np.array([
    [0, 20, 5, 5, 8],
    [8, 0, 9, 12, 18],
    [14, 5, 0, 4, 13],
    [13, 13, 18, 0, 4],
    [12, 3, 16, 13, 0],
], dtype=np.float32)


def check_result(table):
    import numpy as np
    import networkx as nx

    # Create a complete graph with the given distances as edge weights
    G = nx.DiGraph(table)
    tour = nx.algorithms.approximation.traveling_salesman.held_karp_ascent(G)
    print("Optimal cost:", tour[0])
    print("Optimal tour:", tour[1].edges)


print('## My Program ##')
solver = BranchAndBoundAlgorithm(table)
cost, edges = solver.solve()
print("Optimal cost:", cost)
print("Optimal tour:", edges)


print('\n## networkx ##')
check_result(table)
