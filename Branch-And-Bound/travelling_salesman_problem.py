import numpy as np


class Node:
    def __init__(self, A, di, dj, bound, route):
        self.A = A
        self.di = di
        self.dj = dj
        self.bound = bound
        self.route = route


class BranchAndBoundAlgorithm:
    def __init__(self, table):
        self.costs = table
        self.num_cities = table.shape[0]

    def solve(self):
        A = self.costs.copy()
        for i in range(A.shape[0]):
            A[i][i] = np.inf

        A, di, dj, H = self.get_bound(A)
        best_cost = np.inf
        best_route = None
        root_node = Node(A, di, dj, H, [])
        active_nodes = [root_node]
        while active_nodes:
            node = min(active_nodes, key=lambda x: x.bound)
            active_nodes.remove(node)

            if node.bound >= best_cost:
                continue

            if len(node.route) == self.num_cities:
                node_cost = self.get_route_cost(node.route)
                if node_cost < best_cost:
                    best_cost = node_cost
                    best_route = node.route
            else:
                edge = self.get_pivot_edge(node.A)
                if node.A[edge] == np.inf:
                    continue

                add_edge_child = self.get_child_node_with_edge(node, edge)
                no_edge_child = self.get_child_node_without_edge(node, edge)

                for child in [add_edge_child, no_edge_child]:
                    if child.bound < best_cost:
                        active_nodes.append(child)

        return best_cost, best_route

    def get_route_cost(self, edges):
        return sum([self.costs[i][j] for i, j in edges])

    def get_bound(self, A):
        di = np.array([np.min(row) for row in A])
        di = np.array([val if val != np.inf else 0 for val in di])
        A = (A.T - di).T

        dj = np.array([np.min(col) for col in A.T])
        dj = np.array([val if val != np.inf else 0 for val in dj])
        A = A - dj

        H = np.sum(di) + np.sum(dj)
        return A, di, dj, H

    def get_pivot_edge(self, A):
        di = np.full(A.shape[0], None)
        dj = np.full(A.shape[1], None)
        zero_cells = list(zip(*np.where(A == 0)))
        for i, j in zero_cells:
            A[i][j] = np.inf
            di[i] = np.min(A[i])
            dj[j] = np.min(A[:, j])
            A[i][j] = 0

        ratios = np.array([di[i] + dj[j] for i, j in zero_cells])
        try:
            pivot_edge_index = np.argmax(ratios)
            i, j = zero_cells[pivot_edge_index]  # edge
        except ValueError:
            i, j = 0, 0
        return (i, j)

    def get_child_node_with_edge(self, parent_node, edge):
        A = parent_node.A.copy()
        di = parent_node.di.copy()
        dj = parent_node.dj.copy()
        route = parent_node.route.copy()

        i, j = edge
        A[i] = np.inf
        A[:, j] = np.inf
        A[j][i] = np.inf
        A, di, dj, H = self.get_bound(A)
        return Node(A, di, dj, H, route + [(i, j)])

    def get_child_node_without_edge(self, parent_node, edge):
        A = parent_node.A.copy()
        di = parent_node.di.copy()
        dj = parent_node.dj.copy()
        route = parent_node.route.copy()

        i, j = edge
        A[i, j] = np.inf
        A, di, dj, H = self.get_bound(A)
        return Node(A, di, dj, H, route)
