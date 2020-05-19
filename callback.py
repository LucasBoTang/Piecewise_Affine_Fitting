#!/usr/bin/env python
# coding: utf-8

import cplex
from cplex.callbacks import LazyConstraintCallback, IncumbentCallback, HeuristicCallback
import numpy as np
import networkx as nx


class multicutCallback(LazyConstraintCallback):
    def __init__(self, *arg):
        super(multicutCallback, self).__init__(*arg)
        print("Multicut callback is registered!")

    def __call__(self):
        """
        check validation of cuts
        add facet defining cycle constraints
        """
        # current cut
        graph, cuts = self.cut_graph()

        # add multicut constraints
        for path in self.get_multicut(graph, cuts):
            names = self.to_edge(path)
            coefs = [1] * (len(names) - 1) + [-1]
            self.add(constraint=cplex.SparsePair(ind=names, val=coefs), sense="G", rhs=0)

    def cut_graph(self):
        """
        use current solution to cut graph
        """
        graph = self._graph.copy()
        values = self.get_values()
        cuts = []

        for name, value in zip(self._names, values):
            # get cuts
            if name[0] == "x" and value > 0.5:
                drc, i, j = name.split("_")
                i, j = int(i), int(j)
                # apply cut
                if drc == "xr":
                    cut = ((i, j), (i, j+1))
                elif drc == "xc":
                    cut = ((i, j), (i+1, j))
                elif drc == "xf":
                    cut = ((i, j), (i+1, j+1))
                """
                else:
                    cut = ((i-1, j+1), (i, j))
                """
                graph.remove_edge(*cut)
                cuts.append(cut)

        return graph, cuts

    def get_multicut(self, graph, cuts):
        """
        find violated cut with all facet defining paths
        """
        # get connected components
        h, w = 1, 1
        for i, j in graph.nodes:
            h, w = max(h, i), max(w, j)
        components_map = np.zeros((h+1, w+1), dtype=int)
        for k, comp in enumerate(nx.connected_components(graph)):
            for i, j in comp:
                components_map[i, j] = k

        # find violation
        violated_cuts = []
        for (i1, j1), (i2, j2) in cuts:
            if components_map[i1, j1] == components_map[i2, j2]:
                violated_cuts.append(((i1, j1), (i2, j2)))

        if violated_cuts:
            print("Adding cuts...")

        # dfs to find facet defining paths
        paths = []
        for s, t in violated_cuts:
            paths += self.find_path(graph, s, t)

        return paths

    def find_path(self, graph, source, sink):
        """
        depth first search to find path between source to sink
        """
        paths = []
        visited = []
        banned = []
        # get all paths with less 10 length
        paths = self.dfs(graph, source, sink, visited, banned, paths)
        # no path has been found
        if len(paths) == 0:
            # get a shortest path
            paths = self.shortest_path(graph, source, sink)

        return paths

    def dfs(self, graph, cur, sink, visited, banned, paths):
        """
        depth first search within 10 lenth to avoid long time search
        """
        # visit current node
        visited = visited + [cur]

        # ban nodes which violate facet defining
        if self._facet:
            if len(visited) > 2:
                banned = banned + list(graph.neighbors(visited[-3]))
            if cur in banned:
                #print("Ban!")
                return paths

        # visualize dfs
        #self._vis_dfs(visited, graph)

        # reach sink
        if cur == sink:
            paths.append(visited)
            #print("Find!")
            return paths

        # cut long path
        if len(visited) > 10:
            return paths

        # dfs
        for i in graph.neighbors(cur):
            if i not in visited:
                paths = self.dfs(graph, i, sink, visited, banned, paths)
        return paths

    def shortest_path(self, graph, cur, sink):
        """
        shortest path to gurantee to find a violated path
        """
        path = nx.shortest_path(graph, cur, sink)
        return [path]

    def to_edge(self, path):
        """
        convert ordered list of nodes to edge variable names
        """
        names = []

        # connect path to cycle
        cycle = path + [path[0]]

        for k in range(len(path)):
            # kth edge
            i1, j1 = cycle[k]
            i2, j2 = cycle[k+1]
            # sort
            if i1 > i2:
                i1, i2 = i2, i1
                j1, j2 = j2, j1
            if j1 > j2:
                i1, i2 = i2, i1
                j1, j2 = j2, j1
            # row edge
            if i1 == i2 and j2 - j1 == 1:
                name = "xr_{}_{}".format(i1, j1)
            # columns edge
            elif i2 - i1 == 1 and j1 == j2:
                name = "xc_{}_{}".format(i1, j1)
            # forward diagonal edge
            elif i2 - i1 == 1 and j2 - j1 == 1:
                name = "xf_{}_{}".format(i1, j1)
            # backward diagonal edge
            elif i2 - i1 == -1 and j2 - j1 == 1:
                name = "xb_{}_{}".format(i1, j1)
            else:
                raise SystemExit("Edge variable is ignored!")
            names.append(name)
        return names

    def _vis_dfs(self, visited, graph):
        """
        visualize dfs path (for debug and test)
        """
        # image size
        h, w = 0, 0
        for i, j in graph.nodes:
            h, w = max(i+1, h), max(j+1, w)
        img = np.zeros((h, w))

        # mark path
        for x, y in visited:
            img[x, y] = 1

        # visualize
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()


class cutremoveCallback(IncumbentCallback, HeuristicCallback):
    def __init__(self, *arg):
        super(cutremoveCallback, self).__init__(*arg)
        print("Cutremove callback is registered!")

    def __call__(self):
        """
        check the validation of cuts in current incumbent solution
        remove unnecessary cut
        """
        derivative = self.get_derivative()
        cuts = self.get_unnecessary_cuts(derivative)
        if cuts:
            print("Remove cuts...")
            names = list(cuts)
            values = [0] * len(names)
            self.set_solution((names, values))

    def get_derivative(self):
        """
        get absolute second derivative matrix
        """
        # get current output
        output = np.zeros((3000, 3000))
        for name, var in zip(self._names, self.get_values()):
            if name[0] == "w":
                i, j = name.split("_")[1:]
                i, j = int(i), int(j)
                output[i, j] = var
        output = output[:i+1, :j+1]

        # calculate derivative
        derivative = np.zeros((*output.shape, 2))
        # second derivative on rows
        for i in range(output.shape[0]):
            for j in range(1, output.shape[1]-1):
                dw2 = output[i, j-1] - 2 * output[i, j] + output[i, j+1]
                derivative[i, j, 0] = dw2
        # edge padding
        derivative[:, 0, 0] = derivative[:, 1, 0]
        derivative[:, -1, 0] = derivative[:, -2, 0]
        # second derivative on columns
        for j in range(output.shape[1]):
            for i in range(1, output.shape[0]-1):
                dw2 = output[i-1, j] - 2 * output[i, j] + output[i+1, j]
                derivative[i, j, 1] = dw2
        # edge padding
        derivative[0, :, 1] = derivative[1, :, 1]
        derivative[-1, :, 1] = derivative[-2, :, 1]

        return np.abs(derivative)

    def get_unnecessary_cuts(self, derivative):
        """
        find cuts without 2rd derivative
        """
        unnecessary_cuts = set()
        # check rows
        for i in range(derivative.shape[0]):
            for j in range(derivative.shape[1]-1):
                if derivative[i, j, 0] < 0.1:
                    unnecessary_cuts.add("xr_{}_{}".format(i, j))
                    unnecessary_cuts.add("xr_{}_{}".format(i, j+1))

        # check columns
        for j in range(derivative.shape[0]-1):
            for i in range(derivative.shape[1]):
                if derivative[i, j, 1] < 0.1:
                    unnecessary_cuts.add("xc_{}_{}".format(i, j))
                    unnecessary_cuts.add("xc_{}_{}".format(i+1, j))

        return unnecessary_cuts
