#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import heuristics
import ilp
import utils
import generator

# random seed
np.random.seed(23)

if __name__ == "__main__":

    for size in [5, 10]:

        for ind in [3, 4]:

            for timelimit in [1800]:

                # choose image and size
                image = generator.generate_images(size)[ind]

                # initialize with hueristic method
                heuristic_graph = heuristics.solve(image, 0.02)
                heuristic_seg, heuristics_output = utils.graph_to_image(heuristic_graph)

                # build ilp
                model = ilp.build_model(image, 1, 0.5, cycle4=True, cycle8=False, facet=True)
                model.parameters.timelimit.set(timelimit)

                # warm start
                ilp.warm_start(model, heuristic_seg)

                # solve ilp
                print("Solving ilp with b&c...")
                model.solve()

                gap = model.solution.MIP.get_mip_relative_gap()
                print("MIP relative gap:", gap)

                obj = model.solution.get_objective_value()
                print("Objective value:", obj)
