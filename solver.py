#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import cv2
import numpy as np
import heuristics
import ilp
import utils
import generator

# random seed
np.random.seed(23)

if __name__ == "__main__":
    # choose image and size
    _, image = generator.generate_images(5)[0]

    # add Guassian noise
    noise = 0.001
    image = image + noise * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    image = np.clip(image, 0, 1)

    # visualize input signal
    plt.imshow(image)
    plt.show()

    # visualize 3d input signal
    X = np.arange(image.shape[1])
    Y = np.arange(image.shape[0])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, image, cmap=cm.jet, linewidth=0, antialiased=False)
    plt.show()

    # initialize with hueristic method
    heuristic_graph = heuristics.solve(image, 0.02)
    heuristic_seg, heuristics_output = utils.graph_to_image(heuristic_graph)
    plt.imshow(heuristic_seg)
    plt.show()

    # build ilp
    model = ilp.build_model(image, 1, 0.5, cycle4=True, cycle8=False, facet=True)
    timelimit = 600
    model.parameters.timelimit.set(timelimit)

    # warm start
    ilp.warm_start(model, heuristic_seg)

    # solve ilp
    print("Solving ilp with b&c...")
    model.solve()

    # get result
    gap = model.solution.MIP.get_mip_relative_gap()
    print("MIP relative gap:", gap)
    obj = model.solution.get_objective_value()
    print("Objective value:", obj)

    # visualize boundaries
    boundaries = utils.vis_cut(image, model)
    plt.imshow(boundaries)
    plt.show()

    # visualize segmentation
    segmentations = utils.vis_seg(image, model)
    plt.imshow(segmentations)
    plt.show()

    # visualize depth
    depth = utils.reconstruct(image, model)
    plt.imshow(depth)
    plt.show()
    cv2.imwrite('/home/bo/Desktop/sample/depth.png', (depth*255).astype(np.uint8))

    # visualize 3d input signal
    X = np.arange(depth.shape[1])
    Y = np.arange(depth.shape[0])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, depth, cmap=cm.jet, linewidth=0, antialiased=False)
    plt.show()

    if not utils.check_plane(segmentations, depth):
        print("The solution includes non-planar surfaces")
