
#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import random
import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

def get_derivative(image):
    """
    get absolute second derivative matrix
    """
    derivative = np.zeros((*image.shape, 2))

    # second derivative on rows
    for i in range(image.shape[0]):
        for j in range(1, image.shape[1]-1):
            dw2 = image[i, j-1] - 2 * image[i, j] + image[i, j+1]
            derivative[i, j, 0] = dw2
    # edge padding
    derivative[:, 0, 0] = derivative[:, 1, 0]
    derivative[:, -1, 0] = derivative[:, -2, 0]

    # second derivative on columns
    for j in range(image.shape[1]):
        for i in range(1, image.shape[0]-1):
            dw2 = image[i-1, j] - 2 * image[i, j] + image[i+1, j]
            derivative[i, j, 1] = dw2
    # edge padding
    derivative[0, :, 1] = derivative[1, :, 1]
    derivative[-1, :, 1] = derivative[-2, :, 1]

    return np.abs(derivative)


def to_graph(image):
    """
    convert image into graph
    """
    print("Bulding graph from image...")
    # initialize geaph
    graph = nx.Graph()

    # add nodes to graph
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            graph.add_node((i, j))

    # add edges to graph
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cur = (i, j)
            # down edge
            down = (i+1, j)
            if down in graph.nodes:
                graph.add_edge(cur, down, connections=1)
            # right edge
            right = (i, j+1)
            if right in graph.nodes:
                graph.add_edge(cur, right, connections=1)

    return graph


def graph_to_image(graph):
    """
    convert graph into segmentation and output image
    """
    seg = np.zeros((3000, 3000), dtype=int)
    image = np.zeros((3000, 3000))

    height, width = 1, 1
    for s, u in enumerate(graph.nodes):
        for i, j, _ in graph.nodes[u]["pixels"]:
            i, j = int(i), int(j)
            seg[i, j] = s
            image[i, j] = graph.nodes[u]["affine_params"][0] + graph.nodes[u]["affine_params"][1] * i + graph.nodes[u]["affine_params"][2] * j
            height, width = max(height, i+1), max(width, j+1)

    return seg[:height, :width], image[:height, :width]


def model_to_image(image, model):
    """
    convert solution into output image
    """
    output = np.zeros_like(image)
    for name, var in zip(model.variables.get_names(), model.solution.get_values()):
        if name[0] == "w":
            i, j = name.split("_")[1:]
            i, j = int(i), int(j)
            output[i, j] = var

    return output


def vis_cut(image, model):
    """
    visualize cut on image
    """
    boundaries = np.zeros_like(image)
    pairs = zip(model.variables.get_names(), model.solution.get_values())
    for name, var in pairs:
        if name[:2] == "xr":
            i, j = name.split("_")[1:]
            i, j = int(i), int(j)
            if var > 0.01:
                boundaries[i, j] = 1
        elif name[:2] == "xc":
            i, j = name.split("_")[1:]
            i, j = int(i), int(j)
            if var > 0.01:
                boundaries[i, j] = 1

    return boundaries


def vis_seg(image, model):
    """
    visualize segmentation of image
    """
    segmentation = np.zeros_like(image)
    graph = to_graph(image)

    # cut graph
    pairs = zip(model.variables.get_names(), model.solution.get_values())
    for name, var in pairs:
        if name[:2] == "xr":
            i, j = name.split("_")[1:]
            i, j = int(i), int(j)
            if var > 0.01:
                graph.remove_edge((i,j), (i,j+1))
        elif name[:2] == "xc":
            i, j = name.split("_")[1:]
            i, j = int(i), int(j)
            if var > 0.01:
                graph.remove_edge((i,j), (i+1,j))

    # get components
    for k, comp in enumerate(nx.connected_components(graph)):
        for i, j in comp:
            segmentation[i, j] = k

    return segmentation


def reconstruct(image, model):
    """
    reconstruct depth map from ilp result
    """
    depth = np.zeros_like(image)

    pairs = zip(model.variables.get_names(), model.solution.get_values())
    for name, var in pairs:
        if name[0] == "w":
            _, i, j = name.split("_")
            i, j = int(i), int(j)
            depth[i, j] = var

    return depth

def check_plane(segmentations, depth):
    """
    check fitting result is piecewise or not
    """
    # collect points
    segmentations_set = defaultdict(list)
    for i in range(segmentations.shape[0]):
        for j in range(segmentations.shape[1]):
            segmentations_set[segmentations[i,j]].append([[i, j], depth[i,j]])
    # check points
    is_plane = True
    for i in segmentations_set:
        X, Y = [], []
        for x, y in random.sample(segmentations_set[i], min(10, len(segmentations_set[i]))):
            X.append(x)
            Y.append(y)
        # linear regression
        reg = LinearRegression().fit(X, Y)
        # predict
        X, Y = [], []
        for x, y in segmentations_set[i]:
            X.append(x)
            Y.append(y)
        Y_pred = reg.predict(X)
        is_plane &= np.all((Y - Y_pred) < 10e-4)

    return is_plane
