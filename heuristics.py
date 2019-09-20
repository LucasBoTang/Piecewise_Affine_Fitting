#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import networkx as nx
import utils

def solve(image, lambd):
    """
    region fusion for piecewise regression
    """
    # build graph with affine parameters
    graph = affine_regression(image)

    # region fusion
    print("Running region fusion heuristics...")
    gamma = 2.2
    iterations = 100
    for i in range(iterations):
        # increase beta for each iteration
        beta = (i / iterations) ** gamma * lambd
        # region fusion
        region_fuse(graph, beta)

    return graph


def affine_regression(image):
    """
    perform a parametric affine fitting
    """
    # build graph
    graph = utils.to_graph(image)

    affine_params = np.zeros((*image.shape, 3))
    print("Fitting affine parameters...")
    for (i, j) in graph.nodes():
        # find the best 4 points fiting plane
        mse = float("inf")
        #======================= right and down ================================
        # avoid out of bound
        if i < image.shape[0] - 1 and j < image.shape[1] - 1:
            X = [[i, j]]
            y = [image[i, j]]
            # down neighbor
            X.append([i+1, j])
            y.append(image[i+1, j])
            # right neighbor
            X.append([i, j+1])
            y.append(image[i, j+1])
            # down right neighbor
            X.append([i+1, j+1])
            y.append(image[i+1, j+1])

            # linear regression
            cur_affine_param, cur_mse = fit(X, y)
            if cur_mse < mse:
                affine_param, mse = cur_affine_param, cur_mse

        #========================== left and down ===============================
        # avoid out of bound
        if i < image.shape[0] - 1 and j:
            X = [[i, j]]
            y = [image[i, j]]
            # down neighbor
            X.append([i+1, j])
            y.append(image[i+1, j])
            # left neighbor
            X.append([i, j-1])
            y.append(image[i, j-1])
            # down left neighbor
            X.append([i+1, j-1])
            y.append(image[i+1, j-1])

            # linear regression
            cur_affine_param, cur_mse = fit(X, y)
            if cur_mse < mse:
                affine_param, mse = cur_affine_param, cur_mse

        #========================== right and up ===============================
        # avoid out of bound
        if i and j < image.shape[1] - 1:
            X = [[i, j]]
            y = [image[i, j]]
            # up neighbor
            X.append([i-1, j])
            y.append(image[i-1, j])
            # right neighbor
            X.append([i, j+1])
            y.append(image[i, j+1])
            # up right neighbor
            X.append([i-1, j+1])
            y.append(image[i-1, j+1])

            # linear regression
            cur_affine_param, cur_mse = fit(X, y)
            if cur_mse < mse:
                affine_param, mse = cur_affine_param, cur_mse

        #========================= left and up =================================
        # avoid out of bound
        if i and j:
            X = [[i, j]]
            y = [image[i, j]]
            # down neighbor
            X.append([i-1, j])
            y.append(image[i-1, j])
            # left neighbor
            X.append([i, j-1])
            y.append(image[i, j-1])
            # down left neighbor
            X.append([i-1, j-1])
            y.append(image[i-1, j-1])

            # linear regression
            cur_affine_param, cur_mse = fit(X, y)
            if cur_mse < mse:
                affine_param, mse = cur_affine_param, cur_mse

        # record best affine parameters
        affine_params[i, j] = affine_param

    # set attribute
    for (i, j) in graph.nodes():
        # number of nodes as weight
        graph.nodes[(i,j)]["weight"] = 1
        # coordinates and depth
        graph.nodes[(i,j)]["pixels"] = np.array([[i, j, image[i, j]]])
        # affine parameters
        graph.nodes[(i,j)]["affine_params"] = affine_params[i, j]

    return graph


def fit(X, y):
    """
    linear regression fitting
    """
    # linear regression
    reg = LinearRegression().fit(X, y)
    # update affine parameters
    y_pred = reg.predict(X)
    mse = mean_squared_error(y, y_pred)
    affine_param = [reg.intercept_, *reg.coef_]

    return affine_param, mse


def region_fuse(graph, beta):
    """
    fuse region under current tolerance beta
    """
    # loop through all nodes
    for u in list(graph.nodes):
        # skip contracted node
        if u not in graph.nodes:
            continue
        # get attributes of u
        wu = graph.nodes[u]["weight"]
        Yu = graph.nodes[u]["affine_params"]

        # go through neighbors
        for v in list(graph.neighbors(u)):
            # get attributes of v
            wv = graph.nodes[v]["weight"]
            Yv = graph.nodes[v]["affine_params"]
            # get attributes of (i, j)
            cuv = graph.edges[u, v]["connections"]

            # fusion criterion
            if wu * wv * np.linalg.norm(Yu - Yv) ** 2 <= beta * cuv * (wu + wv):
                # contract nodes
                contract(graph, v, u)
                # contract plane
                u = v

    return graph


def contract(graph, u, v):
    """
    contract node u and node v into node u
    """
    # attributes of node i
    wu = graph.nodes[u]["weight"]
    Yu = graph.nodes[u]["affine_params"]
    # attributes of node j
    wv = graph.nodes[v]["weight"]
    Yv = graph.nodes[v]["affine_params"]

    # merge node attributes
    graph.nodes[u]["pixels"] = np.concatenate((graph.nodes[u]["pixels"], graph.nodes[v]["pixels"]), axis=0)
    graph.nodes[u]["weight"] = wu + wv
    if len(np.unique(graph.nodes[u]["pixels"][:,0])) > 1 and len(np.unique(graph.nodes[u]["pixels"][:,1])) > 1:
        X, y = [], []
        for pixel in graph.nodes[u]["pixels"]:
            X.append([pixel[0], pixel[1]])
            y.append(pixel[2])
        affine_param, _ = fit(X, y)
        graph.nodes[u]["affine_params"] = np.array(affine_param)
    else:
        graph.nodes[u]["affine_params"] = (wu * Yu + wv * Yv) / (wu + wv)


    # combine edges
    if v in graph.neighbors(u):
        graph.remove_edge(u, v)
    for k in graph.neighbors(v):
        if k in graph.neighbors(u):
            graph.edges[u, k]["connections"] += graph.edges[v, k]["connections"]
        else:
            graph.add_edge(u, k, connections=graph.edges[v, k]["connections"])

    # remove node v
    graph.remove_node(v)
