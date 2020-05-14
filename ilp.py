#!/usr/bin/env python
# coding: utf-8

import cplex
from callback import multicutCallback, cutremoveCallback
import numpy as np
import utils

def build_model(image, param1, param2, cycle4, cycle8, facet):
    """
    build ilp model for piecewise linear
    """
    print("Building Cplex model...")
    # initialize model
    model = cplex.Cplex()
    # set sense
    model.objective.set_sense(model.objective.sense.minimize)
    # get discrete second derivative
    derivative = utils.get_derivative(image)

    # build graph
    graph = utils.to_graph(image)

    # build objective function
    vars = get_varibles(image)
    model.variables.add(names=vars)
    colnames, obj, types = get_obj(derivative, param2)
    model.variables.add(obj=obj, types=types, names=colnames)

    # add constraints
    rows, senses, rhs = get_constraints(image, derivative, param1, cycle4=cycle4, cycle8=cycle8)
    model.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs)

    # parallel
    model.parameters.parallel.set(-1)
    model.parameters.threads.set(32)

    # register callback
    #model.register_callback(cutremoveCallback)
    model.register_callback(multicutCallback)
    # associate additional data
    multicutCallback._graph = graph.copy()
    multicutCallback._names = model.variables.get_names()
    multicutCallback._facet = facet
    #cutremoveCallback._names = model.variables.get_names()

    return model


def get_varibles(image):
    """
    get fittng variables for model
    """
    vars = []

    # fitting value
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            vars.append("w_{}_{}".format(i, j))

    return vars


def get_obj(derivative, param2):
    """
    get coefficients of objective function with colnames
    """
    # fitting part
    f_colnames = []
    f_obj = []
    f_types = ""
    for i in range(derivative.shape[0]):
        for j in range(derivative.shape[1]):
            name = "e_{}_{}".format(i, j)
            f_colnames += [name+"+", name+"-"]
            f_obj += [1, 1]
            f_types += "CC"

    # regularization part
    r_colnames = []
    r_obj = []
    r_types = ""
    # for each row
    for i in range(derivative.shape[0]):
        lambd = param2 * max(np.max(derivative[i,:,0]), 0.2) / 2
        for j in range(derivative.shape[1]-1):
            name = "xr_{}_{}".format(i, j)
            r_colnames.append(name)
            r_obj.append(lambd)
            r_types += "B"
    # for each column
    for j in range(derivative.shape[1]):
        lambd = param2 * max(np.max(derivative[:,j,1]), 0.2) / 2
        for i in range(derivative.shape[0]-1):
            name = "xc_{}_{}".format(i, j)
            r_colnames.append(name)
            r_obj.append(lambd)
            r_types += "B"

    # concatenate
    colnames = f_colnames + r_colnames
    obj = f_obj + r_obj
    types = f_types + r_types

    return colnames, obj, types


def get_constraints(image, derivative, param1, cycle4=True, cycle8=True):
    """
    get constraints
    """
    rows = []
    rhs = []
    senses = ""

    # absolute value constraints
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            vars = ["w_{}_{}".format(i, j),
                    "e_{}_{}+".format(i, j),
                    "e_{}_{}-".format(i, j)]
            coefs = [1, -1, 1]
            yij = image[i, j]
            rows.append([vars, coefs])
            rhs.append(yij)
            senses += "E"

    M = 2
    # big M constraints for rows
    for i in range(derivative.shape[0]):
        #M = param1 * max(np.max(derivative[i,:,0]), 0.5)
        for j in range(1, image.shape[1]-1):
            # second derivative
            vars = ["w_{}_{}".format(i, j-1),
                    "w_{}_{}".format(i, j),
                    "w_{}_{}".format(i, j+1)]
            # big M
            vars += ["xr_{}_{}".format(i, j-1),
                     "xr_{}_{}".format(i, j),]
            # positive derivative
            coefs_p = [1, -2, 1, -M, -M]
            # negative derivative
            coefs_n = [-1, 2, -1, -M, -M]
            rows.append([vars, coefs_p])
            rows.append([vars, coefs_n])
            rhs += [0, 0]
            senses += "LL"

    # big M constraints for columns
    for j in range(derivative.shape[1]):
        #M = param1 * max(np.max(derivative[:,j,1]), 0.5)
        for i in range(1, image.shape[0]-1):
            # second derivative
            vars = ["w_{}_{}".format(i-1, j),
                    "w_{}_{}".format(i, j),
                    "w_{}_{}".format(i+1, j)]
            # big M
            vars += ["xc_{}_{}".format(i-1, j),
                     "xc_{}_{}".format(i, j),]
            # positive derivative
            coefs_p = [1, -2, 1, -M, -M]
            # negative derivative
            coefs_n = [-1, 2, -1, -M, -M]
            rows.append([vars, coefs_p])
            rows.append([vars, coefs_n])
            rhs += [0, 0]
            senses += "LL"

    # 4-edge cycle multicut constraints
    if cycle4:
        print("Add cycle4 constraints")
        for i in range(image.shape[0]-1):
            for j in range(image.shape[1]-1):
                # 4 edges
                vars = ["xr_{}_{}".format(i, j),
                        "xc_{}_{}".format(i, j),
                        "xr_{}_{}".format(i+1, j),
                        "xc_{}_{}".format(i, j+1)]
                for k in range(4):
                    coefs = [1, 1, 1, 1]
                    coefs[k] = -1
                    rows.append([vars, coefs])
                    rhs.append(0)
                    senses += "G"

    # 8-edge cycle multicut constraints
    if cycle8:
        print("Add cycle8 constraints")
        for i in range(image.shape[0]-2):
            for j in range(image.shape[1]-2):
                # 4 edges
                vars = ["xr_{}_{}".format(i, j),
                        "xr_{}_{}".format(i, j+1),
                        "xc_{}_{}".format(i, j+2),
                        "xc_{}_{}".format(i+1, j+2),
                        "xr_{}_{}".format(i+2, j+1),
                        "xr_{}_{}".format(i+2, j),
                        "xc_{}_{}".format(i+1, j),
                        "xc_{}_{}".format(i, j)]
                for k in range(8):
                    coefs = [1, 1, 1, 1, 1, 1, 1, 1]
                    coefs[k] = -1
                    rows.append([vars, coefs])
                    rhs.append(0)
                    senses += "G"

    return rows, senses, rhs


def warm_start(model, segmentation):
    """
    start ilp with given segmentation
    """
    print("Adding initial solution...")
    names = []
    vars = []

    # # cut in row
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]-1):
            names.append("xr_{}_{}".format(i, j))
            if segmentation[i, j] != segmentation[i, j+1]:
                vars.append(1)
            else:
                vars.append(0)

    for i in range(segmentation.shape[0]-1):
        for j in range(segmentation.shape[1]):
            # cut in columns
            names.append("xc_{}_{}".format(i, j))
            if segmentation[i, j] != segmentation[i+1, j]:
                vars.append(1)
            else:
                vars.append(0)

    model.MIP_starts.add(cplex.SparsePair(ind=names, val=vars), model.MIP_starts.effort_level.solve_fixed, "region_fusion")
