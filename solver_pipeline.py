#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
import pandas as pd

import heuristics
import ilp
import utils
import generator

# random seed
np.random.seed(23)

if __name__ == "__main__":

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--timelimit', type=int, default=600)
    parser.add_argument('--type', type=str, default='synth', choices=['synth', 'real'])

    # get args
    args = parser.parse_args()

    # create folder
    if not os.path.isdir('./res'):
        os.mkdir('./res')

    # create table
    df = pd.DataFrame(columns=['time', 'gap', 'obj'])

    # load images
    images = generator.generate_images(args.size)

    # image type
    if args.type == "synth":
        index = [0, 1, 2]
    else:
        index = [3, 4, 5]

    for i in index:
        print("Loading image {}...".format(i))
        image = images[i]
        image = image + args.noise * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
        image = np.clip(image, 0, 1)

        tick = time.time()

        # initialize with hueristic method
        heuristic_graph = heuristics.solve(image, 0.02)
        heuristic_seg, heuristics_output = utils.graph_to_image(heuristic_graph)

        # build ilp
        model = ilp.build_model(image, 1, 0.5, cycle4=True, cycle8=False, facet=True)
        model.parameters.timelimit.set(args.timelimit)

        # warm start
        ilp.warm_start(model, heuristic_seg)

        # solve ilp
        print("Solving ilp with b&c...")
        model.solve()

        tock = time.time()
        elapse = tock - tick
        print("Time elpase:", elapse)

        gap = model.solution.MIP.get_mip_relative_gap()
        print("MIP relative gap:", gap)

        obj = model.solution.get_objective_value()
        print("Objective value:", obj)

        # add data
        df = df.append(pd.Series([elapse, gap, obj], index=df.columns), ignore_index=True)

    # save data
    df.to_csv('./res/log.csv')
