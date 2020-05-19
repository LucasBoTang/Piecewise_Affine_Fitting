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
    parser.add_argument('--size', type=int, default=5, choices=[5, 10, 20])
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--timelimit', type=int, default=600)
    parser.add_argument('--type', type=str, default='synth', choices=['synth', 'real'])
    parser.add_argument('--cycle3', action='store_true', default=False)
    parser.add_argument('--cycle4', action='store_true', default=False)
    parser.add_argument('--cycle8', action='store_true', default=False)

    # get args
    args = parser.parse_args()

    # file name
    filename = 'log_{}_{}_{}'.format(args.size, args.noise, args.timelimit)
    cycles = []
    if args.cycle3:
        cycles.append('3')
    if args.cycle4:
        cycles.append('4')
    if args.cycle8:
        cycles.append('8')
    if len(cycles) > 0:
        filename += '_' + '+'.join(cycles)

    # create folder
    if not os.path.isdir('./res'):
        os.mkdir('./res')

    # create table
    df = pd.DataFrame(columns=['time', 'nodes', 'cuts', 'gap', 'obj'])

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
        model = ilp.build_model(image, 1, 0.5, cycle3=args.cycle3,
                                               cycle4=args.cycle4,
                                               cycle8=args.cycle8,
                                               facet=True)
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

        nodes = model.solution.progress.get_num_nodes_processed()
        print("Number of nodes:", nodes)

        cuts = model.solution.MIP.get_num_cuts(model.solution.MIP.cut_type.user)
        print("Number of user applied cuts:", cuts)

        obj = model.solution.get_objective_value()
        print("Objective value:", obj)

        # add data
        df = df.append(pd.Series([elapse, nodes, cuts, gap, obj], index=df.columns), ignore_index=True)

    # save data
    df.to_csv('./res/{}.csv'.format(filename))
