#!/usr/bin/env python
# coding: utf-8

import argparse
import solver_pipeline

if __name__ == "__main__":

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=5, choices=[5, 10, 20])
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--timelimit', type=int, default=1200)
    parser.add_argument('--type', type=str, default='synth', choices=['synth', 'real'])
    parser.add_argument('--noheur', action='store_true', default=False)
    parser.add_argument('--cycle3', action='store_true', default=False)
    parser.add_argument('--cycle4', action='store_true', default=False)
    parser.add_argument('--cycle8', action='store_true', default=False)
    parser.add_argument('--facet', action='store_true', default=False)

    # get args
    args = parser.parse_args()

    # setting
    #args.timelimit = 1
    settings = {'size': [5, 10, 20],
                'noise': [0, 0.001, 0.005]}

    for size in settings['size']:
        args.size = size
        for noise in settings['noise']:
            args.noise = noise
            # no preadd
            args.noheur, args.cycle3, args.cycle4, args.cycle8, args.facet = False, False, False, False, False
            solver_pipeline.pipeline(args)
            # no hueristic
            args.noheur, args.cycle3, args.cycle4, args.cycle8, args.facet = True, False, False, False, False
            solver_pipeline.pipeline(args)
            # facet-defining
            args.noheur, args.cycle3, args.cycle4, args.cycle8, args.facet = False, False, False, False, True
            solver_pipeline.pipeline(args)
            # cycle3
            args.noheur, args.cycle3, args.cycle4, args.cycle8, args.facet = False, True, False, False, False
            solver_pipeline.pipeline(args)
            # cycle3 + cycle4
            args.noheur, args.cycle3, args.cycle4, args.cycle8, args.facet = False, True, True, False, False
            solver_pipeline.pipeline(args)
