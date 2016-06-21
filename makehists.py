#!/usr/bin/env python3

"""
Read in Julian's files and make plotting files
"""

import numpy as np
from ndhist.hist import Hist, Axis
from ndhist.mpl import Canvas
import argparse
import itertools
from os.path import splitext, join, isdir
import os

_var_struct = [
    "pt", "eta",
    "t2_d0sig", "t3_d0sig",
    "t2_z0sig", "t3_z0sig",
    "n_d0sig", "jet_prob",
    "width_eta", "width_phi",
    "vertex_significance", "n_secondary",
    "n_secondary_tracks", "dr_vertex",
    "mass", "energy_fraction",
    "y_label", "y_binary_label", "track_prediction", "vertex_prediction",
    "tracks_and_vertices_prediction", "high_prediction",
    "high_and_tracks_prediction", "high_and_vertex_prediction",
    "all_prediction"
]

def _get_structured(file_name):
    raw_array =  np.load(file_name)
    assert raw_array.shape[1] == len(_var_struct)
    dtype = [(x, 'float32') for x in _var_struct]
    arr = np.zeros(raw_array.shape[0], dtype=dtype)
    for num, name in enumerate(_var_struct):
        arr[name] = raw_array[:, num]
    return arr

def _get_cut_at_rejection(array, rejection):
    eff = 1 / rejection
    index = int(eff * array.size)
    error = (index / array.size) - eff
    assert error < 0.001
    return np.sort(array)[-index]

def _draw_hist(ax, array, bins, lims):
    counts, edges = np.histogram(array, bins, lims)
    normed = counts / float(counts.sum())
    ax.plot(edges, np.r_[normed, normed[-1]],
            drawstyle='steps-post')


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file")
    parser.add_argument('-e', '--ext', default='.pdf')
    parser.add_argument('-o', '--out-dir', default='plots')
    args = parser.parse_args()

    # load structured array
    struct_name = '{}_struct{}'.format(*splitext(args.input_file))
    if os.path.isfile(struct_name):
        array = np.load(struct_name)
    else:
        array = _get_structured(args.input_file)
        np.save(struct_name, array)

    # sort by prediction
    pred = 'track_prediction'
    array.sort(order=pred)

    for var in ['eta']:
        outfile = join(args.out_dir,
                       '{}{}'.format(var, args.ext))
        if not isdir(args.out_dir):
            os.mkdir(args.out_dir)
        with Canvas(outfile) as canvas:
            for rej in [1, 20, 40]:
                cut = _get_cut_at_rejection(array[pred], rej)
                valid = (array[pred] > cut) & (array['y_binary_label'] == 0)
                print(np.count_nonzero(valid))
                _draw_hist(canvas.ax, array[var][valid], 40, (-3, 3))


if __name__ == '__main__':
    run()
