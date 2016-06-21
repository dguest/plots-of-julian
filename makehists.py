#!/usr/bin/env python3

"""
Read in Julian's files and make plotting files
"""

import numpy as np
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

_ranges = {
    'eta': (-3, 3),
    'pt': (0, 300),
    'jet_prob': (0, 0.5),
    'y_binary_label': (-0.5, 1.5),
    'n_secondary_tracks': (1.5, 14.5),
    'n_secondary': (-0.5, 7.5),
    'n_d0sig': (-0.5, 10.5),
}
_log_vars = {
    'pt', 'jet_prob', 'n_secondary', 'n_secondary_tracks', 'n_d0sig'
}
def _count_bins(name):
    return _ranges[name][1] - _ranges[name][0]
_n_bins = {
    'y_binary_label': 2,
    'n_secondary_tracks': _count_bins('n_secondary_tracks'),
    'n_secondary': _count_bins('n_secondary'),
    'n_d0sig': _count_bins('n_d0sig'),
}

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

def _draw_hist(ax, array, bins, lims, label=''):
    counts, edges = np.histogram(array, bins, lims)
    normed = counts / float(counts.sum())
    opts = {}
    if label:
        opts['label'] = label
    ax.plot(edges, np.r_[normed, normed[-1]],
            drawstyle='steps-post', **opts)


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
    pred_name = 'track_prediction'
    array.sort(order=pred_name)

    sig_index = array['y_binary_label'] > 0.0
    bg_index = ~sig_index
    with Canvas('{}/dist.pdf'.format(args.out_dir)) as dists:
        opts = dict(bins = 100, lims = (0, 1))
        pred_arr = array[pred_name]
        _draw_hist(dists.ax, pred_arr[sig_index], label='sig', **opts)
        _draw_hist(dists.ax, pred_arr[bg_index], label='bg', **opts)
        dists.ax.set_yscale('log')
        dists.ax.legend(framealpha=0)


    for var in ['eta', 'pt', 'jet_prob', 'y_binary_label',
                'n_secondary_tracks', 'n_secondary', 'n_d0sig']:
        outfile = join(args.out_dir,
                       '{}{}'.format(var, args.ext))
        if not isdir(args.out_dir):
            os.mkdir(args.out_dir)
        with Canvas(outfile) as canvas:
            for rej in [1, 20, 50]:
                cut = _get_cut_at_rejection(array[pred_name][bg_index], rej)
                valid = (array[pred_name] > cut) & bg_index
                label = 'rej-{}'.format(rej)
                hrange = _ranges.get(var)
                if not hrange:
                    hrange = array[var].min(), array[var].max()
                nbins = _n_bins.get(var, 40)
                _draw_hist(canvas.ax, array[var][valid], nbins, hrange,
                           label=label)
            _draw_hist(canvas.ax, array[var][sig_index], nbins, hrange,
                       label='sig')
            canvas.ax.legend(framealpha=0)
            canvas.ax.set_xlabel(var)
            if var in _log_vars:
                canvas.ax.set_yscale('log')



if __name__ == '__main__':
    run()
