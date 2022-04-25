# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
from functools import partial
import json
import multiprocessing
import os
import warnings

import cv2
import numpy as np

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from nicr_scene_analysis_datasets import Hypersim

from utils import DEFAULT_DATASET_PATH
from utils import DEFAULT_PREDICTIONS_PATH


def confusion_matrix_fast(pred, gt, n_classes):
    # note: this function is 15x faster than sklearn.metrics.confusion_matrix

    # determine dtype for unique mapping
    n_classes_squared = n_classes**2
    if n_classes_squared < 2**(8-1)-1:
        dtype = np.int8
    elif n_classes_squared < 2**(16-1)-1:
        dtype = np.int16
    else:
        dtype = np.int64    # equal to long

    # convert to dtype
    pred_ = pred.astype(dtype)
    gt_ = gt.astype(dtype)

    # compute confusion matrix
    unique_mapping = (gt_.reshape(-1)*n_classes + pred_.reshape(-1))
    cnts = np.bincount(unique_mapping,
                       minlength=n_classes_squared)

    return cnts.reshape(n_classes, n_classes)


def get_confusion_matrix_for_sample(
    sample_idx,
    dataset,
    prediction_basepath,
    prediction_extension='.png',
    prediction_contains_void=True,
    max_depth_in_m=20    # max 20m
):
    n_classes = dataset.semantic_n_classes    # with void

    # get sample
    sample = dataset[sample_idx]

    # load prediction
    fp = os.path.join(prediction_basepath, *sample['identifier'])
    fp += prediction_extension
    if '.png' == prediction_extension:
        # prediction is given as image
        pred = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if pred is None:
            raise IOError(f"Cannot load '{fp}'")
        if pred.ndim > 2:
            warnings.warn(f"Prediction ('{fp}') has more than one channel. "
                          "Using first channel.")
            pred = pred[..., 0]
    elif '.npy' == prediction_extension:
        # prediction is given as numpy array with shape (h, w, topk)
        pred = np.load(fp)
        pred = pred[0, ...].astype('uint8')    # use top1 only

    if not prediction_contains_void:
        pred += 1

    # create flat views
    gt = sample['semantic'].reshape(-1)
    pred = pred.reshape(-1)

    # mask using max depth
    if max_depth_in_m is not None:
        depth = sample['depth'].reshape(-1)
        mask = depth < (max_depth_in_m*1000)
        gt = gt[mask]
        pred = pred[mask]

    # move invalid pixels in prediction, i.e., pixels that may indicate free
    # space, to class with index i=n_classes
    pred[pred > (n_classes-1)] = n_classes
    n_classes = n_classes + 1    # +1 = invalid pixels

    return confusion_matrix_fast(pred, gt, n_classes=n_classes)


def get_measures(cm, ignore_void=True):
    # cm is gt x pred with void + n_classes + invalid (free space)

    tp = np.diag(cm)
    sum_gt = cm.sum(axis=1)
    sum_pred = cm.sum(axis=0)
    invalid_pixels = cm[:, -1]

    if ignore_void:
        # void is first class (idx=0)
        tp = tp[1:]
        sum_pred = sum_pred[1:]
        sum_gt = sum_gt[1:]
        sum_pred -= cm[0, 1:]    # do not count fp for void
        invalid_pixels = invalid_pixels[1:]

    n_total_pixels = sum_gt.sum()

    # we do want ignore classes without gt pixels
    gt_mask = sum_gt != 0

    # invalid pixels
    invalid_ratio = invalid_pixels.sum() / n_total_pixels
    with np.errstate(divide='ignore', invalid='ignore'):
        invalid_ratios = invalid_pixels / sum_gt
    invalid_mean_ratio_gt_masked = np.mean(invalid_ratios[gt_mask])
    valid_weights = 1 - invalid_ratios

    # intersection over union
    intersections = tp
    unions = sum_pred + sum_gt - tp

    with np.errstate(divide='ignore', invalid='ignore'):
        ious = intersections / unions.astype(np.float32)

    # mean intersection over union and gt masked version
    miou = np.mean(np.nan_to_num(ious, nan=0.0))
    miou_gt_masked = np.mean(ious[gt_mask])

    # frequency weighted intersection over union
    # normal fwiou and gt masked version are equal
    fwiou_gt_masked = np.sum(ious[gt_mask] * tp[gt_mask]/n_total_pixels)

    # pixel accuracy and mean pixel accuracy
    pacc = tp.sum() / sum_gt.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        paccs = tp / sum_gt

    mean_pacc_gt_masked = np.mean(tp[gt_mask] / sum_gt[gt_mask])

    # valid weighted mean intersection over union
    vwmiou_gt_masked = np.mean(ious[gt_mask]*valid_weights[gt_mask])

    # valid weighted mean pixel accuracy
    vwmean_pacc_gt_masked = np.mean(tp[gt_mask] / sum_gt[gt_mask] * valid_weights[gt_mask])

    # build dict of measures
    measures = {
        'cm': cm.tolist(),
        'invalid_ratio': invalid_ratio,
        'invalid_ratios': invalid_ratios.tolist(),
        'invalid_mean_ratio_gt_masked': invalid_mean_ratio_gt_masked,
        'ious': ious.tolist(),
        'miou': miou,
        'miou_gt_masked': miou_gt_masked,
        'fwiou_gt_masked': fwiou_gt_masked,
        'pacc': pacc,
        'paccs': paccs.tolist(),
        'mean_pacc_gt_masked': mean_pacc_gt_masked,
        'vwmiou_gt_masked': vwmiou_gt_masked,
        'vwmean_pacc_gt_masked': vwmean_pacc_gt_masked,
    }

    return measures


def _parse_args():
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the dataset."
    )
    parser.add_argument(
        '--dataset-split',
        type=str,
        default='test',
        help="Dataset split to use."
    )
    parser.add_argument(
        '--predictions-path',
        type=str,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to stored predicted semantic segmentation. Use an empty "
             "string to skip the evaluating the predicted semantic "
             "segmentation."
    )
    parser.add_argument(
        '--result-paths',
        nargs='+',
        type=str,
        help="Paths to further results.",
        default=[]
    )
    parser.add_argument(
        '--force-recomputing',
        action='store_true',
        default=False,
        help="Force recomputing."
    )
    parser.add_argument(
        '--n-worker',
        type=int,
        default=min(multiprocessing.cpu_count(), 48),
        help="Number of workers to use."
    )

    return parser.parse_args()


def main():
    # args
    args = _parse_args()

    # just obtain all sample names
    dataset = Hypersim(dataset_path=args.dataset_path,
                       split=args.dataset_split,
                       subsample=None,
                       sample_keys=('identifier',))
    samples = [s['identifier'] for s in dataset]    # tuple (scene, cam, id)
    scenes = sorted(list(set(s[0] for s in samples)))

    # load dataset
    dataset = Hypersim(dataset_path=args.dataset_path,
                       split=args.dataset_split,
                       subsample=None,
                       sample_keys=('identifier', 'depth', 'semantic'),
                       use_cache=False,
                       cache_disable_deepcopy=False)

    # get paths to evaluate
    paths = []
    if args.predictions_path:
        # evaluate the network prediction
        paths += [
            os.path.join(args.predictions_path, args.dataset_split,
                         Hypersim.SEMANTIC_DIR),
        ]
    paths += [
        os.path.join(path) for path in args.result_paths
    ]

    # run evaluation
    for path in tqdm(paths):
        print(f"Evaluating: '{path}'")
        results_fp = os.path.join(path, 'results.json')

        if os.path.exists(results_fp) and not args.force_recomputing:
            continue

        # get confusion matrices
        if 1 == args.n_worker:
            cms = []
            for i in tqdm(range(len(dataset))):
                cm = get_confusion_matrix_for_sample(
                    i,
                    dataset=dataset,
                    prediction_basepath=path,
                    prediction_extension='.png',
                    prediction_contains_void=True,
                    max_depth_in_m=20
                )
                cms.append(cm)
        else:
            f = partial(get_confusion_matrix_for_sample,
                        dataset=dataset,
                        prediction_basepath=path,
                        prediction_extension='.png',
                        prediction_contains_void=True,
                        max_depth_in_m=20)
            cms = thread_map(f, list(range(len(dataset))),
                             max_workers=args.n_worker,
                             chunksize=10,
                             leave=False)

        # get overall measures
        assert len(cms) == len(samples)
        cm = np.array(cms).sum(axis=0)

        measures = get_measures(cm, ignore_void=True)
        for k in ('miou_gt_masked', 'mean_pacc_gt_masked',
                  'invalid_ratio', 'invalid_mean_ratio_gt_masked',
                  'vwmiou_gt_masked', 'vwmean_pacc_gt_masked'):
            print(f"{k}: {measures[k]}")

        # get results for each scene
        cms_per_scene = {s: [] for s in scenes}
        for cm, sample in zip(cms, samples):
            scene = sample[0]
            cms_per_scene[scene].append(cm)

        measures['per_scene'] = {}
        for scene, cms_scene in cms_per_scene.items():
            cm = np.array(cms_scene).sum(axis=0)
            measures['per_scene'][scene] = get_measures(cm, ignore_void=True)

        # write results to file
        with open(results_fp, 'w') as f:
            json.dump(measures, f, indent=4)


if __name__ == '__main__':
    main()
