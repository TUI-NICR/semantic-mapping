# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os


def _get_default_path(*path_components):
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, *path_components)


DEFAULT_ONNX_FILEPATH = _get_default_path('trained_models',
                                          'model_hypersim.onnx')

DEFAULT_DATASET_PATH = _get_default_path('datasets', 'hypersim')

DEFAULT_PREDICTIONS_PATH = _get_default_path('datasets', 'hypersim_predictions')
