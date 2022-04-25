# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
import os

import cv2
from nicr_scene_analysis_datasets import Hypersim
from nicr_scene_analysis_datasets.utils.img import save_indexed_png
import numpy as np
import onnx
import onnxruntime as ort
from tqdm import tqdm

from utils import DEFAULT_DATASET_PATH
from utils import DEFAULT_ONNX_FILEPATH
from utils import DEFAULT_PREDICTIONS_PATH


def _get_ort_session(onnx_filepath, img_hw, topk=3):
    model = onnx.load(onnx_filepath)

    # get network output shape (same as input shape)
    # note: our optimizations to the resize operations seem to break onnx's
    # shape inference with OpSet >= 13
    model_output_img_shape = (
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_value,
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    )

    # add missing nodes: final upsampling, softmax, and topk
    # see: https://github.com/onnx/onnx/blob/main/docs/Operators.md
    # -> final upsampling
    final_upsampling_node = onnx.helper.make_node(
        'Resize',
        # inputs=['output', 'roi', 'scales'],
        inputs=['output', '', 'scales'],    # '' for 'roi' requires OpSet >= 13
        outputs=['final_upsampling_output'],
        coordinate_transformation_mode='pytorch_half_pixel',
        cubic_coeff_a=-0.75,
        mode='linear',
        nearest_mode='floor',
    )
    # roi = onnx.helper.make_tensor('roi', onnx.TensorProto.FLOAT, [0], [])
    scale_h = img_hw[0] / model_output_img_shape[0]
    scale_w = img_hw[1] / model_output_img_shape[1]
    scales = onnx.helper.make_tensor('scales',
                                     onnx.TensorProto.FLOAT, [4],
                                     [1, 1, scale_h, scale_w])
    # -> softmax (note that softmax op with 4D inputs requires OpSet >= 13)
    softmax_node = onnx.helper.make_node(
        'Softmax',
        inputs=['final_upsampling_output'],
        outputs=['prediction'],
        axis=1
    )
    # topk
    topk_node = onnx.helper.make_node(
        'TopK',
        inputs=['prediction', 'k'],
        outputs=['scores', 'classes'],
        axis=1,
        largest=1,
        sorted=1
    )
    k = onnx.helper.make_tensor('k', onnx.TensorProto.INT64, [1], [int(topk)])

    # add new nodes and initializers to graph
    # model.graph.initializer.append(roi)
    model.graph.initializer.append(scales)
    model.graph.node.append(final_upsampling_node)
    model.graph.node.append(softmax_node)
    model.graph.initializer.append(k)
    model.graph.node.append(topk_node)

    # replace output information
    if model.graph.input[0].type.tensor_type.shape.dim[0].dim_param:
        # dynamic batch axis
        b = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    else:
        # fixed batch axis
        b = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value

    scores_info = onnx.helper.make_tensor_value_info('scores',
                                                     onnx.TensorProto.FLOAT,
                                                     shape=[b, topk, *img_hw])
    classes_info = onnx.helper.make_tensor_value_info('classes',
                                                      onnx.TensorProto.INT64,
                                                      shape=[b, topk, *img_hw])
    model.graph.output.pop(0)
    model.graph.output.append(scores_info)
    model.graph.output.append(classes_info)

    # perform final check
    onnx.checker.check_model(model)
    # onnx.save(model, './model.onnx')

    # create onnxruntime seesion
    ort_session = ort.InferenceSession(
        model.SerializeToString(),
        providers=[
            # 'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
    )
    return ort_session


def _parse_args():
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--onnx-filepath',
        type=str,
        default=DEFAULT_ONNX_FILEPATH,
        help="Path to ONNX model to use."
    )
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
        '--output-path',
        type=str,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path where to store predicted semantic segmentation."
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=3,
        help="TopK classes to consider."
    )
    return parser.parse_args()


def main():
    # args
    args = _parse_args()

    # load data
    dataset = Hypersim(
        dataset_path=args.dataset_path,
        split=args.dataset_split,
        subsample=None,
        sample_keys=('identifier', 'rgb', 'depth'),
        depth_mode='raw'
    )

    RGB_MEAN = np.array((0.485, 0.456, 0.406), dtype='float32') * 255
    RGB_STD = np.array((0.229, 0.224, 0.225), dtype='float32') * 255

    # ensure that the used depth stats are valid for this model (there was a
    # copy and paste issue that we fixed in future versions of
    # nicr_scene_analysis_datasets)
    assert dataset.depth_mean == 6249.621001070915
    assert dataset.depth_std == 6249.621001070915     # <- c&p ^^

    # process files (for simplification with batch size 1)
    ort_session = None
    for sample in tqdm(dataset, desc='Processing files'):
        # load model lazily (we need a sample to get the spatial dimensions)
        if ort_session is None:
            ort_session = _get_ort_session(
                onnx_filepath=args.onnx_filepath,
                img_hw=sample['rgb'].shape[:2],
                topk=args.topk
            )

        # get network input shape (from rgb input)
        h, w = ort_session.get_inputs()[0].shape[-2:]

        # rgb preprocessing
        # -> resize
        rgb = cv2.resize(sample['rgb'], (w, h),
                         interpolation=cv2.INTER_LINEAR)
        # -> normalize
        rgb = rgb.astype('float32')
        rgb -= RGB_MEAN[None, None, ...]
        rgb /= RGB_STD[None, None, ...]
        # -> create tensor (add batch axis, channels first)
        rgb = rgb.transpose(2, 0, 1)[None, ...]

        # depth preprocessing
        # -> resize
        depth = cv2.resize(sample['depth'], (w, h),
                           interpolation=cv2.INTER_NEAREST)
        # -> normalize
        mask_invalid = depth == 0    # mask for invalid depth values
        depth = depth.astype('float32')
        depth -= dataset.depth_mean
        depth /= dataset.depth_std
        # reset invalid values (the network should not be able to learn from
        # these pixels)
        depth[mask_invalid] = 0
        # -> create tensor (add batch and channel axes)
        depth = depth[None, None, ...]

        # apply model
        scores, classes = ort_session.run(None, {'rgb': rgb, 'depth': depth})

        # remove batch axis
        scores = scores[0]
        classes = classes[0]

        # cast classes to uint8 (< 255 classes)
        classes = classes.astype('uint8')

        # create predicted segmentation
        # note that we store the topk predictions as class_idx + score (to
        # save some space), you may further can think about using float16
        scores_clamped = np.clip(scores, a_min=0, a_max=0.9999)
        classes = classes + 1    # add void class (void + 40 classes)
        segmentation = scores_clamped + classes

        # ensure that class is still correct (top0 only)
        assert (segmentation[0].astype('uint8') == classes[0]).all()

        # store predicted segmentation
        # -> topk prediction (for mapping later)
        fp = os.path.join(args.output_path, args.dataset_split,
                          f'{Hypersim.SEMANTIC_DIR}_topk',
                          *sample['identifier'])
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        np.save(f'{fp}.npy', segmentation)

        # -> predicted classes
        for i in range(args.topk):
            dirname = Hypersim.SEMANTIC_DIR
            if i > 0:
                dirname += f'_topk_{i}'
            fp = os.path.join(args.output_path, args.dataset_split,
                              dirname, *sample['identifier'])
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            cv2.imwrite(f'{fp}.png', segmentation[i].astype('uint8'))

        # -> predicted classes as colored images (with color palette, do not
        # load these images later on with OpenCV, PIL is fine)
        for i in range(args.topk):
            dirname = Hypersim.SEMANTIC_COLORED_DIR
            if i > 0:
                dirname += f'_topk_{i}'
            fp = os.path.join(args.output_path, args.dataset_split,
                              dirname, *sample['identifier'])
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            save_indexed_png(f'{fp}.png', segmentation[i].astype('uint8'),
                             colormap=dataset.semantic_class_colors)


if __name__ == '__main__':
    main()
