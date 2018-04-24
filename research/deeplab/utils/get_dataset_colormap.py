# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Visualizes the segmentation results via specified color map.

Visualizes the semantic segmentation results by the color map
defined by the different datasets. Supported colormaps are:

1. PASCAL VOC semantic segmentation benchmark（数据集）.
Website: http://host.robots.ox.ac.uk/pascal/VOC/
没写全啊。。。
"""

import numpy as np

# Dataset names.
_CITYSCAPES = 'cityscapes'
_PASCAL = 'pascal'

# Max number of entries（记录） in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {#字典
    _CITYSCAPES: 19,
    _PASCAL: 256,
}


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    不同类别对应的不同颜色 CITYSCAPES共有19个类别
  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.asarray([
      [128, 64, 128],
      [244, 35, 232],
      [70, 70, 70],
      [102, 102, 156],
      [190, 153, 153],
      [153, 153, 153],
      [250, 170, 30],
      [220, 220, 0],
      [107, 142, 35],
      [152, 251, 152],
      [70, 130, 180],
      [220, 20, 60],
      [255, 0, 0],
      [0, 0, 142],
      [0, 0, 70],
      [0, 60, 100],
      [0, 80, 100],
      [0, 0, 230],
      [119, 11, 32],
  ])
  return colormap


def get_pascal_name():
  return _PASCAL


def get_cityscapes_name():
  return _CITYSCAPES


def bit_get(val, idx):
  """Gets the bit value.
    返回val的第idx位是0还是1
  Args:
    val: Input value, int or numpy int array.
    idx: Which bit of the input val.

  Returns:
    The "idx"-th bit of input val.
  """
  return (val >> idx) & 1


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((_DATASET_MAX_ENTRIES[_PASCAL], 3), dtype=int)#初始化256行3列全为0
  ind = np.arange(_DATASET_MAX_ENTRIES[_PASCAL], dtype=int)#产生一个0-255的数组

  for shift in reversed(range(8)):#[7, 6, 5, 4, 3, 2, 1, 0]
    for channel in range(3):
      colormap[:, channel] |= bit_get(ind, channel) << shift
    ind >>= 3

  return colormap


def create_label_colormap(dataset=_PASCAL):
  """Creates a label colormap for the specified dataset.

  Args:
    dataset: The colormap used in the dataset.

  Returns:
    A numpy array of the dataset colormap.

  Raises:
    ValueError: If the dataset is not supported.
  """
  if dataset == _PASCAL:
    return create_pascal_label_colormap()
  elif dataset == _CITYSCAPES:
    return create_cityscapes_label_colormap()
  else:
    raise ValueError('Unsupported dataset.')


def label_to_color_image(label, dataset=_PASCAL):
  """Adds color defined by the dataset colormap to the label.
    返回某一种类别对应的colormap值 参数和返回值样例：
    label = np.array([[0, 16, 16], [52, 7, 52]])
    expected_result = np.array([
        [[0, 0, 0], [0, 64, 0], [0, 64, 0]],
        [[0, 64, 192], [128, 128, 128], [0, 64, 192]]
    ])
  Args:
    label: A 2D array with integer type, storing the segmentation label.
    dataset: The colormap used in the dataset.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= _DATASET_MAX_ENTRIES[dataset]:#判序列最大值是否越界
    raise ValueError('label value too large.')

  colormap = create_label_colormap(dataset)
  return colormap[label]

r"""
pascal_label_colormap:
[[  0   0   0]
 [128   0   0]
 [  0 128   0]
 [128 128   0]
 [  0   0 128]
 [128   0 128]
 [  0 128 128]
 [128 128 128]
 [ 64   0   0]
 [192   0   0]
 [ 64 128   0]
 [192 128   0]
 [ 64   0 128]
 [192   0 128]
 [ 64 128 128]
 [192 128 128]
 [  0  64   0]
 [128  64   0]
 [  0 192   0]
 [128 192   0]
 [  0  64 128]
 [128  64 128]
 [  0 192 128]
 [128 192 128]
 [ 64  64   0]
 [192  64   0]
 [ 64 192   0]
 [192 192   0]
 [ 64  64 128]
 [192  64 128]
 [ 64 192 128]
 [192 192 128]
 [  0   0  64]
 [128   0  64]
 [  0 128  64]
 [128 128  64]
 [  0   0 192]
 [128   0 192]
 [  0 128 192]
 [128 128 192]
 [ 64   0  64]
 [192   0  64]
 [ 64 128  64]
 [192 128  64]
 [ 64   0 192]
 [192   0 192]
 [ 64 128 192]
 [192 128 192]
 [  0  64  64]
 [128  64  64]
 [  0 192  64]
 [128 192  64]
 [  0  64 192]
 [128  64 192]
 [  0 192 192]
 [128 192 192]
 [ 64  64  64]
 [192  64  64]
 [ 64 192  64]
 [192 192  64]
 [ 64  64 192]
 [192  64 192]
 [ 64 192 192]
 [192 192 192]
 [ 32   0   0]
 [160   0   0]
 [ 32 128   0]
 [160 128   0]
 [ 32   0 128]
 [160   0 128]
 [ 32 128 128]
 [160 128 128]
 [ 96   0   0]
 [224   0   0]
 [ 96 128   0]
 [224 128   0]
 [ 96   0 128]
 [224   0 128]
 [ 96 128 128]
 [224 128 128]
 [ 32  64   0]
 [160  64   0]
 [ 32 192   0]
 [160 192   0]
 [ 32  64 128]
 [160  64 128]
 [ 32 192 128]
 [160 192 128]
 [ 96  64   0]
 [224  64   0]
 [ 96 192   0]
 [224 192   0]
 [ 96  64 128]
 [224  64 128]
 [ 96 192 128]
 [224 192 128]
 [ 32   0  64]
 [160   0  64]
 [ 32 128  64]
 [160 128  64]
 [ 32   0 192]
 [160   0 192]
 [ 32 128 192]
 [160 128 192]
 [ 96   0  64]
 [224   0  64]
 [ 96 128  64]
 [224 128  64]
 [ 96   0 192]
 [224   0 192]
 [ 96 128 192]
 [224 128 192]
 [ 32  64  64]
 [160  64  64]
 [ 32 192  64]
 [160 192  64]
 [ 32  64 192]
 [160  64 192]
 [ 32 192 192]
 [160 192 192]
 [ 96  64  64]
 [224  64  64]
 [ 96 192  64]
 [224 192  64]
 [ 96  64 192]
 [224  64 192]
 [ 96 192 192]
 [224 192 192]
 [  0  32   0]
 [128  32   0]
 [  0 160   0]
 [128 160   0]
 [  0  32 128]
 [128  32 128]
 [  0 160 128]
 [128 160 128]
 [ 64  32   0]
 [192  32   0]
 [ 64 160   0]
 [192 160   0]
 [ 64  32 128]
 [192  32 128]
 [ 64 160 128]
 [192 160 128]
 [  0  96   0]
 [128  96   0]
 [  0 224   0]
 [128 224   0]
 [  0  96 128]
 [128  96 128]
 [  0 224 128]
 [128 224 128]
 [ 64  96   0]
 [192  96   0]
 [ 64 224   0]
 [192 224   0]
 [ 64  96 128]
 [192  96 128]
 [ 64 224 128]
 [192 224 128]
 [  0  32  64]
 [128  32  64]
 [  0 160  64]
 [128 160  64]
 [  0  32 192]
 [128  32 192]
 [  0 160 192]
 [128 160 192]
 [ 64  32  64]
 [192  32  64]
 [ 64 160  64]
 [192 160  64]
 [ 64  32 192]
 [192  32 192]
 [ 64 160 192]
 [192 160 192]
 [  0  96  64]
 [128  96  64]
 [  0 224  64]
 [128 224  64]
 [  0  96 192]
 [128  96 192]
 [  0 224 192]
 [128 224 192]
 [ 64  96  64]
 [192  96  64]
 [ 64 224  64]
 [192 224  64]
 [ 64  96 192]
 [192  96 192]
 [ 64 224 192]
 [192 224 192]
 [ 32  32   0]
 [160  32   0]
 [ 32 160   0]
 [160 160   0]
 [ 32  32 128]
 [160  32 128]
 [ 32 160 128]
 [160 160 128]
 [ 96  32   0]
 [224  32   0]
 [ 96 160   0]
 [224 160   0]
 [ 96  32 128]
 [224  32 128]
 [ 96 160 128]
 [224 160 128]
 [ 32  96   0]
 [160  96   0]
 [ 32 224   0]
 [160 224   0]
 [ 32  96 128]
 [160  96 128]
 [ 32 224 128]
 [160 224 128]
 [ 96  96   0]
 [224  96   0]
 [ 96 224   0]
 [224 224   0]
 [ 96  96 128]
 [224  96 128]
 [ 96 224 128]
 [224 224 128]
 [ 32  32  64]
 [160  32  64]
 [ 32 160  64]
 [160 160  64]
 [ 32  32 192]
 [160  32 192]
 [ 32 160 192]
 [160 160 192]
 [ 96  32  64]
 [224  32  64]
 [ 96 160  64]
 [224 160  64]
 [ 96  32 192]
 [224  32 192]
 [ 96 160 192]
 [224 160 192]
 [ 32  96  64]
 [160  96  64]
 [ 32 224  64]
 [160 224  64]
 [ 32  96 192]
 [160  96 192]
 [ 32 224 192]
 [160 224 192]
 [ 96  96  64]
 [224  96  64]
 [ 96 224  64]
 [224 224  64]
 [ 96  96 192]
 [224  96 192]
 [ 96 224 192]
 [224 224 192]]
"""
