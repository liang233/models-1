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
"""Provides flags that are common to scripts.
	simplify names; bugfix for single-GPU training ？？？  编辑之前 主目录里看到的对代码的描述
  
  Common flags from train/eval/vis/export_model.py are collected in this script.
"""
import collections

import tensorflow as tf

flags = tf.app.flags
# Flags for input preprocessing.
#tf定义了tf.app.flags，用于支持接受命令行传递参数 第一个是参数名称，第二个参数是默认值，第三个是参数描述 如果有四个参数第三个是可选参数的列表，第四个是参数描述
#tf.app.flags.DEFINE_xxx()就是添加命令行的optional argument（可选参数），而tf.app.flags.FLAGS可以从对应的命令行参数取出参数用于打印什么的
#大体用法： python common.py --logits_kernel_size 7
#然后对于'xception_65', 'mobilenet_v2'各种参数的设定就比较方便了
flags.DEFINE_integer('min_resize_value', None,
                     'Desired size of the smaller image side.')

flags.DEFINE_integer('max_resize_value', None,
                     'Maximum allowed size of the larger image side.')

flags.DEFINE_integer('resize_factor', None,
                     'Resized dimensions are multiple of factor plus one.')

# Model dependent flags.

flags.DEFINE_integer('logits_kernel_size', 1,
                     'The kernel size for the convolutional kernel that '
                     'generates logits.')
#logits 评定模型？分类评定模型？softmax的输入？

# When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
# When using 'xception_65', we set atrous_rates = [6, 12, 18] (output stride 16)
# and decoder_output_stride = 4.
flags.DEFINE_enum('model_variant', 'mobilenet_v2',
                  ['xception_65', 'mobilenet_v2'], 'DeepLab model variant.')

flags.DEFINE_multi_float('image_pyramid', None,
                         'Input scales for multi-scale feature extraction.')

flags.DEFINE_boolean('add_image_level_feature', True,
                     'Add image level feature.')

flags.DEFINE_boolean('aspp_with_batch_norm', True,
                     'Use batch norm parameters for ASPP or not.')

flags.DEFINE_boolean('aspp_with_separable_conv', True,
                     'Use separable convolution for ASPP or not.')

flags.DEFINE_multi_integer('multi_grid', None,
                           'Employ a hierarchy of atrous rates for ResNet.')

flags.DEFINE_float('depth_multiplier', 1.0,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

# For `xception_65`, use decoder_output_stride = 4. For `mobilenet_v2`, use
# decoder_output_stride = None.
flags.DEFINE_integer('decoder_output_stride', None,
                     'The ratio of input to output spatial resolution when '
                     'employing decoder to refine segmentation results.')

flags.DEFINE_boolean('decoder_use_separable_conv', True,
                     'Employ separable convolution for decoder or not.')

flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'],
                  'Scheme to merge multi scale features.')

FLAGS = flags.FLAGS

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'

class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'merge_method',
        'add_image_level_feature',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant'
    ])):
  """Immutable 不可改变的 class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              outputs_to_num_classes,
              crop_size=None,
              atrous_rates=None,
              output_stride=8):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.

    Returns:
      A new ModelOptions instance.
    """
#所以在你给出的代码中，用super(Singleton, cls).__new__(cls, *args, **kwargs) 实现对object的__new__方法的调用。
#这是super最常见的用法， 因为子类中定义__new__ 或 __init__等方法会覆盖父类中的同名方法。
#为了获得父类中同名方法的功能，需要这样显示调用父类中的同名方法。
    return super(ModelOptions, cls).__new__(
        cls, outputs_to_num_classes, crop_size, atrous_rates, output_stride,
        FLAGS.merge_method, FLAGS.add_image_level_feature,
        FLAGS.aspp_with_batch_norm, FLAGS.aspp_with_separable_conv,
        FLAGS.multi_grid, FLAGS.decoder_output_stride,
        FLAGS.decoder_use_separable_conv, FLAGS.logits_kernel_size,
        FLAGS.model_variant)
