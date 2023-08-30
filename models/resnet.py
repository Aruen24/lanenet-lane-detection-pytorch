from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    net, end_points = nets.resnet_v1.resnet_v1_50(images, is_training=phase_train, reuse=reuse)
    net = slim.flatten(net)
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None)

    return net, end_points
