from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import os.path
import sys
import numpy as np
import time
from PIL import Image
import cv2
from collections import namedtuple

image_size = (64, 64)
Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint(sys.argv[1], int(sys.argv[2]))
#mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)

mod.bind(for_training=False, data_shapes=[('data', (1,3,64,64))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

paths = sys.argv[3:]
t = time.time()
for path in paths:
    image = Image.open(path)
    image = np.array(image)
    image = cv2.resize(image, image_size)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)

    mod.forward(Batch([mx.nd.array([image])]), is_train=False)
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    print(path, end=' ')
    #print(prob)
    print("mask:"+str(prob[0])+"-----"+"nomask:"+str(prob[1]))

duration = time.time() - t
print('Time: %.3f' % duration)
