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

#paths = sys.argv[3:]
paths = sys.argv[3]
t = time.time()
i = 0
j = 0
with open('./result.txt','w') as write_file:
    if os.path.isfile(paths):
        t1 = time.time()
        image = Image.open(paths)
        image = np.array(image)
        image = cv2.resize(image, image_size)
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)

        mod.forward(Batch([mx.nd.array([image])]), is_train=False)
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        duration = round(time.time() - t1,3)
        if prob[0] > prob[1]:
            write_file.write(paths+','+'mask'+','+str(duration)+'\n')
            i += 1
        else:
            write_file.write(paths+','+'nomask'+','+str(duration)+'\n')
            j += 1
    elif os.path.isdir(paths):
        for im in os.listdir(paths):
            t2 = time.time()
            path = os.path.join(paths, im)
            image = Image.open(path)
            image = np.array(image)
            image = cv2.resize(image, image_size)
            image = np.swapaxes(image, 0, 2)
            image = np.swapaxes(image, 1, 2)

            mod.forward(Batch([mx.nd.array([image])]), is_train=False)
            prob = mod.get_outputs()[0].asnumpy()
            prob = np.squeeze(prob)
            duration = round(time.time() - t2,3)
            if prob[0] > prob[1]:
                write_file.write(path+','+'mask'+','+str(duration)+','+str(prob[0])+'\n')
                i += 1
            else:
            #    write_file.write(path+','+'nomask'+','+str(duration)+'\n')
                j += 1
            #print(path, end=' ')
            #print(prob)
            #print("mask:"+str(prob[0])+"-----"+"nomask:"+str(prob[1]))

#duration = time.time() - t
#print('Time: %.3f' % duration)
print("mask_count:"+str(i)+"-----"+"nomask_count:"+str(j))
