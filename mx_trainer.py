from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import numpy as np
from PIL import Image
import cv2

import mxnet as mx
import argparse
import mxnet.optimizer as optimizer

from models.mobilenet import fmobilenetv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'mobilenet'))
from image_iter import FaceImageIter

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None

class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count+=1
        label = labels[0]
        pred_label = preds[0]
        
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
            label = label[:,0]
        label = label.astype('int32').flatten()
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        gt_label = preds[-2].asnumpy()
        #print(gt_label)

def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--train_data_dir', default='/home/wangyuanwen/data/mxnet_train_recdata_64/mx_train', help='training set directory')
    parser.add_argument('--val_data_dir', default='/home/wangyuanwen/data/mxnet_train_recdata_64/mx_val', help='validate set directory')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--prefix', default='./data/train_output/mxnet_ckpt/mn-v2_0.25', help='directory to save model.')
    parser.add_argument('--pretrained', default='', help='pretrained model to load')
    parser.add_argument('--ckpt', type=int, default=2, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--max_steps', type=int, default=320000, help='max training batches')
    parser.add_argument('--end_epoch', type=int, default=320000, help='training epoch size.')
    parser.add_argument('--image_size', type=int, default=64, help='specify input image height and width')
    parser.add_argument('--lr', type=float, default=0.003, help='start learning rate')
    parser.add_argument('--lr_steps', type=str, default='200000,300000', help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=0.0004, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--emb_size', type=int, default=512, help='embedding length')
    parser.add_argument('--per_batch_size', type=int, default=64, help='batch size in each context')
    parser.add_argument('--ce_loss', default=False, action='store_true', help='if output ce loss')
    parser.add_argument('--ce_loss_factor', type=float, default=0.3)
    args = parser.parse_args()
    return args

def get_symbol(args, arg_params, aux_params):
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label

    embedding = fmobilenetv2.get_symbol(args.emb_size, width_mult=0.25)
    _weight = mx.symbol.Variable("output_weight", shape=(args.num_classes, args.emb_size))
    _bias = mx.symbol.Variable('output_bias')
    output = mx.sym.FullyConnected(data=embedding, weight=_weight, bias=_bias, num_hidden=args.num_classes, name='output')
    softmax = mx.symbol.SoftmaxOutput(data=output, label=gt_label, name='softmax', normalization='valid')

    out_list = [softmax]
    if args.ce_loss:
        body = mx.symbol.SoftmaxActivation(data=output)
        body = mx.symbol.log(body)
        _label = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=-1.0, off_value=0.0)
        body = body * _label
        ce_loss = mx.symbol.sum(body) / args.per_batch_size
        ce_loss = mx.sym.MakeLoss(ce_loss * args.ce_loss_factor)
        out_list.append(mx.symbol.BlockGrad(ce_loss))

    out = mx.symbol.Group(out_list)
    return (out, arg_params, aux_params)

def get_data(path, image_size):
    images = []
    labels = []
    for id in os.listdir(path):
        id_path = os.path.join(path, id)
        for img in os.listdir(id_path):
            img_path = os.path.join(id_path, img)
            image = Image.open(img_path)
            image = np.array(image)
            image = cv2.resize(image, (image_size, image_size))
            images.append(image)
            if id == 'real':
                labels.append(1)
            else:
                labels.append(0)

    return np.array(images), np.array(labels)

def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd) > 0:
        for i in range(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))

    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.batch_size = args.per_batch_size * args.ctx_num
    args.image_channel = 3
    train_path_imgrec = os.path.join(args.train_data_dir, "data_train.rec")
    val_path_imgrec = os.path.join(args.val_data_dir, "data_val.rec")

    print('image_size', args.image_size)
    assert (args.num_classes > 0)
    print('num_classes', args.num_classes)

    print('Called with argument:', args)
    data_shape = (args.image_channel, args.image_size, args.image_size)

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained) == 0:
        arg_params = None
        aux_params = None
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
        vec = args.pretrained.split(',')
        print('loading', vec)
        _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    model = mx.mod.Module(
        context=ctx,
        symbol=sym,
    )

    train_dataiter = mx.io.ImageRecordIter(
        path_imgrec=train_path_imgrec,
        data_name='data',
        label_name='softmax_label',
        batch_size=args.batch_size,
        data_shape=data_shape,
        shuffle=True
    )

    val_dataiter = mx.io.ImageRecordIter(
        path_imgrec=val_path_imgrec,
        data_name='data',
        label_name='softmax_label',
        batch_size=args.batch_size,
        data_shape=data_shape,
        shuffle=True
    )

    metric1 = AccMetric()
    eval_metrics = [mx.metric.create(metric1)]
    if args.ce_loss:
        metric2 = LossValueMetric()
        eval_metrics.append(mx.metric.create(metric2))

    initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0 / args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 50
    _cb = mx.callback.Speedometer(args.batch_size, som)

    lr_steps = [int(x) for x in args.lr_steps.split(',')]

    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    val_dataiter = mx.io.PrefetchingIter(val_dataiter)

    global_step = [0]
    global_epoch = [0]

    def _epoch_callback(param):
        global_epoch[0] += 1
        msave = global_epoch[0]
        print('saving')
        arg, aux = model.get_params()
        mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)

    def _batch_callback(param):
        #global global_step
        global_step[0]+=1
        mbatch = global_step[0]
        for _lr in lr_steps:
            if mbatch==_lr:
                opt.lr *= 0.1
                print('lr change to', opt.lr)
                break

        _cb(param)
        if mbatch%500==0:
            print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

        if args.max_steps>0 and mbatch>args.max_steps:
            sys.exit(0)

    model.fit(train_dataiter,
              begin_epoch=begin_epoch,
              num_epoch=end_epoch,
              eval_data=val_dataiter,
              eval_metric=eval_metrics,
              kvstore='device',
              optimizer=opt,
              # optimizer_params   = optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=_batch_callback,
              epoch_end_callback=mx.callback.do_checkpoint(prefix))

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()
