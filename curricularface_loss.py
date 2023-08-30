# elif args.loss_type == 9:  # CurricularFace
#     s = args.margin_s  # 参数s， 64
#     m = args.margin_m  # 参数m， 0.5
#     # t = mx.sym.Variable("t",shape=(args.per_batch_size, args.num_classes), init=mx.init.Zero())
#     t = mx.sym.Variable("t", shape=(1), init=mx.init.Zero())
#     assert s > 0.0
#     assert m >= 0.0
#     assert m < (math.pi / 2)
#     # 权重归一化
#     _weight = mx.symbol.L2Normalization(_weight, mode='instance')  # shape = [(类别数目, 512)]
#     # 特征归一化，并放大到 s*x
#     nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
#     fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
#                                 name='fc7')  # args.num_classes:85164   fc7输出的就是角度s*cos(θj)
#
#     zy = mx.sym.pick(fc7, gt_label, axis=1)  # fc7每一行找出gt_label对应的值, 即s*cos(θyi)
#
#     cos_t = zy / s  # 网络输出output = s*x/|x|*w/|w|*cos(θ), 这里将输出除以s，得到实际的cos值，即cos_t=cos（θyi)
#     cos_m = math.cos(m)
#     sin_m = math.sin(m)
#     mm = math.sin(math.pi - m) * m  # sin(pi-m)*m = sin(m) * m  0.2397
#     # threshold = 0.0
#     threshold = math.cos(math.pi - m)  # 这个阈值避免θ+m >= pi, 实际上threshold<0 -cos(m)    -0.8775825618903726
#     if args.easy_margin:  # 将0作为阈值，得到超过阈值的索引
#         cond = mx.symbol.Activation(data=cos_t, act_type='relu')
#     else:
#         cond_v = cos_t - threshold  # 将负数作为阈值
#         cond = mx.symbol.Activation(data=cond_v, act_type='relu')
#     body = cos_t * cos_t  # cos_t^2 + sin_t^2 = 1
#     body = 1.0 - body
#     sin_t = mx.sym.sqrt(body)  # sin_t = sin（θyi)
#     new_zy = cos_t * cos_m  # cos(t+m) = cos(t)cos(m) - sin(t)sin(m)
#     b = sin_t * sin_m
#     new_zy = new_zy - b  # cos（θyi+m)=cos（θyi)cos(m)-sin(θyi)sin(m)
#     new_zy = new_zy * s  # s*cos(θyi + m)
#     if args.easy_margin:
#         zy_keep = zy  # zy_keep为zy，即s*cos(θyi)
#     else:
#         zy_keep = zy - s * mm  # zy-s*sin(m)*m = s*cos(θyi)- s*m*sin(m)
#     new_zy = mx.sym.where(cond, new_zy,
#                           zy_keep)  # cond中>0的保持new_zy=s*cos(θyi+m)不变，<0的裁剪为zy_keep= s*cos(θyi) or s*cos(θyi)-s*m*sin(m)
#
#     diff = new_zy - zy  # s*cos(θyi+m)-s*cos(θyi)
#     diff = mx.sym.expand_dims(diff, 1)
#     gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
#     body = mx.sym.broadcast_mul(gt_one_hot, diff)  # 对应yi处为new_zy - zy
#     # t = mx.sym.broadcast_add(mx.sym.mean(mx.sym.stop_gradient（cos_t）) * 0.01, (1 - 0.01) * t)
#     t = mx.sym.mean(mx.sym.stop_gradient(cos_t)) * 0.01 + (1 - 0.01) * t
#     fc7 = mx.sym.where(zy >= new_zy, fc7 * (t + fc7 / s), fc7)
#     fc7 = fc7 + body  # 对应yi处，fc7=zy + (new_zy - zy) = new_zy，即cond中>0的为s*cos(θj)+s*cos(θyi+m)-s*cos(θyi)，<0的裁剪为s*cos(θj)+s*cos(θyi) or s*cos(θj)+s*cos(θyi)-s*m*sin(m)
#     # fc7 = s*[cos(θj)+cos(θyi+m)-cos(θyi)]
#
