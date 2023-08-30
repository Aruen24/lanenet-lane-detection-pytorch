# mask_detect
## 1、模型训练
```shell
执行./mx_trainer.sh
训练数据图片大小64*64
训练数据所在路径:/home/wangyuanwen/data/mxnet_train_recdata_64/mx_train  验证数据所在路径:/home/wangyuanwen/data/mxnet_train_recdata_64/mx_val
模型保存位置 ./data/train_output/mxnet_ckpt/mn-v2_0.25
```

## 2、测试模型
```shell
单张图片检测执行./mx_predict.sh
批量检测执行./mx_predict_data.sh
```

## 3、口罩检测种类与颜色
```shell
医用外科：蓝色、白色、黑色、粉色、绿色
N95：白色、绿色、灰色
```

## 4、测试的mask、nomask各1w张，口罩检测0.1%误识率的准确率：99.65%（mask）、99.60（nomask）
