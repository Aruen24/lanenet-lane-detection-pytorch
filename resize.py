import cv2
import sys
import os

in_path = sys.argv[1]
out_path = sys.argv[2]

# 二级目录
#for ori_id in os.listdir(in_path):
#    ori_path = os.path.join(in_path, ori_id)
#    for img_name in os.listdir(ori_path):
#        img_path = os.path.join(ori_path,img_name)
#        img = cv2.imread(img_path)
#        img = cv2.resize(img, (112,112))
#        print(img_path)
#        #print(os.path.join(out_path,ori_id,img_name))
#        out_dir = os.path.join(out_path,ori_id)
#        if not os.path.exists(out_dir):
#            os.makedirs(out_dir)
#        cv2.imwrite(os.path.join(out_dir,img_name), img)





# 一级目录
for ori_id in os.listdir(in_path):
    img_path = os.path.join(in_path, ori_id)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64,64))
    #print(ori_id)
    print(img_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cv2.imwrite(os.path.join(out_path,ori_id), img)
