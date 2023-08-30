import os
import shutil
# import magic # Detect image type from buffer contents (disabled, all are jpg)

# parser = argparse.ArgumentParser()
# parser.add_argument('MSCelebTXT', type=str)
# parser.add_argument('--outputDir', type=str, default='raw')
# args = parser.parse_args()

# MS-Celeb-1M_clean_list.txt中数据格式m.0109kg/14-FaceId-0.jpg 0
# with open(args.MSCelebTXT, 'r') as txtF:
# f1 = open('./MS-Celeb-1M_clean_list.txt', 'r')
with open('./test_result_nomask_withmask.txt','r') as f:
    lines = f.readlines()
    i = 0
    for row in lines:
        if row.strip():
            result = row.split(",")
            picPath = result[0]
            #faceID = result[1].strip()
            picname = picPath.split("/")[6]
            # ori_img = cv2.imread("./data/"+picPath.strip())

            saveDir = "/home/wangyuanwen/data/test_result_nomask_nomask/"

            #if os.path.exists(saveDir+picname):
             #   picname = str(i)+"-"+picname

            if not os.path.exists(saveDir):
                os.mkdir(saveDir)

            # cv2.imwrite(saveDir+picname, ori_img)
            #shutil.move(picPath.strip(), saveDir+picname)
            shutil.copy(picPath.strip(), saveDir+picname)


            i += 1

            if i % 50 == 0:
                print("Extracted {} images.".format(i))
