import cv2
import os
# 读取PNG图片
# img = cv2.imread('./data/drug/label/val/____all_classes_20231206____busstation____adult____mask__label__start__1__10__1____1700729173.671000.png', cv2.IMREAD_UNCHANGED)
# img = cv2.imread('./data/drug/label/val/____all_classes_20231206____busstation____adult____mask__label__start__1__10__1____1700729173.671000.png')
img_path = './data/drug/label/val/'
for img_name in os.listdir(img_path):
    img = cv2.imread(img_path + img_name, cv2.IMREAD_GRAYSCALE)
    # 检查图片是否成功读取
    if img is None:
        print('无法读取图片')
    else:
        print('成功读取图片')
        # cv2.imshow("1", img)
        # cv2.waitKey(0)
        print(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] > 10:
                    print("ERROR")
                    break
