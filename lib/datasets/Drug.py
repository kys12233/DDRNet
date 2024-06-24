# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

save_qt_img_path = '/home/keyousheng/code/makemodel/export/data_my/data480_480_semantic100_train_qt'
if not os.path.exists(save_qt_img_path):
    os.makedirs(save_qt_img_path)

class Drug(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=11,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 # 以下的是要先image / 255 :  在image = self.input_transform(image)操作
                 # mean=[0.485, 0.456, 0.406], # ori
                 # std=[0.229, 0.224, 0.225] # ori
                 # mean=[0.0, 0.0, 0.0],
                 # std=[1.0, 1.0, 1.0]
                 # 以下的是直接讀取image，不做操作
                 mean = [0, 0, 0],
                 std = [255, 255, 255]
                 ):

        super(Drug, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10}
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539]).cuda()
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        save_train_qt = 'train_qt/' + name
        if not os.path.exists(save_train_qt):
            os.makedirs(save_train_qt)
        # exit()
        print("save_train_qt:",save_train_qt)
        print(name)
        # print(os.path.join(self.root,'drug',item["img"]))
        # exit()
        image = cv2.imread(os.path.join(self.root,'drug',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape
        # print(size)
        # image_pil = Image.fromarray(image)
        # image_pil.save('0.png')
        # exit()
        # cv2.imwrite("test_results/demo_03.jpg",image)
        # exit()
        image = cv2.resize(image, (480, 480))
        # 作量化的时候需要保存数据，所以在这操作
        # 使用训练的数据进行量化,所以在这将其resize到480*480，然后再进行保存
        # cv
        cv2.imwrite(save_train_qt + '/' + item["img"].split("/train/")[1] ,image)
        # print("dsfhdfhdfa:", save_train_qt + '/' + item["img"])
        # exit()
        print("img name is : ",item["img"])
        # exit()
        if "/val" in item["img"]:
            a = item["img"].split("/val")[1]
            # print(a)
            img_save_path = save_qt_img_path + a
            # print(img_save_path)
            cv2.imwrite(img_save_path,image)
            with open(save_qt_img_path + '.txt', "a") as file:
                file.write(img_save_path + '\n')
                file.close()
        a = item["img"].split("/train")[1]
        # print(a)
        img_save_path = save_qt_img_path + a
        # print(img_save_path)
        # cv2.imwrite(img_save_path, image)
        with open(save_qt_img_path + '.txt', "a") as file:
            file.write(img_save_path + '\n')
            file.close()
        # exit()
        # /home/keyousheng/code/makemodel/export/data_my/data1024x2048_semantic100

        label = cv2.imread(os.path.join(self.root, 'drug', item["label"]),
                           cv2.IMREAD_GRAYSCALE)

        # print(label.shape)
        # exit()
        # print(image.shape)
        label = cv2.resize(label, (480, 480))
        # save_label_path = save_train_qt + '/' + item["label"].split("/train/")[1]
        # print(save_label_path)
        # exit()
        # cv2.imwrite(save_label_path, label)
        # exit()
        # print(type(label))
        # print(image.dtype)
        # print(label.dtype)
        image = np.float32(image)
        # label = label.long()
        label = np.float32(label)
        # print(image.dtype)
        # print(label.dtype)
        # exit()
        # print(size)
        # cv2.imshow("1",image)
        # cv2.waitKey(0)
        # print("size is :", size)

        # print(image.shape)
        # exit()

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            print("1", image.shape)
            # return image.copy(), np.array(size), name # ori
            return np.float32(image.copy()), np.array(size), name
        else:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
        #     print("2", image.shape)
        # exit()
        # print(os.path.join(self.root,'drug',item["label"]))
        # exit()

        # exit()
        # print("1 is : ",label.shape)
        # label = self.convert_label(label)

        # print("2 is : ",label.shape)
        # exit()
        # image = torch.from_numpy(image)
        # exit()
        # label = torch.from_numpy(label)
        # image, label = self.gen_sample(image, label,
        #                         self.multi_scale, self.flip) #ori 不做数据增强先进行注释
        # print(type(image))
        # print(type(label))
        # exit()
        # return image.copy(), label.copy(), np.array(size), name #ori
        # return np.float32(image.copy()), np.float32(label.copy()), np.array(size), name
        return image, label, np.array(size), name

    # def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
    #     batch, _, ori_height, ori_width = image.size()
    #     assert batch == 1, "only supporting batchsize 1."
    #     print(image.shape)
    #     print(image.size())
    #     image = image.numpy()[0].transpose((1,2,0)).copy()
    #     print(image.shape)
    #     exit()
    #     stride_h = np.int(self.crop_size[0] * 1.0) #ori
    #     stride_w = np.int(self.crop_size[1] * 1.0) #ori
    #     final_pred = torch.zeros([1, self.num_classes,
    #                                 ori_height,ori_width]).cuda()
    #     for scale in scales:
    #         new_img = self.multi_scale_aug(image=image,
    #                                        rand_scale=scale,
    #                                        rand_crop=False)
    #         height, width = new_img.shape[:-1]
    #
    #         if scale <= 1.0:
    #             new_img = new_img.transpose((2, 0, 1))
    #             new_img = np.expand_dims(new_img, axis=0)
    #             new_img = torch.from_numpy(new_img)
    #             preds = self.inference(config, model, new_img, flip)
    #             preds = preds[:, :, 0:height, 0:width]
    #         else:
    #             new_h, new_w = new_img.shape[:-1]
    #             rows = np.int(np.ceil(1.0 * (new_h -
    #                             self.crop_size[0]) / stride_h)) + 1
    #             cols = np.int(np.ceil(1.0 * (new_w -
    #                             self.crop_size[1]) / stride_w)) + 1
    #             preds = torch.zeros([1, self.num_classes,
    #                                        new_h,new_w]).cuda()
    #             count = torch.zeros([1,1, new_h, new_w]).cuda()
    #
    #             for r in range(rows):
    #                 for c in range(cols):
    #                     h0 = r * stride_h
    #                     w0 = c * stride_w
    #                     h1 = min(h0 + self.crop_size[0], new_h)
    #                     w1 = min(w0 + self.crop_size[1], new_w)
    #                     h0 = max(int(h1 - self.crop_size[0]), 0)
    #                     w0 = max(int(w1 - self.crop_size[1]), 0)
    #                     crop_img = new_img[h0:h1, w0:w1, :]
    #                     crop_img = crop_img.transpose((2, 0, 1))
    #                     crop_img = np.expand_dims(crop_img, axis=0)
    #                     crop_img = torch.from_numpy(crop_img)
    #                     pred = self.inference(config, model, crop_img, flip)
    #                     preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
    #                     count[:,:,h0:h1,w0:w1] += 1
    #             preds = preds / count
    #             preds = preds[:,:,:height,:width]
    #
    #         preds = F.interpolate(
    #             preds, (ori_height, ori_width),
    #             mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
    #         )
    #         final_pred += preds
    #     return final_pred
    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        # 我传入的数据已经是NCHW了
        # print("11112")
        # exit()
        preds = self.inference(config, model, image, flip)
        # print(preds.shape)
        # exit()
        # preds = preds[:, :, 0:ori_height, 0:ori_width]
        # print("preds shape is :", preds.shape)
        # preds = F.interpolate(
        #         preds, (ori_height, ori_width),
        #         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        #     )
        # final_pred += preds
        # exit()
        # print("dhfjdsafjkd")
        return preds

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
