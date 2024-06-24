from tqdm import *
import torch
import torch.onnx
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as rt
import numpy as np
from torch.nn import functional as F
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test


def inference(config, model, image):
    size = image.size()
    pred = model(image)
    # print(len(pred))
    # print(pred[0].shape)
    # print(pred[1].shape)
    # exit()
    if config.MODEL.NUM_OUTPUTS > 1:
        print("config.TEST.OUTPUT_INDEX is :", config.TEST.OUTPUT_INDEX)
        pred = pred[config.TEST.OUTPUT_INDEX]

    pred = F.interpolate(
        input=pred, size=size[-2:],
        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
    )


    return pred.exp()


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="../experiments/cityscapes/ddrnet23_slim.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    # 实例化模型
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)
    # print(model)

    # 模型的存储路径: pt
    # model_state_file = './save_models/save_3.pt'
    model_state_file = './save_models/save_3_0_255.pt'

    # 模型的存储路径: onnx
    # onnx_path = "./save_models_onnx/ddrnet_save_3_opset_version_12_argmax_float32.onnx"
    onnx_path = "./save_models_onnx/ddrnet_save_3_0_255_opset_version_12_argmax_float32.onnx"

    # model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))
    # 加载模型参数
    model.load_state_dict(torch.load(model_state_file))

    model.eval()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    with torch.no_grad():
        count = 0
        for index, batch in enumerate(tqdm(testloader)):
            image, label, size_image, name, *border_padding = batch
            batch_00, _, ori_height, ori_width = image.size()
            print(image.size())
            # exit()
            assert batch_00 == 1, "only supporting batchsize 1."
            # print(image.shape)
            # exit()
            preds = inference(config, model, image)
            # print(preds.shape)
            # exit()
            output = preds.cpu()
            seg_pred = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
            # seg_pred表示最终输出的结果的经过argmax操作的数据
            # print(seg_pred.shape)
            onnx_pred = onnx_predict(image.numpy(),onnx_path)
            # print(seg_pred.shape)
            # print(len(onnx_pred))
            # print("onnx_pred[0] shape is",onnx_pred[0].shape)
            # print(onnx_pred[0].shape)
            # exit()
            count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            sum_onnx_pred_0 = 0
            with open('./onnx_output_txt/onnx_output_save_3_0_255_train_qt_demo.txt', "w") as file:
                for i in range(onnx_pred[0].shape[0]):
                    for j in range(onnx_pred[0].shape[1]):
                        for k in range(onnx_pred[0].shape[2]):
                            file.write(str(int(onnx_pred[0][i][j][k])) + '\n');
                            sum_onnx_pred_0 += onnx_pred[0][i][j][k]
                            for i_01 in range(11):
                                if onnx_pred[0][i][j][k] == i_01:
                                    count_list[i_01] += 1
            print(count_list)
            print(sum(count_list))
            print(sum_onnx_pred_0)
            if np.testing.assert_allclose(seg_pred, onnx_pred[0], rtol=1e-4) == None:
                count += 1
            else:
                pass
        print(count)
            # exit()
    '''
    # 现在已经获取到直接进入模型的数据了
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config,
                                                       test_dataset,
                                                       testloader,
                                                       model,
                                                       # sv_pred=False
                                                       sv_pred=True
                                                       )

    '''

def onnx_predict(image, onnx_path):

    # 读取onnx模型，安装GPUonnx，并设置providers = ['GPUExecutionProvider']，可以实现GPU运行onnx
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(onnx_path, providers=providers)

    # 推理onnx模型
    output_names = ["outputy"]
    onnx_pred = m.run(output_names, {"inputx": image})

    # 输出结果
    # print('ONNX Predicted shape is :', onnx_pred.shape)
    # return softmax(onnx_pred[0][0])
    return onnx_pred

######################################################################################################################



if __name__ == '__main__':
    # pth_path = "./save_models/save_3.pt"
    # onnx_path = "./save_models_onnx/ddrnet_save_3_opset_version_12_argmax_int32.onnx"

    # 先获取图片（使用1张图）

    main()
    # pth_result = pth_predict(image, pth_path)
    # onnx_result = onnx_predict(image.numpy(), onnx_path)
    # np.testing.assert_allclose(pth_result, onnx_result, rtol=1e-4)
