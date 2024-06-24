# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test

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

class onnx_net(nn.Module):
    def __init__(self, model):
        super(onnx_net, self).__init__()
        self.backone = model

    def forward(self, x):
        x1, x2 = self.backone(x)
        # 由于在eval的时候适合网络的输出的pred的索引为1的值的，所以这里输出的索引也为1
        y = F.interpolate(x2, size=(480,480), mode='bilinear') #
        # y = F.softmax(y, dim=1) # 返回指定维度最大值的索引，所以不使用softmax也能做到
        # print(y.dtype)
        # exit()
        # print(y.shape)
        # exit()
        y = torch.argmax(y, dim=1)
        # print(y.shape)
        # y = y.int() # 将y从int64转换为int32，int()表示int32() #轉出的onnx可以轉換成rknn（第一次
        y = y.float()
        # print(y.dtype)
        # exit()
        # print(type(y))
        # print(y.dtype)
        # y.dtype = torch.int32
        # print(y.dtype)
        # exit()
        return y


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
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    # if config.TEST.MODEL_FILE:
    #     model_state_file = config.TEST.MODEL_FILE
    # else:
    #     model_state_file = os.path.join(final_output_dir, 'best_0.7589.pth')
    #     # model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    # logger.info('=> loading model from {}'.format(model_state_file))
    model_state_file = './save_models/save_3_0_255.pt'
    model.load_state_dict(torch.load(model_state_file))

    # pretrained_dict = torch.load(model_state_file, map_location='cpu')
    # if 'state_dict' in pretrained_dict:
    #     pretrained_dict = pretrained_dict['state_dict']
    # model_dict = model.state_dict()
    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
    #                     if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    net = onnx_net(model)
    net = net.eval() # 转换之前先停止参数更新

    x = torch.randn((1,3,480,480))
    torch_out = net(x)

    output_path = "save_models_onnx/ddrnet_save_3_0_255_opset_version_12_argmax_float32.onnx"
    torch.onnx.export(net,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    output_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['inputx'],   # the model's input names
                    output_names = ['outputy'], # the model's output names
                    verbose=True,
                    )
    # onnx.checker.check_model(output_path)

    


if __name__ == '__main__':
    main()
