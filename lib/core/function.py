# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import Map16, Vedio
# from utils.DenseCRF import DenseCRF

import utils.distributed as dist
import cv2


vedioCap = Vedio('./output/cdOffice.mp4')
map16 = Map16(vedioCap)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    # print(labelmap.shape)
    # print(np.unique(labelmap))
    # exit()
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def save_pred2(image, preds, size_image, sv_path, name):
    print("image.shape is :", image.shape)
    print("preds.shape is :", preds.shape)
    preds = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
    image = image.squeeze(0)
    print("image.shape is :", image.shape)
    print("preds.shape is :", preds.shape)
    # exit()
    image = image.numpy().transpose((1, 2, 0)) # chw to hwc
    print("transpose image shape is: ", image.shape)

    image *= std
    image += mean
    # image *= 255.0 #训练的使用0到255就不需要这一步了，要和训练保持一致

    image = image.astype(np.uint8) # 将数据转换成np.uint8
    colors = np.array([[0, 0, 0],
                       [0, 0, 255],
                       [0, 255, 0],
                       [0, 255, 255],
                       [255, 0, 0],
                       [255, 0, 255],
                       [255, 255, 0],
                       [255, 255, 255],
                       [0, 0, 128],
                       [0, 128, 0],
                       [128, 0, 0],
                       [0, 128, 128],
                       [128, 0, 0],
                       [128, 0, 128],
                       [128, 128, 0],
                       [128, 128, 128],
                       [192, 192, 192]], dtype=np.uint8)
    # print("84 is :", preds.shape)
    # exit()
    pred_color = colorEncode(preds, colors)
    im_vis = image * 0.5 + pred_color * 0.5
    im_vis = im_vis.astype(np.uint8) # 此时尺寸还是480*480的，要转换到原图的尺寸大小进行保存
    # cv2.imshow("1",im_vis)
    # cv2.waitKey(0)
    # exit()
    save_path_train = 'train_qt/' + name.split('______001.png')[0]
    print(save_path_train)
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)
    # cv2.imwrite(save_path_train + '/' + name.split('______001.png')[0]+'_pred.png',im_vis)

    h_00 = int(size_image[0][0].item())
    w_00 = int(size_image[0][1].item())
    # print(h_00, w_00)
    # exit()
    im_vis = cv2.resize(im_vis,(w_00,h_00))
    save_img = Image.fromarray(im_vis)
    # print(os.path.join(sv_path, name))
    print(name)
    # exit()
    # save_img.save(os.path.join(sv_path, name))
    # exit()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size

def pixel_acc(pred, label):
    if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
        pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

'''
def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _, acc = model(images, labels)
        loss = losses.mean()
        acc  = acc.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        ave_acc.update(acc.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
'''

def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, criterion, loss_my_train_01):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    # writer = writer_dict['writer']
    # global_steps = writer_dict['train_global_steps']
    # print("len(trainloader) is :",len(trainloader))
    # exit()
    print(num_epoch)
    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        # print("1")

        # print(images.device)
        # print(labels.device)
        images = images.cuda(0)
        labels = labels.cuda(0)
        # print(images.device)
        # print(labels.device)
        # exit()
        outputs = model(images)

        # print(len(outputs))
        # print(outputs[0].shape)

        # print(outputs[1].shape)
        # exit()
        loss_00 = criterion(outputs, labels.long())
        # exit()
        acc = pixel_acc(outputs[1], labels)
        losses = torch.unsqueeze(loss_00, 0)
        # losses, _, acc = model(images, labels.long())

        loss = losses.mean()
        acc_mean = acc.mean()

        # if dist.is_distributed():
        #     reduced_loss = reduce_tensor(loss)
        # else:
        #     reduced_loss = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        # ave_loss.update(reduced_loss.item())
        ave_acc.update(acc_mean.item())

        # lr = adjust_learning_rate(optimizer,
        #                           base_lr,
        #                           num_iters,
        #                           i_iter+cur_iters)

        # if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
        #     msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
        #           'lr: {}, Loss: {:.6f}, Acc:{:.6f}' .format(
        #               epoch, num_epoch, i_iter, epoch_iters,
        #               batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
        #               ave_acc.average())
        #     logging.info(msg)
        if i_iter % config.PRINT_FREQ == 0:
            print(i_iter)
            print(loss.item())
            print(ave_acc.average())
        # print(loss)
        # exit()
        loss_my_train_01 += loss.item()
    with open("train_loss_0_255.txt", "a") as file:
        file.write("{}      {}".format(epoch, loss_my_train_01 / len(trainloader)) + '\n')
    torch.save(model.state_dict(), 'save_models/save_{}_0_255_2024_06_24.pt'.format(epoch))
    # exit()
    return loss_my_train_01 / len(trainloader) * 64


    # writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    # writer_dict['train_global_steps'] = global_steps + 1


#
# def validate(config, testloader, model, writer_dict):
#     model.eval()
#     ave_loss = AverageMeter()
#     nums = config.MODEL.NUM_OUTPUTS
#     confusion_matrix = np.zeros(
#         (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
#     with torch.no_grad():
#         for idx, batch in enumerate(testloader):
#             image, label, _, _ = batch
#             size = label.size()
#             image = image.cuda()
#             label = label.long().cuda()
#
#             losses, pred, _ = model(image, label)
#             if not isinstance(pred, (list, tuple)):
#                 pred = [pred]
#             for i, x in enumerate(pred):
#                 x = F.interpolate(
#                     input=x, size=size[-2:],
#                     mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
#                 )
#
#                 confusion_matrix[..., i] += get_confusion_matrix(
#                     label,
#                     x,
#                     size,
#                     config.DATASET.NUM_CLASSES,
#                     config.TRAIN.IGNORE_LABEL
#                 )
#
#             if idx % 10 == 0:
#                 print(idx)
#
#             loss = losses.mean()
#             if dist.is_distributed():
#                 reduced_loss = reduce_tensor(loss)
#             else:
#                 reduced_loss = loss
#             ave_loss.update(reduced_loss.item())
#
#     if dist.is_distributed():
#         confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
#         reduced_confusion_matrix = reduce_tensor(confusion_matrix)
#         confusion_matrix = reduced_confusion_matrix.cpu().numpy()
#
#     for i in range(nums):
#         pos = confusion_matrix[..., i].sum(1)
#         res = confusion_matrix[..., i].sum(0)
#         tp = np.diag(confusion_matrix[..., i])
#         IoU_array = (tp / np.maximum(1.0, pos + res - tp))
#         mean_IoU = IoU_array.mean()
#         if dist.get_rank() <= 0:
#             logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
#
#     writer = writer_dict['writer']
#     global_steps = writer_dict['valid_global_steps']
#     writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
#     writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
#     writer_dict['valid_global_steps'] = global_steps + 1
#     return ave_loss.average(), mean_IoU, IoU_array



def validate(config, testloader, model):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred, _ = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    # writer = writer_dict['writer']
    # global_steps = writer_dict['valid_global_steps']
    # writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    # writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    # writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array
#
# def testval(config, test_dataset, testloader, model,
#             sv_dir='', sv_pred=False):
#     model.eval()
#     confusion_matrix = np.zeros(
#         (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
#     with torch.no_grad():
#         for index, batch in enumerate(tqdm(testloader)):
#             image, label, _, name, *border_padding = batch
#             size = label.size()
#             pred = test_dataset.multi_scale_inference(
#                 config,
#                 model,
#                 image,
#                 scales=config.TEST.SCALE_LIST,
#                 flip=config.TEST.FLIP_TEST)
#
#             if len(border_padding) > 0:
#                 border_padding = border_padding[0]
#                 pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]
#
#             if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
#                 pred = F.interpolate(
#                     pred, size[-2:],
#                     mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
#                 )
#
#             # # crf used for post-processing
#             # postprocessor = DenseCRF(   )
#             # # image
#             # mean=[0.485, 0.456, 0.406],
#             # std=[0.229, 0.224, 0.225]
#             # timage = image.squeeze(0)
#             # timage = timage.numpy().copy().transpose((1,2,0))
#             # timage *= std
#             # timage += mean
#             # timage *= 255.0
#             # timage = timage.astype(np.uint8)
#             # # pred
#             # tprob = torch.softmax(pred, dim=1)[0].cpu().numpy()
#             # pred = postprocessor(np.array(timage, dtype=np.uint8), tprob)
#             # pred = torch.from_numpy(pred).unsqueeze(0)
#
#             confusion_matrix += get_confusion_matrix(
#                 label,
#                 pred,
#                 size,
#                 config.DATASET.NUM_CLASSES,
#                 config.TRAIN.IGNORE_LABEL)
#
            # if sv_pred:
            #     sv_path = os.path.join(sv_dir, 'test_results')
            #     if not os.path.exists(sv_path):
            #         os.mkdir(sv_path)
            #     test_dataset.save_pred2(image, pred, sv_path, name)
#
#             if index % 100 == 0:
#                 logging.info('processing: %d images' % index)
#                 pos = confusion_matrix.sum(1)
#                 res = confusion_matrix.sum(0)
#                 tp = np.diag(confusion_matrix)
#                 IoU_array = (tp / np.maximum(1.0, pos + res - tp))
#                 mean_IoU = IoU_array.mean()
#                 logging.info('mIoU: %.4f' % (mean_IoU))
#
#     pos = confusion_matrix.sum(1)
#     res = confusion_matrix.sum(0)
#     tp = np.diag(confusion_matrix)
#     pixel_acc = tp.sum()/pos.sum()
#     mean_acc = (tp/np.maximum(1.0, pos)).mean()
#     IoU_array = (tp / np.maximum(1.0, pos + res - tp))
#     mean_IoU = IoU_array.mean()
#
#     return mean_IoU, IoU_array, pixel_acc, mean_acc


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval() # 模型暂停更新参数（暂停训练）
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, size_image, name, *border_padding = batch
            # print(name)
            # print(size_image)
            # exit()
            # size = label.size()
            # print("size is: ", size)
            size = label.shape

            # print("image shape is: ", image.shape)
            # exit()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)
            sv_path = os.path.join(sv_dir, 'test_results')
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            save_pred2(image, pred, size_image, sv_path, name[0] + '______001.png')
            print(index)
            # exit()
            # print("pred shape is : ",pred.shape)
            # print("label shape is :", label.shape)
            # exit()
            # print("confusion_matrix shape is:", confusion_matrix.shape)
            # exit()
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            # print("jieguoshi:",confusion_matrix)
            # exit()


            # print(name[0] + '______001.png')
            # exit()


            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))


    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                # mean=[0.485, 0.456, 0.406],
                #  std=[0.229, 0.224, 0.225]
                image = image.squeeze(0)
                image = image.numpy().transpose((1,2,0))
                image *= [0.229, 0.224, 0.225]
                image += [0.485, 0.456, 0.406]
                image *= 255.0
                image = image.astype(np.uint8)

                _, pred = torch.max(pred, dim=1)
                pred = pred.squeeze(0).cpu().numpy()
                map16.visualize_result(image, pred, sv_dir, name[0]+'.jpg')
                # sv_path = os.path.join(sv_dir, 'test_results')
                # if not os.path.exists(sv_path):
                #     os.mkdir(sv_path)
                # test_dataset.save_pred(image, pred, sv_path, name)
        vedioCap.releaseCap()
