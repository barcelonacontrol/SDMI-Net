# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

from skimage import measure
from scipy import ndimage
from PIL import Image
import torch
import time

def post_process_evaluate(x, target, name, args):

    JA_sum, AC_sum, DI_sum, SE_sum, SP_sum = [],[],[],[],[]

    for i in range(x.shape[0]):
        x_tmp = x[i]
        target_tmp = target[i]

        x_tmp[x_tmp >= 0.5] = 1
        x_tmp[x_tmp <= 0.5] = 0
        x_tmp = np.array(x_tmp, dtype='uint8')
        x_tmp = ndimage.binary_fill_holes(x_tmp).astype(int)

        # only reserve largest connected component.
        box = []
        [lesion, num] = measure.label(x_tmp, return_num=True)
        if num == 0:
            JA_sum.append(0)
            AC_sum.append(0)
            DI_sum.append(0)
            SE_sum.append(0)
            SP_sum.append(0)
        else:
            region = measure.regionprops(lesion)
            for i in range(num):
                box.append(region[i].area)
            label_num = box.index(max(box)) + 1
            lesion[lesion != label_num] = 0
            lesion[lesion == label_num] = 1

            #  calculate TP,TN,FP,FN
            # print(lesion.shape)
            # print(target_tmp.shape)
            TP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 1)))
            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 0)))

            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 0)))

            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 1)))

            #  calculate JA, Dice, SE, SP
            JA = TP / ((TP + FN + FP + 1e-7))
            AC = (TP + TN) / ((TP + FP + TN + FN + 1e-7))
            DI = 2 * TP / ((2 * TP + FN + FP + 1e-7))
            SE = TP / (TP + FN+1e-7)
            SP = TN / ((TN + FP+1e-7))

            JA_sum.append(JA); AC_sum.append(AC); DI_sum.append(DI); SE_sum.append(SE); SP_sum.append(SP)

    if args.evaluate:
        with open("/".join(args.resume.split('/')[:-1])+ '/result.txt', 'a') as f:
            for item in range(0, len(DI_sum)):
                f.write("\n%s " % name[item].split("/")[-1])
                f.write("JA %.4f " % JA_sum[item])
                f.write("DI %.4f " % DI_sum[item])
                f.write("AC %.4f " % AC_sum[item])
                f.write("SE %.4f " % SE_sum[item])
                f.write("SP %.4f " % SP_sum[item])

    return sum(JA_sum), sum(AC_sum), sum(DI_sum), sum(SE_sum), sum(SP_sum)


def post_process_evaluate_pre(x, target):

    JA_sum, AC_sum, DI_sum, SE_sum, SP_sum = [],[],[],[],[]


    x_tmp = x
    target_tmp = target


    x_tmp[x_tmp >= 0.5] = 1
    x_tmp[x_tmp <= 0.5] = 0
    x_tmp = np.array(x_tmp, dtype='uint8')
    x_tmp = ndimage.binary_fill_holes(x_tmp).astype(int)

    # only reserve largest connected component.
    box = []
    [lesion, num] = measure.label(x_tmp, return_num=True)
    if num == 0:
        JA_sum.append(0)
        AC_sum.append(0)
        DI_sum.append(0)
        SE_sum.append(0)
        SP_sum.append(0)
    else:
        region = measure.regionprops(lesion)
        for i in range(num):
            box.append(region[i].area)
        label_num = box.index(max(box)) + 1
        lesion[lesion != label_num] = 0
        lesion[lesion == label_num] = 1

        #  calculate TP,TN,FP,FN
        TP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 255)))
        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 0)))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 0)))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 255)))

        #  calculate JA, Dice, SE, SP
        JA = TP / ((TP + FN + FP + 1e-7))
        AC = (TP + TN) / ((TP + FP + TN + FN + 1e-7))
        DI = 2 * TP / ((2 * TP + FN + FP + 1e-7))
        SE = TP / (TP + FN+1e-7)
        SP = TN / ((TN + FP+1e-7))

        JA_sum.append(JA); AC_sum.append(AC); DI_sum.append(DI); SE_sum.append(SE); SP_sum.append(SP)

    return sum(JA_sum), sum(AC_sum), sum(DI_sum), sum(SE_sum), sum(SP_sum)


def multi_validate(valloader, model, criterion, epoch, use_cuda, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = []

    JA = 0
    DI = 0
    AC = 0
    SE = 0
    SP = 0
    MIOU = 0
    i = 0

    # val_ground = np.vstack(val_ground)
    # print (val_ground.shape)
    # switch to evaluate mode
    model.eval()
    model.mode = 'test'

    end = time.time()
    # bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        kk = 0
        idx = 0
        for batch_idx, (inputs, targets, name) in enumerate(valloader):


            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                targets = targets.long()
                targets[targets == 255] = 1
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # init
            score_box = np.zeros((inputs.shape[0], 248, 248), dtype='float32')
            num_box = np.zeros((inputs.shape[0], 248, 248), dtype='uint8')
            nowMIOU = 0

            # compute output
            outputs, _ = model(x_l=inputs[:, :, 0:224,0:224], dropout=False)
            # outputs = torch.sigmoid(outputs)

            # Lx1 = criterion(outputs, targets[:,0:224,0:224].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 0:224, 0:224] = outputs[:, 1, :, :].cpu().detach().numpy()
            num_box[:, 0:224, 0:224] = 1
            outputs = torch.softmax(outputs, dim=1).max(1)[1]
            nowMIOU += cal_mIou(outputs.cpu().numpy(), targets[:, 0:224, 0:224].cpu().numpy())

            outputs, _ = model(x_l=inputs[:, :, 24:248, 24:248], dropout=False)
            # outputs = torch.sigmoid(outputs)
            # Lx2 = criterion(outputs, targets[:, 24:248, 24:248].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 24:248, 24:248] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 24:248, 24:248] += 1
            outputs = torch.softmax(outputs, dim=1).max(1)[1]
            nowMIOU += cal_mIou(outputs.cpu().numpy(), targets[:, 24:248, 24:248].cpu().numpy())

            outputs, _ = model(x_l=inputs[:, :, 0:224, 24:248], dropout=False)
            # outputs = torch.sigmoid(outputs)
            # Lx3 = criterion(outputs, targets[:, 0:224,24:248].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 0:224, 24:248] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 0:224, 24:248] += 1
            outputs = torch.softmax(outputs, dim=1).max(1)[1]
            nowMIOU += cal_mIou(outputs.cpu().numpy(), targets[:, 0:224, 24:248].cpu().numpy())

            outputs, _ = model(x_l=inputs[:, :, 24:248, 0:224], dropout=False)
            # outputs = torch.sigmoid(outputs)
            # Lx4 = criterion(outputs, targets[:, 24:248,0:224].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 24:248, 0:224] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 24:248, 0:224] += 1
            outputs = torch.softmax(outputs, dim=1).max(1)[1]
            nowMIOU += cal_mIou(outputs.cpu().numpy(), targets[:, 24:248,0:224].cpu().numpy())

            score = score_box / (num_box + 1e-5)

            # loss = (Lx1 + Lx2 + Lx3 + Lx4) / 4.0
            # losses.append(loss.item())

            # measure accuracy and record loss
            x = score

            nowMIOU = nowMIOU / 4.0

            # y = val_ground[batch_idx]
            # x = cv2.resize(x[0], (y.shape[1], y.shape[0]),
            #                      interpolation=cv2.INTER_NEAREST)


            y = targets.cpu().detach().numpy()
            results = post_process_evaluate(x, y, name, args)

            JA += results[0]
            AC += results[1]
            DI += results[2]
            SE += results[3]
            SP += results[4]
            MIOU += nowMIOU

            i = i + inputs.shape[0]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # outputs = torch.sigmoid(outputs)
            # Lx3 = criterion(outputs, targets[:, 0:224,24:248].long())

            idx += 1
            # bar.next()
        # bar.finish()

    return (0, [JA/i, AC/i,DI/i, SE/i, SP/i, MIOU/idx])


def cal_mIou(seg, gt, classes=2, background_id=0):
    channel_iou = []
    for i in range(classes):
        if i == background_id:
            continue
        cond = i ** 2
        inter = len(np.where(seg * gt == cond)[0])
        union = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0]) - inter
        if union == 0:
            iou = 0
        else:
            iou = inter / union

        channel_iou.append(iou)

    res = np.array(channel_iou).mean()
    return res



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss



def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf