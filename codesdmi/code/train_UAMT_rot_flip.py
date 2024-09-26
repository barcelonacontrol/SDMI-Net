import os
import sys
from collections import OrderedDict

from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import MSELoss
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from networks.SandT import SNet, TNet
from networks.entire_model import S_model, T_model
from test_two_classes import test_calculate_metric, test_calculate_metric_memory
from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses, rot_flip
from dataloaders.process import Pack, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
# 当前运行的方法
parser.add_argument('--method', type=str, default='UAMT_flip_new', help='model_name')

# 更换数据集务必更改这三条
parser.add_argument('--dataset', type=str, default='LA', help='Name of Experiment')
parser.add_argument('--labelnum', type=int, default=16, help='random seed')
parser.add_argument('--trainnum', type=int, default=80, help='random seed')
parser.add_argument('--testnum', type=int, default=20, help='random seed')
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--max_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--numclasses', type=int, default=2, help='class number')
args = parser.parse_args()

train_data_path = args.root_path + "/" + args.dataset

snapshot_path = "../model/" + args.dataset + "/" + args.method + '/'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

bestDice = 0.0
bestDicet = 0.0
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
warmup_iter = 500


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

@torch.no_grad()
def update_teachers(student_encoder, student_decoder, teacher_encoder, teacher_decoder, keep_rate=0.996):
    student_encoder_dict = student_encoder.state_dict()
    student_decoder_dict = student_decoder.state_dict()
    new_teacher_encoder_dict = OrderedDict()
    new_teacher_decoder_dict = OrderedDict()
    for key, value in teacher_encoder.state_dict().items():
        if key in student_encoder_dict.keys():
            new_teacher_encoder_dict[key] = (
                    student_encoder_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student encoder model".format(key))

    for key, value in teacher_decoder.state_dict().items():

        if key in student_decoder_dict.keys():
            new_teacher_decoder_dict[key] = (
                    student_decoder_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student decoder model".format(key))
    teacher_encoder.load_state_dict(new_teacher_encoder_dict, strict=True)
    teacher_decoder.load_state_dict(new_teacher_decoder_dict, strict=True)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(my_model="S_model", ema=False):
        # Network definition
        if my_model == "S_model":
            net = SNet(n_channels=1, n_classes=args.numclasses, normalization='batchnorm', has_dropout=True)
        if my_model == "T_model":
            net = TNet(n_channels=1, n_classes=args.numclasses, normalization='batchnorm', has_dropout=True)
        if my_model == "VNet":
            net = VNet(n_channels=1, n_classes=args.numclasses, normalization='batchnorm', has_dropout=True)
        net = net.cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
        return net

    model = create_model(my_model="VNet")
    ema_model = create_model(my_model="VNet", ema=True)
    ema_model2 = create_model(my_model="VNet", ema=True)
    db_train = Pack(base_dir=train_data_path,
                    split='train',
                    transform=transforms.Compose([
                        RandomRotFlip(),
                        RandomCrop(patch_size),
                        ToTensor(),
                    ]))

    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.trainnum))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    ce_loss = BCEWithLogitsLoss()

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))w
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            # 只取 -0.2 ～0.2的区域，randn_like(返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
            ema_inputs = unlabeled_volume_batch + noise  # 往ema输入添加噪声
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                #     print(ema_output.shape) #[2, 2, 112, 112, 80])
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)  # 复制一倍数据 无标签数据
            # print(volume_batch_r.shape)  ->  [4, 1, 112, 112, 80] 意思是4 batchsize
            stride = volume_batch_r.shape[0] // 2  # 取矩阵第一维度的长度,即数组的行数
            # print(stride)  stride = 2
            preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()  #[16]
            x = torch.zeros([T, batch_size//2, 1, 112, 112, 80]).cuda() #[8, 2 ...]
            opt = []
            for i in range(T // 2):  # 取整除 等于4
                # rot = random.randint(1, 4)
                # flip = random.randint(0, 1)
                x[2 * i:2 * i + 1], opt[4 * i:4 * i + 2] = rot_flip.rot_num_flip(volume_batch_r[:batch_size // 2], i, i % 2)
                x[2 * i + 1:2 * i + 2], opt[4 * i + 2:4 * i + 4] = rot_flip.rot_num_flip(volume_batch_r[batch_size // 2:batch_size], i, (1+i) % 2)
                ema_inputs = torch.cat((x[2 * i], x[2 * i + 1]), dim=0)
                ema_inputs = ema_inputs + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)  # 无标签数据 每次都加入噪声
                with torch.no_grad():
                    ema_out = ema_model(ema_inputs)
                    outA = rot_flip.get_origin(ema_out[:batch_size // 2], opt[4 * i:4 * i + 2])
                    outB = rot_flip.get_origin(ema_out[batch_size // 2:batch_size], opt[4 * i + 2:4 * i + 4])
                    pred = torch.cat((outA, outB), dim=0)
                    preds[2 * stride * i:2 * stride * (i + 1)] = pred
                    # print(preds[2 * stride * i:2 * stride * (i + 1)].shape) [4, 2, 112, 112, 80])
                    # print(preds.shape) #[16, 2, 112, 112, 80])
            preds = F.softmax(preds, dim=1)  # 激活 把输出转化为概率 也就是 Pc / dim等于1的意思是对每个分类进行softmax
            preds = preds.reshape(T, stride, 2, 112, 112, 80)  # 8 2 2 112 112 80
            preds = torch.mean(preds, dim=0)  # (batch, 2, 112,112,80)  取每次前传平均值 { Uc = 1/T sum(Pc) }
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            # (batch, 1, 112,112,80)  -plogp 不确定性用熵来衡量，dim=1, 得到的是每种分类的不确定熵
            # torch.log 对每个元素求 ln  ,keepdim=True是为了保证维度元素为1时保留维度，

            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])  # 像素预测图的损失
            outputs_soft = F.softmax(outputs, dim=1)  # 有标签和无标签变成4batch得到的学生模型的输出

            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output)  # (batch, 2, 112,112,80)
            # losses.softmax_mse_loss  无标签不确定估计一致性

            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)  # 一个标量控制指示函数的阈值

            mask = (uncertainty < threshold).float()  # 用于过滤不确定度高的掩码
            consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            loss = supervised_loss + consistency_weight * consistency_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            # logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %(iter_num, loss.item(), const_loss.item(), consistency_weight))
            logging.info('iteration %d : loss : %f , lsup: %f , lcons: %f , loss_weight: %f, lr: %f' % (
            iter_num, loss.item(), supervised_loss.item(), consistency_dist.item(), consistency_weight, lr_))

            ## change lr
            # if iter_num % 2500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_

            if iter_num % 250 == 0:
                save_test_path = snapshot_path + "test/"
                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path)
                save_mode_path1 = os.path.join(snapshot_path, '_iter_' + str(iter_num) + '.pth')
                save_mode_path2 = os.path.join(snapshot_path, '_iter_' + str(iter_num) + '_teacher.pth')
                torch.save(model.state_dict(), save_mode_path1)
                torch.save(ema_model.state_dict(), save_mode_path2)
                logging.info("save model to {}".format(save_mode_path1))
                save_mode_path = [save_mode_path1]
                metric = test_calculate_metric(save_mode_path, train_data_path, save_test_path, args.testnum,
                                               args.numclasses)  # 6000
                logging.info(
                    'iteration %d student: Dice: %f, JA: %f, 95HD: %f, ASD: %f' %
                    (iter_num, metric[0], metric[1], metric[2], metric[3]))

                bestDice = max(bestDice, metric[0])

                save_mode_path = [save_mode_path2]
                metric = test_calculate_metric(save_mode_path, train_data_path, save_test_path, args.testnum,
                                               args.numclasses)  # 6000
                bestDicet = max(bestDicet, metric[0])
                logging.info(
                    'iteration %d teacher: Dice: %f, JA: %f, 95HD: %f, ASD: %f' %
                    (iter_num, metric[0], metric[1], metric[2], metric[3]))
                logging.info( 'nowbest student Dice: %f   nowbest teacher Dice: %f' % (bestDice, bestDicet))
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
