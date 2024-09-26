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
from test import test_calculate_metric

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses, rot_flip
from dataloaders.process import Pack, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.util import compute_sdf
from utils.rmi import RMILoss


parser = argparse.ArgumentParser()
# 当前运行的方法
parser.add_argument('--method', type=str, default='XIAO_WO_UAMT', help='model_name')

# 更换数据集务必更改这三条
parser.add_argument('--dataset', type=str, default='LA', help='Name of Experiment')

parser.add_argument('--labelnum', type=int, default=16, help='random seed')
parser.add_argument('--trainnum', type=int, default=80, help='random seed')
parser.add_argument('--testnum', type=int, default=90, help='random seed')

# parser.add_argument('--labelnum', type=int, default=42, help='random seed')
# parser.add_argument('--trainnum', type=int, default=210, help='random seed')
# parser.add_argument('--testnum', type=int, default=90, help='random seed')

parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--max_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=3401, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--num_classes', type=int, default=2, help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--numclasses', type=int, default=2, help='class number')
args = parser.parse_args()

train_data_path = args.root_path + "/" + args.dataset

snapshot_path = "../model/" + args.dataset + "/" + args.method + '/'
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
if os.path.exists(snapshot_path + '/code'):
    shutil.rmtree(snapshot_path + '/code')
shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

bestDice = 0.0

test_save_path = os.path.join(snapshot_path, "test/")

cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

with open(train_data_path + '/test.list', 'r') as f:
    test_image_list = f.readlines()
test_image_list = [train_data_path + "/" + item.replace('\n', '')+"/mri_norm2.h5" for item in test_image_list]


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

def update_ema_variables_noise(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        nowstd = torch.std(param.data, unbiased=False).item() / 10.0
        noise = get_noise_ema(param.data, nowstd)
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data + noise)

def get_noise_ema(inputs, std):
    inputs = inputs.cpu()
    gaussian = np.random.normal(0, std, inputs.shape)
    gaussian = torch.from_numpy(gaussian).float().cuda()

    return gaussian.cuda()


if __name__ == "__main__":

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code')

    # if os.path.exists(snapshot_path + "/log.txt"):
    #     os.remove(snapshot_path + "/log.txt")

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        net = VNet(n_channels=1, n_classes=args.numclasses, normalization='batchnorm', has_dropout=True)
        net = net.cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
        return net

    model = create_model()
    ema_model = create_model(ema=True)
    ema_model2 = create_model(ema=True)

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
    ema_model2.train()
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

    rmi = RMILoss(num_classes=2)

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            # 只取 -0.2 ～0.2的区域，randn_like(返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
            ema_inputs = unlabeled_volume_batch + noise  # 往ema输入添加噪声
            with torch.no_grad():
                ema_outputs1 = ema_model(ema_inputs)
                ema_outputs2 = ema_model2(ema_inputs)

                ema_output = 0.5 * (ema_outputs2 + ema_outputs1)
                #     print(ema_output.shape) #[2, 2, 112, 112, 80])

            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])  # 像素预测图的损失
            outputs_soft = F.softmax(outputs, dim=1)  # 有标签和无标签变成4batch得到的学生模型的输出

            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output)  # (batch, 2, 112,112,80)
            # losses.softmax_mse_loss  无标签不确定估计一致性

            # ema_output_arg = torch.argmax(ema_output, dim=1)
            # L_rmi = rmi(outputs[labeled_bs:], ema_output_arg)

            loss = supervised_loss + consistency_weight * consistency_dist # + 0.5 * L_rmi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_num % 2 == 1:
                update_ema_variables_noise(model, ema_model, args.ema_decay, iter_num)
            else:
                update_ema_variables_noise(model, ema_model2, args.ema_decay, iter_num)

            # update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            # update_ema_variables(model, ema_model2, args.ema_decay, iter_num)
            iter_num = iter_num + 1

            # logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %(iter_num, loss.item(), const_loss.item(), consistency_weight))
            logging.info('iteration %d : loss : %f , lsup: %f , lcons: %f,  loss_weight: %f, lr: %f' % (
            iter_num, loss.item(), supervised_loss.item(), consistency_dist.item(), consistency_weight, lr_))

            ## change lr
            # if iter_num % 2500 == 0 and iter_num <= 6000:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_

            if iter_num > 2500 and iter_num % 200 == 0:
                metric = test_calculate_metric(model, args, test_save_path=test_save_path,
                                               image_list=test_image_list)  # 6000
                if bestDice < metric[0]:
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.method))
                    torch.save(model.state_dict(), save_best_path)
                    bestDice = metric[0]

                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}'.format(iter_num, bestDice))
                    file = open(save_mode_path, 'w')
                    file.close()

                logging.info(
                    'iteration %d : Dice: %f, JA: %f, 95HD: %f, ASD: %f' %
                    (iter_num, metric[0], metric[1], metric[2], metric[3]))

                print('nowbest Dice:', bestDice)

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
