import os
import argparse
import torch
from networks.vnet import VNet
from test_util_1 import test_calculate_metric

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='LA', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=0, help='detail')
parser.add_argument('--method', type=str,  default='UAMT_unlabel', help='detail')
parser.add_argument('--num-classes', type=int,  default=2, help='num classes')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = "./{}".format(args.model)
test_save_path = os.path.join(snapshot_path, "test/")


if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

with open(args.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [args.root_path + "/" + item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]


if __name__ == '__main__':
    save_mode_path = os.path.join('../model/' + args.model + '/' + args.method + '/bestDice1.pth')
    print('best model path: ', save_mode_path)
    state_dict = torch.load(save_mode_path)

    model = VNet(n_channels=1, n_classes=args.num_classes, normalization='batchnorm', has_dropout=False).cuda()
    model.load_state_dict(torch.load(save_mode_path))

    metric = test_calculate_metric(model, args, test_save_path=test_save_path, image_list=image_list)  # 6000
    print(metric)


