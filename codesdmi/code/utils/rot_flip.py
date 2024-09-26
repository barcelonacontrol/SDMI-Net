import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import cv2 # 转换成图
import torch

def rot_num_flip(t, num_rot, is_flip):
    if is_flip:
        t = torch.flip(t, dims=(2, 3))

    for i in range(num_rot):
        t = torch.rot90(t, 1, dims=(2, 3))

    return t, [num_rot, is_flip]

def get_origin(t, opt):
    num_rot = opt[0]
    is_flip = opt[1]
    for i in range(num_rot):
        t = torch.rot90(t, -1, dims=(2, 3))

    if is_flip:
        t = torch.flip(t, dims=(2, 3))

    return t
"""

bs = rot_num_flip(bs, 0, False)
bs_rot1 = rot_num_flip(bs, 1, False)
bs_rot2 = rot_num_flip(bs, 2, False)
bs_rot3 = rot_num_flip(bs, 3, False)

bs_f = rot_num_flip(bs, 0 ,True)
bs_rot1_f = rot_num_flip(bs, 1 ,True)
bs_rot2_f = rot_num_flip(bs, 2 ,True)
bs_rot3_f = rot_num_flip(bs, 3 ,True)

"""
def nii_to_image_single(filepath):

    img = nib.load(filepath)  # 读取nii

    img_fdata = img.get_fdata()

    test_num = img_fdata[img_fdata > 0]
    test_num[test_num > 0] = 1

    print(test_num[test_num > 0].sum())

    bs = torch.from_numpy(img_fdata)
    bs = bs.unsqueeze(0)
    bs = torch.stack([bs, bs])
    print(bs.shape)

    bs, opt1 = rot_num_flip(bs, 0, False)
    bs_rot1, opt2 = rot_num_flip(bs, 1, False)
    bs_rot2, opt3 = rot_num_flip(bs, 2, False)
    bs_rot3, opt4 = rot_num_flip(bs, 3, False)

    bs_f, opt5 = rot_num_flip(bs, 0, True)
    bs_rot2_f, opt6 = rot_num_flip(bs, 1, True)
    bs_rot3_f, opt7 = rot_num_flip(bs, 2, True)
    bs_rot3_f, opt8 = rot_num_flip(bs, 3, True)
    print((bs_rot1 == bs_f).sum())
    bs_rot1 = get_origin(bs_rot1, opt2)
    bs_f = get_origin(bs_f, opt5)
    print((bs_f == bs).sum())



if __name__ == '__main__':
    nii_to_image_single(r'./data/BraTS19_2013_0_1_t2.nii.gz')

