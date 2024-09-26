import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import cv2 # 转换成图


def nii_to_image(niifile):
    filenames = os.listdir(filepath)  # 读取nii文件夹
    slice_trans = []

    for f in filenames:
        # 开始读取nii文件
        if f[0] == '.': continue
        for i in os.listdir(filepath+f):
            if i[0]=='.' : continue
            if i.split("_")[-1]!="seg.nii" : continue
            img_path = os.path.join(filepath, f, i)
            img = nib.load(img_path)  # 读取nii

            img_fdata = img.get_fdata()
            # img_fdata = img_fdata % 1
            # img_fdata = np.mod(img_fdata, 1)
            fname = i.replace('.nii', '')  # 去掉nii的后缀名

            # img_f_path = os.path.join(imgfile, fname)
            # os.mkdir(img_f_path)
            # 创建nii对应的图像的文件夹
            img_3d_max = np.max(img_fdata)
            img_fdata = img_fdata > 0
            # img_fdata = img_fdata / img_3d_max
            # img_fdata = np.rot90(img_fdata,k=-1,axes=(0,1))
            img_fdata = img_fdata * 255.
            # img_fdata = cv2.flip(img_fdata,1)
            # 开始转换为图像
            (x, y, z) = img.shape
            for p in range(z):  # z是图像的序列
                silce = img_fdata[:,:,p]  # 选择哪个方向的切片都可以
                cv2.imwrite(imgfile + fname+"_"+str(p)+".png", silce)
                # 保存图像


if __name__ == '__main__':
    filepath = 'end_nii/'
    imgfile = 'mask/'
    nii_to_image(filepath)

