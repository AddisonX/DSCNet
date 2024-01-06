# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import join
import numpy as np
import SimpleITK as sitk

"""
The purpose of this code is to calculate the "mean" and "std" of the image, 
which will be used in the subsequent normalization process

Take the image ending with "nii.gz" as an example (using SimpleITK)
"""


def Getmeanstd(args, image_path, meanstd_name):
    """
    :param args: Parameters
    :param image_path: Address of image
    :param meanstd_name: save name of "mean" and "std"  (using ".npy" format to save)
    :return: None
    """
    root_dir = args.root_dir
    file_names = [f for f in os.listdir(image_path) if f.lower().endswith(('.nii', '.nii.gz'))]  # 确保只处理 NIfTI 图像
    mean, std, length = 0.0, 0.0, 0.0

    for file_name in file_names:
        full_path = os.path.join(image_path, file_name)  # 使用 os.path.join
        if os.path.isdir(full_path):  # 检查这是否是一个目录
            continue  # 如果是目录，跳过
        image = sitk.ReadImage(full_path)  # 现在这里使用完整路径
        image = sitk.GetArrayFromImage(image).astype(np.float32)
        length += image.size
        mean += np.sum(image)

    mean = mean / length

    for file_name in file_names:
        full_path = os.path.join(image_path, file_name)  # 使用 os.path.join
        if os.path.isdir(full_path):  # 检查这是否是一个目录
            continue  # 如果是目录，跳过
        image = sitk.ReadImage(full_path)  # 现在这里使用完整路径
        image = sitk.GetArrayFromImage(image).astype(np.float32)
        std += np.sum(np.square((image - mean)))

    std = np.sqrt(std / length)
    print("1 Finish Getmeanstd: ", meanstd_name)
    print("Mean and std are: ", mean, std)
    np.save(os.path.join(root_dir, meanstd_name), [mean, std])  # 保存 mean 和 std
