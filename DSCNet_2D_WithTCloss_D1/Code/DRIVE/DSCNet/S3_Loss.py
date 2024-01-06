# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import gudhi as gd


class TCloss(nn.Module):
    def __init__(self):
        super(TCloss, self).__init__()

    def hausdorff_distance(self, dgm1, dgm2):
        # 计算两个持续图之间的Hausdorff距离
        dgm1 = np.array(dgm1)
        dgm2 = np.array(dgm2)
        if len(dgm1) == 0 or len(dgm2) == 0:
            # 如果一个持续图为空，则距离为0
            return 0
        # 计算所有点对之间的距离
        return gd.bottleneck_distance(dgm1, dgm2)

    def CEloss(self, y_true, y_pred):
        smooth = 1e-6
        y_pred = torch.clamp(y_pred, smooth, 1.0 - smooth)  # 防止log(0)的情况
        loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return loss
    
    def forward(self, y_true, y_pred):
        # 确保输入在CPU上，并转换为numpy数组
        y_true_np = y_true.detach().cpu().numpy().squeeze()
        y_pred_np = y_pred.detach().cpu().numpy().squeeze()

        # 这里简化处理，将输入图像转换为点集
        points_true = np.column_stack(np.where(y_true_np > 0.5))
        points_pred = np.column_stack(np.where(y_pred_np > 0.5))

        # 创建Rips复形并计算持续同调
        rips_complex_true = gd.RipsComplex(points=points_true, max_edge_length=2)
        simplex_tree_true = rips_complex_true.create_simplex_tree(max_dimension=2)
        persistence_true = simplex_tree_true.persistence()

        rips_complex_pred = gd.RipsComplex(points=points_pred, max_edge_length=2)
        simplex_tree_pred = rips_complex_pred.create_simplex_tree(max_dimension=2)
        persistence_pred = simplex_tree_pred.persistence()

        # 将持续同调特征转换为持续图格式
        dgm1 = simplex_tree_true.persistence_intervals_in_dimension(0)
        dgm2 = simplex_tree_pred.persistence_intervals_in_dimension(0)

        # 计算Hausdorff距离
        dH = self.hausdorff_distance(dgm1, dgm2)

        # 计算TCloss
        TC_loss = dH + self.CEloss(y_true, y_pred)

        return TC_loss
