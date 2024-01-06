import torch
from torch import nn
import numpy as np
import gudhi as gd

class TCloss(nn.Module):
    def __init__(self):
        super(TCloss, self).__init__()

    def hausdorff_distance(self, A, B):
        # 将numpy.ndarray转换为Tensor
        A_tensor = torch.from_numpy(A)
        B_tensor = torch.from_numpy(B)

        # 计算 A 到 B 的最小距离
        if A_tensor.numel() > 0 and B_tensor.numel() > 0:
            min_distances_A_to_B = torch.cdist(A_tensor, B_tensor).min(dim=1)[0]
        else:
            min_distances_A_to_B = torch.tensor([0.0])

        # 计算 B 到 A 的最小距离
        if B_tensor.numel() > 0 and A_tensor.numel() > 0:
            min_distances_B_to_A = torch.cdist(B_tensor, A_tensor).min(dim=1)[0]
        else:
            min_distances_B_to_A = torch.tensor([0.0])

        # 计算Hausdorff距离
        if A_tensor.numel() > 0 or B_tensor.numel() > 0:
            hausdorff_distance = torch.max(min_distances_A_to_B.max(), min_distances_B_to_A.max())
        else:
            hausdorff_distance = torch.tensor([0.0])

        return hausdorff_distance.item()


    def CEloss(self, y_true, y_pred):
        smooth = 1e-6
        y_pred = torch.clamp(y_pred, smooth, 1.0 - smooth)
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
