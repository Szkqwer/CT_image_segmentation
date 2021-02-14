import torch

from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU_loss
from loss.msssimLoss import MSSSIM_loss


# 深监督分支递减权重混合loss，beta为递减率
class MixLoss(torch.nn.Module):

    def __init__(self, gama_list, size_average=True):
        super(MixLoss, self).__init__()
        # 设置各种loss权重
        self.BCE_weight = gama_list[0]
        self.MSSSIM_weight = gama_list[0]
        self.IOU_weight = gama_list[0]

        self.size_average = size_average

    def forward(self, pred_all, label, beta):
        # 总损失
        loss = 0
        # 分支权重
        branch_weight = 1
        # 权重和
        sum_weight = 0

        for pred in pred_all:
            loss += (self.BCE_weight * BCE_loss(pred, label) + self.MSSSIM_weight * MSSSIM_loss(pred, label) + self.IOU_weight * IOU_loss(pred, label)) * branch_weight
            sum_weight += branch_weight

            # 分支权重递减
            branch_weight *= beta

        loss /= sum_weight
        return loss
