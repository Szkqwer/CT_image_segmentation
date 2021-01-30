import os
import torch.nn.functional as F
import cv2
import torch
from torchsummary import summary
from torch.utils.data import DataLoader

from data_utils import CTDataset
from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU_loss
from loss.msssimLoss import MSSSIM_loss
from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup_CGM, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_Res, UNet_3Plus_DeepSup_Res2, UNet_3Plus_DeepSup_Res2x, \
    UNet_3Plus_DeepSup_Attention_Res2
import numpy as np


# 交叉熵训练基本unet3+
def train_baseline(input_model, input_device, loss_fun, model_path, lr=5e-4, batch_size=11, epoch=200, width=128, height=128):
    # 加载各种数据
    input_model = input_model.to(input_device)
    # summary(model, (3,height,width))

    input_model.train()
    # 数据集
    dataset = CTDataset('./train_data/thrombus_train_data.csv', width, height, False)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # 定义模型参数
    optimizer = torch.optim.Adam(input_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.SGD(input_model.parameters(), lr=lr, momentum=0.3)
    criterion = loss_fun

    # 训练epoch轮
    for train_round in range(0, epoch):
        batch_loss = []
        print('train round:', train_round)
        for input_images, masks in train_loader:
            # 预处理数据
            input_images = torch.tensor(input_images, dtype=torch.float)
            input_images = input_images.to(input_device)

            # masks.type(torch.FloatTensor)
            masks = torch.tensor(masks, dtype=torch.long)
            masks = masks.to(input_device)

            # 梯度置零
            optimizer.zero_grad()
            # 模型输出
            outputs = input_model(input_images)
            # 计算loss
            loss = criterion(outputs, masks)
            # loss反向传播
            loss.backward()
            # 反向传播后参数更新
            optimizer.step()
            batch_loss.append(loss.item())
        print('Epoch loss:', str(np.mean(batch_loss)))
        # print(loss)

        # 保存模型
        torch.save(input_model.state_dict(), model_path)
        print('round train over')


# 训练复杂unet3+
def train(input_model, input_device, loss_fun, model_path, lr=1e-3, batch_size=3, epoch=400, width=256, height=256):
    # 加载各种数据
    input_model = input_model.to(input_device)
    # summary(model, (3,height,width))

    input_model.train()
    # 数据集
    dataset = CTDataset(r'./train_data/thrombus_train_data.csv', width, height, False)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # 定义模型参数
    optimizer = torch.optim.Adam(input_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.SGD(input_model.parameters(), lr=lr, momentum=0.3)
    criterion = loss_fun

    # 训练epoch轮
    for train_round in range(0, epoch):
        batch_loss = []
        print('train round:', train_round)
        for input_images, masks in train_loader:
            # 预处理数据
            input_images = torch.tensor(input_images, dtype=torch.float)
            input_images = input_images.to(input_device)

            # masks.type(torch.FloatTensor)
            masks = torch.tensor(masks, dtype=torch.float)
            masks = masks.to(input_device)

            # 梯度置零
            optimizer.zero_grad()
            # 模型输出
            outputs = input_model(input_images)
            # 计算loss
            loss = criterion(outputs, masks)
            # loss反向传播
            loss.backward()
            # 反向传播后参数更新
            optimizer.step()
            batch_loss.append(loss.item())
        print('Epoch loss:', str(np.mean(batch_loss)))
        # print(loss)

        # 保存模型
        torch.save(input_model.state_dict(), model_path)
        print('round train over')


def loss_fun(pred_all, label):
    loss = 0
    for pred in pred_all:
        loss += IOU_loss(pred, label) + BCE_loss(pred, label) + MSSSIM_loss(pred, label)
        # loss += BCE_loss(pred, label) + MSSSIM_loss(pred, label)
        # loss+=IOU_loss(pred, label)

    return loss / len(pred_all)


def loss_fun_iou(pred_all, label):
    loss = 0
    for pred in pred_all:
        loss += IOU_loss(pred, label)

    return loss / len(pred_all)


def loss_fun_2(pred_all, label):
    loss = 0
    for pred in pred_all:
        loss += BCE_loss(pred, label) + MSSSIM_loss(pred, label)

    return loss / len(pred_all)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行

    # 使用交叉熵作为损失函数
    # model_CELoss = UNet_3Plus(in_channels=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # CELoss_model_path = r'./checkpoints/model_CELoss.pth'
    # criterion = torch.nn.CrossEntropyLoss()
    # if os.path.exists(CELoss_model_path):
    #     model_CELoss.load_state_dict(torch.load(CELoss_model_path))
    # train_baseline(model_CELoss, device,criterion, CELoss_model_path)

    # 使用论文loss和模型
    # # cgm
    # model = UNet_3Plus_DeepSup_CGM(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_CGM_model.pth'
    # # dsp
    # model = UNet_3Plus_DeepSup(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_model.pth'

    # 使用自定义模型
    # res
    # model = UNet_3Plus_DeepSup_Res(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_Res_model.pth'

    # res2 #256好像大了
    # model = UNet_3Plus_DeepSup_Res2(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_Res2_model.pth'

    # res2next #256好像大了
    # model = UNet_3Plus_DeepSup_Res2x(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_Res2x_model.pth'

    # res2加入attention
    model = UNet_3Plus_DeepSup_Attention_Res2(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = r'./checkpoints/DeepSup_Attention_Res2_model.pth'

    # 加载数据
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    # 定义损失函数等信息
    lr = 1e-3
    batch_size = 1
    epoch = 400
    width = 128
    height = 128
    # criterion = loss_fun_2
    # train(model, device, criterion, model_path, lr=lr, epoch=epoch, width=width, height=height)

    lr = 1e-5
    criterion = loss_fun_iou
    train(model, device, criterion, model_path, lr=lr, epoch=epoch, width=width, height=height)