import argparse
import os
import torch
from torchsummary import summary
from torch.utils.data import DataLoader

from data_utils import CTDataset
from loss.iouLoss import IOU_loss
from loss.mixLoss import MixLoss

from models.UNet3P_Series import UNet3P, DeepSupCGMUNet3P, DeepSupUNet3P, DeepSupResUNet3P, DeepSupRes2UNet3P, DeepSupRes2XUNet3P, DeepSupAR2UNet3P

import numpy as np


# 训练其他损失函数的改进unet3+
def train(input_model, input_device, loss_fun, model_path, csv_path, lr=1e-3, batch_size=3, epoch=400, width=256, height=256, beta=0.1, dec_epoch=10, dec_rate=0.9,
          save_epoch=5):
    input_model = input_model.to(input_device)

    input_model.train()
    # 数据集
    dataset = CTDataset(csv_path, width, height)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    # 定义模型参数
    optimizer = torch.optim.Adam(input_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = loss_fun

    # 训练epoch轮
    for train_round in range(0, epoch):
        all_loss = []
        print('train round:', train_round)

        for input_images, masks in train_loader:
            # 对有病灶图片训练
            # if have_diease.item():
            # 预处理数据
            input_images = input_images.to(input_device)

            masks = masks.to(input_device)

            # 梯度置零
            optimizer.zero_grad()
            # 模型输出
            outputs = input_model(input_images)

            # 计算loss
            loss = criterion(outputs, masks, beta)
            # loss反向传播
            loss.backward()
            # 反向传播后参数更新
            optimizer.step()
            all_loss.append(loss.item())

        print('mean epoch loss:', str(np.mean(all_loss)))

        # 降低beta
        if train_round % dec_epoch == dec_epoch - 1:
            beta *= dec_rate
            print('decrease beta to', beta)

        # 保存模型
        if train_round % save_epoch == save_epoch - 1:
            torch.save(input_model.state_dict(), model_path)
            print('save model over')
        print('round train over')
        print('')
    return input_model, beta


# 使用分段函数训练
def step_train(input_model, input_device, model_path, csv_path, batch_size=3, epoch=400, width=256, height=256):
    input_model = input_model.to(input_device)
    # summary(model, (3,height,width))

    # 加载各模型数据
    if os.path.exists(model_path):
        input_model.load_state_dict(torch.load(model_path))
        print('load', input_model.__class__.__name__, 'over')

    # 初始化beta
    # beta = 0.7
    beta = 0.1
    # 定义beta降低速度和轮数
    dec_epoch = 5
    dec_rate = 0.98
    # 保存间隔轮数
    save_epoch = 1

    # 第一步训练
    lr = 1e-4
    gama_list = [0.5, 0.5, 0]
    criterion = MixLoss(gama_list)
    input_model, beta = train(input_model, input_device, criterion, model_path, csv_path, lr=lr, batch_size=batch_size, epoch=epoch, width=width, height=height,
                              beta=beta, dec_epoch=dec_epoch, dec_rate=dec_rate, save_epoch=save_epoch)

    # 第二步训练
    # lr = 1e-6
    # gama_list = [0, 0, 1]
    # criterion = MixLoss(gama_list)
    # input_model, beta = train(input_model, input_device, criterion, model_path, csv_path, lr=lr, batch_size=batch_size, epoch=epoch, width=width, height=height,
    #                           beta=beta, dec_epoch=dec_epoch, dec_rate=dec_rate, save_epoch=save_epoch)

    # 最终保存
    torch.save(input_model.state_dict(), model_path)
    print('save model over')


if __name__ == '__main__':
    # 定义基本数据
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行

    # 定义数据集名字
    dataset_dict = {'tumor': r'./csv_data/tumor_train_data.csv', 'thrombus': r'./csv_data/thrombus_train_data.csv'}
    # dataset_name = 'thrombus'
    dataset_name = 'tumor'
    csv_path = dataset_dict[dataset_name]
    checkpoint_folder = r'./checkpoints'
    # csv_path = r'./csv_data/thrombus_train_data.csv'

    # 训练数据
    epoch = 400
    width = 256
    height = 256

    # 基本unet3+
    # batch_size = 3
    # model = UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    # step_train(model, device, model_path,csv_path, batch_size=batch_size, epoch=epoch, width=width, height=height)

    # 使用论文loss和模型
    # cgm
    # batch_size = 3
    # model = DeepSupCGMUNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path=checkpoint_folder+'/'+model.__class__.__name__+'_'+dataset_name+'.pth'
    # step_train(model, device, model_path,csv_path, batch_size=batch_size, epoch=epoch, width=width, height=height)

    # dsp
    # batch_size = 3
    # model = DeepSupUNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    # step_train(model, device, model_path, csv_path, batch_size=batch_size, epoch=epoch, width=width, height=height)

    # 使用自定义模型
    # res
    # batch_size = 3
    # model = DeepSupResUNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path=checkpoint_folder+'/'+model.__class__.__name__+'_'+dataset_name+'.pth'
    # step_train(model, device, model_path,csv_path, batch_size=batch_size, epoch=epoch, width=width, height=height)

    # res2
    # batch_size = 2
    # model = DeepSupRes2UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path=checkpoint_folder+'/'+model.__class__.__name__+'_'+dataset_name+'.pth'
    # step_train(model, device, model_path,csv_path, batch_size=batch_size, epoch=epoch, width=width, height=height)

    # AR2UNet3P
    batch_size = 1
    model = DeepSupAR2UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    step_train(model, device, model_path, csv_path, batch_size=batch_size, epoch=epoch, width=width, height=height)
