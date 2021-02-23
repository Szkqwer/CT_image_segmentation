import os
import numpy as np
import cv2
import torch

from models.UNet3P_Series import UNet3P, DeepSupUNet3P, DeepSupResUNet3P, DeepSupAR2UNet3P, DeepSupRes2UNet3P, DeepSupRes2XUNet3P, DeepSupCGMUNet3P


# 计算dice
def calculate_dice(disease_place, label_disease):
    # 用乘法求交集
    TP_place = disease_place & label_disease
    # 准确预测数
    share_num = TP_place.sum()
    # 数量和
    # 加小尾缀防止0/0
    suffix_data = 1e-10
    union_num = disease_place.sum() + label_disease.sum()
    dice = (2 * share_num + suffix_data) / (union_num + suffix_data)
    return dice


# 计算mPA
def calculate_mPA(disease_place, label_disease):
    # 加小尾缀防止0/0
    suffix_data = 1e-10
    # 求病灶区域PA
    TP_place = disease_place & label_disease
    diease_PA = (TP_place.sum() + suffix_data) / (label_disease.sum() + suffix_data)

    # 非病灶区域PA
    not_diease_place = ~label_disease
    TP_place = (~disease_place) & not_diease_place
    other_PA = (TP_place.sum() + suffix_data) / (not_diease_place.sum() + suffix_data)

    mPA = (diease_PA + other_PA) / 2
    return mPA


# 计算评分并保存图片
def get_result(outputs, label_disease, input_img_name, picture_root, model_name):
    result_dice = 0
    result_mPA = 0
    result_score = 0

    if type(outputs) == tuple:
        for output_index in range(0, len(outputs)):
            output = np.array(outputs[output_index].detach().cpu())

            # 转换回图片样式
            result = output[0].transpose((1, 2, 0))
            # 获取分类
            disease_place = (result > 0.5).all(axis=2)

            # 设置颜色
            result[~disease_place] = 0
            result[disease_place] += 127

            result[label_disease] += 63

            # 计算主分支评分
            if output_index == 0:
                dice = calculate_dice(disease_place, label_disease)
                mPA = calculate_mPA(disease_place, label_disease)
                score = (dice + mPA) / 2
                result_dice = dice
                result_mPA = mPA
                result_score = score

                result_img = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                cv2.imwrite(picture_root + '/' + model_name + '_' + str(output_index) + '_' + input_img_name, result_img.astype(np.uint8))
    else:
        # 转换回图片样式
        output = np.array(outputs[0].detach().cpu())
        result = output.transpose((1, 2, 0))
        # 获取分类
        disease_place = (result > 0.5).all(axis=2)

        # 设置颜色
        result[~disease_place] = 0
        result[disease_place] += 127

        result[label_disease] += 63

        # 计算主分支评分

        dice = calculate_dice(disease_place, label_disease)
        mPA = calculate_mPA(disease_place, label_disease)
        score = (dice + mPA) / 2
        result_dice = dice
        result_mPA = mPA
        result_score = score

        result_img = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(picture_root + '/' + model_name + '_' + input_img_name, result_img.astype(np.uint8))

    return result_dice, result_mPA, result_score


# 评估模型
def evaluate_model(model, model_path, device, csv_path, picture_root, score_root, width=256, height=256):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('load model over')

    model = model.to(device)
    model.eval()

    model_name = model.__class__.__name__

    input_root_list = []
    mask_root_list = []
    f = open(csv_path, 'r')
    all_data = f.readlines()
    for data in all_data:
        temp = data.split('\n')[0]
        temp = temp.split(',')
        input_root_list.append(temp[0])
        mask_root_list.append(temp[1])

    all_score = []
    all_dice = []
    all_mPA = []

    # 遍历所有测试图片
    for input_root_index in range(0, len(input_root_list)):
        # 将图片转换为合适的输入
        input_img_name = input_root_list[input_root_index].split('\\')[-1]
        input_img = cv2.imread(input_root_list[input_root_index])
        input_img = cv2.resize(input_img, (width, height))
        input_img = input_img.transpose((2, 0, 1)) / 255.0

        # 转换标签
        mask_img = cv2.imread(mask_root_list[input_root_index])
        mask_img = cv2.resize(mask_img, (width, height))
        label_disease = (mask_img == 255).all(axis=2)

        # 获取所有分支输出
        input_tensor = torch.Tensor([input_img]).to(device)
        outputs = model(input_tensor)

        # 获取结果并保存图片
        dice, mPA, score = get_result(outputs, label_disease, input_img_name, picture_root, model_name)

        all_dice.append(dice)
        all_mPA.append(mPA)
        all_score.append(score)

    # 在测试集上的平均评分
    mean_dice = sum(all_dice) / len(all_dice)
    mean_mPA = sum(all_mPA) / len(all_mPA)
    mean_score = sum(all_score) / len(all_score)
    print(model.__class__.__name__)
    print('mean_dice:', mean_dice, 'mean_mPA:', mean_mPA, 'mean_score:', mean_score)

    score_path = score_root + '/' + model_name + '.txt'
    f = open(score_path, 'w')
    f.write(str(mean_dice) + ',' + str(mean_mPA) + ',' + str(mean_score))
    f.close()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    # 定义各种路径
    # 数据集
    dataset_dict = {'tumor': r'./csv_data/tumor_test_data.csv.csv', 'thrombus': r'./csv_data/thrombus_test_data.csv'}
    dataset_name = 'thrombus'
    csv_path = dataset_dict[dataset_name]

    # 结果位置
    picture_root = r'./results/picture_' + dataset_name
    score_root = r'./results/score_' + dataset_name
    if not os.path.exists(picture_root):
        os.makedirs(picture_root)
    if not os.path.exists(score_root):
        os.makedirs(score_root)

    # 模型位置
    checkpoint_folder = r'./checkpoints'

    # 定义图片宽高
    width = 256
    height = 256

    # 基本unet3+
    model = UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    evaluate_model(model, model_path, device, csv_path, picture_root, score_root, width=width, height=height)

    # 使用论文loss和模型
    # cgm
    # model = DeepSupCGMUNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    # evaluate_model(model, model_path, device, csv_path, picture_root, score_root, width=width, height=height)

    # dsp
    model = DeepSupUNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    evaluate_model(model, model_path, device, csv_path, picture_root, score_root, width=width, height=height)

    # 使用自定义模型
    # res
    model = DeepSupResUNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    evaluate_model(model, model_path, device, csv_path, picture_root, score_root, width=width, height=height)

    # res2
    model = DeepSupRes2UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    evaluate_model(model, model_path, device, csv_path, picture_root, score_root, width=width, height=height)

    # AR2UNet3P
    model = DeepSupAR2UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = checkpoint_folder + '/' + model.__class__.__name__ + '_' + dataset_name + '.pth'
    evaluate_model(model, model_path, device, csv_path, picture_root, score_root, width=width, height=height)
