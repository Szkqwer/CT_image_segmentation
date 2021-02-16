import os
import numpy as np
import cv2
import torch

from models.UNet3P_Series import UNet3P, DeepSup_UNet3P, DeepSup_ResUNet3P, DeepSup_AR2UNet3P, DeepSup_Res2UNet3P


# 计算dice
def calculate_dice(disease_place, label_disease):
    # 用乘法求交集
    TP_place = disease_place & label_disease
    # 准确预测数
    share_num = TP_place.sum()
    # 数量和
    union_num = disease_place.sum() + label_disease.sum()
    dice = (2 * share_num) / union_num
    return dice


# 计算mPA
def calculate_mPA(disease_place, label_disease):
    # 求病灶区域PA
    TP_place = disease_place & label_disease
    diease_PA = TP_place.sum() / label_disease.sum()

    # 非病灶区域PA
    not_diease_place = ~label_disease
    TP_place = (~disease_place) & not_diease_place
    other_PA = TP_place.sum() / not_diease_place.sum()

    mPA = (diease_PA + other_PA) / 2
    return mPA


# 计算综合评分
def calculate_score(disease_place, label_disease):
    score = (calculate_dice(disease_place, label_disease) + calculate_mPA(disease_place, label_disease)) / 2

    return score


def evaluate_model_baseline(model, device, image_root, result_root, label_root, width=256, height=256):
    model = model.to(device)
    model.eval()
    input_img_list = os.listdir(image_root)
    mask_img_list = os.listdir(label_root)
    # 遍历所有测试图片
    for input_img_index in range(0, len(input_img_list)):
        # 将图片转换为合适的输入
        input_img = cv2.imread(image_root + '/' + input_img_list[input_img_index])
        input_img = cv2.resize(input_img, (width, height))
        input_img = input_img.transpose((2, 0, 1)) / 255.0

        mask_img = cv2.imread(label_root + '/' + mask_img_list[input_img_index])
        mask_img = cv2.resize(mask_img, (width, height))
        label_disease = (mask_img == 255).all(axis=2)

        # 获取输出
        input_tensor = torch.Tensor([input_img]).to(device)
        outputs = model(input_tensor).detach()
        outputs = np.array(outputs.cpu())

        # 转换回图片样式
        result = outputs[0].transpose((1, 2, 0))
        # 获取分类
        result = np.argmax(result, axis=2)
        disease_place = (result == 1).all(axis=2)
        result[disease_place] = 255
        result[~disease_place] = 0
        result[label_disease] = 127

        # result_img = result * 255
        # result[label_disease] = 127
        result_img = result.astype(np.uint8)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(result_root + '/result_' + input_img_list[input_img_index], result_img.astype(np.uint8))


def evaluate_model(model, device, image_root, result_root, label_root, width=256, height=256):
    model = model.to(device)
    model.eval()
    input_img_list = os.listdir(image_root)
    mask_img_list = os.listdir(label_root)
    all_score = []
    all_dice = []
    all_mPA = []
    # 遍历所有测试图片
    for input_img_index in range(0, len(input_img_list)):

        # 将图片转换为合适的输入
        input_img = cv2.imread(image_root + '/' + input_img_list[input_img_index])
        input_img = cv2.resize(input_img, (width, height))
        input_img = input_img.transpose((2, 0, 1)) / 255.0

        # 转换标签
        mask_img = cv2.imread(label_root + '/' + mask_img_list[input_img_index])
        mask_img = cv2.resize(mask_img, (width, height))
        label_disease = (mask_img == 255).all(axis=2)

        # 获取所有分支输出
        input_tensor = torch.Tensor([input_img]).to(device)
        outputs = model(input_tensor)
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
                score = (dice+mPA)/2
                all_dice.append(dice)
                all_mPA.append(mPA)
                all_score.append(score)

            result_img = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.imwrite(result_root + '/result_' + str(output_index) + '_' + input_img_list[input_img_index], result_img.astype(np.uint8))

    # 在测试集上的平均评分
    mean_dice = sum(all_dice) / len(all_dice)
    mean_mPA = sum(all_mPA) / len(all_mPA)
    mean_score = sum(all_score) / len(all_score)
    print(mean_dice, mean_mPA, mean_score)
# 0.514780701314058 0.7090536218407361 0.6119171615773968
# 0.6319317774189223 0.7619652029576374 0.6969484901882798

# 0.0273972602739726 0.5074626865671642 0.2674299734205684



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # image_root=r'./test_images'
    image_root = r'F:/dataset/medical/thrombus/image/chenxianhong-right'
    label_root = r'F:/dataset/medical/thrombus/mask/chenxianhong-right'
    result_root = r'./results'

    # 测试深监督unet3+
    # model = UNet_3Plus_DeepSup(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_model.pth'
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    # evaluate_model(model, device, image_root, result_root)

    # 使用自定义模型
    # model = DeepSup_ResUNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_ResUNet3P.pth'

    model = DeepSup_AR2UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = r'checkpoints/DeepSup_AR2UNet3P.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('load model over')
    width = 256
    height = 256
    evaluate_model(model, device, image_root, result_root, label_root=label_root, width=width, height=height)

