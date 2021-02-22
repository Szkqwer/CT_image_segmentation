import os
import numpy as np
import cv2
import torch

from models.UNet3P_Series import UNet3P, DeepSup_UNet3P, DeepSup_ResUNet3P, DeepSup_AR2UNet3P, DeepSup_Res2UNet3P, DeepSup_Res2XUNet3P, DeepSup_CGM_UNet3P


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


# # 计算综合评分
# def calculate_score(disease_place, label_disease):
#     score = (calculate_dice(disease_place, label_disease) + calculate_mPA(disease_place, label_disease)) / 2
#
#     return score


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
def evaluate_model(model, model_path, device, csv_root, picture_root, score_root, width=256, height=256):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('load model over')

    model = model.to(device)
    model.eval()

    model_name = model.__class__.__name__

    input_root_list = []
    mask_root_list = []
    f = open(csv_root, 'r')
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


# 0.514780701314058 0.7090536218407361 0.6119171615773968
# 0.6319317774189223 0.7619652029576374 0.6969484901882798

# 0.8167270655469886 0.9181498657464561 0.8674384656467223
# mean_dice: 0.9441329243881572 mean_mPA: 0.970669604123489 mean_score: 0.957401264255823
# mean_dice: 0.6147382262603662 mean_mPA: 0.7816819053782629 mean_score: 0.6982100658193146
# mean_dice: 0.7317746759935629 mean_mPA: 0.8606408418849936 mean_score: 0.7962077589392782

if __name__ == '__main__':
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    # 定义各种路径
    picture_root = r'./results/picture'
    score_root = r'./results/score'
    csv_root = r'./csv_data/thrombus_test_data.csv'

    # 定义图片宽高
    width = 256
    height = 256

    # 基本unet3+
    model = UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = r'checkpoints/UNet3P.pth'
    evaluate_model(model, model_path, device, csv_root, picture_root, score_root, width=width, height=height)

    # 使用论文loss和模型
    # cgm
    # model = DeepSup_CGM_UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_CGM_UNet3P.pth'
    # evaluate_model(model, model_path, device, csv_root, picture_root, score_root, width=width, height=height)

    # dsp
    # model = DeepSup_UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_UNet3P.pth'
    # evaluate_model(model, model_path, device, csv_root, picture_root, score_root, width=width, height=height)

    # 使用自定义模型
    # res
    model = DeepSup_ResUNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = r'./checkpoints/DeepSup_ResUNet3P.pth'
    evaluate_model(model, model_path, device, csv_root, picture_root, score_root, width=width, height=height)

    # res2
    model = DeepSup_Res2UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = r'./checkpoints/DeepSup_Res2UNet3P.pth'
    evaluate_model(model, model_path, device, csv_root, picture_root, score_root, width=width, height=height)

    # AR2UNet3P
    model = DeepSup_AR2UNet3P(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = r'checkpoints/DeepSup_AR2UNet3P.pth'
    evaluate_model(model, model_path, device, csv_root, picture_root, score_root, width=width, height=height)
