import os
import numpy as np
import cv2
import torch

from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_Res


def evaluate_model_baseline(model, device, image_root, result_root, width=128, height=128):
    model = model.to(device)
    model.eval()
    path_list = os.listdir(image_root)
    # 遍历所有测试图片
    for path in path_list:
        # 将图片转换为合适的输入
        input_img = cv2.imread(image_root + '/' + path)
        input_img = cv2.resize(input_img, (width, height))
        input_img = input_img.transpose((2, 0, 1)) / 255.0

        # 获取输出
        input_tensor = torch.Tensor([input_img]).to(device)
        outputs = model(input_tensor).detach()
        outputs = np.array(outputs.cpu())

        # 转换回图片样式
        result = outputs[0].transpose((1, 2, 0))
        # 获取分类
        result = np.argmax(result, axis=2)

        result_img = result * 255
        result_img = result_img.astype(np.uint8)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(result_root + '/result_' + path, result_img.astype(np.uint8))


def evaluate_model(model, device, image_root, result_root, label_root, width=128, height=128):
    model = model.to(device)
    model.eval()
    input_img_list = os.listdir(image_root)
    mask_img_list = os.listdir(label_root)
    # 遍历所有测试图片
    for input_img_index in range(0,len(input_img_list)):

        # 将图片转换为合适的输入
        input_img = cv2.imread(image_root + '/' + input_img_list[input_img_index])
        input_img = cv2.resize(input_img, (width, height))
        input_img = input_img.transpose((2, 0, 1)) / 255.0

        mask_img = cv2.imread(label_root + '/' + mask_img_list[input_img_index])
        mask_img = cv2.resize(mask_img, (width, height))

        # 获取输出
        input_tensor = torch.Tensor([input_img]).to(device)
        outputs = model(input_tensor)
        for output_index in range(0, len(outputs)):
            output = np.array(outputs[output_index].detach().cpu())

            # 转换回图片样式
            result = output[0].transpose((1, 2, 0))
            # 获取分类
            disease_place = (result > 0.5).all(axis=2)

            label_disease = (mask_img == 255).all(axis=2)

            result[disease_place] = 255
            result[~disease_place] = 0
            result[label_disease] = 127
            # result=np.argmax(result, axis=2)
            #
            # result_img=result*255
            # result_img = result_img
            result_img = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.imwrite(result_root + '/result_' + str(output_index) + '_' + input_img_list[input_img_index], result_img.astype(np.uint8))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行

    # image_root=r'./test_images'
    image_root = r'F:/dataset/medical/CT/image/chenxianhong-right'
    label_root = r'F:/dataset/medical/CT/mask/chenxianhong-right'
    result_root = r'./results'

    # 测试深监督unet3+
    # model = UNet_3Plus_DeepSup(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # model_path = r'./checkpoints/DeepSup_model.pth'
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    # evaluate_model(model, device, image_root, result_root)

    # 使用自定义模型
    model = UNet_3Plus_DeepSup_Res(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_path = r'./checkpoints/DeepSup_Res_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    width = 256
    height = 256
    evaluate_model(model, device, image_root, result_root, label_root=label_root, width=width, height=height)
