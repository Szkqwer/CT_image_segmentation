import random

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2


# 将图片路径写入csv
def write_root(picture_root, mask_root, csv_root):
    # 输入图像路径
    p_root_list = []
    for p_root, p_dirs, file_names in os.walk(picture_root):
        for file_name in file_names:
            p_path = os.path.join(p_root, file_name)
            p_root_list.append(p_path)

    # ground truth路径
    gt_root_list = []
    for gt_root, gt_dirs, file_names in os.walk(mask_root):
        for file_name in file_names:
            gt_path = os.path.join(gt_root, file_name)
            gt_root_list.append(gt_path)

    # 写入
    f = open(csv_root, 'w')
    for i in range(0, len(gt_root_list)):
        str_w = p_root_list[i] + ',' + gt_root_list[i] + '\n'
        f.write(str_w)
    f.close()


# 点(x,y) 绕(cx,cy)点旋转
# def rotate_xy(x, y, angle, cx, cy):
#     angle = angle * pi / 180
#     x_new = (x - cx) * cos(angle) + (y - cy) * sin(angle) + cx
#     y_new = -(x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
#     return x_new, y_new


# 绕图片中心旋转，返回新图片已经对应坐标，角度为90倍数
def rotate_img(img, angle):
    height, width = img.shape[:2]
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    imgRotation = cv2.warpAffine(img, matRotation, (width, height), borderValue=(0, 0, 0))
    return imgRotation


# 在图片上添加椒盐噪声
def add_noise(img, prob):
    output = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            temp = random.random()
            if temp <= prob:
                output[i][j] = img[i][j] + random.randint(-20, 20)
                if output[i][j][0] < 0:
                    output[i][j] = np.array([0, 0, 0])
                elif output[i][j][0] > 255:
                    output[i][j] = np.array([255, 255, 255])
            else:
                output[i][j] = img[i][j]
    return output


# 数据增强
def data_enhance(input_img, label_img):
    # 随机旋转
    i = random.randint(0, 3)
    degree = 90 * i
    new_img = rotate_img(input_img, degree)
    new_label = rotate_img(label_img, degree)

    # 随机加噪声
    prob = 0.1
    new_img = add_noise(new_img, prob)

    return new_img, new_label


# 读取CT的数据类
class CTDataset(Dataset):
    """
     root：图像存放地址根路径
     width: 图片宽度
     height：图片高度
     is_use_cel:是否以 便于计算交叉熵损失函数形式输出
    """

    def __init__(self, csv_root, width, height, is_use_cel=True):
        self.width = width
        self.height = height
        self.is_use_CEL = is_use_cel
        # 这个list存放所有图像的地址
        csv_file = open(csv_root, 'r')
        all_lines = csv_file.readlines()
        csv_file.close()
        self.input_list = []
        self.mask_list = []
        for line in all_lines:
            self.input_list.append(line.split(',')[0])
            self.mask_list.append(line.split(',')[1].split('\n')[0])

    def __getitem__(self, index):
        # 读取图像数据并返回
        # 输入
        input_img = cv2.imread(self.input_list[index])
        input_img = cv2.resize(input_img, (self.width, self.height))

        # label
        mask_img = cv2.imread(self.mask_list[index])
        mask_img = cv2.resize(mask_img, (self.width, self.height))

        # 数据增强
        input_img, mask_img = data_enhance(input_img, mask_img)

        if self.is_use_CEL:
            label_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) / 255.0
        else:
            label_img = np.zeros((self.height, self.width, 1))
            disease_place = (mask_img == np.array([255, 255, 255])).all(axis=2)
            label_img[disease_place] = np.array([1])
            label_img = label_img.transpose((2, 0, 1))

        # temp=cv2.cvtColor(mask_img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        # cv2.imwrite('a.png',mask_img)
        # cv2.imwrite('b.png',input_img)

        input_img = input_img.transpose((2, 0, 1)) / 255.0
        return input_img.astype('float32'), label_img

    def __len__(self):
        # 返回图像的数量
        return len(self.input_list)


if __name__ == '__main__':
    write_root(picture_root=r'F://dataset/medical/thrombus/image', mask_root=r'F://dataset/medical/thrombus/mask', csv_root=r'./thrombus_train_data.csv')
