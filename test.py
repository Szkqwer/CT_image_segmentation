import torch.nn as nn
import numpy as np
import torch

# in_coin = input().split(',')
# result = [0, 0, 0]
# is_ok = True
# for data in in_coin:
#     money = int(data)
#     if money == 1:
#         result[2] += 1
#     elif money == 5:
#
#         if result[2] >= 4:
#             result[1] += 1
#             result[2] -= 4
#         else:
#             is_ok = False
#             break
#     else:
#         if result[1] >= 1 and result[2] >= 4:
#             result[0] += 1
#             result[1] -= 1
#             result[2] -= 4
#         elif result[2] >= 9 and result[1] == 0:
#             result[0] += 1
#
#             result[2] -= 9
#         else:
#             is_ok = False
#             break
#
# s = str(result)
# if is_ok:
#     s = 'TRUE:' + s
#
# else:
#     s = 'FALSE:' + s
# print(s)

'''
第一题
时间o（n）
空间o（币种）

第二题
优化方法
1、只找可能的开始，前面能塞东西的不可能（假如1，2可以在3，4前则不可能为3，4）
2、按照开始时间排序，再按结束时间排序
新建一个记录状态的list
若第一次遍历
1   n   3   4

3   n   4   n

4   n   n   n
第二次可以直接调用3和4

···

第三题
动态规划
新建m*n每个格子存max(up,left)和到达这里选用的路径（将这个路径上的清零）
做2次
可以简化的方式
1 -1  1（不用考虑可以设为-1）
1  -1 1（不用考虑可以设为-1）
1  1  1


'''
# def score(index, start_time, end_time, profit):
#     s = profit[index]
#     if end_time[index] >= max(start_time):
#         return s
#     r = [0] * (len(start_time) - index - 1)
#     for i in range(index + 1, len(start_time)):
#         if end_time[index] <= start_time[i]:
#             r[i - index - 1] += score(index, start_time, end_time, profit)
#     return max(r) + s
#
#
# if __name__ == '__main__':
#
#
#     game_num = int(input())
#     start_time = input().split()
#     end_time = input().split()
#     profit = input().split()
#     result = []
#     no_poss_start = []
#     i = 0
#     while i < game_num:
#         if end_time[i] > max(start_time):
#             no_poss_start.append(i)
#         i += 1
#
#     for i in range(0,game_num):
#         if i not in no_poss_start:
#             result.append(score(i, start_time, end_time, profit))
#
#     print(max(result))

# def get_diff(word1,word2):
#     diff_num=0
#     for i in range(len(word1)):
#         if word1[i]!=word2[i]:
#             diff_num+=1
#     return diff_num
#
# def find_route(start,end,word_list,all_ok):
#     if start==end:
#         return True
#     if start in word_list:
#         con=all_ok[word_list.index(start)+1]
#     else:
#         con =all_ok[0]
#     if con==[]:
#         return False
#     is_ok=False
#     for w in con:
#         is_ok=find_route(w, end, word_list, all_ok)
#     return is_ok
#
# begin_word='hit'
# end_word='cog'
# word_list=['hot','dot','dog','lot','log','cog']
# all_ok=[]
# ok_list=[]
# for index in range(len(word_list)):
#     if get_diff(begin_word, word_list[index]) == 1:
#         ok_list.append(word_list[index])
# all_ok.append(ok_list)
# for word in word_list:
#     ok_list=[]
#     for index in range(len(word_list)):
#         if get_diff(word,word_list[index])==1:
#             ok_list.append(word_list[index])
#         all_ok.append(ok_list)
#
# now_word=begin_word
# find_route(begin_word,end_word,word_list,all_ok)


# def is_same_type(c1,c2):
#     if c1=='(' or c1==')':
#         if c2=='[' or c2==']':
#             return False
#         else:
#             return True
#     else:
#         if c2=='(' or c2==')':
#             return False
#         else:
#             return True
#
# def is_pair(c1,c2):
#     if (c1=='(' and c2==')')or (c1=='[' and c2==']'):
#         return True
#
# def match(input_str):
#     mid_num=0
#     small_num=0
#     last_c=''
#     change_num=0
#     change_list=[]
#     for i in range(0,len(input_str)):
#         c=input_str[i]
#         if c=='[':
#             mid_num+=1
#         elif c==']':
#             mid_num -= 1
#         elif c=='(':
#             small_num+=1
#         else:
#             small_num -= 1
#
#         if not is_same_type(last_c,c):
#             if change_list!=[]:
#                 if is_pair(input_str[change_list[-1]],c):
#                     change_num -= 1
#                     del change_list[-1]
#                 else:
#                     change_num+=1
#                     change_list.append(i)
#
#     return abs(mid_num)+abs(small_num)+change_num
#
# input_num=int(input())
# str_list=[]
# for i in range(input_num):
#     print(match(input()))



# input_str=input()
# time_dict={}
# for c in input_str:
#     if c not in time_dict:
#         time_dict[c]=1
#     else:
#         time_dict[c]+=1
#
# result_list=[]
# last_num=0
# sorted_list=list(sorted(time_dict.items(),key=lambda x:x[1],reverse=True))
# for k in sorted_list:
#     if k[1]!=last_num:
#         result_list.append([k[0]])
#
#     else:
#         result_list[-1].append(k[0])
#     last_num = k[1]
# s=''
# for r in result_list:
#     r.sort()
#     s+=''.join(r)
# pass



import SimpleITK as sitk
import cv2
import numpy as np
import os
import glob

img = []
img1 = cv2.imread('1.png',0)
img1 = np.array(img1)
img.append(img1)

width = img1.shape[1]
height = img1.shape[0]

img_path = glob.glob(r'./*.png')#512_496_482是存放png图片的地址，有很多张二维图片
chanel = len(img_path)

img_resize = np.zeros([chanel,height,width],dtype=np.uint8)

for i in range(chanel):
    print(i)
    img = cv2.imread(img_path[i])
    img_resize[i,(height- img1.shape[0]) // 2:(height - img1.shape[0]) // 2 + img1.shape[0],
    (width - img1.shape[1]) // 2:(width - img1.shape[1]) // 2 + img1.shape[1]] = img

img_resize=np.reshape(img_resize,[chanel,height, width])
mhd_data = sitk.GetImageFromArray(img_resize)
sitk.WriteImage(mhd_data, "1.mhd")









