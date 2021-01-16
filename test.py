#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50

if __name__ == "__main__":
    # model = CenterNet_HourglassNet({'hm': 80, 'wh': 2, 'reg':2}).train().cuda()
    # summary(model,(3,128,128))
    model = CenterNet_Resnet50().train().cuda()
    summary(model,(3,512,512))
