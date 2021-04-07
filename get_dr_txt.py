import colorsys
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

from centernet import CenterNet
from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from utils.utils import (centernet_correct_boxes, decode_bbox, letterbox_image,
                         nms)

'''
这里设置的门限值较低是因为计算map需要用到不同门限条件下的Recall和Precision值。
所以只有保留的框足够多，计算的map才会更精确，详情可以了解map的原理。
计算map时输出的Recall和Precision值指的是门限为0.5时的Recall和Precision值。

此处获得的./input/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。

这里的self.nms_threhold指的是非极大抑制所用到的iou，具体的可以了解非极大抑制的原理，
如果低分框与高分框的iou大于这里设定的self.nms_threhold，那么该低分框将会被剔除。

可能有些同学知道有0.5和0.5:0.95的mAP，这里的self.nms_threhold=0.5不代表mAP0.5。
如果想要设定mAP0.x，比如设定mAP0.75，可以去get_map.py设定MINOVERLAP。
'''
def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std
    
class mAP_CenterNet(CenterNet):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        f = open("./input/detection-results/"+image_id+".txt","w") 
        self.confidence = 0.01
        self.nms_threhold = 0.5

        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = letterbox_image(image, [self.image_size[0],self.image_size[1]])
        #----------------------------------------------------------------------------------#
        #   将RGB转化成BGR，这是因为原始的centernet_hourglass权值是使用BGR通道的图片训练的
        #----------------------------------------------------------------------------------#
        photo = np.array(crop_img,dtype = np.float32)[:,:,::-1]
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)), [1, self.image_size[2], self.image_size[0], self.image_size[1]])
        
        with torch.no_grad():
            images = Variable(torch.from_numpy(np.asarray(photo)).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            outputs = self.centernet(images)
            if self.backbone=='hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            #-----------------------------------------------------------#
            #   利用预测结果进行解码
            #-----------------------------------------------------------#
            outputs = decode_bbox(outputs[0],outputs[1],outputs[2], self.image_size, self.confidence, self.cuda)

            #-------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            #-------------------------------------------------------#
            try:
                if self.nms:
                    outputs = np.array(nms(outputs, self.nms_threhold))
            except:
                pass
            
            output = outputs[0]
            if len(output)<=0:
                return image

            batch_boxes, det_conf, det_label = output[:,:4], output[:,4], output[:,5]
            det_xmin, det_ymin, det_xmax, det_ymax = batch_boxes[:, 0], batch_boxes[:, 1], batch_boxes[:, 2], batch_boxes[:, 3]
            #-----------------------------------------------------------#
            #   筛选出其中得分高于confidence的框 
            #-----------------------------------------------------------#
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
            
            #-----------------------------------------------------------#
            #   去掉灰条部分
            #-----------------------------------------------------------#
            boxes = centernet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.image_size[0],self.image_size[1]]),image_shape)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

centernet = mAP_CenterNet()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # image.save("./input/images-optional/"+image_id+".jpg")
    centernet.detect_image(image_id,image)
    

print("Conversion completed!")
