import colorsys
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.autograd import Variable

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from utils.utils import (centernet_correct_boxes, decode_bbox, letterbox_image,
                         nms)


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std
    
#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class CenterNet(object):
    _defaults = {
        "model_path"        : 'model_data/centernet_resnet50_voc.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        # "model_path"        : 'model_data/centernet_hourglass_coco.h5',
        # "classes_path"      : 'model_data/coco_classes.txt',
        "backbone"          : "resnet50",
        "image_size"        : [512,512,3],
        "confidence"        : 0.3,
        # backbone为resnet50时建议设置为True
        # backbone为hourglass时建议设置为False
        # 也可以根据检测效果自行选择
        "nms"               : True,
        "nms_threhold"      : 0.3,
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化centernet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #----------------------------------------#
        #   计算种类数量
        #----------------------------------------#
        self.num_classes = len(self.class_names)

        #----------------------------------------#
        #   创建centernet模型
        #----------------------------------------#
        assert self.backbone in ['resnet50', 'hourglass']
        if self.backbone == "resnet50":
            self.centernet = CenterNet_Resnet50(num_classes=self.num_classes, pretrain=False)
        else:
            self.centernet = CenterNet_HourglassNet({'hm': self.num_classes, 'wh': 2, 'reg':2})

        #----------------------------------------#
        #   载入权值
        #----------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.centernet.load_state_dict(state_dict,strict=True)
        self.centernet = self.centernet.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.centernet = nn.DataParallel(self.centernet)
            self.centernet.cuda()
                                    
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
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

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.image_size[0], 1)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
