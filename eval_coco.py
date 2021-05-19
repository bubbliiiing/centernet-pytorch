import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from centernet import CenterNet
from utils.utils import (centernet_correct_boxes, decode_bbox, letterbox_image,
                         nms)

coco_classes = {'person': 1, 'bicycle': 2, 'car': 3, 'motorbike': 4, 'aeroplane': 5, 
    'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 
    '': 83, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 
    'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 
    'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 
    'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 
    'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 
    'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 
    'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 
    'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'sofa': 63, 'pottedplant': 64, 'bed': 65, 
    'diningtable': 67, 'toilet': 70, 'tvmonitor': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 
    'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 
    'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 
    'hair drier': 89, 'toothbrush': 90
}

clsid2catid = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
               15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
               27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43,
               39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
               51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72,
               63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85,
               75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std
    
class mAP_CenterNet(CenterNet):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_id, image, results):
        self.confidence = 0.01
        self.nms_threhold = 0.5

        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = letterbox_image(image, [self.image_size[1], self.image_size[0]])
        #----------------------------------------------------------------------------------#
        #   将RGB转化成BGR，这是因为原始的centernet_hourglass权值是使用BGR通道的图片训练的
        #----------------------------------------------------------------------------------#
        photo = np.array(crop_img, dtype = np.float32)[:,:,::-1]
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)), [1, self.image_size[2], self.image_size[0], self.image_size[1]])
        
        with torch.no_grad():
            images = torch.from_numpy(np.asarray(photo)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.centernet(images)
            if self.backbone=='hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            #-----------------------------------------------------------#
            #   利用预测结果进行解码
            #-----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

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
                return results

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
            result = {}
            predicted_class = self.class_names[int(c)]
            top, left, bottom, right = boxes[i]

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            result["image_id"] = int(image_id)
            result["category_id"] = clsid2catid[c]
            result["bbox"] = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"] = float(top_conf[i])
            results.append(result)

        return results

centernet = mAP_CenterNet()

jpg_names = os.listdir("./coco_dataset/val2017")

with open("./coco_dataset/eval_results.json","w") as f:
    results = []
    for jpg_name in tqdm(jpg_names):
        if jpg_name.endswith("jpg"):
            image_path = "./coco_dataset/val2017/" + jpg_name
            image = Image.open(image_path)
            # 开启后在之后计算mAP可以可视化
            results = centernet.detect_image(jpg_name.split(".")[0], image, results)
    json.dump(results,f)
