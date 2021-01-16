import math
from random import shuffle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils.utils import draw_gaussian, gaussian_radius


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class CenternetDataset(Dataset):
    def __init__(self, train_lines, input_size, num_classes, is_train):
        super(CenternetDataset, self).__init__()

        self.train_lines = train_lines
        self.input_size = input_size
        self.output_size = (int(input_size[0]/4) , int(input_size[1]/4))
        self.num_classes = num_classes
        self.is_train = is_train

    def __len__(self):
        return len(self.train_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box

        return image_data, box_data

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines

        #-------------------------------------------------#
        #   进行数据增强
        #-------------------------------------------------#
        img, y = self.get_random_data(lines[index], [self.input_size[0],self.input_size[1]], random=self.is_train)
        
        batch_hm = np.zeros((self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_reg = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
        
        if len(y) != 0:
            #-------------------------------------------------#
            #   转换成相对于特征层的大小
            #-------------------------------------------------#
            boxes = np.array(y[:,:4],dtype=np.float32)
            boxes[:,0] = boxes[:,0] / self.input_size[1] * self.output_size[1]
            boxes[:,1] = boxes[:,1] / self.input_size[0] * self.output_size[0]
            boxes[:,2] = boxes[:,2] / self.input_size[1] * self.output_size[1]
            boxes[:,3] = boxes[:,3] / self.input_size[0] * self.output_size[0]

        for i in range(len(y)):
            bbox = boxes[i].copy()
            bbox = np.array(bbox)
            #-------------------------------------------------#
            #   防止超出特征层的范围
            #-------------------------------------------------#
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size[1] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size[0] - 1)

            cls_id = int(y[i, -1])
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                #-------------------------------------------------#
                #   计算真实框所属的特征点
                #-------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                #----------------------------#
                #   绘制高斯热力图
                #----------------------------#
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        img = np.array(img, dtype=np.float32)[:,:,::-1]
        img = np.transpose(preprocess_image(img), (2, 0, 1))

        return img, batch_hm, batch_wh, batch_reg, batch_reg_mask


# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs = np.array(imgs)
    batch_hms = np.array(batch_hms)
    batch_whs = np.array(batch_whs)
    batch_regs = np.array(batch_regs)
    batch_reg_masks = np.array(batch_reg_masks)
    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks

