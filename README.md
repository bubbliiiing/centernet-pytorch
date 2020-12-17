## CenterNet:Objects as Points目标检测模型在Keras当中的实现
---

### 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [参考资料 Reference](#Reference)

### 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12 | [centernet_resnet50_voc.pth](https://github.com/bubbliiiing/centernet-pytorch/releases/download/v1.0/centernet_resnet50_voc.pth) | VOC-Test07 | 512x512 | - | 77.1
| COCO-Train2017 | [centernet_hourglass_coco.pth](https://github.com/bubbliiiing/centernet-pytorch/releases/download/v1.0/centernet_hourglass_coco.pth) | COCO-Val2017 | 512x512 | 38.4 | 56.8 

### 所需环境
torch==1.2.0

### 注意事项
代码中的centernet_resnet50_voc.pth是使用voc数据集训练的。    
代码中的centernet_hourglass_coco.pth是使用voc数据集训练的。   
**注意不要使用中文标签，文件夹中不要有空格！**     
**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**。     

### 文件下载 
训练所需的centernet_resnet50_voc.pth、centernet_hourglass_coco.pth可在百度网盘中下载。    
链接: https://pan.baidu.com/s/1QBBgRb_TH8kJdSCQGgcXmQ 提取码: phnc  
centernet_resnet50_voc.pth是voc数据集的权重。    
centernet_hourglass_coco.pth是coco数据集的权重。    

### 预测步骤
#### 1、使用预训练权重
a、下载完库后解压，在百度网盘下载centernet_resnet50_voc.pth或者centernet_hourglass_coco.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
可完成预测。  
b、利用video.py可进行摄像头检测。  
#### 2、使用自己训练的权重
a、按照训练步骤训练。  
b、在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
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
```
c、运行predict.py，输入  
```python
img/street.jpg
```
可完成预测。  
d、利用video.py可进行摄像头检测。  

### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2centernet.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7、**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：   
```python
classes_path = 'model_data/new_classes.txt'    
```
model_data/new_classes.txt文件内容为：   
```python
cat
dog
...
```
8、运行train.py即可开始训练。

### mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

### Reference
https://github.com/xuannianz/keras-CenterNet      
https://github.com/see--/keras-centernet      
https://github.com/xingyizhou/CenterNet    
