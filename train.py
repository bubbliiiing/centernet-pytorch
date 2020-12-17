#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from nets.centernet_training import focal_loss, reg_l1_loss
from utils.dataloader import CenternetDataset, centernet_dataset_collate


#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0
    start_time = time.time()

    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break

            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            optimizer.zero_grad()

            if backbone=="resnet50":
                hm, wh, offset = net(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                wh_loss = 0.1*reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
                loss = c_loss + wh_loss + off_loss

                total_loss += loss.item()
                total_c_loss += c_loss.item()
                total_r_loss += wh_loss.item() + off_loss.item()
            else:
                outputs = net(batch_images)
                loss = 0
                c_loss_all = 0
                r_loss_all = 0
                index = 0
                for output in outputs:
                    hm, wh, offset = output["hm"].sigmoid(), output["wh"], output["reg"]
                    c_loss = focal_loss(hm, batch_hms)
                    wh_loss = 0.1*reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss += c_loss + wh_loss + off_loss

                    c_loss_all += c_loss
                    r_loss_all += wh_loss + off_loss
                    index += 1
                total_loss += loss.item()/index
                total_c_loss += c_loss_all.item()/index
                total_r_loss += r_loss_all.item()/index
            loss.backward()
            optimizer.step()
            
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                'total_c_loss'  : total_c_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer),
                                's/step'        : waste_time})
            pbar.update(1)

            start_time = time.time()

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

                if backbone=="resnet50":
                    hm, wh, offset = net(batch_images)
                    c_loss = focal_loss(hm, batch_hms)
                    wh_loss = 0.1*reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    loss = c_loss + wh_loss + off_loss
                    val_loss += loss.item()
                else:
                    outputs = net(batch_images)
                    index = 0
                    loss = 0
                    for output in outputs:
                        hm, wh, offset = output["hm"].sigmoid(), output["wh"], output["reg"]
                        c_loss = focal_loss(hm, batch_hms)
                        wh_loss = 0.1*reg_l1_loss(wh, batch_whs, batch_reg_masks)
                        off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                        loss += c_loss + wh_loss + off_loss
                        index += 1
                    val_loss += loss.item()/index


            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)
    
if __name__ == "__main__":
    #-------------------------------------------#
    #   输入图片的大小
    #-------------------------------------------#
    input_shape = (512,512,3)
    annotation_path = '2007_train.txt'
    #-------------------------------------------#
    #   类对应的txt
    #-------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'  
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    #-------------------------------------------#
    #   用于设定在使用resnet50作为主干网络时，
    #   是否使用imagenet-resnet50的预训练权重。
    #   仅在主干网络为resnet50时有作用。
    #   默认为False
    #-------------------------------------------#
    pretrain = False
    #-------------------------------------------#
    #   主干特征提取网络的选择
    #   resnet50和hourglass
    #-------------------------------------------#
    backbone = "resnet50"  

    Cuda = True

    assert backbone in ['resnet50', 'hourglass']
    if backbone == "resnet50":
        model = CenterNet_Resnet50(num_classes, pretrain=pretrain)
    else:
        model = CenterNet_HourglassNet({'hm': num_classes, 'wh': 2, 'reg':2})

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    model_path = r"model_data/centernet_resnet50_voc.pth"
    # 加快模型训练的效率
    print('Loading weights into state dict...')
    model_dict = model.state_dict() 
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-3
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 50
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = CenternetDataset(lines[:num_train], input_shape, num_classes)
        val_dataset = CenternetDataset(lines[num_train:], input_shape, num_classes)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                                drop_last=True, collate_fn=centernet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8,pin_memory=True, 
                                drop_last=True, collate_fn=centernet_dataset_collate)


        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        model.freeze_backbone()

        for epoch in range(Init_Epoch,Freeze_Epoch):
            val_loss = fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-4
        Batch_size = 4
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = CenternetDataset(lines[:num_train], input_shape, num_classes)
        val_dataset = CenternetDataset(lines[num_train:], input_shape, num_classes)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                                drop_last=True, collate_fn=centernet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8,pin_memory=True, 
                                drop_last=True, collate_fn=centernet_dataset_collate)

                        
        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        model.unfreeze_backbone()

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            val_loss = fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)
