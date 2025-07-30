import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageFilter
import cv2
from  torchvision import utils as vutils
import cv2 as cv
import math
import torchvision.transforms as transforms
import imageio
import sys

imsave=imageio.imsave

class VIFDataset(Dataset):
    """ RGBT dataset
 
    :dataset_root_dir: Root directory of the RGBT dataset
    :upsample: Whether to perform upsampling images within the network X2
    :dataset_dict: Dictionary storing names and paths of VIF task datasets
    :rgb_list: list of rgb images
    :t_list: list of t images
    :arbitrary_input_size: Whether the images inside the dataset are dynamic in size or not
    """
    def __init__(self,dataset_root_dir,dataset_dict,train=True,upsample=False,arbitrary_input_size=True,crop_size=256):

        self.dataset_root_dir = dataset_root_dir
        self.upsample =upsample
        self.dataset_dict = dataset_dict   #dict type
        self.train=train
        self.rgb_list,self.t_list = self.get_RGBT()
        self.crop_size=crop_size
        self.train=train
        # self.transform_normalize = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        #
   
        self.arbitrary_input_size = arbitrary_input_size
        if self.upsample:
            self.win_HW = 112
        else:
            self.win_HW = 224
        
    def __len__(self):
        return len(self.rgb_list)


    def __getitem__(self, idx):
        #load multi-source images
        rgb = Image.open(self.rgb_list[idx]).convert('YCbCr')
        t = Image.open(self.t_list[idx]).convert('YCbCr')

        # rgb = Image.open(self.rgb_list[idx]).convert('L')
        # t = Image.open(self.t_list[idx]).convert('L')

        if self.train:
            #cut to 256x256
            rgb=np.array(rgb)
            t=np.array(t)

            # 随机生成裁剪起点
            h, w, c= rgb.shape
            crop_h = np.random.randint(0, h - self.crop_size + 1)
            crop_w = np.random.randint(0, w - self.crop_size + 1)
            
            # 裁剪图像
            t = t[crop_h:crop_h + self.crop_size, crop_w:crop_w + self.crop_size]
            rgb = rgb[crop_h:crop_h + self.crop_size, crop_w:crop_w + self.crop_size]
        else:
            w,h=rgb.size

        #Data Augmentation
        rgb=transforms.ToTensor()(rgb)
        t=transforms.ToTensor()(t)



        # normalize_rgb = transforms.Normalize(mean=[0.1856], std=[0.150])
        # normalize_t = transforms.Normalize(mean=[0.280], std=[0.172])

        # rgb_y=rgb[0].unsqueeze(0)
        # t_y=t[0].unsqueeze(0)

        # # rgb_y=rgb[0]
        # # t_y=t[0]

        # rgb[0] = normalize_rgb(rgb_y).squeeze()
        # t[0]= normalize_t(t_y).squeeze()
        
        image_name = self.rgb_list[idx].split("/")[-1]
        #Some info important or useful during training
        train_info = {'H':h,'W':w,                  #image origin size
                    'name':image_name,              #image name in dataset
            }
        return rgb,t,train_info
    # def __getitem__(self, idx):
    #     #load multi-source images
    #     rgb = Image.open(self.rgb_list[idx]).convert('YCbCr')
    #     t = Image.open(self.t_list[idx]).convert('YCbCr')
    #     rgb=transforms.ToTensor()(rgb)
    #     rgb_y=rgb[0]
    #     t=transforms.ToTensor()(t)
    #     t_y=t[0]

    #     return rgb_y,t_y

    #
    def get_RGBT(self):
        """ imports each dataset in dataset_dict sequentially
            Returns a list of sample paths for each modality
        """
        rgb_list=[]
        t_list=[]
        if self.train:
            rgb_dir = os.path.join(self.dataset_dict,"visible","train")
            t_dir = os.path.join(self.dataset_dict,"infrared","train")
        else:
            rgb_dir = os.path.join(self.dataset_dict,"visible","test")
            t_dir = os.path.join(self.dataset_dict,"infrared","test")

        for path in os.listdir(rgb_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(rgb_dir, path)):
                rgb_list.append(os.path.join(rgb_dir, path))
                t_list.append(os.path.join(t_dir, path))

        # rgb_list = 2*rgb_list
        # t_list = 2*t_list
        
        return rgb_list,t_list
    


    def get_img_list(self,x):
        """ Cut the input tensor by window size 
            input (3,H,W)
            Return tensor for winows list (N,3,win_HW,win_HW)
        """
        _,H,W = x.shape
        win_HW = self.win_HW
        H_len = math.ceil(H/win_HW)
        W_len = math.ceil(W/win_HW)
        #print(H,W,H_len,W_len)

        img_list = []

        for i in range(H_len):
            if i==H_len-1:
                str_H = H - win_HW
                end_H = H
            else:
                str_H = i*win_HW
                end_H = (i+1)*win_HW
            for j in range(W_len):
                if j==W_len-1:
                    str_W = W - win_HW
                    end_W = W
                else:
                    str_W = j*win_HW
                    end_W = (j+1)*win_HW
                #print(str_H,end_H,str_W,end_W)
                img_list.append(x[:,str_H:end_H,str_W:end_W])
       # print(len(img_list))
        img_list = torch.stack(img_list)
        return img_list
    
    def recover_img(self,img_list,train_info):
        """ Recover the tensor of the winows list into a single image tensor.
            input (N,3,win_HW,win_HW)
            return (3,H,W)
        """
        win_HW = self.win_HW
        H_len,W_len = train_info['H_len'],train_info['W_len']
        resize_H = H_len*win_HW
        resize_W = W_len*win_HW

        img = torch.zeros(3, resize_H,resize_W)
        for i in range(H_len):
            if i==H_len-1:
                str_H = resize_H - win_HW
                end_H = resize_H
            else:
                str_H = i*win_HW
                end_H = (i+1)*win_HW
            for j in range(W_len):
                if j==W_len-1:
                    str_W = resize_W - win_HW
                    end_W = resize_W
                else:
                    str_W = j*win_HW
                    end_W = (j+1)*win_HW
                img[:,str_H:end_H,str_W:end_W] = img_list[i*W_len+j]
       # img = img.permute(1,2,0)
        return img


    def save_img(self,img_tensor,path,train_info,name=None):
        """ Save an image tensor to a specified location

        """
        #print(train_info)
        H,W = train_info['H'][0].item(),train_info['W'][0].item()
        if not os.path.exists(path):
            os.makedirs(path)
        #print(img_tensor.shape)
        #img = img_tensor.permute(2,0,1)
        re_transform = transforms.Compose([
            transforms.Resize([H,W]),
            ])
        img = re_transform(img_tensor)
        img = img.permute(1,2,0)

        if name!=None:
            img_path = os.path.join(path,name)
        else:
            img_path = os.path.join(path,train_info['name'])

        imsave(img_path, img)



