from PIL import Image
import numpy as np
import torch
import sys
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def save_feature(features,dir):
    b,c,_,_=features.shape
    for i in range(c):
        single_channel = features[0, i, :, :]
        normalized_map = (single_channel - single_channel.min()) / (single_channel.max() - single_channel.min())
        image = Image.fromarray((normalized_map.cpu().numpy() * 255).astype('uint8'))
        image.save(dir+'/feature_map_'+str(i)+'.png')

    sys.exit()
    sum=torch.sum(features,dim=1)
    sum=sum[0,:,:]
    sum = (sum - sum.min()) / (sum.max() - sum.min())
    to_pil = transforms.ToPILImage()
    image = to_pil(sum)
    contrast = transforms.functional.adjust_contrast(image, contrast_factor=2.0)  # 2.0 表示增加对比度

    # 转回张量
    to_tensor = transforms.ToTensor()
    adjusted_tensor = to_tensor(contrast)

    image = Image.fromarray((adjusted_tensor[0,:,:].cpu().numpy() * 255).astype('uint8'))
    image.save(dir+'/feature_map_sum.png')

    binary_image=torch.where(adjusted_tensor>0.5,torch.tensor(1.0),torch.tensor(0.0))
    image2 = Image.fromarray((binary_image[0,:,:].cpu().numpy() * 255).astype('uint8'))
    image2.save(dir+'/feature_map_mask.png')

def getMask(features):
    sum=torch.sum(features,dim=1)
    sum=sum[0,:,:]
    sum = (sum - sum.min()) / (sum.max() - sum.min())
    print(sum.mean())
    # image2 = Image.fromarray((sum.cpu().numpy() * 255).astype('uint8'))
    # image2.save('./feature_map_sum'+str(num)+'.png')
    to_pil = transforms.ToPILImage()
    image = to_pil(sum)
    contrast = transforms.functional.adjust_contrast(image, contrast_factor=3.0)  # 2.0 表示增加对比度

    # 转回张量
    to_tensor = transforms.ToTensor()
    adjusted_tensor = to_tensor(contrast)
    c1=adjusted_tensor>0.1
    binary_image=torch.where(c1,torch.tensor(1.0),torch.tensor(0.0))
    # image2 = Image.fromarray((binary_image[0,:,:].cpu().numpy() * 255).astype('uint8'))
    # image2.save('./feature_map_mask'+str(num)+'.png')
    return binary_image.unsqueeze(0)



