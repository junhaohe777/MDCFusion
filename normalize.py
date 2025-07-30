import torch,gc
import MSD_CSC_ISTA as msdcsc
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from dataloader import VIFDataset
import torch.utils.data as Data
import os
from recover import recover_Norm

BATCH_SIZE=1

os.environ["CUDA_VISIBLE_DEVICES"]="1"

datasets = VIFDataset(dataset_root_dir=None, \
                    dataset_dict='/media/ssd_2t/home/py/DesCSC_Fusion/data/LLVIP', \
                    train=True,upsample = False,arbitrary_input_size = True)

data_loader = Data.DataLoader(dataset=datasets, batch_size=BATCH_SIZE, shuffle=True)


with torch.no_grad():
    mean_rgb = 0.0
    std_rgb = 0.0

    mean_t = 0.0
    std_t = 0.0

    for step,( rgb,t) in enumerate(data_loader):
        if step%10==0:
            print(f"step: {step}")
        rgb=rgb.cuda()
        t=t.cuda()

        mean_rgb += rgb.mean()
        std_rgb += rgb.std()

        mean_t += t.mean()
        std_t += t.std()

    size=len(datasets)
    mean_rgb /= size
    std_rgb /= size

    mean_t /= size
    std_t /= size

    print(f"Total size of dataset: {size}")
    print(f"Mean of infrared Y channel: {mean_t.item()}, Std of infrared Y channel: {std_t.item()}")
    print(f"Mean of visible Y channel: {mean_rgb.item()}, Std of visible Y channel: {std_rgb.item()}")
