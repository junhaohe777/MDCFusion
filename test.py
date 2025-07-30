import torch,gc
import MSD_CSC_ISTA as msdcsc
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from dataloader import VIFDataset
import torch.utils.data as Data
import os
from recover import recover_Norm_y,ycbcr_to_rgb
from feature_to_image import save_feature,getMask
import sys

BATCH_SIZE=1

os.environ["CUDA_VISIBLE_DEVICES"]="1"

testset = VIFDataset(dataset_root_dir=None, \
                    dataset_dict='/media/ssd_2t/home/py/dataset/TNO', \
                    train=False,upsample = False,arbitrary_input_size = True)

test_loader = Data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)

checkpoint = torch.load('./model/checkpoint4.pth')
state_dict=checkpoint['model_state_dict']
encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}

encoder=msdcsc.Encoder(16,3,1).cuda()
decoder=msdcsc.Decoder().cuda()
encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

encoder.eval()
decoder.eval()

with torch.no_grad():
    for step,( rgb,t,info) in enumerate(test_loader):
        rgb_y=rgb[:, 0, :, :]
        rgb_y=torch.unsqueeze(rgb_y,1).cuda()

        t_y=t[:, 0, :, :]
        t_y=torch.unsqueeze(t_y,1).cuda()

        output_rgb=encoder(rgb_y)
        output_t=encoder(t_y)

        # m=getMask(output_t).cuda()
        # output_mask=output_t*m

        output=decoder(output_t+output_rgb)

        output=output.squeeze(1)

        output=output[0].cpu()

        rgb_cb=rgb[0][1]
        rgb_cr=rgb[0][2]
        
        # output_y=recover_Norm_y(output,0)

        result=ycbcr_to_rgb(output,rgb_cb,rgb_cr)
        testset.save_img(result,'./tests/output',info,info['name'][0])