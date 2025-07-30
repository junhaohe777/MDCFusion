import torch
import torch.nn as nn  
import torch.nn.functional as F 
import torchmetrics

class SSIM_loss(nn.Module):
    def __init__(self):  
        super(SSIM_loss, self).__init__()  
  
  
    def forward(self,rgb,fused):
        alpha=100
        
        # weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=fused.dtype, device=fused.device)  
        # weights = weights.view(1, 3, 1, 1)
        # fused_gray = (fused * weights).sum(dim=1, keepdim=True)
        # fused_gray.clamp_(0, 1)
        # fused_gray=fused_gray.repeat(1,3,1,1)
        
        loss_intensity=F.mse_loss(fused,rgb,reduction='mean')
        
        ssim=torchmetrics.StructuralSimilarityIndexMeasure(data_range=6.666,reduction="elementwise_mean",sigma=1.5,kernel_size=11).cuda()
        loss_ssim=1-ssim(fused,rgb)
        
        
        return loss_intensity+alpha*loss_ssim