import torch  

def recover_Norm(image):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    imagenet_mean= imagenet_mean.view([3,1,1])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    imagenet_std = imagenet_std .view([3,1,1])
    return torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).float()

def recover_Norm_y(image,type):
    mean=0
    std=0
    #rgb
    if type==0:
        mean=0.1856
        std=0.150
    #t
    else:
        mean=0.280
        std=0.172
    result=image*std+mean
    return result

def ycbcr_to_rgb(Y, Cb, Cr):
    # 转换公式
    R = Y + 1.402 * (Cr - 0.5)
    G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)

    # Clip values to [0, 255]
    R = torch.clamp(R, 0, 1)
    G = torch.clamp(G, 0, 1)
    B = torch.clamp(B, 0, 1)

    # Stack them into one tensor
    rgb = torch.stack([R, G, B], dim=0)

    return rgb