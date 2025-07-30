import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import torch
import MSD_CSC_ISTA as msdcsc
import torch.utils.data as Data
from dataloader import VIFDataset
from ssim_loss import SSIM_loss
from recover import recover_Norm_y,ycbcr_to_rgb
import matplotlib.pyplot as plt  
import sys




EPOCH = 4
BATCH_SIZE_TRAIN = 6
BATCH_SIZE_TEST = 1
cudaopt = True

cur=0

trainset = VIFDataset(dataset_root_dir=None, \
                    dataset_dict='/media/ssd_2t/home/py/DesCSC_Fusion/data/LLVIP', \
                    train=True,upsample = False,arbitrary_input_size = True,crop_size=512)
testset = VIFDataset(dataset_root_dir=None, \
                    dataset_dict='/media/ssd_2t/home/py/DesCSC_Fusion/data/LLVIP', \
                    train=False,upsample = False,arbitrary_input_size = True,crop_size=256)

train_loader = Data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE_TRAIN, shuffle = True) 
test_loader = Data.DataLoader(dataset=testset, batch_size=1, shuffle=True)


torch.cuda.set_device(0)
model = msdcsc.FusionNet()
model.cuda()
l2=SSIM_loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5 , betas=(0.9, 0.95))

if os.path.exists('./model/checkpoint'+str(cur)+'.pth'):  
    print('use: ./model/checkpoint'+str(cur)+'.pth')
    checkpoint = torch.load('./model/checkpoint'+str(cur)+'.pth')  
    model.load_state_dict(checkpoint['model_state_dict'])  
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

# train

losses=[]
for epoch in range(cur,EPOCH):
    model.train()
    loss_total=0.0
    count=0
    for step,(rgb,t,info) in enumerate(train_loader):
        count=count+1
        if step%10==0:
            print('epoch:'+str(epoch)+'\tstep:'+str(step))
        y=rgb[:, 0, :, :]
        y=torch.unsqueeze(y,1).cuda()
        output=model(y)

        loss=l2(y,output)
        loss_total=loss_total+loss.item()
        if step%10==0:
            print('loss:'+str(loss_total/count))
            losses.append(loss_total/count)
            loss_total=0.0
            count=0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(figsize=(10, 5))  
    plt.plot(range(len(losses)), losses)  
    plt.title('Training Loss over Epochs')  
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')  
    plt.grid(True) 

    filename = f'./loss/loss_plot_epoch_{epoch}.png'  
    plt.savefig(filename)  
    plt.close() 

    model.eval()
    with torch.no_grad():
        count=0
        for step,( rgb,t,info) in enumerate(test_loader):
            if count==10:
                break
            rgb_y=rgb[:, 0, :, :]
            rgb_y=torch.unsqueeze(rgb_y,1).cuda()

            output=model(rgb_y)
            output=output.squeeze(1).cpu()

            for i in range(BATCH_SIZE_TEST):    
                rgb_cur=rgb[i]
                out_cur=output[i]
                rgb_y=rgb_cur[0]
                rgb_cb=rgb_cur[1]
                rgb_cr=rgb_cur[2]

                # output_y=recover_Norm_y(out_cur,0)
                # rgb_y=recover_Norm_y(rgb_y,0)

                rgb_out=ycbcr_to_rgb(rgb_y,rgb_cb,rgb_cr)
                result=ycbcr_to_rgb(out_cur,rgb_cb,rgb_cr)

                testset.save_img(rgb_out,'./results',info,'vis'+str(count*BATCH_SIZE_TEST+i)+'_epoch_'+str(epoch)+'.jpg')
                testset.save_img(result,'./results',info,'out'+str(count*BATCH_SIZE_TEST+i)+'_epoch_'+str(epoch)+'.jpg')
            count=count+1
        checkpoint = {  
            'model_state_dict': model.state_dict(),  
            'optimizer_state_dict': optimizer.state_dict(),  
        } 
    torch.save(checkpoint, "./model/checkpoint"+str(epoch+1)+".pth")
print('train done!')
        