import argparse
import logging
from re import T
import sys
from pathlib import Path
import os
import cv2
from transformer import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from utils.data_loader import BasicDataset, CarvanaDataset
from losses import *
from natsort import natsorted
import ssl
from torchsummary import summary




ssl._create_default_https_context = ssl._create_unverified_context

#训练数据路径
dir_img = Path('')  
dir_mask = Path('')
#模型保存路径
dir_checkpoint = Path('')
#测试数据路径
test_img =  Path('')

test_mask = Path('')

os.makedirs(dir_checkpoint,exist_ok=True)#自动生成模型保存文件
#测试代码
def test_net(net,
              device,
              batch_size=1,
              img_size: int=512):
    try:
        dataset = CarvanaDataset(test_img, test_mask,img_size)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(test_img, test_mask,img_size)
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=False, **loader_args)
    model_path=''  #测试模型保存路径
    model_dir = natsorted(os.listdir(model_path))
    num = 40
    for i in range(0,num):
      
   
        print(os.path.join(model_path,'epoch_%d.pth'%(i+1)))
        net.load_state_dict(torch.load(os.path.join(model_path,'epoch_%d.pth'%(i+1))))
        outs=[]
       
        net.eval()
      
      
        for batch in train_loader:
                
                    
                    images = batch['image']
                    true_masks = batch['mask']
                    
                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                    out1,out2,out3,out4= net(images)
                    

                    
                    out = torch.sigmoid(out4).cpu().detach().numpy()
                    
                    
                    out = np.where(out>0.75,1,0)
                    out= np.array(out,np.uint8)
                   
                    
                    
                
                    outs.append(out)
                 
                    
                
        outs = np.array(outs)
      
        
        outs = np.squeeze(outs,axis=1)
    
     
        np.save("_"+str(i+1)+".npy",outs)  #测试结果保存路径
                
                
    
    
#测试函数 

def train_net(net,
              device,
              epochs: int = 100,    #训练超参数
              batch_size: int = 16,
              learning_rate: float = 1e-4,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_size: int=512,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask,img_size)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask,img_size)

 
  

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images size:  {img_size}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCELoss()
 
    global_step = 0
    n_train = 29160  #训练数据总量
    iter=0
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
     
     
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                # assert images.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp): 

                    out1,out2,out3,out4= net(images)
                
               
                    out1 = torch.sigmoid(out1)
                    out2 = torch.sigmoid(out2)
                    out3 = torch.sigmoid(out3)
                    out4 = torch.sigmoid(out4)
             
                    loss1= criterion(out1, true_masks) +dice_loss(out1,true_masks)  
              
                                 
                    loss2= criterion(out2, true_masks)+dice_loss(out2,true_masks)
                    loss3= criterion(out3, true_masks) +dice_loss(out3,true_masks)  
                    loss4= criterion(out4, true_masks)+dice_loss(out4,true_masks) 
                   
                    loss = loss1+loss2+loss3+loss4
                    iter += 1
                    
                   
                if iter%100==0:
                   print("Epoch:{},Loss:{:.4f},loss1:{:.4f},loss2:{:.4f},loss3:{:.4f},loss4:{:.4f}".format(epoch+1,loss,loss1,loss2,loss3,loss4))
                  
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
          
                pbar.set_postfix(**{'loss (batch)': loss.item()})


        if save_checkpoint:
            print("------------save model------------" )
            save_dir = os.path.join(dir_checkpoint, 'epoch_{}.pth'.format(epoch+1))
            torch.save(net.state_dict(), save_dir)
        
            

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
  
    parser.add_argument('--size', '-s', type=float, default=224, help='Downscaling factor of the images')
   
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda')
    logging.info(f'Using device {device}')
 
    net = FAT_Net(imgsize=args.size)

    

    net.to(device=device)
    train= True   #训练时 True 测试时 False
    test = False  #测试时 True  训练时 False
    if train:
        try:
            train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_size=args.size,
                  amp=args.amp)
        except KeyboardInterrupt:
           torch.save(net.state_dict(), 'INTERRUPTED.pth')
           logging.info('Saved interrupt')
           sys.exit(0)
    if test:
        test_net(net=net,
                 device=device,
                 img_size=args.size,
                 batch_size=args.batch_size)
    