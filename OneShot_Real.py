import torch
import torchvision
import imageio
import model
import argparse
import os
from cv2.ximgproc import guidedFilter
from utils.utils import _np2Tensor, _augment, t_matting
import torch.nn as nn
import torchvision.transforms as trans
from tqdm import tqdm
def Transf(image, H,W):
    transfomer = trans.Compose([
        trans.Resize((H, W))
    ])
    img = transfomer(image)
    return img

def TransF(image,H,W):
    transfomer = trans.Compose([
        trans.Resize((H, W))
    ])
    img = transfomer(image)
    return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1: #0.02
        m.weight.data.normal_(0.0, 0.001)
    if classname.find('Linear') != -1: #0.02
        m.weight.data.normal_(0.0, 0.001)

parser = argparse.ArgumentParser(description='Single Image Dehazing')
parser.add_argument('--TestFolderPath', type=str, default='./datasets/HSTS/real-world', help='Hazy Image folder name') 
parser.add_argument('--SavePath', type=str, default='./results/real-world', help='SavePath Name')
args = parser.parse_args()

itr_no = 10000
l1_loss = nn.MSELoss()

def test(args):
    input_img = os.listdir(args.TestFolderPath+'/hazy/')
    input_img.sort()
    os.makedirs(args.SavePath+'/', exist_ok=True)

    for i in range(len(input_img)):
        print("Images Processed: %d/ %d  \r" % (i+1, len(input_img)))

        #### define model ####
        net = model.Model('hazemodel')
        net.apply(weights_init)
        net.cuda(1)
        net_p = model.Model('hazeproducemodel')
        net_p.apply(weights_init)
        net_p.cuda(1)


        #### define optimizer ####
        optimizer = torch.optim.Adam([
                    {'params': net.parameters()},
                    {'params': net_p.parameters()},],
                    lr=1e-4, betas=(0.99, 0.999), eps=1e-08, weight_decay=1e-2)
        
        #### data process ####
        Hazy = imageio.imread(args.TestFolderPath+'/hazy/'+input_img[i])
        print('name:', input_img[i])

        Input = _np2Tensor(Hazy)
        Input = (Input/255.).cuda(1)

        Hx, Wx = Input.shape[2], Input.shape[3]
        _Hx = Hx - Hx%64
        _Wx = Wx - Wx%64
        if _Hx >= 2000 and Wx >= 2000:
            print('缩小前:shape:', Input.shape, _Hx, _Wx)
            Input = Transf(Input,int(_Hx/2),int(_Wx/2))
            print('缩小后:shape:', Input.shape)
        elif _Hx < 2000 or Wx < 2000:
            print('不改变:shape:', Input.shape, _Hx, _Wx)
            Input = Transf(Input, _Hx, _Wx)
            print('不改变:shape:', Input.shape)

        if Input.shape[1] == 4:
            Input = torch.cat((Input[0][0].unsqueeze(0).unsqueeze(0), Input[0][1].unsqueeze(0).unsqueeze(0), Input[0][2].unsqueeze(0).unsqueeze(0)), dim=1)

        #### train ####
        for k in tqdm(range(itr_no), desc="Loading..."):

            #### net and Discriminator start train ####
            net.train()
            net_p.train()
            optimizer.zero_grad()

            #### augment ####
            Inputmage, gt= _augment(Input, Input)

            #### two ways removal or produce fog ####
            trans, atm, HazefreeImage = net(Inputmage, 'train')
            transX, atmX ,HazeProducemage = net_p(HazefreeImage, 'train')

            #### define Airlight ####
            A = Input.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)

            #### 上溢 下溢 ####
            otensor = torch.ones(HazefreeImage.shape).cuda(1)
            ztensor = torch.zeros(HazefreeImage.shape).cuda(1)
            lossMx = torch.sum(torch.max(HazefreeImage, otensor))  - torch.sum(otensor)
            lossMn = - torch.sum(torch.min(HazefreeImage, ztensor))
            lossMnx = lossMx + lossMn

            #### cycle loss ####
            loss_cycle = l1_loss(HazeProducemage, Inputmage)

            #### transmission loss ###
            loss_trans = l1_loss(trans, transX)

            #### Airlight loss ####
            loss_air = l1_loss(atm,A) + l1_loss(atmX,A)

            #### dcp loss ####
            dcp_prior = torch.min(HazefreeImage.permute(0, 2, 3, 1), 3)[0]
            dcp_loss =  l1_loss(dcp_prior, torch.zeros_like(dcp_prior)) - 0.005

            #### total loss ####
            loss = loss_cycle + loss_trans + loss_air + 0.001*lossMnx + 0.01*dcp_loss
            if (k+1) % 200 == 1 or (k+1) % 200 == 0:
                refinet = t_matting(Inputmage.detach().cpu().numpy(), trans[0].detach().cpu().numpy())
                J = (Inputmage - (1 - torch.from_numpy(refinet).cuda(1))*atm)/torch.from_numpy(refinet).cuda(1)
                print('loss:', loss, loss_cycle, loss_trans, loss_air, 0.001*lossMnx, 'dcp_loss:', 0.01*dcp_loss)
                torchvision.utils.save_image(torch.cat((HazeProducemage,Inputmage,HazefreeImage,gt),dim=0), args.SavePath+'/'+input_img[i].split('.')[0]+'_H.png')
            
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            _trans, _atm, _out = net(Input, 'train')
            _out = torch.clamp(_out, 0, 1)

            _trans, _atm = net(Input, 'test')
            refine_t = t_matting(Input.detach().cpu().numpy(), _trans[0].detach().cpu().numpy())
            _GT = (Input - (1 - torch.from_numpy(refine_t).cuda(1))*_atm)/torch.from_numpy(refine_t).cuda(1)
            _GT = torch.clamp(_GT, 0, 1)

        _out = TransF(_out, Hx, Wx)
        _GT = TransF(_GT, Hx, Wx)

        torchvision.utils.save_image(_GT, args.SavePath+'/'+input_img[i].split('.')[0]+'_G.png')
        torchvision.utils.save_image(_out, args.SavePath+'/'+input_img[i].split('.')[0]+'_out.png')

test(args) 
