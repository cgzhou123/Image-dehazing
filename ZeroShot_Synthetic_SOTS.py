import torch
import torchvision
import imageio
import model
import argparse
import os
from psnr import ssim
from cv2.ximgproc import guidedFilter
from utils.utils import _np2Tensor, _augment, psnr, t_matting, write_log
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
parser.add_argument('--TestFolderPath', type=str, default='./datasets/SOTS', help='Hazy Image folder name') 
parser.add_argument('--SavePath', type=str, default='./results/SOTS', help='SavePath Name')
args = parser.parse_args()

itr_no = 10000
l1_loss = nn.MSELoss()

def test(args):
    input_img = os.listdir(args.TestFolderPath+'/hazy/')
    input_img.sort()
    os.makedirs(args.SavePath+'/', exist_ok=True)
    
    total_psnr, total_ssim = 0, 0
    total_n_psnr, total_n_ssim = 0, 0

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
        ground_truth = imageio.imread(args.TestFolderPath+'/gt/'+input_img[i].split('_')[0]+'.png')
        print('name:', input_img[i])

        Input = _np2Tensor(Hazy)
        Input = (Input/255.).cuda(1)
        ground_truth = _np2Tensor(ground_truth)
        clear = (ground_truth/255.).cuda(1)

        Hx, Wx = Input.shape[2], Input.shape[3]
        _Hx = Hx - Hx%32
        _Wx = Wx - Wx%32
        Input = Transf(Input, _Hx, _Wx)

        HY, WY = clear.shape[2], clear.shape[3]
        _HY = HY - HY%32
        _WY = WY - WY%32
        ground_truth = Transf(clear, _HY, _WY)

        if Input.shape[1] == 4:
            Input = torch.cat((Input[0][0].unsqueeze(0).unsqueeze(0), Input[0][1].unsqueeze(0).unsqueeze(0), Input[0][2].unsqueeze(0).unsqueeze(0)), dim=1)
        if ground_truth.shape[1] == 4:
            ground_truth = torch.cat((ground_truth[0][0].unsqueeze(0).unsqueeze(0), ground_truth[0][1].unsqueeze(0).unsqueeze(0), ground_truth[0][2].unsqueeze(0).unsqueeze(0)), dim=1)

        best_psnr = 0
        best_ssim = 0
        #### train ####
        for k in tqdm(range(itr_no), desc="Loading..."):

            #### net and Discriminator start train ####
            net.train()
            net_p.train()
            optimizer.zero_grad()

            #### augment ####
            Inputmage, gt= _augment(Input, ground_truth)

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
                if psnr(J, gt)>best_psnr:
                    best_psnr = psnr(J, gt)
                if ssim(J, gt)>best_ssim:
                    best_ssim = ssim(J,gt)
                print('loss:', loss, 'dcp_loss:', 0.01*dcp_loss, 'current psnr:', psnr(HazefreeImage, gt), psnr(J,gt), 'current ssim:', ssim(HazefreeImage, gt), ssim(J,gt), 'best_psnr:', best_psnr, best_ssim )
                torchvision.utils.save_image(torch.cat((HazeProducemage,Inputmage,HazefreeImage,gt),dim=0), args.SavePath+'/'+input_img[i].split('_')[0]+'_H.png')
            
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
        
        psnr_matting, ssim_matting = psnr(_GT, ground_truth), ssim(_GT, ground_truth)
        total_psnr += psnr_matting
        total_ssim += ssim_matting

        psnr_normal, ssim_normal = psnr(_out, ground_truth), ssim(_out, ground_truth)
        total_n_psnr += psnr_normal
        total_n_ssim += ssim_normal

        print('保存后的图像PSNR和SSIM:', psnr_normal, psnr_matting, ssim_normal, ssim_matting)

        write_log('./log/SOTS_matting.txt', input_img[i].split('_')[0], psnr_matting, ssim_matting)
        write_log('./log/SOTS_nor.txt', input_img[i].split('_')[0], psnr_normal, ssim_normal)


        torchvision.utils.save_image(_GT, args.SavePath+'/'+input_img[i].split('_')[0]+'_G.png')
        torchvision.utils.save_image(_out, args.SavePath+'/'+input_img[i].split('_')[0]+'_out.png')
    
    avg_psnr = total_psnr/len(input_img)
    avg_ssim = total_ssim/len(input_img)
    avg_n_psnr = total_n_psnr/len(input_img)
    avg_n_ssim = total_n_ssim/len(input_img)
    write_log('./log/SOTS_matting.txt', 'average', avg_psnr, avg_ssim)
    write_log('./log/SOTS_nor.txt', 'average', avg_n_psnr, avg_n_ssim)
    print('total_psnr:',total_psnr, total_ssim, 'total_n_psnr:',total_n_psnr, total_n_ssim, 'avg_psnr:', avg_psnr, avg_ssim, 'avg_n_psnr:', avg_n_psnr, avg_n_ssim)

test(args) 

