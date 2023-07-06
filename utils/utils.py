'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-11-03 10:43:14
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-12-07 14:17:41
FilePath: /one_shot/utils/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import math
import torch
import random
import numpy as np
import torch.nn as nn
from cv2.ximgproc import guidedFilter
import torch.nn.functional as F

def _np2Tensor(img):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    return torch.unsqueeze(tensor, 0)


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def Atm(fn):
    src = cv2.imread(fn)
    I = src.astype('float')/255
    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)

    return A

def psnr(imgS, imgG):
	diff = imgS - imgG
	mse = diff.pow(2).mean()
	return -10 * math.log10(mse)

def _augment(_image, _gt):
	it = random.randint(0, 7)
	if it==1: 
		_image = _image.rot90(1, [2, 3])
		_gt = _gt.rot90(1, [2, 3])
	if it==2: 
		_image = _image.rot90(2, [2, 3])
		_gt = _gt.rot90(2, [2, 3])
	if it==3: 
		_image = _image.rot90(3, [2, 3])
		_gt = _gt.rot90(3, [2, 3])
	if it==4: 
		_image = _image.flip(2).rot90(1, [2, 3])
		_gt = _gt.flip(2).rot90(1, [2, 3])
	if it==5: 
		_image = _image.flip(3).rot90(1, [2, 3])
		_gt = _gt.flip(3).rot90(1, [2, 3])
	if it==6: 
		_image = _image.flip(2)
		_gt = _gt.flip(2)
	if it==7: 
		_image = _image.flip(3)
		_gt = _gt.flip(3)
	return _image, _gt


def t_matting(image, mask_out_np):

    refine_t0 = guidedFilter(image[0].transpose(1, 2, 0).astype(np.float32),
                            mask_out_np[0].astype(np.float32), 50, 1e-4)
    refine_t1 = guidedFilter(image[0].transpose(1, 2, 0).astype(np.float32),
                            mask_out_np[1].astype(np.float32), 50, 1e-4)
    refine_t2 = guidedFilter(image[0].transpose(1, 2, 0).astype(np.float32),
                            mask_out_np[2].astype(np.float32), 50, 1e-4)
    refine_t = np.stack([refine_t0,refine_t1,refine_t2], axis=0)
    # print('refine_t:', refine_t.shape)
    if True:
        return np.array([np.clip(refine_t, 0.1, 1)])
    else:
        return np.array([np.clip(refine_t, 0, 1)])


def write_log(file_name, title, psnr, ssim):
    fp = open(file_name, "a+")
    fp.write(title+ ':\n')
    fp.write('PSNR:%0.6f\n'%psnr)
    fp.write('SSIM:%0.6f\n'%ssim)
    fp.close()
