import os, sys
import argparse
import scipy.misc
import torch
import h5py

import numpy as np
import math
import random
from time import time
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms


import torchsrc
import generate_sublist
from img_loader import img_loader

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='')
parser.add_argument('--image_path',default='')
parser.add_argument('--batchSize_lmk', type=int, default=4, help='input batch size for lmk detection')
parser.add_argument('--loss_fun',default='MSE',help='MSE | Dice | Dice_norm | cross_entropy')
opt = parser.parse_args()



osize = [512, 512]
fineSize = [512,512]


transform_list = []
transform_list.append(transforms.Scale(osize, Image.BICUBIC))
transforms_scale = transforms.Compose(transform_list)

transform_list = []
transform_list.append(transforms.Scale(osize, Image.NEAREST))
transforms_seg_scale = transforms.Compose(transform_list)

transform_list = []
transform_list.append(transforms.ToTensor())
transforms_toTensor = transforms.Compose(transform_list)

transform_list = []
transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5)))
transforms_normalize = transforms.Compose(transform_list)


code_path = os.getcwd()
fcn_torch_code_path = os.path.join(code_path, 'torchfcn')
if fcn_torch_code_path not in sys.path:
     sys.path.insert(0, fcn_torch_code_path)



model = torchsrc.models.Unet_BN(n_class=1) #lr = 0.0014
model.load_state_dict(torch.load(opt.model_path))

A_img = Image.open(opt.image_path)

A_img = transforms_scale(A_img)
A_img = transforms_toTensor(A_img)
A_img = transforms_normalize(A_img)
data = torch.Tensor(3, 512, 512)
data = A_img


cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)
	model = model.cuda()

if cuda:
    data = data.cuda()
data = torch.unsqueeze(data,0)
data = Variable(data)

pred = model(data)
pred_imgs = pred.data.cpu().numpy()

osize2 = [326, 490]
transform_list2 = []
transform_list2.append(transforms.Scale(osize2, Image.NEAREST))
transforms_seg_scale2 = transforms.Compose(transform_list2)

for i in range(pred_imgs.shape[0]) :
    pred_img = pred_imgs[i,0,:,:]
    pred_img = Image.fromarray(pred_img)
    pred_img = transforms_seg_scale2(pred_img).convert('RGB')
    output_file = ''
    pred_img.save(output_file)
    img_np = np.array(pred_img)[:,:,0]
    idx = np.argmax(img_np)
    y_coord = idx / 490
    x_coord = idx - int(y_coord) * 490

    result_x = x_coord / float(490)
    result_y = y_coord / float(326)
    print("Result: x : %.4f, y : %.4f" % (result_x, result_y))
