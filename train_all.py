import os, sys
import argparse
import torch
import h5py

import numpy as np
import math
import random
from time import time
from torch.autograd import Variable
import torchsrc
import generate_sublist
from img_loader import img_loader



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter(subs,viewName):
	img_subs = []
	for i in range(len(subs)):
		if (subs[i][1]==viewName):
			img_subs.append(subs[i])
	return img_subs


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=int,default=205, help='the network that been used')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize_lmk', type=int, default=2, help='input batch size for lmk detection')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for, default=50')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--compete',type=bool,default=False,help='True: do compete training')
parser.add_argument('--augment',type=bool,default=False,help='True: do use augmented data')
parser.add_argument('--GAN',type=bool,default=False,help='True: add GAN loss')
parser.add_argument('--accreEval',type=bool,default=False,help='True: only evaluate accre results')
parser.add_argument('--viewName',help='viewall | view1 | view2 | view3')
parser.add_argument('--loss_fun',default='MSE',help='MSE | Dice | Dice_norm | cross_entropy')
parser.add_argument('--LSGAN',type=bool,default=False,help='True: use LSGAN')




opt = parser.parse_args()
print(opt)
# task_name = opt.task #0: do single task lmk, 1 do single task clss, 2 do multi task lmk, 3 do multi task lmk+clss
epoch_num = opt.epoch
lmk_batch_size = opt.batchSize_lmk
learning_rate = opt.lr
network_num = opt.network
num_workers = 1
compete = opt.compete
augment = opt.augment
GAN = opt.GAN
onlyEval = opt.accreEval
viewName =  opt.viewName
lmk_num = 1
loss_fun = opt.loss_fun
noLSGAN = not opt.LSGAN

code_path = os.getcwd()
fcn_torch_code_path = os.path.join(code_path, 'torchfcn')
if fcn_torch_code_path not in sys.path:
     sys.path.insert(0, fcn_torch_code_path)


img_root_dir = '/home/xujuan/yucheng/project1/assignment1_data'
train_sublist_name = 'train_list.txt'
test_sublist_name = 'test_list.txt'

train_img_root_dir = os.path.join(img_root_dir, 'train')
test_img_root_dir = os.path.join(img_root_dir, 'validation')


working_root_dir = '/home/xujuan/yucheng/project1/working'

sublist_dir = os.path.join(working_root_dir,'sublist')
mkdir(sublist_dir)


train_img_list_file = os.path.join(sublist_dir,train_sublist_name)
train_subs = generate_sublist.dir2list(train_img_root_dir,train_img_list_file)
test_img_list_file = os.path.join(sublist_dir,test_sublist_name)
test_subs = generate_sublist.dir2list(test_img_root_dir,test_img_list_file)


results_path = os.path.join(working_root_dir, 'results')
mkdir(results_path)


# network setting, 1XX clss, 2XX lmk, 3xx MTL
if network_num == 205:
	model = torchsrc.models.Unet_BN(n_class=lmk_num) #lr = 0.0014
elif network_num == 206:
	model = torchsrc.models.FCNGCN(num_classes=lmk_num)
elif network_num == 207:
	model = torchsrc.models.FCNGCNAD(num_classes=lmk_num)
elif network_num == 208:
	model = torchsrc.models.ResUnet50(n_classes=lmk_num, pretrained=False)
elif network_num == 202:
	model = torchsrc.models.ResNet50(n_classes=lmk_num,pretrained=True)  #lr = 0.001 or 0.0001
elif network_num == 502:
	model = torchsrc.models.ResNetFCN(num_classes=lmk_num)  #lr = 0.001 or 0.0001
elif network_num == 601:
	model = torchsrc.models.SSNet(num_classes=lmk_num)  #lr = 0.001 or 0.0001
elif network_num == 602:
	model = torchsrc.models.ResNetFCNFinal(num_classes=lmk_num)  #lr = 0.001 or 0.0001
elif network_num == 603:
	model = torchsrc.models.GCNFinal(num_classes=lmk_num)  #lr = 0.001 or 0.0001
elif network_num == 604:
	model = torchsrc.models.SSNetFinal(num_classes=lmk_num)  #lr = 0.001 or 0.0001


out = os.path.join(results_path,str(network_num),loss_fun)
mkdir(out)

train_set = img_loader(train_subs)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=lmk_batch_size,shuffle=True,num_workers=num_workers)
test_set = img_loader(test_subs)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=lmk_batch_size,shuffle=False,num_workers=num_workers)



cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)
	model = model.cuda()

start_epoch = 0
start_iteration = 1

optim = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))

trainer = torchsrc.Trainer(
	cuda=cuda,
	model=model,
	optimizer=optim,
	train_loader=train_loader,
	test_loader=test_loader,
	out=out,
	network_num = network_num,
	max_epoch = epoch_num,
	compete = compete,
	GAN = GAN,
	batch_size = lmk_batch_size,
	lmk_num = lmk_num,
	onlyEval = onlyEval,
	view = viewName,
	loss_fun = loss_fun,
	noLSGAN = noLSGAN,
)


print("==start training==")
print("==view is == %s "%viewName)

start_epoch = 0
start_iteration = 1
trainer.epoch = start_epoch
trainer.iteration = start_iteration
# if opt.test:
# 	trainer.test()
# else:
trainer.train_epoch()

# model_output_file = torch.save(model,os.path.join(out,'model.pth.tar'))


