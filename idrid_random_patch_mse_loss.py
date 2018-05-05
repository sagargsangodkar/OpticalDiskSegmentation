import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import torch.optim as optim
import math
import random

def extractPatch(npTrain,npGt,patchSize):
	ImgSize = npTrain.shape
	X = (np.linspace(0,ImgSize[3]-patchSize,5)).astype(int)
	Y = (np.linspace(0,ImgSize[2]-patchSize,5)).astype(int)
	for batchItem in range(ImgSize[0]):
		for i in range(4):
		    for j in range(4):
			if(i==0 and j==0 and batchItem==0):
			    indexX = random.randrange(X[i],X[i+1])
			    indexY = random.randrange(Y[j],Y[j+1])
			    stackedTrain = npTrain[batchItem,:,indexY:indexY+patchSize,indexX:indexX+patchSize]
			    stackedTrain = np.expand_dims(stackedTrain,axis=0)
			    stackedGt = npGt[batchItem,indexY:indexY+patchSize,indexX:indexX+patchSize]
			    stackedGt = np.expand_dims(stackedGt,axis=0)
			else:
			    indexX = random.randrange(X[i],X[i+1])
			    indexY = random.randrange(Y[j],Y[j+1])
			    patchTrain = npTrain[batchItem,:,indexY:indexY+patchSize,indexX:indexX+patchSize]
			    patchTrain = np.expand_dims(patchTrain,axis=0)
			    stackedTrain = np.concatenate((stackedTrain,patchTrain),axis=0)  
			    patchGt = npGt[batchItem,indexY:indexY+patchSize,indexX:indexX+patchSize]
			    patchGt = np.expand_dims(patchGt,axis=0)
			    stackedGt = np.concatenate((stackedGt,patchGt),axis=0)  
    	return(torch.Tensor(stackedTrain),torch.Tensor(stackedGt))

class IDRIDDataset(Dataset):
	def __init__(self,path_train,path_gt,num_of_images):
		self.path_train_file=path_train
		self.path_gt_file=path_gt
		self.num_of_images = num_of_images
	def __len__(self):
		return (self.num_of_images)
	def __getitem__(self, idx):
		f_train = open(self.path_train_file)
		f_gt = open(self.path_gt_file)
		image_path = f_train.readlines()
		im = Image.open(image_path[idx].split('\n')[0])
		width, height = im.size
		width_new=int(math.ceil(width/4.0))
		height_new=int(math.ceil(height/4.0))
		np_img=np.array(im.resize((width_new,height_new)))
		np_img = np_img.transpose(2,0,1)
		image_gt_path = f_gt.readlines()
		im = Image.open(image_gt_path[idx].split('\n')[0])
		width, height = im.size
		width_new=int(math.ceil(width/4.0))
		height_new=int(math.ceil(height/4.0))
		np_gt_img=np.array(im.resize((width_new,height_new)))
		return (torch.Tensor(np_img), torch.Tensor(np_gt_img)*255)
                


path_train = '/media/asr-gpu/hdd/sachin/ml_project/path_training_list.txt'
path_gt = '/media/asr-gpu/hdd/sachin/ml_project/path_gt_list.txt'
num_of_images = 54
dataset = IDRIDDataset(path_train,path_gt,num_of_images)

train_loader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=2)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
	self.bn1=nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
	self.bn2=nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
	self.bn3=nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=5)
	self.bn4=nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 16, kernel_size=5)
	self.bn5=nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, 1, kernel_size=5)


    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.conv6(x))
        return x

model = Net()
model=model.cuda()



criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001,momentum=0.9)
patchSize = 180

batchNo=0
# Training loop
while(1):
	for epoch in range(1):
	    for i, data in enumerate(train_loader, 0):
		# get the inputs
		batchNo+=1
		trainImage,GtImage = data        
		trainImagePatches,GtImagePatches = extractPatch(np.array(trainImage),np.array(GtImage),patchSize)
		trainImagePatchesVar, GtImagePatchesVar = Variable(trainImagePatches.cuda()), Variable(GtImagePatches.cuda()[:,12:12+156,12:12+156].unsqueeze_(1))

		# Forward pass: Compute predicted y by passing x to the model
		y_pred = model(trainImagePatchesVar)
		
		loss=torch.sum((y_pred-GtImagePatchesVar )**2)/96.0
		loss.backward()

		print('Batch Number :: ',batchNo,'Training Loss ::',loss.data.cpu().numpy()[0])

		optimizer.step()
		optimizer.zero_grad()
