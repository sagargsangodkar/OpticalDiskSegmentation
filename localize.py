import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from numpy import linalg as LA
import pywt
from matplotlib import pyplot as plt
import numpy
import io
import pandas
import math




filepath="/media/asr-gpu/hdd/sachin/ml_project/training_c/"
'''
image=Image.open(filepath+"IDRiD_01.jpg").resize((1500,1000))
img=np.array(image)[:,:,2]
img1=np.array(image)[:,:,0]
img2=np.array(image)[:,:,1]
print(img.shape)
h,w=img.shape
index_h=0
index_w=0
temp=0
count=0
z=0
'''
list = os.listdir(filepath) # dir is your directory path
number_files = len(list)
l=-1
text_file = open("Annotations.txt", "w")
for fileName in sorted(os.listdir(filepath)):
	l=l+1
	image=Image.open(filepath+fileName)
	h,w=image.size
	h=int(math.ceil(h/16.0))
	w=int(math.ceil(w/16.0))
	img=np.array(image)[:,:,2]
	img1=np.array(image)[:,:,0]
	img2=np.array(image)[:,:,1]
	print(img.shape)
	
	index_h=0
	index_w=0
	temp=0
	count=0
	z=0
	for height in range(h):
		for width in range(w):
			
			if w-width>16 and h-height>16:
				for i in range(16):	
					for j in range(16):
						count=count+img[i+height,j+width]
					
				if temp<count:
					temp=count
					index_h=height+8
					index_w=width+8
				
				count=0
			elif w-width<=16 :
				width=0	
				break
		

		if h-height<=16:
			break
	text_file.write(fileName+","+str(index_w)+","+str(index_h))
	print(fileName+","+str(index_w)+","+str(index_h))
	
text_file.close()	
localize=np.dstack((img1[index_h-90:index_h+90,index_w-90:index_w+90],img2[index_h-90:index_h+90,index_w-90:index_w+90],img[index_h-90:index_h+90,index_w-90:index_w+90]))
print(localize.shape)

plt.imshow(localize,cmap=plt.cm.brg)
plt.show()

