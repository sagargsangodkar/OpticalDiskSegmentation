import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class IDRIDDataset(Dataset):
    
	def __init__(self,path,num_of_images):
		self.path_file=path
		self.num_of_images = num_of_images
		print('init')

	def __len__(self):
		print('len')
		return (self.num_of_images)

	def __getitem__(self, idx):
        	with open(self.path_file) as f:
        		image_path = f.readlines()
        		im = Image.open(image_path[idx].split('\n')[0])
			width, height = im.size
			np_img=np.array(im.resize((100,100)))
			np_img = np_img.transpose(2,0,1)

    			return torch.Tensor(np_img)
                


path = '/media/asr-gpu/hdd/sachin/ml_project/path_list.txt'
num_of_images = 413
dataset = IDRIDDataset(path,num_of_images)


train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9680, 500)
	self.fc2 = nn.Linear(500, 3)
        self.sigmoid = torch.nn.Sigmoid()



    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc1(x)
	x = self.fc2(x)
        return F.softmax(x)

model = Net()




criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1):
    for i, data in enumerate(train_loader, 0):
        inputs = data
        print(inputs.shape)
        # wrap them in Variable
	inputs = Variable(inputs)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)
	print(y_pred)
        # Compute and print loss
        #loss = criterion(y_pred, labels)
        #print(epoch, i, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        #optimizer.zero_grad()
        #loss.backward()	
#optimizer.step()
