import os
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import datetime
import shutil
import matplotlib.pyplot as plt

import logging
import pickle

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The current device is {device}")

"""
ResNet 50 model & Bottle neck layer
32 initial channel showed better performance than 64 initial channel
"""
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(
        self, in_channels, out_channels, stride = 1, downsample=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x.clone()
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    

class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=5, zero_init_residual=False, chan=None):
        super(ResNet50, self).__init__()
        self.in_channels = 32
        if chan == '5ch':
            self.conv1 = nn.Conv2d(5, 32, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif chan == '1ch':
            self.conv1 = nn.Conv2d(1,32, kernel_size=7, stride=2, padding=3,
                                  bias=False)
        elif chan == 'time':
            self.conv1 = nn.Conv2d(13,32, kernel_size=7, stride=2, padding=3,
                                  bias=False)
        elif chan == '5time':
            self.conv1 = nn.Conv2d(65,32, kernel_size=7, stride=2, padding=3,
                                  bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0],stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256 * block.expansion, 50)
        self.fc2 = nn.Linear(50, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, planes * block.expansion,stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x
    

"""
train, validate, test fuctions
"""   
def train(net, optimizer, criterion, trainloader, epoch):
    print('\nEpoch: %d'% epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader,0):
        inputs = inputs.to(device=device,dtype=torch.float)
        targets = np.asarray(targets)
        targets = targets.astype(float)
        targets = torch.from_numpy(targets).to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        
        targets = targets.type(torch.LongTensor).to(device)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        batch_count += 1
    
    print("Train Loss : {:.3f} | Train Acc: {:.3f}".format(train_loss / batch_count, 100.*correct/total))
    train_loss = train_loss / batch_count
    train_acc = 100 * correct / total
    return train_loss, train_acc

def validate(net, criterion, valloader, epoch):
    net.eval()
    
    correct = 0
    total = 0
    val_loss = 0 
    batch_count = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.to(device=device,dtype=torch.float)
            labels = np.asarray(labels)
            labels = labels.astype(float)
            labels = torch.from_numpy(labels).to(device)
            
            outputs = net(images)
            
            labels = labels.type(torch.LongTensor).to(device)

            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1
        val_loss = val_loss / len(valloader)
        val_acc = 100 * correct / total
            
    print("Test Loss : {:.3f} | Test Acc: {:.3f}".format(val_loss / batch_count, 100.*correct/total))
    
    return val_loss, val_acc

def test(net, criterion, testloader, epoch):
    
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data
            inputs = inputs.to(device=device, dtype=torch.float)
            targets = np.asarray(targets)
            targets = targets.astype(float)
            targets = torch.from_numpy(targets).to(device)
            targets = targets.type(torch.LongTensor).to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _,predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_loss /= len(testloader.dataset)
    test_acc = 100 * correct / total
    print("Test acc : {:.3f}".format(test_acc))
    
    return test_loss, test_acc
    
"""
Dataset, Dataloaders
"""    
class eegDataset(Dataset):
    def __init__(self, path, data_list, transform=None):
        self.path = path
        
        self.data = data_list[0]
        self.label = data_list[1]
        
        self.image_list = [path+'/'+img+'.npy' for img in self.data]
        
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = np.load(img_path)
        label = self.label[idx]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label

def load_eeg_data(participants, directory, filename, num_folds):
    train_all = []
    test_all = []

    for i, participant in enumerate(participants):
        train_tmp, test_tmp = utils.load_cv_data(directory, participant, filename, num_folds)
        if i == 0:
            train_all = train_tmp
            test_all = test_tmp
        else:
            train_all = np.hstack([train_all,train_tmp])
            test_all = np.hstack([test_all,test_tmp])
    
    return train_all, test_all

# Summary writer fuction to save model records
def mkwriter(attempt):
    if not os.path.exists('./train_history'):
        os.mkdir('./train_history')
    save_result_loc = os.path.join('./train_history',attempt)
    if not os.path.exists(save_result_loc):
        os.mkdir(save_result_loc)
    print("result saved location: ",save_result_loc)
    return SummaryWriter(save_result_loc)

# train_validate separation function
def train_test_splits(traindata):
    all_indices = np.arange(len(traindata[0]))
    
    train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
    
    train_all = np.array([traindata[0][train_indices], traindata[1][train_indices]])
    val_all = np.array([traindata[0][val_indices], traindata[1][val_indices]])
    return train_all, val_all

# data paths
norm5ch50dir = './data/5ch/50x50/normalize'
norm50dir = './data/1ch/50x50/normalize'
normtime50dir = './data/time_series/50x50/normalize(1wave)'
normtime5w50dir = './data/time_series/50x50/normalize(5wave)'

# csv files, cv_num = which session tobe test session
csv_name1 = 'eeg_image(1ch)_dataset.csv'
csv_name2 = 'eeg_image(5ch)_dataset.csv'
csv_name3 = 'eeg_image(time)_dataset.csv'
csv_name4 = 'eeg_image(5time)_dataset.csv'
cv_num = 3

"""
split and load train, test dataset

change data_path, csv_file to train other inputs, 
change cv_num to choose other session as test set

ex) train_all_tmp, test_all = load_eeg_data(participants,data_path,
                                            csv_file, cv_num)
"""

eeg_dir = './SEED-V/Normalized_EEG'
participants = os.listdir(eeg_dir)
train_all_tmp, test_all = load_eeg_data(participants,norm5ch50dir,csv_name2, cv_num)


# split train, test set
train_all, val_all = train_test_splits(train_all_tmp)

np.take(train_all, np.random.permutation(train_all.shape[1]), axis = 1, out= train_all)
np.take(val_all, np.random.permutation(val_all.shape[1]), axis = 1, out= val_all)
np.take(test_all, np.random.permutation(test_all.shape[1]), axis = 1, out= test_all)

img_file_path = norm5ch50dir

transform = transforms.ToTensor()
trainset = eegDataset(path=img_file_path,data_list=train_all, transform=transform)
trainloader = DataLoader(dataset=trainset,
                        batch_size=16,
                        shuffle=True,
                        drop_last=False)
valset = eegDataset(path=img_file_path,data_list=val_all, transform=transform)
valloader = DataLoader(dataset=valset,
                        batch_size=16,
                        shuffle=True,
                        drop_last=False)
testset = eegDataset(path=img_file_path,data_list=test_all, transform=transform)
testloader = DataLoader(dataset=testset,
                        batch_size=16,
                        shuffle=True,
                        drop_last=False)


"""
summary writer to save model data 
(need to change every run or delete directory because of summarywriter)
"""

writer = mkwriter('norm1_50')

net = ResNet50(block=Bottleneck, layers=[3,4,6,3],chan='5ch') # chan ='1ch','5ch','time','5time'
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(),lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# save model shape
data,_ = next(iter(trainloader))
data = data.to(device=device,dtype=torch.float)
writer.add_graph(net,data)

best_acc = 0
best_epoch = 0
model_path = os.path.join('./train_history/norm1_50','best_model.pth')

for epoch in tqdm(range(100)):
    train_loss, train_acc = train(net, optimizer, criterion,trainloader,epoch)
    val_loss, val_acc = validate(net,criterion,valloader,epoch)
    test_loss, test_acc = test(net,criterion, testloader, epoch=epoch)

    writer.add_scalars('Loss', {"Train Loss" : train_loss, "val Loss" : val_loss, "Test Loss" : test_loss}, epoch)
    writer.add_scalars('Accuracy', {"Train acc" : train_acc, "val acc" : val_acc, "Test acc" : test_acc}, epoch)
    #writer.add_scalars('Accuracy',{"Test acc" : test_acc})
    
    scheduler.step()
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        torch.save(net.state_dict(), model_path)

writer.close()

print(f'Best Test Accuracy: {best_acc:.3f} at epoch {best_epoch}')



# create and save confusion matrix

model_path = './train_history/norm1_50/best_model.pth'
net = ResNet50(block=Bottleneck, layers=[3,4,6,3],chan='5ch')
net.load_state_dict(torch.load(model_path))
net = net.to(device)
utils.create_and_save_confusion_matrix(net=net,device=device, testloader=testloader, num_classes=5,
                                       title='Confusion Matrix',path=model_path, cmap=plt.cm.Blues)
