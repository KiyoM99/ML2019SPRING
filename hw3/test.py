
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.utils.data
import random as ran
import pickle as pk
import time
import sys


# In[ ]:


test_data = sys.argv[1]
out= sys.argv[2]


# In[2]:


def get_variable(x):
    x = Variable(x)
    return x.cuda()


# In[3]:


def get_histogram(image, bins):
    histogram = np.zeros(bins)
    
    for pixel in image:
        histogram[pixel] += 1
    
    return histogram


# In[4]:


def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


# In[5]:


def css(cs):
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    cs = nj / N

    cs = cs.astype('uint8')
    return cs


# In[6]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.ZeroPad2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.ZeroPad2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.ZeroPad2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2))
        self.fc1 = nn.Linear(4608,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,7)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.dropout(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)


# In[7]:


cnn1=torch.load('cnn_21.pt')
cnn1.eval()
cnn2=torch.load('cnn_48.pt')
cnn2.eval()
cbcvb=1


# In[8]:


tr=open(sys.argv[1],'r',encoding='utf8')
train=tr.read()
train.replace(',',' ')
train1=train.split("\n")
train1=train1[1:-1]
x22=np.zeros((len(train1),48*48))
x21=np.zeros((len(train1),1,48,48),dtype=np.uint8)
for i in range(len(train1)):
    if i<10:
        train1[i]=train1[i][2:]
    elif i>=10 and i<100:
        train1[i]=train1[i][3:]
    elif i>=100 and i<1000:
        train1[i]=train1[i][4:]
    else:
        train1[i]=train1[i][5:]
    x22[i]=np.fromstring(train1[i],dtype=np.uint8, sep=' ')


# In[9]:


x22=x22.astype(int)


# In[10]:


for i in range(len(x22)):
    hist = get_histogram(x22[i], 256)
    cs = cumsum(hist)
    look=css(cs)
    x22[i]=look[x22[i]]
    x21[i][0]=np.reshape(x22[i],(48,48))


# In[11]:


x21=x21.astype(np.uint8)


# In[12]:


mean=np.mean(x21)
std=np.std(x21)
x21=(x21-mean)/std


# In[13]:


test_loader2 = torch.utils.data.DataLoader(dataset=x21,
                                           batch_size=16,
                                           shuffle=False)


# In[14]:


tr=open(sys.argv[1],'r',encoding='utf8')
train=tr.read()
train.replace(',',' ')
train1=train.split("\n")
train1=train1[1:-1]
x22=np.zeros((len(train1),48*48))
x21=np.zeros((len(train1),1,48,48),dtype=np.uint8)
for i in range(len(train1)):
    if i<10:
        train1[i]=train1[i][2:]
    elif i>=10 and i<100:
        train1[i]=train1[i][3:]
    elif i>=100 and i<1000:
        train1[i]=train1[i][4:]
    else:
        train1[i]=train1[i][5:]
    x22[i]=np.fromstring(train1[i],dtype=np.uint8, sep=' ')
    x21[i]=x22[i].reshape((48,48))
x21=torch.from_numpy(x21)
x21=x21.type("torch.FloatTensor")
x21=x21/255
test_loader1 = torch.utils.data.DataLoader(dataset=x21,
                                           batch_size=16,
                                           shuffle=False)


# In[15]:


out0=np.zeros((1,7))
for images in test_loader1:
    images = get_variable(images)
    outputs = cnn1(images)
    outn=outputs.cpu().detach().numpy()
    #for i in range(16):
    out0=np.append(out0,outn,axis=0)
    #_, predicted = torch.max(outputs.data, 1)
    #fin1=torch.cat((fin1,predicted))
out1=np.zeros((1,7))
for images in test_loader2:
    images = get_variable(images)
    images=images.type("torch.cuda.FloatTensor")
    outputs = cnn2(images)
    outn=outputs.cpu().detach().numpy()
    #for i in range(16):
    out1=np.append(out1,outn,axis=0)
    #_, predicted = torch.max(outputs.data, 1)
    #predicted=predicted.type("torch.cuda.FloatTensor")
    #fin=torch.cat((fin,predicted))


# In[16]:


out0=out0[1:]
out1=out1[1:]


# In[19]:


n1=(0.44*out0+0.15*out1)/0.59
nn=[]
for tt in range(len(n1)):
    nn.append(np.where(max(n1[tt])==n1[tt])[0][0])
sub=open(sys.argv[2],"w",encoding="BIG5")
sub.write('id,label\n')
for i in range(len(n1)):
    sub.write(str(i)+','+str(nn[i])+'\n')
sub.close()

