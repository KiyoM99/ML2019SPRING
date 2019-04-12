
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


# In[2]:


def get_variable(x):
    x = Variable(x)
    return x.cuda()


# In[3]:


a=ran.randint(0,1000000001)
torch.manual_seed(a)
np.random.seed(441599383)
print(a)


# In[4]:


with open('../input/train_x.pk', 'rb') as file:
    x2 =pk.load(file)
with open('../input/label_y.pk', 'rb') as file:
    y =pk.load(file)
with open('../input/vadx.pk', 'rb') as file:
    vadx =pk.load(file)
with open('../input/vady.pk', 'rb') as file:
    vady =pk.load(file)
print(x2[0])


# In[5]:


x11=torch.from_numpy(x2)
y11=torch.from_numpy(y)
vadx=torch.from_numpy(vadx)
vady=torch.from_numpy(vady)
x11=x11.type("torch.FloatTensor")
y11=y11.long()
vadx=vadx.type("torch.FloatTensor")
vady=vady.long()
#vady=vady.cuda()
train_dataset = torch.utils.data.TensorDataset(x11,y11)
vad_dataset = torch.utils.data.TensorDataset(vadx,vady)


# In[6]:


batch=256
itr=30


# In[7]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch,
                                           shuffle=True)

vad_loader = torch.utils.data.DataLoader(dataset=vad_dataset,
                                           batch_size=32,
                                           shuffle=False)


# In[8]:


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


# In[9]:


maxco=0
loss1=[]


# In[10]:


ti=time.time()
u=0
for epoch in range(50):
    cnn.train()
    for n, (x11, y11) in enumerate(train_loader):
        x11 = get_variable(x11)
        y11= get_variable(y11)

        outputs = cnn(x11)
        loss = loss_func(outputs, y11)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch%1 == 0:
        cnn.eval()
        val=torch.zeros(0)
        val=val.type("torch.LongTensor")
        val=val.cuda()
        for i, (vadx, vady1) in enumerate(vad_loader):
            vadx = get_variable(vadx)
            vady1 = get_variable(vady1)
            
            outputv = cnn(vadx)
            _, predicted = torch.max(outputv.data, 1)
            val=torch.cat((val,predicted))
        val=val.cpu()
        print(loss_func(outputs, y11),loss_func(outputv, vady1))
        co=0
        for i in range(len(vady)):
            if val[i]==vady[i]:
                co+=1
        loss1.append([epoch,loss_func(outputv, vady1),co/1500])
        if co>maxco:
            maxco=co
            u=0
            torch.save(cnn,'cnn_'+str(epoch)+'.pt')
        else:
            u+=1
        
        print(epoch,co/1500,time.time()-ti)
        ti=time.time()
        if u>20:
            break


# In[11]:


print(vady)


# In[12]:


print(val)


# In[13]:


f=open(str(a)+'.txt','w',encoding="utf8")
for i in range(len(loss1)):
    f.write(str(loss1[i])+'\n')
f.close()

