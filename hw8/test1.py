
# coding: utf-8

# In[1]:


import numpy as np
import time
import pickle as pk
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


# In[2]:


def get_variable(x):
    x = Variable(x)
    x=x.type(torch.FloatTensor)
    return x.cuda()


# In[3]:


def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    
    # return our final result
    return histogram

# execute our histogram function
#hist = get_histogram(x1[0], 256)
#print(hist)


# In[4]:


def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

# execute the fn
#cs = cumsum(hist)

# display the result
#plt.plot(cs)


# In[5]:


def css(cs):# numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    # re-normalize the cumsum
    cs = nj / N

    # cast it back to uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')
    return cs


# In[6]:


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
    #x21[i]=x22[i].reshape((48,48))


# In[7]:


x22=x22.astype(int)


# In[8]:


for i in range(len(x22)):
    hist = get_histogram(x22[i], 256)
    cs = cumsum(hist)
    look=css(cs)
    x22[i]=look[x22[i]]
    x21[i][0]=np.reshape(x22[i],(48,48))


# In[9]:


x21=x21.astype(np.uint8)


# In[10]:


mean=np.mean(x21)
std=np.std(x21)
x21=(x21-mean)/std


# In[11]:


f = open('testnew.pk', 'wb')
pk.dump(x21, f)
f.close()


# In[12]:


with open('testnew.pk', 'rb') as file:
    testnew =pk.load(file)


# In[13]:


testnew=torch.from_numpy(testnew)


# In[14]:


test_loader = torch.utils.data.DataLoader(dataset=testnew,
                                           batch_size=16,
                                           shuffle=False)


# In[15]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,stride=2)
            )
        self.model = nn.Sequential(
            conv_bn(  1,  32, 2), 
            nn.Dropout(0.3),
            conv_dw( 32,  32, 1),
            nn.Dropout(0.3),
            conv_dw( 32,  64, 2),
            nn.Dropout(0.3),
            conv_dw( 64,  64, 1),
            nn.Dropout(0.3),
            conv_dw( 64,  64, 1),
            nn.Dropout(0.3),
            conv_dw( 64, 128, 2),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(128, 7)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)


# In[16]:


cnnc= torch.load('cnn1.pth')
cnnc.eval()
asdsadas=1


# In[17]:


#fin1=torch.zeros(0)
#fin1=fin1.type("torch.LongTensor")
#fin1=fin1.cuda()
out0=np.zeros((1,7))
print(out0)
for images in test_loader:
    images = get_variable(images)
    outputs = cnnc(images)
    outn=outputs.cpu().detach().numpy()
    #for i in range(16):
    out0=np.append(out0,outn,axis=0)
    #_, predicted = torch.max(outputs.data, 1)
    #fin1=torch.cat((fin1,predicted))


# In[18]:


out0=out0[1:]


# In[19]:


f = open('out0.pk', 'wb')
pk.dump(out0, f)
f.close()


# In[20]:


with open('out0.pk', 'rb') as file:
    out0 =pk.load(file)


# In[21]:


aa=1
n1=out0
nn=[]
for tt in range(len(n1)):
    nn.append(np.where(max(n1[tt])==n1[tt])[0][0])
sub=open(sys.argv[2],"w",encoding="BIG5")
sub.write('id,label\n')
for i in range(len(n1)):
    sub.write(str(i)+','+str(nn[i])+'\n')
sub.close()

