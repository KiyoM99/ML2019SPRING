#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import random as ran
import pickle as pk
from sklearn.cluster import KMeans
import sys


# In[ ]:


x1=[]
x2=[]


# In[ ]:


t=time.time()
for i in range(20000):
    n='00000'+str(i+1)
    n=n[-6:]
    x1.append(io.imread(sys.argv[1]+n+'.jpg'))
    n='00000'+str(i+1+20000)
    n=n[-6:]
    x2.append(io.imread(sys.argv[1]+n+'.jpg'))
    if i%1000==0:
        print(time.time()-t)
        t=time.time()


# In[ ]:


x3=[]
x3.append(x1)
x3.append(x2)


# In[ ]:


x3=np.asarray(x1)
x4=np.asarray(x2)


# In[ ]:


x5=np.concatenate((x3,x4))


# In[ ]:


f = open('train.pk', 'wb')
pk.dump(x5, f)
f.close()


# In[3]:


with open('train.pk', 'rb') as file:
    x =pk.load(file)


# In[ ]:


x=np.asarray(x)


# In[ ]:


x=x/255


# In[ ]:


x1=torch.from_numpy(x)


# In[ ]:


test_dataset = torch.utils.data.TensorDataset(x1)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=False)


# In[ ]:


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(32*32*3, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 32*32*3),
            nn.Sigmoid(),  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# In[ ]:


torch.manual_seed(452345637)#1245234
autoencoder = torch.load('auto70.pt')
autoencoder=autoencoder.cuda()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
loss_func = nn.MSELoss()


# In[ ]:


b1=np.zeros((1,256))
for step, (x2) in enumerate(test_loader):
    b_x = x2[0].view(-1, 32*32*3)   # batch x, shape (batch, 28*28)

    b_x=b_x.type(torch.FloatTensor)

    b_x=get_variable(b_x)

    encoded, decoded = autoencoder(b_x)
    encoded=encoded.cpu().detach().numpy()
    b1=np.concatenate((b1,encoded))


# In[ ]:


f = open('out.pk', 'wb')
pk.dump(b1, f)
f.close()


# In[ ]:


with open('out.pk', 'rb') as file:
    x =pk.load(file)


# In[ ]:


x=x[1:]


# In[ ]:


clf = KMeans(n_clusters=2, random_state=9545242)


# In[ ]:


clf.fit(x)


# In[ ]:


a=clf.labels_


# In[ ]:


te=open(sys.argv[2],'r',encoding='utf8')
test=te.read()
test=test.split('\n')
test=test[1:-1]


# In[ ]:


for i in range(len(test)):
    test[i]=test[i].split(',')[1:]
    for n in range(2):
        test[i][n]=int(test[i][n])


# In[ ]:


fin=[]
for i in range(len(test)):
    if a[test[i][0]-1]==a[test[i][1]-1]:
        fin.append(1)
    else:
        fin.append(0)


# In[ ]:


sub=open(sys.argv[3],"w",encoding="BIG5")
sub.write('id,label\n')
for i in range(len(fin)):
    sub.write(str(i)+','+str(fin[i])+'\n')
sub.close()

