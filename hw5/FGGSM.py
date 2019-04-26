
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import pickle as pk
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import random as ran
import time
import torchvision.transforms as transforms
import sys


# In[2]:


def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )


# In[3]:


a=np.zeros((1,224,224,3))
for i in range(200):
    b=load_image(sys.argv[1]+'/'+str(1000+i)[1:]+'.png')
    b=np.reshape(b,(1,224,224,3))
    a=np.concatenate((a,b))
print(np.shape(a))
a=a/255
a1=a[1:]


# In[4]:


a3=np.zeros((200,3,224,224))
for i in range(200):
    for l in range(3):
        for n in range(224):
            for p in range(224):
                a3[i][l][n][p]=a1[i][n][p][l]


# In[5]:


a2=np.copy(a3)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
for i in range(200):
    a2[i][0]=(a2[i][0]-mean[0])/std[0]
    a2[i][1]=(a2[i][1]-mean[1])/std[1]
    a2[i][2]=(a2[i][2]-mean[2])/std[2]


# In[6]:


aaaaa='305 883 243 559 438 990 949 853 609 582 915 455 619 961 630 741 455 707 854 922 129 537 672 476 299 99 476 251 520 923 760 582 525 317 464 478 667 961 865 324 33 922 142 312 302 582 948 360 789 440 746 764 949 480 792 900 733 327 441 882 920 839 955 555 519 510 888 990 430 396 97 78 140 362 705 659 640 967 489 937 991 887 603 467 498 879 807 708 967 472 287 853 971 805 719 854 471 890 572 883 476 581 603 967 311 873 582 16 672 780 489 685 366 746 599 912 950 614 348 353 21 84 437 946 746 646 544 469 597 81 734 719 51 293 897 416 544 415 814 295 829 759 971 306 637 471 94 984 708 863 391 383 417 442 38 858 716 99 546 137 980 517 322 765 632 595 754 805 873 475 455 442 734 879 685 521 640 663 720 759 535 582 607 859 532 113 695 565 554 311 8 385 570 480 324 897 738 814 253 751'
aaaaa=aaaaa.split(' ')
aaaaa=np.asarray(aaaaa)


# In[7]:


aaaaa=aaaaa.astype(np.int)


# In[8]:


import torchvision.models as models
VGG19 = models.resnet50(True)


# In[9]:


x=a2
y=aaaaa


# In[10]:


VGG19.eval()
VGG19=VGG19.cuda()


# In[11]:


x1=torch.from_numpy(x)
y1=torch.from_numpy(y)
x1=x1.type("torch.FloatTensor")
y1=y1.long()
train_dataset = torch.utils.data.TensorDataset(x1,y1)


# In[12]:


test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=8,
                                           shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=False)


# In[13]:


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    return perturbed_image


# In[14]:


def test( model, test_loader, epsilon ):
    nnn=[]
    correct = 0

    for data, target in test_loader:

        data, target = data.cuda(), target.cuda()

        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        #if init_pred.item() != target.item():
            #continue

        loss = F.nll_loss(output, target)
        
        model.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        ppp=perturbed_data.cpu().detach().numpy()
        nnn.append(ppp)
        
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1

    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, nnn


# In[15]:


torch.manual_seed(1654822)
a,nnn=test(VGG19,train_loader,0.1)
nnn1=np.asarray(nnn)
nnn1=np.reshape(nnn1,(200, 3, 224, 224))
nnn2=torch.from_numpy(nnn1)
nnn2=nnn2.type("torch.FloatTensor")
nnn_dataset = torch.utils.data.TensorDataset(nnn2,y1)
train_loader = torch.utils.data.DataLoader(dataset=nnn_dataset,
                                           batch_size=1,
                                           shuffle=False)


# In[16]:


x=nnn1


# In[17]:


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
for i in range(200):
    x[i][0]=x[i][0]*std[0]+mean[0]
    x[i][1]=x[i][1]*std[1]+mean[1]
    x[i][2]=x[i][2]*std[2]+mean[2]


# In[18]:


a=np.zeros((200,224,224,3))
tt=time.time()
for i in range(200):
    if i%40==0:
        print(i,time.time()-tt)
        tt=time.time()
    for l in range(224):
        for n in range(224):
            for p in range(3):
                a[i][l][n][p]=x[i][p][l][n]
                if a[i][l][n][p]>1:
                    a[i][l][n][p]=1
                elif a[i][l][n][p]<0:
                    a[i][l][n][p]=0


# In[19]:


a=a*255
a=np.rint(a)
a=a.astype(np.uint8)


# In[20]:


x=a


# In[21]:


x=x.astype(np.uint8)


# In[22]:


for i in range(200):
    img = Image.fromarray(x[i])
    img.save(sys.argv[2]+'/'+str(1000+i)[1:]+'.png')

