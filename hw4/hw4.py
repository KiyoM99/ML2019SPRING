
# coding: utf-8

# In[38]:


import sys
import csv
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import pickle as pk
from lime import lime_image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import pickle as pk
import time
import sys


# In[39]:


train_data = sys.argv[1]
out= sys.argv[2]


# In[2]:


def get_variable(x):
    x = Variable(x)
    return x.cuda()


# In[3]:


abb=[0,387,2,8,42,15,4]


# In[4]:


tr=open(sys.argv[1],'r',encoding='utf8')
train=tr.read()
train.replace(',',' ')
train1=train.split("\n")
train1=train1[1:-1]
train3=train1[:]
for i in range(len(train1)):
    train3[i]=train3[i][2:]


# In[5]:


train2=[]
for i in range(len(train1)):
    train2.append(train3[i].split(' '))


# In[6]:


tim=time.time()
for i in range(len(train2)):
    for n in range(len(train2[0])):
        train2[i][n]=int(train2[i][n])
    if i%200==0:
        print(time.time()-tim)


# In[7]:


for i in range(len(train2)):
    train2[i].sort()


# In[8]:


a=[]
for i in range(len(train2)):
    if train2[i][1300]==0:
        a.append(i)


# In[9]:


coo=0
for i in a:
    train1.pop(i-coo)
    coo+=1


# In[10]:


x1=np.zeros((len(train1),48*48))
x=np.zeros((len(train1),1,48,48))
y=np.zeros(len(train1))
for i in range(len(train1)):
    y[i]=int(train1[i][0])
    train1[i]=train1[i][2:]


# In[11]:


for i in range(len(train1)):    
    x1[i]=np.fromstring(train1[i],dtype=int, sep=' ')
    #x[i][0]=x1[i].reshape((48,48))


# In[12]:


x1=x1.astype(int)


# In[13]:


def get_histogram(image, bins):
    histogram = np.zeros(bins)
    
    for pixel in image:
        histogram[pixel] += 1
    
    return histogram


# In[14]:


def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


# In[15]:


def css(cs):
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    cs = nj / N

    cs = cs.astype('uint8')
    return cs


# In[16]:


for i in range(len(x1)):
    hist = get_histogram(x1[i], 256)
    cs = cumsum(hist)
    look=css(cs)
    x1[i]=look[x1[i]]
    x[i][0]=np.reshape(x1[i],(48,48))


# In[17]:


x=x.astype(np.uint8)


# In[18]:


x2=x[:2700]
y2=y[:2700]


# In[19]:


print(y2)


# In[20]:


mean=np.mean(x2)
std=np.std(x2)


# In[21]:


print(mean,std)


# In[22]:


x3=np.zeros((7,48,48,3))
x4=np.zeros((7,1,48,48))
y4=np.zeros(7)
for p in range(7):
    for i in range(48):
        for n in range(48):
            for o in range(3):
                x3[p][i][n][o]=x2[abb[p]][0][i][n]
            x4[p][0][i][n]=x2[abb[p]][0][i][n]
    y4[p]=y2[abb[p]]


# In[23]:


print(y4)


# In[24]:


print(np.shape(x2))


# In[25]:


x21=torch.from_numpy(x2)
y21=torch.from_numpy(y2)
x21=x21.type("torch.FloatTensor")
y21=y21.long()
x41=torch.from_numpy(x4)
y41=torch.from_numpy(y4)
x41=x41.type("torch.FloatTensor")
y41=y41.long()


# In[26]:


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


# In[27]:


cnnc= torch.load('cnn_48.pt')
cnnc.eval()
mmm=11


# In[28]:


def compute_saliency_maps(x, y, model):
    model.eval()
    x.requires_grad_()
    y_pred = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliency = x.grad.abs().squeeze().data
    return saliency


# In[29]:


def show_saliency_maps(x, y, model):
    x=x.detach()
    y=y.detach()
    x_org = x.squeeze().numpy()
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(x, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().cpu().numpy()
    
    num_pics = x_org.shape[0]
    for i in range(num_pics):
        plt.imsave(sys.argv[2]+'fig1_'+ str(i)+".jpg", 5*saliency[i], cmap=plt.cm.jet)

torch.manual_seed(170487252)
np.random.seed(100)
show_saliency_maps(x41, y41, cnnc)


# In[30]:


def predict(x3):
    x2=np.zeros((len(x3),1,48,48))
    for p in range(len(x3)):
        for i in range(48):
            for n in range(48):
                x2[p][0][i][n]=x3[p][i][n][0]
    #print(x2)
    x21=torch.from_numpy(x2)
    x21=x21.type("torch.FloatTensor")
    x21=(x21-mean)/std
    return F.softmax(cnnc(x21.cuda()),dim=-1).detach().cpu().numpy()


# In[31]:


def seg(x):
    return slic(x,n_segments=50, compactness=10)


# In[32]:


explainer = lime_image.LimeImageExplainer()


# In[33]:


def explain(image, classifier_fn, **kwargs):
    np.random.seed(1110)
    return explainer.explain_instance(image, classifier_fn, **kwargs)


# In[34]:


for i in range(7):
    explaination = explain(image=x3[i], 
                                classifier_fn=predict,
                                segmentation_fn=seg,
                                batch_size=10
                            )
    image, mask = explaination.get_image_and_mask(
                                    label=y4[i],
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=7,
                                    min_weight=0.0
                                )

    # save the image
    plt.imsave(sys.argv[2]+'fig3_'+ str(i)+'.jpg', image/255)


# In[37]:


for pp in range(50):
    torch.manual_seed(170487252)
    np.random.seed(100)
    ranim=np.zeros((48,48,3))
    x=np.zeros((1,1,48,48))
    for i in range(48):
        for n in range(48):
            a=np.random.randint(100,110)
            ranim[i][n]=np.uint8(a)
            x[0][0][i][n]=np.uint8(a)
    x=(x-mean)/std
    x=torch.from_numpy(x)
    x=x.type("torch.FloatTensor")
    x=x.cuda()

    optimizer = torch.optim.Adam([x.requires_grad_()], lr=0.03)

    for i in range(1, 50):
        optimizer.zero_grad()
        
        y = cnnc.conv1(x)
        y = cnnc.conv2(y)
        #y = cnnc.conv3(y)
        conv_output = y[0,pp]

        loss = -torch.mean(conv_output)
        loss.backward()
        optimizer.step()
    x1=x*std+mean
    x1 = torch.clamp(x1, 0, 255)
    x1=x1.detach().cpu().numpy()
    x1=np.rint(x1)
    x1=np.uint8(x1)
    #print(x)
    out=np.zeros((48,48,3))
    for i in range(48):
        for n in range(48):
            out[i][n]=x1[0][0][i][n]
    #plt.imsave('vi/img_'+str(pp)+'.png', out/255)
    plt.subplot(5,10,pp+1)
    plt.imshow(out)
    plt.axis('off')
plt.savefig(sys.argv[2]+"fig2_1.jpg")
plt.show()


# In[36]:


torch.manual_seed(170487252)
np.random.seed(100)
x=x2[29:30]
# Process image and return variable
x=(x-mean)/std
x=torch.from_numpy(x)
x=x.type("torch.FloatTensor")
x=x.cuda()

x = cnnc.conv1(x)
for pp in range(21):
    x1= x[0,pp]
    x1=x1*std+mean
    x1 = torch.clamp(x1, 0, 255)
    x1=x1.detach().cpu().numpy()
    x1=np.rint(x1)
    x1=np.uint8(x1)
    out=np.zeros((len(x1),len(x1),3))
    for i in range(len(x1)):
        for n in range(len(x1)):
            out[i][n]=x1[i][n]
    #plt.imsave('vi/img_'+str(pp)+'.png', out/255)
    plt.subplot(3,7,pp+1)
    plt.imshow(out)
    plt.axis('off')
plt.title('layer1',loc='center')
plt.savefig(sys.argv[2]+"fig2_2.jpg")
plt.show()

