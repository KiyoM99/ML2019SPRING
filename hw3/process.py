
# coding: utf-8

# In[1]:


import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import pickle as pk
import sys


# In[25]:


tr=open(sys.argv[1],'r',encoding='utf8')
train=tr.read()
train.replace(',',' ')
train1=train.split("\n")
train1=train1[1:-1]
train3=train1[:]
for i in range(len(train1)):
    train3[i]=train3[i][2:]


# In[26]:


train2=[]
for i in range(len(train1)):
    train2.append(train3[i].split(' '))


# In[27]:


tim=time.time()
for i in range(len(train2)):
    for n in range(len(train2[0])):
        train2[i][n]=int(train2[i][n])
    if i%200==0:
        print(time.time()-tim)


# In[28]:


for i in range(len(train2)):
    train2[i].sort()


# In[29]:


a=[]
for i in range(len(train2)):
    if train2[i][1300]==0:
        a.append(i)


# In[30]:


coo=0
for i in a:
    train1.pop(i-coo)
    coo+=1


# In[23]:


len(train1)


# In[31]:


print(train1[0])


# In[32]:


x1=np.zeros((len(train1),48*48))
x=np.zeros((len(train1),1,48,48))
y=np.zeros(len(train1))
for i in range(len(train1)):
    y[i]=int(train1[i][0])
    train1[i]=train1[i][2:]


# In[34]:


for i in range(len(train1)):    
    x1[i]=np.fromstring(train1[i],dtype=int, sep=' ')
    #x[i][0]=x1[i].reshape((48,48))


# In[42]:


x1=x1.astype(int)


# In[2]:


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


# In[3]:


def css(cs):# numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    # re-normalize the cumsum
    cs = nj / N

    # cast it back to uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')
    return cs


# In[59]:


for i in range(len(x1)):
    hist = get_histogram(x1[i], 256)
    cs = cumsum(hist)
    look=css(cs)
    x1[i]=look[x1[i]]
    x[i][0]=np.reshape(x1[i],(48,48))


# In[85]:


x=x.astype(np.uint8)


# In[86]:


print(x[2334][0][0])
img = Image.fromarray(x[2334][0])
img.save('1123.bmp')


# In[87]:


vadx=x[:2700]
vady=y[:2700]


# In[88]:


x=x[2700:]
y=y[2700:]


# In[96]:


f = open('train_x.pk', 'wb')
pk.dump(x, f)
f.close()
f = open('label_y.pk', 'wb')
pk.dump(y, f)
f.close()
f = open('vadx.pk', 'wb')
pk.dump(vadx, f)
f.close()
f = open('vady.pk', 'wb')
pk.dump(vady, f)
f.close()

