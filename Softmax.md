## Softmax regression cost function 구현하기


```python
import torch
import torch.nn.functional as F

torch.manual_seed(1)
```




    <torch._C.Generator at 0x1cea9a319f0>



### 1. 파이토치로 소프트맥스 비용함수 구현 ( Low - level )


```python
z = torch.FloatTensor([1,2,3]) 

hypothesis = F.softmax(z, dim=0)
print(hypothesis)
hypothesis.sum() # 합이 1인지 확인
```

    tensor([0.0900, 0.2447, 0.6652])
    




    tensor(1.)




```python
# 비용함수 직접 구현

z = torch.rand(3,5,requires_grad=True) #3x5 random data

hypothesis = F.softmax(z, dim=1) #합이 1이되는 vector들로 변경
print(hypothesis)

#임의의 실제값 만들기
y = torch.randint(5, (3,)).long()
print(y)

#one-hot encoding
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1),1) #연산뒤에 _를 붙이면 덮어쓰기가 됨 

#Softmax 함수 구현
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
```

    tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
            [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
            [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)
    tensor([0, 2, 1])
    tensor(1.4689, grad_fn=<MeanBackward0>)
    

### 파이토치로 소프트맥수의 비용함수 구현하기 ( High - level )


```python
#low-level
torch.log(F.softmax(z, dim=1))


```




    tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
            [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
            [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]], grad_fn=<LogBackward>)




```python
# High-level
F.log_softmax(z, dim=1)
```




    tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
            [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
            [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]],
           grad_fn=<LogSoftmaxBackward>)




```python
#Low-level
(y_one_hot*-torch.log(F.softmax(z,dim=1))).sum(dim=1).mean()


```




    tensor(1.4689, grad_fn=<MeanBackward0>)




```python
#High-level
(y_one_hot*-F.log_softmax(z,dim=1)).sum(dim=1).mean()
```




    tensor(1.4689, grad_fn=<MeanBackward0>)




```python
#더 간단히
F.nll_loss(F.log_softmax(z,dim=1),y) 

#F.nll_loss()는 Negative Log Likelihood의 약자로, F.log_softmax()를 수행한 후 남은 수식들을 수행함
#F.cross_entropy()는 F.log_softmax()와 F.nll_loss()를 포함하고 있음.
```




    tensor(1.4689, grad_fn=<NllLossBackward>)




```python
#제일 간단히
F.cross_entropy(z,y)
```




    tensor(1.4689, grad_fn=<NllLossBackward>)



## Softmax 회귀 구현하기

소프트맥스 회귀를 low level과 F.cross_entropy를 사용해서 구현해보자


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

print(x_train.shape)
print(y_train.shape)
```

    torch.Size([8, 4])
    torch.Size([8])
    

### Low level 구현


```python
# set Weight and get z
W = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
z = x_train.matmul(W)+b 


# Optimizer
optimizer = optim.SGD([W, b], lr = 0.1)

# one-hot encoding
y_one_hot = torch.zeros_like(z)
y_one_hot.scatter_(1,y_train.unsqueeze(1),1)
print(y_one_hot)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    
    #H(X)
    z = x_train.matmul(W)+b
    
    #Cost function
    cost = (y_one_hot*-torch.log(F.softmax(z, dim=1))).sum(dim=0).mean()
    
    #Update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch: {}/{} Cost: {:.4f}'.format(epoch,nb_epochs,cost.item()))
    
```

    tensor([[0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [1., 0., 0.]])
    Epoch: 0/2000 Cost: 2.9296
    Epoch: 100/2000 Cost: 6.0195
    Epoch: 200/2000 Cost: 5.6420
    Epoch: 300/2000 Cost: 5.3287
    Epoch: 400/2000 Cost: 5.0231
    Epoch: 500/2000 Cost: 4.7239
    Epoch: 600/2000 Cost: 4.4342
    Epoch: 700/2000 Cost: 4.1555
    Epoch: 800/2000 Cost: 3.8880
    Epoch: 900/2000 Cost: 3.6309
    Epoch: 1000/2000 Cost: 3.3833
    Epoch: 1100/2000 Cost: 3.1438
    Epoch: 1200/2000 Cost: 2.9103
    Epoch: 1300/2000 Cost: 2.6809
    Epoch: 1400/2000 Cost: 2.4345
    Epoch: 1500/2000 Cost: 1.8837
    Epoch: 1600/2000 Cost: 1.2294
    Epoch: 1700/2000 Cost: 0.2150
    Epoch: 1800/2000 Cost: 0.2075
    Epoch: 1900/2000 Cost: 0.2006
    Epoch: 2000/2000 Cost: 0.1942
    

### High level 구현


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# Model Initialization
model = nn.Linear(4,3)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epoch = 2000
for epoch in range(nb_epoch+1):
    
    #H(x)
    z = model(x_train)
    
    #cost
    cost = F.cross_entropy(z,y_train)
    
    #update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch: {}/{} Cost: {:.4f}'.format(epoch, nb_epoch, cost.item()))
    
    
```

    Epoch: 0/2000 Cost: 1.6168
    Epoch: 10/2000 Cost: 1.4127
    Epoch: 20/2000 Cost: 1.2778
    Epoch: 30/2000 Cost: 1.1718
    Epoch: 40/2000 Cost: 1.0892
    Epoch: 50/2000 Cost: 1.0252
    Epoch: 60/2000 Cost: 0.9759
    Epoch: 70/2000 Cost: 0.9375
    Epoch: 80/2000 Cost: 0.9073
    Epoch: 90/2000 Cost: 0.8830
    Epoch: 100/2000 Cost: 0.8631
    Epoch: 110/2000 Cost: 0.8465
    Epoch: 120/2000 Cost: 0.8323
    Epoch: 130/2000 Cost: 0.8199
    Epoch: 140/2000 Cost: 0.8090
    Epoch: 150/2000 Cost: 0.7991
    Epoch: 160/2000 Cost: 0.7902
    Epoch: 170/2000 Cost: 0.7819
    Epoch: 180/2000 Cost: 0.7743
    Epoch: 190/2000 Cost: 0.7671
    Epoch: 200/2000 Cost: 0.7603
    Epoch: 210/2000 Cost: 0.7540
    Epoch: 220/2000 Cost: 0.7479
    Epoch: 230/2000 Cost: 0.7421
    Epoch: 240/2000 Cost: 0.7366
    Epoch: 250/2000 Cost: 0.7312
    Epoch: 260/2000 Cost: 0.7261
    Epoch: 270/2000 Cost: 0.7212
    Epoch: 280/2000 Cost: 0.7164
    Epoch: 290/2000 Cost: 0.7118
    Epoch: 300/2000 Cost: 0.7074
    Epoch: 310/2000 Cost: 0.7030
    Epoch: 320/2000 Cost: 0.6988
    Epoch: 330/2000 Cost: 0.6947
    Epoch: 340/2000 Cost: 0.6908
    Epoch: 350/2000 Cost: 0.6869
    Epoch: 360/2000 Cost: 0.6832
    Epoch: 370/2000 Cost: 0.6795
    Epoch: 380/2000 Cost: 0.6759
    Epoch: 390/2000 Cost: 0.6725
    Epoch: 400/2000 Cost: 0.6691
    Epoch: 410/2000 Cost: 0.6657
    Epoch: 420/2000 Cost: 0.6625
    Epoch: 430/2000 Cost: 0.6593
    Epoch: 440/2000 Cost: 0.6562
    Epoch: 450/2000 Cost: 0.6532
    Epoch: 460/2000 Cost: 0.6503
    Epoch: 470/2000 Cost: 0.6474
    Epoch: 480/2000 Cost: 0.6445
    Epoch: 490/2000 Cost: 0.6418
    Epoch: 500/2000 Cost: 0.6390
    Epoch: 510/2000 Cost: 0.6364
    Epoch: 520/2000 Cost: 0.6338
    Epoch: 530/2000 Cost: 0.6312
    Epoch: 540/2000 Cost: 0.6287
    Epoch: 550/2000 Cost: 0.6262
    Epoch: 560/2000 Cost: 0.6238
    Epoch: 570/2000 Cost: 0.6215
    Epoch: 580/2000 Cost: 0.6191
    Epoch: 590/2000 Cost: 0.6169
    Epoch: 600/2000 Cost: 0.6146
    Epoch: 610/2000 Cost: 0.6124
    Epoch: 620/2000 Cost: 0.6103
    Epoch: 630/2000 Cost: 0.6081
    Epoch: 640/2000 Cost: 0.6060
    Epoch: 650/2000 Cost: 0.6040
    Epoch: 660/2000 Cost: 0.6020
    Epoch: 670/2000 Cost: 0.6000
    Epoch: 680/2000 Cost: 0.5980
    Epoch: 690/2000 Cost: 0.5961
    Epoch: 700/2000 Cost: 0.5942
    Epoch: 710/2000 Cost: 0.5924
    Epoch: 720/2000 Cost: 0.5905
    Epoch: 730/2000 Cost: 0.5887
    Epoch: 740/2000 Cost: 0.5869
    Epoch: 750/2000 Cost: 0.5852
    Epoch: 760/2000 Cost: 0.5835
    Epoch: 770/2000 Cost: 0.5818
    Epoch: 780/2000 Cost: 0.5801
    Epoch: 790/2000 Cost: 0.5785
    Epoch: 800/2000 Cost: 0.5768
    Epoch: 810/2000 Cost: 0.5752
    Epoch: 820/2000 Cost: 0.5736
    Epoch: 830/2000 Cost: 0.5721
    Epoch: 840/2000 Cost: 0.5705
    Epoch: 850/2000 Cost: 0.5690
    Epoch: 860/2000 Cost: 0.5675
    Epoch: 870/2000 Cost: 0.5661
    Epoch: 880/2000 Cost: 0.5646
    Epoch: 890/2000 Cost: 0.5632
    Epoch: 900/2000 Cost: 0.5617
    Epoch: 910/2000 Cost: 0.5603
    Epoch: 920/2000 Cost: 0.5590
    Epoch: 930/2000 Cost: 0.5576
    Epoch: 940/2000 Cost: 0.5562
    Epoch: 950/2000 Cost: 0.5549
    Epoch: 960/2000 Cost: 0.5536
    Epoch: 970/2000 Cost: 0.5523
    Epoch: 980/2000 Cost: 0.5510
    Epoch: 990/2000 Cost: 0.5497
    Epoch: 1000/2000 Cost: 0.5485
    Epoch: 1010/2000 Cost: 0.5472
    Epoch: 1020/2000 Cost: 0.5460
    Epoch: 1030/2000 Cost: 0.5448
    Epoch: 1040/2000 Cost: 0.5436
    Epoch: 1050/2000 Cost: 0.5424
    Epoch: 1060/2000 Cost: 0.5412
    Epoch: 1070/2000 Cost: 0.5400
    Epoch: 1080/2000 Cost: 0.5389
    Epoch: 1090/2000 Cost: 0.5377
    Epoch: 1100/2000 Cost: 0.5366
    Epoch: 1110/2000 Cost: 0.5355
    Epoch: 1120/2000 Cost: 0.5344
    Epoch: 1130/2000 Cost: 0.5333
    Epoch: 1140/2000 Cost: 0.5322
    Epoch: 1150/2000 Cost: 0.5311
    Epoch: 1160/2000 Cost: 0.5301
    Epoch: 1170/2000 Cost: 0.5290
    Epoch: 1180/2000 Cost: 0.5280
    Epoch: 1190/2000 Cost: 0.5270
    Epoch: 1200/2000 Cost: 0.5259
    Epoch: 1210/2000 Cost: 0.5249
    Epoch: 1220/2000 Cost: 0.5239
    Epoch: 1230/2000 Cost: 0.5229
    Epoch: 1240/2000 Cost: 0.5219
    Epoch: 1250/2000 Cost: 0.5210
    Epoch: 1260/2000 Cost: 0.5200
    Epoch: 1270/2000 Cost: 0.5190
    Epoch: 1280/2000 Cost: 0.5181
    Epoch: 1290/2000 Cost: 0.5171
    Epoch: 1300/2000 Cost: 0.5162
    Epoch: 1310/2000 Cost: 0.5153
    Epoch: 1320/2000 Cost: 0.5144
    Epoch: 1330/2000 Cost: 0.5135
    Epoch: 1340/2000 Cost: 0.5125
    Epoch: 1350/2000 Cost: 0.5117
    Epoch: 1360/2000 Cost: 0.5108
    Epoch: 1370/2000 Cost: 0.5099
    Epoch: 1380/2000 Cost: 0.5090
    Epoch: 1390/2000 Cost: 0.5081
    Epoch: 1400/2000 Cost: 0.5073
    Epoch: 1410/2000 Cost: 0.5064
    Epoch: 1420/2000 Cost: 0.5056
    Epoch: 1430/2000 Cost: 0.5047
    Epoch: 1440/2000 Cost: 0.5039
    Epoch: 1450/2000 Cost: 0.5031
    Epoch: 1460/2000 Cost: 0.5023
    Epoch: 1470/2000 Cost: 0.5014
    Epoch: 1480/2000 Cost: 0.5006
    Epoch: 1490/2000 Cost: 0.4998
    Epoch: 1500/2000 Cost: 0.4990
    Epoch: 1510/2000 Cost: 0.4982
    Epoch: 1520/2000 Cost: 0.4975
    Epoch: 1530/2000 Cost: 0.4967
    Epoch: 1540/2000 Cost: 0.4959
    Epoch: 1550/2000 Cost: 0.4951
    Epoch: 1560/2000 Cost: 0.4944
    Epoch: 1570/2000 Cost: 0.4936
    Epoch: 1580/2000 Cost: 0.4929
    Epoch: 1590/2000 Cost: 0.4921
    Epoch: 1600/2000 Cost: 0.4914
    Epoch: 1610/2000 Cost: 0.4906
    Epoch: 1620/2000 Cost: 0.4899
    Epoch: 1630/2000 Cost: 0.4892
    Epoch: 1640/2000 Cost: 0.4884
    Epoch: 1650/2000 Cost: 0.4877
    Epoch: 1660/2000 Cost: 0.4870
    Epoch: 1670/2000 Cost: 0.4863
    Epoch: 1680/2000 Cost: 0.4856
    Epoch: 1690/2000 Cost: 0.4849
    Epoch: 1700/2000 Cost: 0.4842
    Epoch: 1710/2000 Cost: 0.4835
    Epoch: 1720/2000 Cost: 0.4828
    Epoch: 1730/2000 Cost: 0.4821
    Epoch: 1740/2000 Cost: 0.4814
    Epoch: 1750/2000 Cost: 0.4807
    Epoch: 1760/2000 Cost: 0.4801
    Epoch: 1770/2000 Cost: 0.4794
    Epoch: 1780/2000 Cost: 0.4787
    Epoch: 1790/2000 Cost: 0.4781
    Epoch: 1800/2000 Cost: 0.4774
    Epoch: 1810/2000 Cost: 0.4768
    Epoch: 1820/2000 Cost: 0.4761
    Epoch: 1830/2000 Cost: 0.4755
    Epoch: 1840/2000 Cost: 0.4748
    Epoch: 1850/2000 Cost: 0.4742
    Epoch: 1860/2000 Cost: 0.4736
    Epoch: 1870/2000 Cost: 0.4729
    Epoch: 1880/2000 Cost: 0.4723
    Epoch: 1890/2000 Cost: 0.4717
    Epoch: 1900/2000 Cost: 0.4710
    Epoch: 1910/2000 Cost: 0.4704
    Epoch: 1920/2000 Cost: 0.4698
    Epoch: 1930/2000 Cost: 0.4692
    Epoch: 1940/2000 Cost: 0.4686
    Epoch: 1950/2000 Cost: 0.4680
    Epoch: 1960/2000 Cost: 0.4674
    Epoch: 1970/2000 Cost: 0.4668
    Epoch: 1980/2000 Cost: 0.4662
    Epoch: 1990/2000 Cost: 0.4656
    Epoch: 2000/2000 Cost: 0.4650
    

### Softmax 회귀 클래스로 구현하기


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class Softmax_classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)
        
    def forward(self,x):
        return self.linear(x)

model = Softmax_classification()

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    
    #H(x)
    z = model(x_train)
    
    #cost
    cost = F.cross_entropy(z,y_train)
    
    #update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch: {}/{} Cost: {:.4f}'.format(epoch, nb_epochs, cost.item()))
```

    Epoch: 0/1000 Cost: 1.6168
    Epoch: 100/1000 Cost: 0.6589
    Epoch: 200/1000 Cost: 0.5734
    Epoch: 300/1000 Cost: 0.5182
    Epoch: 400/1000 Cost: 0.4733
    Epoch: 500/1000 Cost: 0.4335
    Epoch: 600/1000 Cost: 0.3966
    Epoch: 700/1000 Cost: 0.3609
    Epoch: 800/1000 Cost: 0.3254
    Epoch: 900/1000 Cost: 0.2892
    Epoch: 1000/1000 Cost: 0.2541
    

## Softmax regression으로 MNIST Data 분류하기


```python
# import library

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", device)
```

    다음 기기로 학습합니다: cuda
    


```python
# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
```


```python
# hyperparameters
training_epochs = 15
batch_size = 100
```


```python
#MNIST classifier 구현하기

#dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                        train = False,
                        transform=transforms.ToTensor(),
                        download=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 503: Service Unavailable
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to MNIST_data/MNIST\raw\train-images-idx3-ubyte.gz
    

    31.0%IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    90.5%IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    102.8%
    

    Extracting MNIST_data/MNIST\raw\train-labels-idx1-ubyte.gz to MNIST_data/MNIST\raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 503: Service Unavailable
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST\raw\t10k-images-idx3-ubyte.gz
    

    100.0%
    

    Extracting MNIST_data/MNIST\raw\t10k-images-idx3-ubyte.gz to MNIST_data/MNIST\raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 503: Service Unavailable
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST\raw\t10k-labels-idx1-ubyte.gz
    

    112.7%
    C:\Users\ImedisynRnD2\anaconda3\envs\pytorch\lib\site-packages\torchvision\datasets\mnist.py:502: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:143.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
    

    Extracting MNIST_data/MNIST\raw\t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST\raw
    
    Processing...
    Done!
    


```python
# Dataset loader
data_loader = DataLoader(dataset=mnist_train,
                        batch_size = batch_size,
                        shuffle=True,
                        drop_last=True)

#MNIST data image of shape : 28 * 28 = 784
linear = nn.Linear(784,10, bias = True).to(device)

#cost and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    
    for X, Y in data_loader:
        
        #배치크기가 100이므로 아래의 연산에서 X는 (100,784)의 텐서가 된다.
        X = X.view(-1,28*28).to(device)
        
        # 레이블은 one-hot encoding이 된 상태가 아니라 0~9의 정수.
        Y = Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis,Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
        
    print('Epoch:', '%04d' % (epoch +1), 'cost =', '{:.9f}'.format(avg_cost))
    
print('Learning finished')


```

    Epoch: 0001 cost = 0.536015809
    Epoch: 0002 cost = 0.359202534
    Epoch: 0003 cost = 0.331243694
    Epoch: 0004 cost = 0.316479772
    Epoch: 0005 cost = 0.306780636
    Epoch: 0006 cost = 0.300162762
    Epoch: 0007 cost = 0.295002848
    Epoch: 0008 cost = 0.290735900
    Epoch: 0009 cost = 0.287426829
    Epoch: 0010 cost = 0.284311414
    Epoch: 0011 cost = 0.281867415
    Epoch: 0012 cost = 0.279607654
    Epoch: 0013 cost = 0.277803063
    Epoch: 0014 cost = 0.276044399
    Epoch: 0015 cost = 0.274502218
    Learning finished
    
