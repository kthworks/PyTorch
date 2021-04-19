## Convolutional Neural Network (CNN) 

MNIST data classification



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device=='cuda':
    torch.cuda.manual_seed_all(777)
    
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 50
    
    
# Dataset
inputs = torch.Tensor(1, 1, 28, 28) # batchsize x channel x height x width

mnist_train = dsets.MNIST(root='MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)


# Model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        #Input = 
        
        # 첫번째 층
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  #()
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        # 두번째 층
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        # Fully Connected Layer        
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        torch.nn.init.xavier_uniform(self.fc.weight)
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # FC를 위해서 Flatten
        out = self.fc(out)
        return out
        
        
total_batch = len(data_loader)
print(total_batch)
# Model 
model = CNN().to(device)

#Cost and optim
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
```

    1200
    

    <ipython-input-106-6d39276fbee3>:62: UserWarning: nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.
      torch.nn.init.xavier_uniform(self.fc.weight)
    

    [Epoch:    1] cost = 0.175038129
    [Epoch:    2] cost = 0.0548392795
    [Epoch:    3] cost = 0.0419386476
    [Epoch:    4] cost = 0.0322068594
    [Epoch:    5] cost = 0.0257292558
    [Epoch:    6] cost = 0.0217912029
    [Epoch:    7] cost = 0.0169503037
    [Epoch:    8] cost = 0.013955079
    [Epoch:    9] cost = 0.0110185612
    [Epoch:   10] cost = 0.010185793
    [Epoch:   11] cost = 0.00821572263
    [Epoch:   12] cost = 0.00729327742
    [Epoch:   13] cost = 0.00555294147
    [Epoch:   14] cost = 0.00504562398
    [Epoch:   15] cost = 0.00539928395
    


```python
# 학습을 진행하지 않을 것이므로 torch.no_grad()

with torch.no_grad():
    X_test = mnist_test.test_data.view(10000, 1, 28, 28).float().to(device)
    X_test = X_test[0:1000,:,:,:]
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test[0:1000]
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
```

    Accuracy: 0.9790000319480896
    

## Deep CNN



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device == 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 50

# Datasets
mnist_train = dsets.MNIST(root = 'MNIST_data/',               # 다운로드 경로지정
                          train = True,                       # True를 지정하면 훈련 데이터로 다운로드
                          transform = transforms.ToTensor(),  # 텐서로 변환
                          download = True)                    

mnist_test = dsets.MNIST(root = 'MNIST_data/',
                         train = False,
                         transform = transforms.ToTensor(),
                         download = True)

dataloader = torch.utils.data.DataLoader(dataset=mnist_train,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)

# Model
class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.keep_prob = 0.5
        
        #Layer1 Image Shape = (?, 28, 28, 1)
        #       Conv        = (?, 28, 28, 32)
        #       Pool        = (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2))
        
        #Layer2 Image Shape = (?,14,14, 32)
        #       Conv        = (?,14,14, 64)
        #       Pool        = (?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2))
        
        #Layer3 Image Shape = (?, 7, 7, 64)
        #       Conv        = (?, 7, 7, 128)
        #       Pool        = (?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding = 1))
        
        #Layer4 FC 4x4x128 inputs => 625 outputs
        self.fc1 = torch.nn.Linear(4*4*128,625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-self.keep_prob))
        
        #Layer5 final FC 625 inputs => 10 outputs
        self.fc2 = torch.nn.Linear(625,10,bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
                
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)    # Flatten them for FC
        out = self.layer4(out)    
        out = self.fc2(out)
        return out

            

  
```


```python
# DCNN Model 정의
model = CNN().to(device)

#Criterion and optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    
```

    총 배치의 수 : 1200
    [Epoch:    1] cost = 0.151509419
    [Epoch:    2] cost = 0.0467769802
    [Epoch:    3] cost = 0.0347131342
    [Epoch:    4] cost = 0.025673449
    [Epoch:    5] cost = 0.0220221486
    [Epoch:    6] cost = 0.0180410165
    [Epoch:    7] cost = 0.0135856671
    [Epoch:    8] cost = 0.0152827185
    [Epoch:    9] cost = 0.0106577203
    [Epoch:   10] cost = 0.011598805
    [Epoch:   11] cost = 0.00935635902
    [Epoch:   12] cost = 0.00848897826
    [Epoch:   13] cost = 0.00811122358
    [Epoch:   14] cost = 0.00895662606
    [Epoch:   15] cost = 0.00675837416
    
