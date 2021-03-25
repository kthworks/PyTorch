# Logistic Regression

이진 분류를 위한 s자 모양의 그래프가 필요함 -> Sigmoid 함수 사용

$ H(x) = f(Wx+b) $ 

$ H(x) = sigmoid(Wx+b) = \frac{1}{1+e^-(Wx+b)} = \sigma(Wx + b) $

우선,W와 b에 따라 Sigmoid 함수가 어떻게 변하는지 살펴보자


```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): #시그모이드 함수 정의
    return 1/(1+np.exp(-x))
```

## 1. W와 b에 따른 sigmoid 그래프



```python
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = sigmoid(0.5*x) # W값 줄였을 떄
y3 = sigmoid(2*x) # W값 늘였을 때

plt.plot(x,y1,'r')
plt.plot(x,y2,'g') # W값 줄였을 때
plt.plot(x,y3,'b') # W값 늘였을 때
plt.plot([0,0], [1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# W 값에 따라 
```


![png](/Logistic Regression/output_3_0.png)



```python
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = sigmoid(-2+x) # b값 줄였을 떄
y3 = sigmoid(2+x) # b값 늘였을 때

plt.plot(x,y1,'r')
plt.plot(x,y2,'g') # b값 줄였을 때
plt.plot(x,y3,'b') # b값 늘였을 때
plt.plot([0,0], [1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

#B값에 따라 좌우로 이동
```


![png](/Logistic Regression/output_4_0.png)


## Cost function

Logistic regression의 Hypothesis는 $ H(x) = sigmoid(Wx+b) $ 이다.

선형 회귀에서 사용했던 cost function은 $ cost(W,b) = \frac{1}{n}\sum^{n}_{i=1}[y^{i}-H(x^{(i)})]^2 $ 였다.

그러나, sigmoid 함수가 들어가면서 미분을 하면 local minimum에 빠질수 있는 non-convex 형태의 그래프가 나온다.

시그모이드 함수의 특징을 고려했을 때, 실제값이 1일 때 예측값이 0에 가까워지면 오차가 커져야하고, 실제값이 0일 때 예측값이 1에 가까워지면 오차가 커져야한다. 이를 충족하는 함수는 로그함수다.

if $ y = 1, cost(H(x),y) = -log(H(x)) $   

if $ y = 0, cost(H(x),y) = -log(1-H(x)) $

따라서, 이 두 식을 통합하면,

$ Cost (H(x), y) = - [ylogH(x) + (1-y)log(1-H(x))] $ 이 된다.

선형회귀에서 했던 MSE처럼, 여기서도 모든 오차의 평균을 구함.

$ Cost(W) = -\frac{1}{n}\sum^{n}_{i=1}[y^{(i)}log(H(x^{(i)}) + (1-y^{(i)})log(1-H(x^{(i)}))] $ 


```python
import numpy as np
from matplotlib import pyplot as plt
import math

x = np.arange(0, 1, 0.01)
plt.plot(np.arange(0, 1, 0.01), -np.log(x))
plt.plot(np.arange(0,1,0.01), -np.log(1-x))
plt.plot([0.5,0.5], [5,0.0], ':') # 가운데 점선 추가
plt.show()
```

    c:\users\imedisynrnd2\appdata\local\programs\python\python36\lib\site-packages\ipykernel_launcher.py:6: RuntimeWarning: divide by zero encountered in log
      
    


![png](/Logistic Regression/output_6_1.png)


## 파이토치로 Logistic Regression 구현하기

파이토치로 다중 logistic regression을 구현해 보자.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)
```

    torch.Size([6, 2])
    torch.Size([6, 1])
    

현재 x_train은 6x2 행렬이며, y_train은 6x1 벡터이므로, XW = Y 가 성립되기 위해서는 W = 2x1 가 되어야함.


```python
W = torch.zeros((2,1), requires_grad = True) 
b = torch.zeros(1, requires_grad = True)

hypothesis = 1/ (1+torch.exp(-x_train.matmul(W) + b))
# or torch.sigmoid(x_train.matmul(W)+b)

print(hypothesis)
```

    tensor([[0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000]], grad_fn=<MulBackward0>)
    

현재 총 6개의 원소가 존재하지만 하나의 샘플, 즉 하나의 원소에 대해서만 오차를 구하는 식을 작성해보자.


```python
-(y_train[0] * torch.log(hypothesis[0])) + (1-y_train[0]) * torch.log(1-hypothesis[0])
```




    tensor([-0.6931], grad_fn=<AddBackward0>)



이런 식으로, 모든 원소에 대해서 오차를 구해보자


```python
losses = -(y_train * torch.log(hypothesis) + (1-y_train)*torch.log(1 - hypothesis))
print(losses)

cost = losses.mean()
```

    tensor([[0.6931],
            [0.6931],
            [0.6931],
            [0.6931],
            [0.6931],
            [0.6931]], grad_fn=<NegBackward>)
    tensor(0.6931, grad_fn=<MeanBackward0>)
    

위에서는 cost function을 직접 구현했지만, 라이브러리 함수를 이용해서 간편하게 구할 수도 있다.


```python
F.binary_cross_entropy(hypothesis, y_train)
```




    tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)




```python
# Training 전체 코드
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

optimizer = optim.SGD([W,b], lr=1)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    
    # H(x)
    hypothesis = torch.sigmoid(x_train.matmul(W)+b)
    
    # Cost
    cost =  -(y_train * torch.log(hypothesis) + 
             (1 - y_train) * torch.log(1 - hypothesis)).mean()
    
    # H(x) Update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch: {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))                                                   
    
    
```

    Epoch:    0/2000 Cost: 0.693147
    Epoch:  100/2000 Cost: 0.134722
    Epoch:  200/2000 Cost: 0.080643
    Epoch:  300/2000 Cost: 0.057900
    Epoch:  400/2000 Cost: 0.045300
    Epoch:  500/2000 Cost: 0.037261
    Epoch:  600/2000 Cost: 0.031673
    Epoch:  700/2000 Cost: 0.027556
    Epoch:  800/2000 Cost: 0.024394
    Epoch:  900/2000 Cost: 0.021888
    Epoch: 1000/2000 Cost: 0.019852
    Epoch: 1100/2000 Cost: 0.018165
    Epoch: 1200/2000 Cost: 0.016743
    Epoch: 1300/2000 Cost: 0.015528
    Epoch: 1400/2000 Cost: 0.014479
    Epoch: 1500/2000 Cost: 0.013562
    Epoch: 1600/2000 Cost: 0.012755
    Epoch: 1700/2000 Cost: 0.012039
    Epoch: 1800/2000 Cost: 0.011400
    Epoch: 1900/2000 Cost: 0.010825
    Epoch: 2000/2000 Cost: 0.010305
    


```python
# 훈련이 잘 되었는지 확인

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
```

    tensor([[5.4423e-05],
            [1.6844e-02],
            [2.0160e-02],
            [9.7644e-01],
            [9.9951e-01],
            [9.9994e-01]], grad_fn=<SigmoidBackward>)
    tensor([[False],
            [False],
            [False],
            [ True],
            [ True],
            [ True]])
    

## nn.Module로 구현하는 Logistic Regression


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Dataset
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# Model
model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    
    #H(x)
    hypothesis = model(x_train)
    
    #Cost function
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    #H(x) Update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch%10 ==0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch: {:4d}/{} Cost: {:.4f} Accuracy: {:2.2f}%'.format(epoch,nb_epochs,cost.item(), accuracy * 100))

```

    Epoch:    0/1000 Cost: 0.5397 Accuracy: 83.33%
    Epoch:   10/1000 Cost: 0.6149 Accuracy: 66.67%
    Epoch:   20/1000 Cost: 0.4419 Accuracy: 66.67%
    Epoch:   30/1000 Cost: 0.3731 Accuracy: 83.33%
    Epoch:   40/1000 Cost: 0.3164 Accuracy: 83.33%
    Epoch:   50/1000 Cost: 0.2661 Accuracy: 83.33%
    Epoch:   60/1000 Cost: 0.2205 Accuracy: 100.00%
    Epoch:   70/1000 Cost: 0.1821 Accuracy: 100.00%
    Epoch:   80/1000 Cost: 0.1573 Accuracy: 100.00%
    Epoch:   90/1000 Cost: 0.1441 Accuracy: 100.00%
    Epoch:  100/1000 Cost: 0.1343 Accuracy: 100.00%
    Epoch:  110/1000 Cost: 0.1258 Accuracy: 100.00%
    Epoch:  120/1000 Cost: 0.1183 Accuracy: 100.00%
    Epoch:  130/1000 Cost: 0.1117 Accuracy: 100.00%
    Epoch:  140/1000 Cost: 0.1058 Accuracy: 100.00%
    Epoch:  150/1000 Cost: 0.1005 Accuracy: 100.00%
    Epoch:  160/1000 Cost: 0.0957 Accuracy: 100.00%
    Epoch:  170/1000 Cost: 0.0914 Accuracy: 100.00%
    Epoch:  180/1000 Cost: 0.0874 Accuracy: 100.00%
    Epoch:  190/1000 Cost: 0.0838 Accuracy: 100.00%
    Epoch:  200/1000 Cost: 0.0805 Accuracy: 100.00%
    Epoch:  210/1000 Cost: 0.0774 Accuracy: 100.00%
    Epoch:  220/1000 Cost: 0.0746 Accuracy: 100.00%
    Epoch:  230/1000 Cost: 0.0720 Accuracy: 100.00%
    Epoch:  240/1000 Cost: 0.0695 Accuracy: 100.00%
    Epoch:  250/1000 Cost: 0.0672 Accuracy: 100.00%
    Epoch:  260/1000 Cost: 0.0651 Accuracy: 100.00%
    Epoch:  270/1000 Cost: 0.0631 Accuracy: 100.00%
    Epoch:  280/1000 Cost: 0.0612 Accuracy: 100.00%
    Epoch:  290/1000 Cost: 0.0595 Accuracy: 100.00%
    Epoch:  300/1000 Cost: 0.0578 Accuracy: 100.00%
    Epoch:  310/1000 Cost: 0.0562 Accuracy: 100.00%
    Epoch:  320/1000 Cost: 0.0548 Accuracy: 100.00%
    Epoch:  330/1000 Cost: 0.0534 Accuracy: 100.00%
    Epoch:  340/1000 Cost: 0.0520 Accuracy: 100.00%
    Epoch:  350/1000 Cost: 0.0508 Accuracy: 100.00%
    Epoch:  360/1000 Cost: 0.0495 Accuracy: 100.00%
    Epoch:  370/1000 Cost: 0.0484 Accuracy: 100.00%
    Epoch:  380/1000 Cost: 0.0473 Accuracy: 100.00%
    Epoch:  390/1000 Cost: 0.0463 Accuracy: 100.00%
    Epoch:  400/1000 Cost: 0.0453 Accuracy: 100.00%
    Epoch:  410/1000 Cost: 0.0443 Accuracy: 100.00%
    Epoch:  420/1000 Cost: 0.0434 Accuracy: 100.00%
    Epoch:  430/1000 Cost: 0.0425 Accuracy: 100.00%
    Epoch:  440/1000 Cost: 0.0417 Accuracy: 100.00%
    Epoch:  450/1000 Cost: 0.0408 Accuracy: 100.00%
    Epoch:  460/1000 Cost: 0.0401 Accuracy: 100.00%
    Epoch:  470/1000 Cost: 0.0393 Accuracy: 100.00%
    Epoch:  480/1000 Cost: 0.0386 Accuracy: 100.00%
    Epoch:  490/1000 Cost: 0.0379 Accuracy: 100.00%
    Epoch:  500/1000 Cost: 0.0372 Accuracy: 100.00%
    Epoch:  510/1000 Cost: 0.0366 Accuracy: 100.00%
    Epoch:  520/1000 Cost: 0.0360 Accuracy: 100.00%
    Epoch:  530/1000 Cost: 0.0354 Accuracy: 100.00%
    Epoch:  540/1000 Cost: 0.0348 Accuracy: 100.00%
    Epoch:  550/1000 Cost: 0.0342 Accuracy: 100.00%
    Epoch:  560/1000 Cost: 0.0337 Accuracy: 100.00%
    Epoch:  570/1000 Cost: 0.0331 Accuracy: 100.00%
    Epoch:  580/1000 Cost: 0.0326 Accuracy: 100.00%
    Epoch:  590/1000 Cost: 0.0321 Accuracy: 100.00%
    Epoch:  600/1000 Cost: 0.0316 Accuracy: 100.00%
    Epoch:  610/1000 Cost: 0.0312 Accuracy: 100.00%
    Epoch:  620/1000 Cost: 0.0307 Accuracy: 100.00%
    Epoch:  630/1000 Cost: 0.0303 Accuracy: 100.00%
    Epoch:  640/1000 Cost: 0.0299 Accuracy: 100.00%
    Epoch:  650/1000 Cost: 0.0294 Accuracy: 100.00%
    Epoch:  660/1000 Cost: 0.0290 Accuracy: 100.00%
    Epoch:  670/1000 Cost: 0.0287 Accuracy: 100.00%
    Epoch:  680/1000 Cost: 0.0283 Accuracy: 100.00%
    Epoch:  690/1000 Cost: 0.0279 Accuracy: 100.00%
    Epoch:  700/1000 Cost: 0.0275 Accuracy: 100.00%
    Epoch:  710/1000 Cost: 0.0272 Accuracy: 100.00%
    Epoch:  720/1000 Cost: 0.0268 Accuracy: 100.00%
    Epoch:  730/1000 Cost: 0.0265 Accuracy: 100.00%
    Epoch:  740/1000 Cost: 0.0262 Accuracy: 100.00%
    Epoch:  750/1000 Cost: 0.0259 Accuracy: 100.00%
    Epoch:  760/1000 Cost: 0.0256 Accuracy: 100.00%
    Epoch:  770/1000 Cost: 0.0252 Accuracy: 100.00%
    Epoch:  780/1000 Cost: 0.0250 Accuracy: 100.00%
    Epoch:  790/1000 Cost: 0.0247 Accuracy: 100.00%
    Epoch:  800/1000 Cost: 0.0244 Accuracy: 100.00%
    Epoch:  810/1000 Cost: 0.0241 Accuracy: 100.00%
    Epoch:  820/1000 Cost: 0.0238 Accuracy: 100.00%
    Epoch:  830/1000 Cost: 0.0236 Accuracy: 100.00%
    Epoch:  840/1000 Cost: 0.0233 Accuracy: 100.00%
    Epoch:  850/1000 Cost: 0.0231 Accuracy: 100.00%
    Epoch:  860/1000 Cost: 0.0228 Accuracy: 100.00%
    Epoch:  870/1000 Cost: 0.0226 Accuracy: 100.00%
    Epoch:  880/1000 Cost: 0.0223 Accuracy: 100.00%
    Epoch:  890/1000 Cost: 0.0221 Accuracy: 100.00%
    Epoch:  900/1000 Cost: 0.0219 Accuracy: 100.00%
    Epoch:  910/1000 Cost: 0.0217 Accuracy: 100.00%
    Epoch:  920/1000 Cost: 0.0214 Accuracy: 100.00%
    Epoch:  930/1000 Cost: 0.0212 Accuracy: 100.00%
    Epoch:  940/1000 Cost: 0.0210 Accuracy: 100.00%
    Epoch:  950/1000 Cost: 0.0208 Accuracy: 100.00%
    Epoch:  960/1000 Cost: 0.0206 Accuracy: 100.00%
    Epoch:  970/1000 Cost: 0.0204 Accuracy: 100.00%
    Epoch:  980/1000 Cost: 0.0202 Accuracy: 100.00%
    Epoch:  990/1000 Cost: 0.0200 Accuracy: 100.00%
    Epoch: 1000/1000 Cost: 0.0198 Accuracy: 100.00%
    

## 03. 클래스로 파이토치 모델 구현하기


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#앞에서 구현한 방식
model = nn.Sequential( 
    nn.Linear(2,1)
    nn.Sigmoid()
)

#클래스로 구현

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        return self.sigmoid(self.linear(x))



```

### Logistic Regression 클래스로 구현하기


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Dataset
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# Model

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        return self.sigmoid(self.linear(x))
    
model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epoch = 1000
for epoch in range(nb_epoch+1):
    
    #H(x)
    hypothesis = model(x_train)
    
    #cost
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    #H(x) update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch%10 == 0:
        predict = hypothesis >= torch.FloatTensor([0.5])
        correct_predict = predict.float() == y_train
        accuracy = correct_predict.sum().item() / len(correct_predict)
        
        print('Epoch: {:4d}/{} Cost: {:.4f} Accuracy: {:2.2f}%'.format(epoch, nb_epoch, cost.item(), accuracy*100))
        

```

    Epoch:    0/1000 Cost: 0.5397 Accuracy: 83.33%
    Epoch:   10/1000 Cost: 0.6149 Accuracy: 66.67%
    Epoch:   20/1000 Cost: 0.4419 Accuracy: 66.67%
    Epoch:   30/1000 Cost: 0.3731 Accuracy: 83.33%
    Epoch:   40/1000 Cost: 0.3164 Accuracy: 83.33%
    Epoch:   50/1000 Cost: 0.2661 Accuracy: 83.33%
    Epoch:   60/1000 Cost: 0.2205 Accuracy: 100.00%
    Epoch:   70/1000 Cost: 0.1821 Accuracy: 100.00%
    Epoch:   80/1000 Cost: 0.1573 Accuracy: 100.00%
    Epoch:   90/1000 Cost: 0.1441 Accuracy: 100.00%
    Epoch:  100/1000 Cost: 0.1343 Accuracy: 100.00%
    Epoch:  110/1000 Cost: 0.1258 Accuracy: 100.00%
    Epoch:  120/1000 Cost: 0.1183 Accuracy: 100.00%
    Epoch:  130/1000 Cost: 0.1117 Accuracy: 100.00%
    Epoch:  140/1000 Cost: 0.1058 Accuracy: 100.00%
    Epoch:  150/1000 Cost: 0.1005 Accuracy: 100.00%
    Epoch:  160/1000 Cost: 0.0957 Accuracy: 100.00%
    Epoch:  170/1000 Cost: 0.0914 Accuracy: 100.00%
    Epoch:  180/1000 Cost: 0.0874 Accuracy: 100.00%
    Epoch:  190/1000 Cost: 0.0838 Accuracy: 100.00%
    Epoch:  200/1000 Cost: 0.0805 Accuracy: 100.00%
    Epoch:  210/1000 Cost: 0.0774 Accuracy: 100.00%
    Epoch:  220/1000 Cost: 0.0746 Accuracy: 100.00%
    Epoch:  230/1000 Cost: 0.0720 Accuracy: 100.00%
    Epoch:  240/1000 Cost: 0.0695 Accuracy: 100.00%
    Epoch:  250/1000 Cost: 0.0672 Accuracy: 100.00%
    Epoch:  260/1000 Cost: 0.0651 Accuracy: 100.00%
    Epoch:  270/1000 Cost: 0.0631 Accuracy: 100.00%
    Epoch:  280/1000 Cost: 0.0612 Accuracy: 100.00%
    Epoch:  290/1000 Cost: 0.0595 Accuracy: 100.00%
    Epoch:  300/1000 Cost: 0.0578 Accuracy: 100.00%
    Epoch:  310/1000 Cost: 0.0562 Accuracy: 100.00%
    Epoch:  320/1000 Cost: 0.0548 Accuracy: 100.00%
    Epoch:  330/1000 Cost: 0.0534 Accuracy: 100.00%
    Epoch:  340/1000 Cost: 0.0520 Accuracy: 100.00%
    Epoch:  350/1000 Cost: 0.0508 Accuracy: 100.00%
    Epoch:  360/1000 Cost: 0.0495 Accuracy: 100.00%
    Epoch:  370/1000 Cost: 0.0484 Accuracy: 100.00%
    Epoch:  380/1000 Cost: 0.0473 Accuracy: 100.00%
    Epoch:  390/1000 Cost: 0.0463 Accuracy: 100.00%
    Epoch:  400/1000 Cost: 0.0453 Accuracy: 100.00%
    Epoch:  410/1000 Cost: 0.0443 Accuracy: 100.00%
    Epoch:  420/1000 Cost: 0.0434 Accuracy: 100.00%
    Epoch:  430/1000 Cost: 0.0425 Accuracy: 100.00%
    Epoch:  440/1000 Cost: 0.0417 Accuracy: 100.00%
    Epoch:  450/1000 Cost: 0.0408 Accuracy: 100.00%
    Epoch:  460/1000 Cost: 0.0401 Accuracy: 100.00%
    Epoch:  470/1000 Cost: 0.0393 Accuracy: 100.00%
    Epoch:  480/1000 Cost: 0.0386 Accuracy: 100.00%
    Epoch:  490/1000 Cost: 0.0379 Accuracy: 100.00%
    Epoch:  500/1000 Cost: 0.0372 Accuracy: 100.00%
    Epoch:  510/1000 Cost: 0.0366 Accuracy: 100.00%
    Epoch:  520/1000 Cost: 0.0360 Accuracy: 100.00%
    Epoch:  530/1000 Cost: 0.0354 Accuracy: 100.00%
    Epoch:  540/1000 Cost: 0.0348 Accuracy: 100.00%
    Epoch:  550/1000 Cost: 0.0342 Accuracy: 100.00%
    Epoch:  560/1000 Cost: 0.0337 Accuracy: 100.00%
    Epoch:  570/1000 Cost: 0.0331 Accuracy: 100.00%
    Epoch:  580/1000 Cost: 0.0326 Accuracy: 100.00%
    Epoch:  590/1000 Cost: 0.0321 Accuracy: 100.00%
    Epoch:  600/1000 Cost: 0.0316 Accuracy: 100.00%
    Epoch:  610/1000 Cost: 0.0312 Accuracy: 100.00%
    Epoch:  620/1000 Cost: 0.0307 Accuracy: 100.00%
    Epoch:  630/1000 Cost: 0.0303 Accuracy: 100.00%
    Epoch:  640/1000 Cost: 0.0299 Accuracy: 100.00%
    Epoch:  650/1000 Cost: 0.0294 Accuracy: 100.00%
    Epoch:  660/1000 Cost: 0.0290 Accuracy: 100.00%
    Epoch:  670/1000 Cost: 0.0287 Accuracy: 100.00%
    Epoch:  680/1000 Cost: 0.0283 Accuracy: 100.00%
    Epoch:  690/1000 Cost: 0.0279 Accuracy: 100.00%
    Epoch:  700/1000 Cost: 0.0275 Accuracy: 100.00%
    Epoch:  710/1000 Cost: 0.0272 Accuracy: 100.00%
    Epoch:  720/1000 Cost: 0.0268 Accuracy: 100.00%
    Epoch:  730/1000 Cost: 0.0265 Accuracy: 100.00%
    Epoch:  740/1000 Cost: 0.0262 Accuracy: 100.00%
    Epoch:  750/1000 Cost: 0.0259 Accuracy: 100.00%
    Epoch:  760/1000 Cost: 0.0256 Accuracy: 100.00%
    Epoch:  770/1000 Cost: 0.0252 Accuracy: 100.00%
    Epoch:  780/1000 Cost: 0.0250 Accuracy: 100.00%
    Epoch:  790/1000 Cost: 0.0247 Accuracy: 100.00%
    Epoch:  800/1000 Cost: 0.0244 Accuracy: 100.00%
    Epoch:  810/1000 Cost: 0.0241 Accuracy: 100.00%
    Epoch:  820/1000 Cost: 0.0238 Accuracy: 100.00%
    Epoch:  830/1000 Cost: 0.0236 Accuracy: 100.00%
    Epoch:  840/1000 Cost: 0.0233 Accuracy: 100.00%
    Epoch:  850/1000 Cost: 0.0231 Accuracy: 100.00%
    Epoch:  860/1000 Cost: 0.0228 Accuracy: 100.00%
    Epoch:  870/1000 Cost: 0.0226 Accuracy: 100.00%
    Epoch:  880/1000 Cost: 0.0223 Accuracy: 100.00%
    Epoch:  890/1000 Cost: 0.0221 Accuracy: 100.00%
    Epoch:  900/1000 Cost: 0.0219 Accuracy: 100.00%
    Epoch:  910/1000 Cost: 0.0217 Accuracy: 100.00%
    Epoch:  920/1000 Cost: 0.0214 Accuracy: 100.00%
    Epoch:  930/1000 Cost: 0.0212 Accuracy: 100.00%
    Epoch:  940/1000 Cost: 0.0210 Accuracy: 100.00%
    Epoch:  950/1000 Cost: 0.0208 Accuracy: 100.00%
    Epoch:  960/1000 Cost: 0.0206 Accuracy: 100.00%
    Epoch:  970/1000 Cost: 0.0204 Accuracy: 100.00%
    Epoch:  980/1000 Cost: 0.0202 Accuracy: 100.00%
    Epoch:  990/1000 Cost: 0.0200 Accuracy: 100.00%
    Epoch: 1000/1000 Cost: 0.0198 Accuracy: 100.00%
    
