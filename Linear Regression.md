# 선형 회귀 (Linear Regression)

### 1. Basic setting



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 부여함.
torch.manual_seed(1)
```




    <torch._C.Generator at 0x14c3db72348>



### 2. 변수 선언


```python
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

print(x_train)
print(x_train.shape)
```

    tensor([[1.],
            [2.],
            [3.]])
    torch.Size([3, 1])
    

### 3. 가중치와 편향의 초기화


```python
# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
W = torch.zeros(1, requires_grad=True)

# 가중치 W를 출력
print(W)

b = torch.zeros(1,requires_grad=True)
```

    tensor([0.], requires_grad=True)
    

### 4. 가설 세우기

$ H(x) = Wx+b $


```python
hypothesis = x_train * W + b
print(hypothesis)
```

    tensor([[0.],
            [0.],
            [0.]], grad_fn=<AddBackward0>)
    

### 5. Cost function 선언하기

$cost(W,b) = \frac{1}{n} [\sum^{n}_{i=1}(y^{(i)}-H(x^{(i)})]^{2}$


```python
cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)
```

    tensor(18.6667, grad_fn=<MeanBackward0>)
    

### 6. Gradient descent 구현하기


```python
optimizer = optim.SGD([W,b], lr=0.01)

# gradient initializing
optimizer.zero_grad()

# cost function 미분해서 gradient 계산
cost.backward()

# W와 b를 업데이트
optimizer.step()
```

### 7. 전체 코드


```python
# data setting

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# model initialize
W = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# Optimizer 
optimizer = optim.SGD([W,b], lr = 0.01)

nb_epochs = 4000;
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * W + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), b.item(), cost.item()))
```

    Epoch    0/4000 W: 0.187, b: 0.080 Cost: 18.666666
    Epoch  100/4000 W: 1.746, b: 0.578 Cost: 0.048171
    Epoch  200/4000 W: 1.800, b: 0.454 Cost: 0.029767
    Epoch  300/4000 W: 1.843, b: 0.357 Cost: 0.018394
    Epoch  400/4000 W: 1.876, b: 0.281 Cost: 0.011366
    Epoch  500/4000 W: 1.903, b: 0.221 Cost: 0.007024
    Epoch  600/4000 W: 1.924, b: 0.174 Cost: 0.004340
    Epoch  700/4000 W: 1.940, b: 0.136 Cost: 0.002682
    Epoch  800/4000 W: 1.953, b: 0.107 Cost: 0.001657
    Epoch  900/4000 W: 1.963, b: 0.084 Cost: 0.001024
    Epoch 1000/4000 W: 1.971, b: 0.066 Cost: 0.000633
    Epoch 1100/4000 W: 1.977, b: 0.052 Cost: 0.000391
    Epoch 1200/4000 W: 1.982, b: 0.041 Cost: 0.000242
    Epoch 1300/4000 W: 1.986, b: 0.032 Cost: 0.000149
    Epoch 1400/4000 W: 1.989, b: 0.025 Cost: 0.000092
    Epoch 1500/4000 W: 1.991, b: 0.020 Cost: 0.000057
    Epoch 1600/4000 W: 1.993, b: 0.016 Cost: 0.000035
    Epoch 1700/4000 W: 1.995, b: 0.012 Cost: 0.000022
    Epoch 1800/4000 W: 1.996, b: 0.010 Cost: 0.000013
    Epoch 1900/4000 W: 1.997, b: 0.008 Cost: 0.000008
    Epoch 2000/4000 W: 1.997, b: 0.006 Cost: 0.000005
    Epoch 2100/4000 W: 1.998, b: 0.005 Cost: 0.000003
    Epoch 2200/4000 W: 1.998, b: 0.004 Cost: 0.000002
    Epoch 2300/4000 W: 1.999, b: 0.003 Cost: 0.000001
    Epoch 2400/4000 W: 1.999, b: 0.002 Cost: 0.000001
    Epoch 2500/4000 W: 1.999, b: 0.002 Cost: 0.000000
    Epoch 2600/4000 W: 1.999, b: 0.001 Cost: 0.000000
    Epoch 2700/4000 W: 2.000, b: 0.001 Cost: 0.000000
    Epoch 2800/4000 W: 2.000, b: 0.001 Cost: 0.000000
    Epoch 2900/4000 W: 2.000, b: 0.001 Cost: 0.000000
    Epoch 3000/4000 W: 2.000, b: 0.001 Cost: 0.000000
    Epoch 3100/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 3200/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 3300/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 3400/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 3500/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 3600/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 3700/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 3800/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 3900/4000 W: 2.000, b: 0.000 Cost: 0.000000
    Epoch 4000/4000 W: 2.000, b: 0.000 Cost: 0.000000
    

# 02. 자동 미분 (Autograd)

경사하강법의 requires_grad = True, backward() 등은 파이토치에서 제공하고 있는 미분 (Autograd) 기능을 수행하고 있는 것.
파이토치 학습 과정을 깊이 이해하기 위해서 자동 미분에 대해 익혀보자.

## (1) 경사하강법 리뷰 - 생략

## (2) 자동 미분 (Autograd) 실습하기

자동 미분을 실습을 통해 이해해보자. 임의로 $2w^2+5$ 라는 식을 세워보고, w에 대해 미분해보자.


```python
import torch

# 스칼라 텐서 w 선언. 이 텐서에 대한 기울기를 저장하겠다는 의미임.
w = torch.tensor(2.0, requires_grad=True)

# 수식 정의
y = w**2
z = 2*y + 5

# 해당 수식 미분
z.backward()

print('수식을 w로 미분한 값: {}'.format(w.grad))
```

    수식을 w로 미분한 값: 8.0
    

# 03. 다중 선형 회귀 (Multivariable Linear regression)

다수의 x로부터 y를 예측하는 다중 선형 회귀 (Multivariable Linear Regression)에 대해서 이해하자

## (1) 데이터 이해

Quiz1 (x1), Quiz2 (x2), Quiz3 (x3), Final(y)

$H(x) = w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + b$

## (2) PyTorch로 구현하기

필요한 도구 import 및 random seed 고정


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```




    <torch._C.Generator at 0x14c3db72348>




```python
# Training data
x1_train = torch.FloatTensor([[73],[93],[89],[96],[73]])
x2_train = torch.FloatTensor([[80],[88],[91],[98],[66]])
x3_train = torch.FloatTensor([[75],[91],[90],[100],[70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])




w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Optimizer 
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    
    # Cost 계산
    cost = torch.mean((hypothesis-y_train)**2)
    
    # Cost로 H(x) 업데이트
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch : {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost{:.6f}'.format(epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()))
```

    Epoch :    0/1000 w1: 0.294 w2: 0.294 w3: 0.296 b: 0.003 Cost29661.800781
    Epoch :  100/1000 w1: 0.677 w2: 0.662 w3: 0.674 b: 0.008 Cost2.533385
    Epoch :  200/1000 w1: 0.685 w2: 0.655 w3: 0.674 b: 0.008 Cost2.434982
    Epoch :  300/1000 w1: 0.691 w2: 0.649 w3: 0.674 b: 0.008 Cost2.341614
    Epoch :  400/1000 w1: 0.698 w2: 0.642 w3: 0.674 b: 0.008 Cost2.252968
    Epoch :  500/1000 w1: 0.705 w2: 0.636 w3: 0.674 b: 0.008 Cost2.168830
    Epoch :  600/1000 w1: 0.711 w2: 0.630 w3: 0.673 b: 0.009 Cost2.088943
    Epoch :  700/1000 w1: 0.717 w2: 0.623 w3: 0.673 b: 0.009 Cost2.013134
    Epoch :  800/1000 w1: 0.724 w2: 0.618 w3: 0.673 b: 0.009 Cost1.941131
    Epoch :  900/1000 w1: 0.730 w2: 0.612 w3: 0.672 b: 0.009 Cost1.872806
    Epoch : 1000/1000 w1: 0.735 w2: 0.606 w3: 0.672 b: 0.009 Cost1.807898
    

# 04. nn.Module로 구현하는 선형 회귀

구현되어 있는 module을 이용하여 더 쉽고 빠르게 구현하기


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# model = nn.Linear(input_dim, output_dim)
# cost = F.mse_loss(prediction, y_train)

# data
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

model = nn.Linear(1,1)
print(list(model.parameters()))  # 첫번째 값 W, 두번째 값 b

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training
nb_epochs = 2000
for epoch in range(nb_epochs+1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch,nb_epochs, cost.item()))

# 임의의 입력 4를 선언
new_var = torch.FloatTensor([[4]])
pred_y = model(new_var)
print("훈련 후 입력이 4일때의 예측값 :", pred_y)

# 학습된 모델 파라미터 출력
print(list(model.parameters()))
```

    [Parameter containing:
    tensor([[0.5153]], requires_grad=True), Parameter containing:
    tensor([-0.4414], requires_grad=True)]
    Epoch    0/2000 Cost: 13.103541
    Epoch  100/2000 Cost: 0.002791
    Epoch  200/2000 Cost: 0.001724
    Epoch  300/2000 Cost: 0.001066
    Epoch  400/2000 Cost: 0.000658
    Epoch  500/2000 Cost: 0.000407
    Epoch  600/2000 Cost: 0.000251
    Epoch  700/2000 Cost: 0.000155
    Epoch  800/2000 Cost: 0.000096
    Epoch  900/2000 Cost: 0.000059
    Epoch 1000/2000 Cost: 0.000037
    Epoch 1100/2000 Cost: 0.000023
    Epoch 1200/2000 Cost: 0.000014
    Epoch 1300/2000 Cost: 0.000009
    Epoch 1400/2000 Cost: 0.000005
    Epoch 1500/2000 Cost: 0.000003
    Epoch 1600/2000 Cost: 0.000002
    Epoch 1700/2000 Cost: 0.000001
    Epoch 1800/2000 Cost: 0.000001
    Epoch 1900/2000 Cost: 0.000000
    Epoch 2000/2000 Cost: 0.000000
    훈련 후 입력이 4일때의 예측값 : tensor([[7.9989]], grad_fn=<AddmmBackward>)
    [Parameter containing:
    tensor([[1.9994]], requires_grad=True), Parameter containing:
    tensor([0.0014], requires_grad=True)]
    

### 다중선형회귀 구현 (by module)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# data
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# model
model = nn.Linear(3,1)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    
    # H(x) 계산
    predict = model(x_train)
    
    # Cost 계산
    cost = F.mse_loss(predict, y_train)
    
    # H(x) 개선
    # optimizer gradient initialize
    optimizer.zero_grad()
    # cost function 을 이용한 backprop
    cost.backward()
    # parameter update
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch: {:4d}/{} Cost{:.6f}'.format(epoch, nb_epochs, cost.item()))
        
print(list(model.parameters()))
    
```

    Epoch:    0/2000 Cost31667.599609
    Epoch:  100/2000 Cost0.225993
    Epoch:  200/2000 Cost0.223911
    Epoch:  300/2000 Cost0.221941
    Epoch:  400/2000 Cost0.220059
    Epoch:  500/2000 Cost0.218271
    Epoch:  600/2000 Cost0.216575
    Epoch:  700/2000 Cost0.214950
    Epoch:  800/2000 Cost0.213413
    Epoch:  900/2000 Cost0.211952
    Epoch: 1000/2000 Cost0.210559
    Epoch: 1100/2000 Cost0.209230
    Epoch: 1200/2000 Cost0.207967
    Epoch: 1300/2000 Cost0.206762
    Epoch: 1400/2000 Cost0.205618
    Epoch: 1500/2000 Cost0.204529
    Epoch: 1600/2000 Cost0.203481
    Epoch: 1700/2000 Cost0.202486
    Epoch: 1800/2000 Cost0.201539
    Epoch: 1900/2000 Cost0.200634
    Epoch: 2000/2000 Cost0.199770
    [Parameter containing:
    tensor([[0.9778, 0.4539, 0.5768]], requires_grad=True), Parameter containing:
    tensor([0.2802], requires_grad=True)]
    

# 05. 클래스로 파이토치 모델 구현하기

파이토치의 대부분 구현체들은 모델을 생성할 때 클래스(Class)를 사용함. 앞서 배운 선형 회귀를 클래스로 구현해보자.

## (1) 모델을 클래스로 구현

앞서 단순 선형 회귀 모델은 다음과 같이 구현했었음.


```python
# 모델 선언 및 초기화. 단순 선형 회귀이므로 input_dim =1 , output_dim =1
model = nn.Linear(1,1)
```

이를 클래스로 구현하면 아래와 같음


```python
class LinearRegressionModel(nn.Module): #torch.nn.Module을 상속받는 파이썬 클래스
    
    def _init_(self): #생성자 : 생성자(Constructor)란 객체가 생성될 때 자동으로 호출되는 메서드를 의미
        super().__init__()
        self.linear = nn.Linear(1,1) # 단순 선형 회귀이므로 input_dim=1, output_dim=1.
        
    def forward(self,x):
        return self.linear(x)

    
model = LinearRegressionModel()
```

**위와 같은 클래스를 사용한 모델 구현 형식은 대부분의 파이토치 구현체에서 사용하고 있는 방식으로 반드시 숙지해야함**

클래스(class) 형태의 모델은 nn.Module을 상속받음.
__init__() 에서의 모델의 구조와 동적을 정의하는 생성자를 정의함.
이는 파이썬에서 객체가 갖는 속성값을 초기화하는 역할로, 객체가 생성될 때 자동으로 호출됨. 
super() 함수를 부르면 여기서 만든 클래스는 nn.Module 클래스의 속성들을 가지고 초기화됨.
forward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수임.

### 단순 선형 회귀 클래스로 구현하기


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# set data 
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# Model by class

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
        
    def forward(self,x_train):
        return self.linear(x_train)
    
model = LinearRegression()

# Optimization
optimization = torch.optim.SGD(model.parameters(), lr = 0.01)

nb_epochs = 10000
for epoch in range(nb_epochs+1):
    
    # H(x)
    predict = model.forward(x_train)

    # Cost 
    cost = F.mse_loss(predict, y_train)

    # H(x) update
    optimization.zero_grad()
    cost.backward()
    optimization.step()
    
    if epoch % 100 == 0:
        print('Epoch: {:4d}/{} cost: {:.4f}'.format(epoch, nb_epochs, cost.item() ))
        
        
print(list(model.forward(x_train)))
        
```

    Epoch:    0/10000 cost: 13.1035
    Epoch:  100/10000 cost: 0.0028
    Epoch:  200/10000 cost: 0.0017
    Epoch:  300/10000 cost: 0.0011
    Epoch:  400/10000 cost: 0.0007
    Epoch:  500/10000 cost: 0.0004
    Epoch:  600/10000 cost: 0.0003
    Epoch:  700/10000 cost: 0.0002
    Epoch:  800/10000 cost: 0.0001
    Epoch:  900/10000 cost: 0.0001
    Epoch: 1000/10000 cost: 0.0000
    Epoch: 1100/10000 cost: 0.0000
    Epoch: 1200/10000 cost: 0.0000
    Epoch: 1300/10000 cost: 0.0000
    Epoch: 1400/10000 cost: 0.0000
    Epoch: 1500/10000 cost: 0.0000
    Epoch: 1600/10000 cost: 0.0000
    Epoch: 1700/10000 cost: 0.0000
    Epoch: 1800/10000 cost: 0.0000
    Epoch: 1900/10000 cost: 0.0000
    Epoch: 2000/10000 cost: 0.0000
    Epoch: 2100/10000 cost: 0.0000
    Epoch: 2200/10000 cost: 0.0000
    Epoch: 2300/10000 cost: 0.0000
    Epoch: 2400/10000 cost: 0.0000
    Epoch: 2500/10000 cost: 0.0000
    Epoch: 2600/10000 cost: 0.0000
    Epoch: 2700/10000 cost: 0.0000
    Epoch: 2800/10000 cost: 0.0000
    Epoch: 2900/10000 cost: 0.0000
    Epoch: 3000/10000 cost: 0.0000
    Epoch: 3100/10000 cost: 0.0000
    Epoch: 3200/10000 cost: 0.0000
    Epoch: 3300/10000 cost: 0.0000
    Epoch: 3400/10000 cost: 0.0000
    Epoch: 3500/10000 cost: 0.0000
    Epoch: 3600/10000 cost: 0.0000
    Epoch: 3700/10000 cost: 0.0000
    Epoch: 3800/10000 cost: 0.0000
    Epoch: 3900/10000 cost: 0.0000
    Epoch: 4000/10000 cost: 0.0000
    Epoch: 4100/10000 cost: 0.0000
    Epoch: 4200/10000 cost: 0.0000
    Epoch: 4300/10000 cost: 0.0000
    Epoch: 4400/10000 cost: 0.0000
    Epoch: 4500/10000 cost: 0.0000
    Epoch: 4600/10000 cost: 0.0000
    Epoch: 4700/10000 cost: 0.0000
    Epoch: 4800/10000 cost: 0.0000
    Epoch: 4900/10000 cost: 0.0000
    Epoch: 5000/10000 cost: 0.0000
    Epoch: 5100/10000 cost: 0.0000
    Epoch: 5200/10000 cost: 0.0000
    Epoch: 5300/10000 cost: 0.0000
    Epoch: 5400/10000 cost: 0.0000
    Epoch: 5500/10000 cost: 0.0000
    Epoch: 5600/10000 cost: 0.0000
    Epoch: 5700/10000 cost: 0.0000
    Epoch: 5800/10000 cost: 0.0000
    Epoch: 5900/10000 cost: 0.0000
    Epoch: 6000/10000 cost: 0.0000
    Epoch: 6100/10000 cost: 0.0000
    Epoch: 6200/10000 cost: 0.0000
    Epoch: 6300/10000 cost: 0.0000
    Epoch: 6400/10000 cost: 0.0000
    Epoch: 6500/10000 cost: 0.0000
    Epoch: 6600/10000 cost: 0.0000
    Epoch: 6700/10000 cost: 0.0000
    Epoch: 6800/10000 cost: 0.0000
    Epoch: 6900/10000 cost: 0.0000
    Epoch: 7000/10000 cost: 0.0000
    Epoch: 7100/10000 cost: 0.0000
    Epoch: 7200/10000 cost: 0.0000
    Epoch: 7300/10000 cost: 0.0000
    Epoch: 7400/10000 cost: 0.0000
    Epoch: 7500/10000 cost: 0.0000
    Epoch: 7600/10000 cost: 0.0000
    Epoch: 7700/10000 cost: 0.0000
    Epoch: 7800/10000 cost: 0.0000
    Epoch: 7900/10000 cost: 0.0000
    Epoch: 8000/10000 cost: 0.0000
    Epoch: 8100/10000 cost: 0.0000
    Epoch: 8200/10000 cost: 0.0000
    Epoch: 8300/10000 cost: 0.0000
    Epoch: 8400/10000 cost: 0.0000
    Epoch: 8500/10000 cost: 0.0000
    Epoch: 8600/10000 cost: 0.0000
    Epoch: 8700/10000 cost: 0.0000
    Epoch: 8800/10000 cost: 0.0000
    Epoch: 8900/10000 cost: 0.0000
    Epoch: 9000/10000 cost: 0.0000
    Epoch: 9100/10000 cost: 0.0000
    Epoch: 9200/10000 cost: 0.0000
    Epoch: 9300/10000 cost: 0.0000
    Epoch: 9400/10000 cost: 0.0000
    Epoch: 9500/10000 cost: 0.0000
    Epoch: 9600/10000 cost: 0.0000
    Epoch: 9700/10000 cost: 0.0000
    Epoch: 9800/10000 cost: 0.0000
    Epoch: 9900/10000 cost: 0.0000
    Epoch: 10000/10000 cost: 0.0000
    [tensor([2.0000], grad_fn=<UnbindBackward>), tensor([4.], grad_fn=<UnbindBackward>), tensor([6.0000], grad_fn=<UnbindBackward>)]
    

### 다중선형회귀 클래스로 구현하기


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Dataset
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Model

class MultiRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    
    def forward(self,x):
        return self.linear(x)
        
model = MultiRegression()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
nb_epoch = 5000
for epoch in range(nb_epoch):
    
    #H(x)
    prediction = model.forward(x_train)
    
    #cost
    cost = F.mse_loss(prediction, y_train)
    
    #H(x) update
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch: {:4d}/{} Cost: {:.4f}'.format(epoch, nb_epoch, cost.item()))



print(list(model.forward(x_train)))
new_input = torch.FloatTensor([1,3,5])
print(list(model.forward(new_input)))
```

    Epoch:    0/5000 Cost: 31667.5996
    Epoch:  100/5000 Cost: 0.2260
    Epoch:  200/5000 Cost: 0.2239
    Epoch:  300/5000 Cost: 0.2219
    Epoch:  400/5000 Cost: 0.2201
    Epoch:  500/5000 Cost: 0.2183
    Epoch:  600/5000 Cost: 0.2166
    Epoch:  700/5000 Cost: 0.2150
    Epoch:  800/5000 Cost: 0.2134
    Epoch:  900/5000 Cost: 0.2120
    Epoch: 1000/5000 Cost: 0.2106
    Epoch: 1100/5000 Cost: 0.2092
    Epoch: 1200/5000 Cost: 0.2080
    Epoch: 1300/5000 Cost: 0.2068
    Epoch: 1400/5000 Cost: 0.2056
    Epoch: 1500/5000 Cost: 0.2045
    Epoch: 1600/5000 Cost: 0.2035
    Epoch: 1700/5000 Cost: 0.2025
    Epoch: 1800/5000 Cost: 0.2015
    Epoch: 1900/5000 Cost: 0.2006
    Epoch: 2000/5000 Cost: 0.1998
    Epoch: 2100/5000 Cost: 0.1989
    Epoch: 2200/5000 Cost: 0.1982
    Epoch: 2300/5000 Cost: 0.1974
    Epoch: 2400/5000 Cost: 0.1967
    Epoch: 2500/5000 Cost: 0.1960
    Epoch: 2600/5000 Cost: 0.1953
    Epoch: 2700/5000 Cost: 0.1947
    Epoch: 2800/5000 Cost: 0.1941
    Epoch: 2900/5000 Cost: 0.1935
    Epoch: 3000/5000 Cost: 0.1930
    Epoch: 3100/5000 Cost: 0.1925
    Epoch: 3200/5000 Cost: 0.1919
    Epoch: 3300/5000 Cost: 0.1915
    Epoch: 3400/5000 Cost: 0.1910
    Epoch: 3500/5000 Cost: 0.1905
    Epoch: 3600/5000 Cost: 0.1901
    Epoch: 3700/5000 Cost: 0.1897
    Epoch: 3800/5000 Cost: 0.1893
    Epoch: 3900/5000 Cost: 0.1889
    Epoch: 4000/5000 Cost: 0.1885
    Epoch: 4100/5000 Cost: 0.1882
    Epoch: 4200/5000 Cost: 0.1878
    Epoch: 4300/5000 Cost: 0.1875
    Epoch: 4400/5000 Cost: 0.1872
    Epoch: 4500/5000 Cost: 0.1869
    Epoch: 4600/5000 Cost: 0.1866
    Epoch: 4700/5000 Cost: 0.1863
    Epoch: 4800/5000 Cost: 0.1860
    Epoch: 4900/5000 Cost: 0.1858
    [tensor([151.3324], grad_fn=<UnbindBackward>), tensor([184.7329], grad_fn=<UnbindBackward>), tensor([180.5544], grad_fn=<UnbindBackward>), tensor([196.3119], grad_fn=<UnbindBackward>), tensor([141.9234], grad_fn=<UnbindBackward>)]
    [tensor(5.4936, grad_fn=<UnbindBackward>)]
    
