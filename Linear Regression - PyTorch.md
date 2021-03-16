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
    


```python

```
