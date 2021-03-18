# 미니 배치와 데이터 로드 (Mini Batch and Data Load)

## (1) 미니 배치와 배치 크기 (Mini Batch and Batch Size)

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
```

이정도로 데이터 수가 적으면 optimize를 진행하는데 무리가 없지만, 데이터 사이즈가 크면 굉장히 느리고 메모리가 많이 듬

전체 데이터를 더 작은 단위로 나누어서 학습하는 개념 -> 미니배치 (Mini Batch)

미니배치로 학습하면 미니배치 만큼만 가져가서 미니배치에 대한 cost를 계산하고, optimize 수행함.
다음 미니배치를 가져가서 또 수행하고 마지막 배치까지 이를 반복함. 
전체 데이터에 대한 학습이 1회 끝나면 1 epoch이 끝남.
배치 크기는 보통 2의 제곱수 사용.

## (2) 이터레이션(Iteration)
미니 배치와 배치 크기의 정의에 대해서 이해하였다면 iteration을 정의할 수 있음.

iteration은 한번의 epoch 내에서 이루어지는 매개변수인 가중치 W와 b의 업데이트 횟수.
전체 데이터가 2000일때 배치 크기를 200으로 한다면 iteration 수는 총 10개.

즉, 전체 데이터 수 = batch_size * num_iter

## (3) 데이터 로드하기(Data Load)
파이토치에서는 데이터를 좀 더 쉽게 다룰 수 있도록 데이터셋(Dataset)과 데이터로더(DataLoader)를 제공함.

이를 사용하면 미니배치 학습, 데이터 셔플, 병렬처리까지 간단히 수행 가능.
기본적인 사용 방법은 Dataset을 정의하고, 이를 DataLoader에 전달하는 것.

Dataset을 커스텀으로 만들수도 있지만, 여기서는 텐서를 입력받아 dataset의 형태로 변환해주는 TensorDataset을 사용.


```python
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

데이터셋을 정의하고 나면 데이터로더 사용 가능. 기본적으로 2개의 인자를 입력받음 (데이터셋, 배치 사이즈)
추가적으로 많이 사용되는 인자는 shuffle. shuffle=True면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿈


```python
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# Training 

nb_epochs = 20
for epoch in range(nb_epochs +1):
    
    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        xtrain, ytrain = samples
        
        # H(x) 계산
        prediction = model(x_train)
        
        # cost 계산
        cost = F.mse_loss(prediction, y_train)
        
        # H(x) update
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        
        print('Epoch: {:4d}/{} Batch {}/{} Cost: {:.5f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))
        
        
```

    Epoch:    0/20 Batch 1/3 Cost: 31667.59961
    Epoch:    0/20 Batch 2/3 Cost: 9926.26660
    Epoch:    0/20 Batch 3/3 Cost: 3111.51392
    Epoch:    1/20 Batch 1/3 Cost: 975.45135
    Epoch:    1/20 Batch 2/3 Cost: 305.90854
    Epoch:    1/20 Batch 3/3 Cost: 96.04250
    Epoch:    2/20 Batch 1/3 Cost: 30.26075
    Epoch:    2/20 Batch 2/3 Cost: 9.64170
    Epoch:    2/20 Batch 3/3 Cost: 3.17867
    Epoch:    3/20 Batch 1/3 Cost: 1.15287
    Epoch:    3/20 Batch 2/3 Cost: 0.51786
    Epoch:    3/20 Batch 3/3 Cost: 0.31880
    Epoch:    4/20 Batch 1/3 Cost: 0.25639
    Epoch:    4/20 Batch 2/3 Cost: 0.23682
    Epoch:    4/20 Batch 3/3 Cost: 0.23066
    Epoch:    5/20 Batch 1/3 Cost: 0.22872
    Epoch:    5/20 Batch 2/3 Cost: 0.22809
    Epoch:    5/20 Batch 3/3 Cost: 0.22788
    Epoch:    6/20 Batch 1/3 Cost: 0.22780
    Epoch:    6/20 Batch 2/3 Cost: 0.22776
    Epoch:    6/20 Batch 3/3 Cost: 0.22773
    Epoch:    7/20 Batch 1/3 Cost: 0.22771
    Epoch:    7/20 Batch 2/3 Cost: 0.22768
    Epoch:    7/20 Batch 3/3 Cost: 0.22767
    Epoch:    8/20 Batch 1/3 Cost: 0.22765
    Epoch:    8/20 Batch 2/3 Cost: 0.22762
    Epoch:    8/20 Batch 3/3 Cost: 0.22760
    Epoch:    9/20 Batch 1/3 Cost: 0.22757
    Epoch:    9/20 Batch 2/3 Cost: 0.22755
    Epoch:    9/20 Batch 3/3 Cost: 0.22753
    Epoch:   10/20 Batch 1/3 Cost: 0.22751
    Epoch:   10/20 Batch 2/3 Cost: 0.22749
    Epoch:   10/20 Batch 3/3 Cost: 0.22747
    Epoch:   11/20 Batch 1/3 Cost: 0.22745
    Epoch:   11/20 Batch 2/3 Cost: 0.22742
    Epoch:   11/20 Batch 3/3 Cost: 0.22739
    Epoch:   12/20 Batch 1/3 Cost: 0.22738
    Epoch:   12/20 Batch 2/3 Cost: 0.22735
    Epoch:   12/20 Batch 3/3 Cost: 0.22733
    Epoch:   13/20 Batch 1/3 Cost: 0.22731
    Epoch:   13/20 Batch 2/3 Cost: 0.22729
    Epoch:   13/20 Batch 3/3 Cost: 0.22727
    Epoch:   14/20 Batch 1/3 Cost: 0.22725
    Epoch:   14/20 Batch 2/3 Cost: 0.22722
    Epoch:   14/20 Batch 3/3 Cost: 0.22720
    Epoch:   15/20 Batch 1/3 Cost: 0.22718
    Epoch:   15/20 Batch 2/3 Cost: 0.22715
    Epoch:   15/20 Batch 3/3 Cost: 0.22714
    Epoch:   16/20 Batch 1/3 Cost: 0.22712
    Epoch:   16/20 Batch 2/3 Cost: 0.22709
    Epoch:   16/20 Batch 3/3 Cost: 0.22707
    Epoch:   17/20 Batch 1/3 Cost: 0.22705
    Epoch:   17/20 Batch 2/3 Cost: 0.22703
    Epoch:   17/20 Batch 3/3 Cost: 0.22700
    Epoch:   18/20 Batch 1/3 Cost: 0.22698
    Epoch:   18/20 Batch 2/3 Cost: 0.22695
    Epoch:   18/20 Batch 3/3 Cost: 0.22694
    Epoch:   19/20 Batch 1/3 Cost: 0.22691
    Epoch:   19/20 Batch 2/3 Cost: 0.22690
    Epoch:   19/20 Batch 3/3 Cost: 0.22687
    Epoch:   20/20 Batch 1/3 Cost: 0.22685
    Epoch:   20/20 Batch 2/3 Cost: 0.22683
    Epoch:   20/20 Batch 3/3 Cost: 0.22681
    
