# 커스텀 데이터셋(Custom Dataset)

torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋(Custom Dataset)을 만들어보자.

기본적인 뼈대는 아래와 같다.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
    #데이터셋의 전처리를 해주는 부분
        
    def __len__(self):
    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    
    def __getitem__(self, idx):
    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
```

## (1) 커스텀 데이터셋 (Custom Dataset)으로 선형 회귀 구현하기


```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(1)

# Dataset 상속
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    
    # 총 데이터 개수 리턴
    def __len__(self):
        return len(self.x_data)
    
    # 인덱스를 입력받아 그에 매핑되는 입출력 데이터를 파이토치의 tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y
    
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        
        prediction = model(x_train)
        
        cost = F.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch: {:4d}/{} Batch: {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item() ) )
```

    Epoch:    0/20 Batch: 1/3 Cost: 28494.531250
    Epoch:    0/20 Batch: 2/3 Cost: 10967.986328
    Epoch:    0/20 Batch: 3/3 Cost: 5100.810547
    Epoch:    1/20 Batch: 1/3 Cost: 682.938171
    Epoch:    1/20 Batch: 2/3 Cost: 230.594070
    Epoch:    1/20 Batch: 3/3 Cost: 91.501534
    Epoch:    2/20 Batch: 1/3 Cost: 22.564547
    Epoch:    2/20 Batch: 2/3 Cost: 5.264287
    Epoch:    2/20 Batch: 3/3 Cost: 0.666619
    Epoch:    3/20 Batch: 1/3 Cost: 0.254771
    Epoch:    3/20 Batch: 2/3 Cost: 0.412526
    Epoch:    3/20 Batch: 3/3 Cost: 1.510413
    Epoch:    4/20 Batch: 1/3 Cost: 0.508034
    Epoch:    4/20 Batch: 2/3 Cost: 0.028076
    Epoch:    4/20 Batch: 3/3 Cost: 0.157345
    Epoch:    5/20 Batch: 1/3 Cost: 0.537085
    Epoch:    5/20 Batch: 2/3 Cost: 0.217656
    Epoch:    5/20 Batch: 3/3 Cost: 0.058551
    Epoch:    6/20 Batch: 1/3 Cost: 0.043198
    Epoch:    6/20 Batch: 2/3 Cost: 0.588473
    Epoch:    6/20 Batch: 3/3 Cost: 0.011543
    Epoch:    7/20 Batch: 1/3 Cost: 0.121354
    Epoch:    7/20 Batch: 2/3 Cost: 0.603847
    Epoch:    7/20 Batch: 3/3 Cost: 0.005891
    Epoch:    8/20 Batch: 1/3 Cost: 0.439329
    Epoch:    8/20 Batch: 2/3 Cost: 0.191190
    Epoch:    8/20 Batch: 3/3 Cost: 0.009326
    Epoch:    9/20 Batch: 1/3 Cost: 0.475450
    Epoch:    9/20 Batch: 2/3 Cost: 0.245828
    Epoch:    9/20 Batch: 3/3 Cost: 0.070225
    Epoch:   10/20 Batch: 1/3 Cost: 0.083207
    Epoch:   10/20 Batch: 2/3 Cost: 0.005543
    Epoch:   10/20 Batch: 3/3 Cost: 1.172741
    Epoch:   11/20 Batch: 1/3 Cost: 0.510159
    Epoch:   11/20 Batch: 2/3 Cost: 0.045958
    Epoch:   11/20 Batch: 3/3 Cost: 0.193334
    Epoch:   12/20 Batch: 1/3 Cost: 0.517951
    Epoch:   12/20 Batch: 2/3 Cost: 0.305695
    Epoch:   12/20 Batch: 3/3 Cost: 0.000302
    Epoch:   13/20 Batch: 1/3 Cost: 0.522243
    Epoch:   13/20 Batch: 2/3 Cost: 0.227066
    Epoch:   13/20 Batch: 3/3 Cost: 0.065910
    Epoch:   14/20 Batch: 1/3 Cost: 0.103471
    Epoch:   14/20 Batch: 2/3 Cost: 0.578098
    Epoch:   14/20 Batch: 3/3 Cost: 0.183205
    Epoch:   15/20 Batch: 1/3 Cost: 0.570738
    Epoch:   15/20 Batch: 2/3 Cost: 0.050819
    Epoch:   15/20 Batch: 3/3 Cost: 0.005355
    Epoch:   16/20 Batch: 1/3 Cost: 0.016758
    Epoch:   16/20 Batch: 2/3 Cost: 0.544686
    Epoch:   16/20 Batch: 3/3 Cost: 0.119511
    Epoch:   17/20 Batch: 1/3 Cost: 0.528579
    Epoch:   17/20 Batch: 2/3 Cost: 0.221694
    Epoch:   17/20 Batch: 3/3 Cost: 0.067613
    Epoch:   18/20 Batch: 1/3 Cost: 0.540206
    Epoch:   18/20 Batch: 2/3 Cost: 0.063319
    Epoch:   18/20 Batch: 3/3 Cost: 0.007752
    Epoch:   19/20 Batch: 1/3 Cost: 0.478905
    Epoch:   19/20 Batch: 2/3 Cost: 0.243084
    Epoch:   19/20 Batch: 3/3 Cost: 0.069274
    Epoch:   20/20 Batch: 1/3 Cost: 0.592551
    Epoch:   20/20 Batch: 2/3 Cost: 0.107780
    Epoch:   20/20 Batch: 3/3 Cost: 0.196000
    
