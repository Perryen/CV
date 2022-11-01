# cv第六次作业

## 1、作业要求

在W6_MNIST_FC.ipynb基础上，增加卷积层结构/增加 dropout或者BN技术等，训练出尽可能高的MNIST分类效果。

## 2、代码阅读（部分）

 **名词解释：**

| 名词      | 定义                                                         |
| --------- | ------------------------------------------------------------ |
| Epoch     | 使用训练集的全部数据对模型进行一次完整训练，被称为“一代训练” |
| Batch     | 使用训练集中的一小部分样本对模型权重进行一次反向传播的参数更新，这一小部分样本被称为“一批数据” |
| Iteration | 使用一个Batch数据对模型进行依次参数更新的过程，被称之为“一次训练” |

```python
EPOCH = 2
LR = 0.001
DOWNLOAD_MNIST = True

#初始化一些参数：2代训练、0.001的学习率，并声明一个布尔常量
```

```python
train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST, )
"""
MNIST
dset.MNIST(root, train=True, transform=None, target_transform=None, download=False)

参数说明： 
- root : processed/training.pt 和 processed/test.pt 的主目录 
- train : True = 训练集, False = 测试集 
- download : True = 从互联网上下载数据集，并把数据集放在root目录下. 如果数据集之前下载过，将处理过的数据（minist.py中有相关函数）放在processed文件夹下。
"""
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

print(train_data.train_data.shape)

```

```python
train_x = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor) / 255.
train_y = train_data.train_labels
print(train_x.shape)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.  # Tensor on GPU
test_y = test_data.test_labels[:2000]

# 对输入数据和测试数据的格式进行增维操作，卷积层的数据必须是四维的。
```

## 3、增加卷积层结构

### 增加前的代码：

```python
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
#         self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)

        output = x
        return output


fc = FC()

optimizer = torch.optim.Adam(fc.parameters(), lr=LR)
# loss_func = nn.MSELoss()
loss_func = nn.CrossEntropyLoss()

data_size = 20000
batch_size = 50

for epoch in range(EPOCH):
    random_indx = np.random.permutation(data_size)
    for batch_i in range(data_size // batch_size):
        indx = random_indx[batch_i * batch_size:(batch_i + 1) * batch_size]

        b_x = train_x[indx, :]
        b_y = train_y[indx]
#         print(b_x.shape)
#         print(b_y.shape)
#         pdb.set_trace()

        output = fc(b_x)
    
        loss = loss_func(output, b_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_i % 50 == 0:
            test_output = fc(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.3f' % accuracy)

test_output = fc(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()  # move the computation in GPU

print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
```

结果：

```
Epoch:  0 | train loss: 2.3274 | test accuracy: 0.244
Epoch:  0 | train loss: 0.4754 | test accuracy: 0.803
Epoch:  0 | train loss: 0.3038 | test accuracy: 0.863
Epoch:  0 | train loss: 0.2705 | test accuracy: 0.868
Epoch:  0 | train loss: 0.4862 | test accuracy: 0.886
Epoch:  0 | train loss: 0.4480 | test accuracy: 0.891
Epoch:  0 | train loss: 0.1842 | test accuracy: 0.891
Epoch:  0 | train loss: 0.2281 | test accuracy: 0.900
Epoch:  1 | train loss: 0.1103 | test accuracy: 0.910
Epoch:  1 | train loss: 0.1764 | test accuracy: 0.912
Epoch:  1 | train loss: 0.0724 | test accuracy: 0.907
Epoch:  1 | train loss: 0.2743 | test accuracy: 0.914
Epoch:  1 | train loss: 0.1204 | test accuracy: 0.923
Epoch:  1 | train loss: 0.2403 | test accuracy: 0.923
Epoch:  1 | train loss: 0.2604 | test accuracy: 0.924
Epoch:  1 | train loss: 0.1817 | test accuracy: 0.928
tensor([7, 2, 1, 0, 4, 1, 4, 9, 6, 9]) prediction number
tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9]) real number
```

### 增加后的代码：

```python
class FC(nn.Module):
    # 初始化进行了修改
    def __init__(self):
        super(FC, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=(3,3),stride = (1,1),padding = 1)
        self.relu1 = nn.ReLU(inplace = True)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=(3,3),stride = (1,1),padding = 1)
        self.relu2 = nn.ReLU(inplace = True)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        
        self.fc1 = nn.Linear(7*7*32,256)
        self.relu3 = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(256,10)
    
      

	# 前向传播进行了修改
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        output = x
        return output


fc = FC()

optimizer = torch.optim.Adam(fc.parameters(), lr=LR)
# loss_func = nn.MSELoss()
loss_func = nn.CrossEntropyLoss()

data_size = 20000
batch_size = 50

for epoch in range(EPOCH):
    random_indx = np.random.permutation(data_size)
    for batch_i in range(data_size // batch_size):
        indx = random_indx[batch_i * batch_size:(batch_i + 1) * batch_size]

        b_x = train_x[indx, :]
        b_y = train_y[indx]
#         print(b_x.shape)
#         print(b_y.shape)
#         pdb.set_trace()

        output = fc(b_x)
    
        loss = loss_func(output, b_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_i % 50 == 0:
            test_output = fc(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.3f' % accuracy)

test_output = fc(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()  # move the computation in GPU

print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
```

结果：

```
Epoch:  0 | train loss: 2.3068 | test accuracy: 0.191
Epoch:  0 | train loss: 0.7662 | test accuracy: 0.821
Epoch:  0 | train loss: 0.3963 | test accuracy: 0.887
Epoch:  0 | train loss: 0.1821 | test accuracy: 0.906
Epoch:  0 | train loss: 0.1534 | test accuracy: 0.913
Epoch:  0 | train loss: 0.0572 | test accuracy: 0.946
Epoch:  0 | train loss: 0.2439 | test accuracy: 0.948
Epoch:  0 | train loss: 0.1579 | test accuracy: 0.943
Epoch:  1 | train loss: 0.1594 | test accuracy: 0.961
Epoch:  1 | train loss: 0.2206 | test accuracy: 0.956
Epoch:  1 | train loss: 0.2899 | test accuracy: 0.960
Epoch:  1 | train loss: 0.0593 | test accuracy: 0.947
Epoch:  1 | train loss: 0.0355 | test accuracy: 0.967
Epoch:  1 | train loss: 0.0345 | test accuracy: 0.965
Epoch:  1 | train loss: 0.0125 | test accuracy: 0.967
Epoch:  1 | train loss: 0.1015 | test accuracy: 0.964
tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9]) prediction number
tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9]) real number
```

## 4、实验结论

对比上面的两个实验结果可以发现，增加卷积层结构后MNIST分类效果有所提高。



















