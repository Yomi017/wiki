---
{"dg-publish":true,"permalink":"/notion/theoretical-knowledge/computer-science/artificial-intelligence/pytorch/"}
---

# 1. 基本操作

## 1.0 Tool
```
dir(package) # list all in this package (e.g. dir(p): 1, 2, 3)
dir(package.x) # list all in x, which is a part of package (e.g. dir(p.3): a, b)
# if the list like: __xxx__, __yyy__, it mean that the package.x is a function
help(package.x) # tell how to use this function or package
```

## 1.1 Load Data
```
torch.utils.data.Dataset & torch.utils.data.DataLoader
```
Dataset: stores data samples and expected values
Dataloader: group data in batches, enables multiprocessing
```
dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size, shuffle=True) 
											# ↑
											# Training: True
											# Testing: False
```
如何定义自己的Dataset：
```
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
	def __init__(self, dile):
		self.data = ...             # Read data & preprocess
	
	def __getitem__(self, index):
		return self.data[index]     # Returns one sample at a time
	
	def __len__(self):
		return len(self.data)       # Returns the size of the dataset
```

## 1.2