---
{"dg-publish":true,"permalink":"/notion/theoretical-knowledge/computer-science/artificial-intelligence/pytorch/"}
---

# 1. 基本操作 (Basic Operations)

## 1.0 工具 (Tools)

这些是 Python 内置的实用函数，对于探索库和对象非常有用：

*   `dir(package_or_module)`: 列出指定包或模块中可用的所有名称（变量、函数、类等）。
    *   例如: `dir(torch)` 会显示 `torch` 命名空间下的所有内容。
    *   例如: `dir(torch.optim)` 会显示 `torch.optim` 模块下的所有内容。
*   `dir(object)`: 列出对象的所有属性和方法。
    *   如果列表主要包含 `__xxx__` 形式的名称（称为 "dunder" 或 "magic" methods/attributes），这通常表明该对象是一个类的实例、一个类本身，或者一个内置类型/函数。例如，`dir(my_list_object)` 或 `dir(str)`.
*   `help(item)`: 显示关于 `item`（可以是模块、类、函数、方法或变量）的帮助文档/文档字符串 (docstring)。
    *   例如: `help(torch.nn.Linear)` 会告诉你如何使用 `Linear` 层。
    *   例如: `help(my_dataset_instance)` 会显示 `MyDataset` 类的文档字符串（如果定义了的话）。

## 1.1 加载数据 (Load Data)

### 1.1.1 基本函数 (Basic Function)

PyTorch 提供了两个核心工具来简化和优化数据加载流程：`torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader`。

*   **`Dataset`**:
    *   是一个抽象类，代表一个数据集。
    *   你需要继承它并重写 `__len__` (返回数据集大小) 和 `__getitem__` (支持从0到 `len(self)-1` 的整数索引，返回一个数据样本)。
    *   它负责存储样本及其对应的标签（或其他目标值），并提供按索引访问单个样本的能力。

*   **`DataLoader`**:
    *   是一个迭代器，它将 `Dataset` 包装起来。
    *   功能：
        *   **批量处理 (Batching)**: 将数据组织成小批量 (mini-batches)。
        *   **打乱数据 (Shuffling)**: 在每个 epoch 开始时打乱数据顺序，有助于模型训练。
        *   **并行加载 (Multiprocessing)**: 使用多个子进程并行加载数据，避免数据加载成为训练瓶颈。
        *   自动整理 (Collation): 将从 `Dataset` 获取的单个样本聚合成批次。

**基本使用流程:**

1.  **定义你自己的 `Dataset` 类**:
    ```python
    from torch.utils.data import Dataset, DataLoader
    import torch # 假设你的数据会转换为 PyTorch 张量
    # import numpy as np # 如果你从 numpy 数组加载
    # import pandas as pd # 如果你从 csv 加载

    class MyDataset(Dataset):
        def __init__(self, data_source, transform=None):
            # 1. 初始化阶段：通常在这里加载和预处理所有数据
            #    data_source 可以是文件路径、数据列表、或其他数据源指示符
            #    例如：
            #    - self.data = pd.read_csv(data_source).values
            #    - self.image_paths = glob.glob(os.path.join(data_source, '*.jpg'))
            #    - self.raw_data = data_source # 如果数据已经加载好传入
            self.data = ... # 示例：假设 self.data 存储了所有样本
            self.labels = ... # 示例：假设 self.labels 存储了所有标签
            # self.n_samples = len(self.data) # 如果数据是列表或类似结构

            self.transform = transform # 可选的数据转换/增强操作

        def __getitem__(self, index):
            # 2. 按索引获取单个样本
            #    必须返回一个样本，通常是 (features, label) 的元组
            sample = self.data[index]
            label = self.labels[index]

            if self.transform:
                sample = self.transform(sample) # 应用转换

            # 确保返回的是 PyTorch 张量 (通常在 Dataset,DataLoader 的 collate_fn 中处理)
            # return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            return sample, label

        def __len__(self):
            # 3. 返回数据集的总样本数
            return len(self.data) # 或 self.n_samples
    ```

2.  **实例化 `Dataset` 和 `DataLoader`**:
    ```python
    # 假设 'my_data_file.csv' 是你的数据文件
    dataset = MyDataset(data_source='my_data_file.csv') # 或者其他数据源

    # 对于训练数据，通常启用 shuffle
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=64,    # 每个批次包含的样本数
                                  shuffle=True,     # 在每个 epoch 开始时打乱数据
                                  num_workers=2)    # 使用多少子进程加载数据 (0表示在主进程中加载)

    # 对于测试/验证数据，通常不打乱
    # test_dataset = MyDataset(data_source='my_test_data.csv')
    # test_dataloader = DataLoader(dataset=test_dataset,
    #                              batch_size=64,
    #                              shuffle=False,
    #                              num_workers=2)
    ```
    **注意 `shuffle` 参数**:
    *   **训练 (Training) 时**: `shuffle=True` 是推荐的，有助于模型泛化，防止模型按特定顺序学习数据。
    *   **测试 (Testing) / 验证 (Validation) 时**: `shuffle=False` 是推荐的，以确保每次评估的顺序一致，便于比较和复现结果。

**示例 :**
* 1. 假设有一个 `wine.csv` 文件，前N-1列是特征，最后一列是标签。
```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class WineDataset(Dataset):
    def __init__(self, filepath, transform=None):
        # 加载数据: 使用 numpy 加载 CSV 文件，跳过标题行，指定分隔符
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # 将数据分为特征 (X) 和标签 (y)
        # 这里假设最后一列是标签，其余是特征
        self.x_data = torch.from_numpy(xy[:, :-1]) # 所有行, 除了最后一列
        self.y_data = torch.from_numpy(xy[:, [-1]]) # 所有行, 只有最后一列

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index]
        label = self.y_data[index]

        if self.transform:
            # 注意：这里的 transform 通常用于图像增强等
            # 对于表格数据，转换可能在 __init__ 中完成，或有特定于表格的转换
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return self.n_samples

# 使用
# dataset = WineDataset('wine.csv')
# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# # 迭代 DataLoader 获取批次数据
# data_iter = iter(dataloader)
# features, labels = next(data_iter)
# print(features)
# print(labels)
```
* 2. label 与文件夹名称相同![Image/Computer-Science/Pytorch/1.png](/img/user/Image/Computer-Science/Pytorch/1.png)
* 3. label 描述在文本文件里![Image/Computer-Science/Pytorch/3.png](/img/user/Image/Computer-Science/Pytorch/3.png)

### 1.1.2 数据集操作 (Dataset Operations)

*   **数据集串联 (Concatenation)**:
    PyTorch 的 `Dataset` 对象可以直接使用 `+` 操作符进行拼接，这实际上是使用了 `torch.utils.data.ConcatDataset`。
    ```python
    dataset1 = MyDataset(source_A)
    dataset2 = MyDataset(source_B)

    # 合并两个数据集
    combined_dataset = dataset1 + dataset2
    # 或者显式使用 ConcatDataset
    # from torch.utils.data import ConcatDataset
    # combined_dataset = ConcatDataset([dataset1, dataset2])

    print(len(dataset1))
    print(len(dataset2))
    print(len(combined_dataset)) # 输出: len(dataset1) + len(dataset2)

    # combined_dataloader 可以基于 combined_dataset 创建
    combined_dataloader = DataLoader(dataset=combined_dataset, batch_size=32, shuffle=True)
    ```
    **示例：**
    ![Image/Computer-Science/Pytorch/2.png](/img/user/Image/Computer-Science/Pytorch/2.png)
    这在需要合并来自不同来源的训练数据，或者合并训练集和部分验证集进行训练时非常有用。

## 1.3 TensorBoard

TensorBoard 是一个强大的可视化工具，最初为 TensorFlow 开发，但 PyTorch 通过 `torch.utils.tensorboard` 模块提供了出色的支持。它能帮助你可视化模型图、跟踪训练指标（如损失和准确率）、查看权重和梯度的分布、展示图像等，对于调试、理解和优化神经网络至关重要。

### 1.3.1 介绍 (Introduction)

*   **目的 (Purpose)**: TensorBoard 提供了一系列可视化工具，使你能够：
    *   实时跟踪和可视化训练过程中的指标（例如损失函数值、准确率）。
    *   可视化模型的计算图结构。
    *   查看模型权重、偏置或梯度的直方图，了解它们的分布和变化。
    *   展示输入的图像、生成的图像、特征图等。
    *   分析词嵌入或图像特征嵌入在高维空间中的分布（例如使用 t-SNE 或 PCA 降维投影）。
*   **工作原理 (How it works)**:
    1.  在 PyTorch 脚本中，你使用 `SummaryWriter` 对象将训练过程中的各种数据（标量、图像、直方图等）写入到指定的日志目录 (`logdir`)。
    2.  然后，在你的终端（命令行）中启动 TensorBoard 服务，并让它指向这个日志目录。
    3.  最后，通过浏览器访问 TensorBoard 提供的网址（通常是 `http://localhost:6006`）来查看和分析这些可视化数据。

### 1.3.2 基本操作 (Basic Operations)

首先，你需要导入 `SummaryWriter` 并实例化它。

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision # For image examples
import numpy as np # For generating example data
```

**初始化 `SummaryWriter`**

`SummaryWriter` 是将数据写入 TensorBoard 事件文件的核心类。

```python
# writer = SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
```

**`SummaryWriter` 构造函数参数详解:**

*   `log_dir (str, optional)`: 指定事件文件存储的目录。
    *   如果为 `None` (默认)，日志将保存在 `runs/CURRENT_DATETIME_HOSTNAME` 格式的目录中。
    *   如果提供路径，例如 `'runs/my_experiment'`，则日志将保存在该目录下。TensorBoard 启动时需要指向这个 `log_dir` 或其父目录。
*   `comment (str, optional)`: 一个可选的后缀，会附加到默认 `log_dir` 的末尾（如果 `log_dir` 未指定）。例如，如果 `comment='_my_model'`，则默认目录可能变为 `runs/CURRENT_DATETIME_HOSTNAME_my_model`。这有助于区分在同一时间戳附近生成的不同运行。
*   `purge_step (int, optional)`: TensorBoard 可能因意外关闭（如Ctrl+C）而留下未完成的事件文件。当下一次使用相同的 `log_dir` 启动 `SummaryWriter` 时，如果 `purge_step` 被设置，它会清除在该步骤（`global_step`）之后的所有事件。这有助于清理损坏的运行。
*   `max_queue (int, optional)`: 在将事件异步写入磁盘之前，内存中事件队列的最大大小。默认为 10。
*   `flush_secs (int, optional)`: 将队列中的事件刷新（写入）到磁盘的频率（以秒为单位）。默认为 120 秒（2分钟）。
*   `filename_suffix (str, optional)`: 可选的文件名后缀，会附加到事件文件的名称上。默认为空。

**示例实例化:**

```python
# 推荐做法：为每次实验指定一个清晰的日志目录
writer = SummaryWriter('runs/mnist_experiment_1')

# 或者使用默认目录
# writer_default = SummaryWriter()
```

现在，让我们看看如何使用 `writer` 对象记录不同类型的数据。

---

**1. 记录标量 (Logging Scalars)**

用于跟踪单个数值随时间（或训练步骤）的变化，如损失、准确率、学习率等。

`writer.add_scalar(tag, scalar_value, global_step=None, walltime=None)`
*   `tag (str)`: 数据标识符，例如 `'Loss/train'` 或 `'Accuracy/validation'`。斜杠 `/` 用于在 TensorBoard UI 中创建层级结构。
*   `scalar_value (float or int)`: 要记录的标量值。
*   `global_step (int, optional)`: 记录该数据点时对应的全局步骤，通常是迭代次数或 epoch 数。这将是图表中的 x 轴。
*   `walltime (float, optional)`: 事件发生的时间戳，默认为 `time.time()`。一般不需要手动设置。

```python
print("Logging scalars...")
for n_iter in range(100):
    loss = 0.99**n_iter # 模拟损失下降
    accuracy = 1.0 - loss * 0.8
    writer.add_scalar('Training/Loss', loss, n_iter)
    writer.add_scalar('Training/Accuracy', accuracy, n_iter)

# 如果想在同一张图上比较多个标量，可以使用 add_scalars
# writer.add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
# - main_tag (str): 这组标量图的主标题。
# - tag_scalar_dict (dict): 一个字典，键是子标签，值是对应的标量值。
print("Logging grouped scalars...")
for n_iter in range(50):
    writer.add_scalars('Loss_Variations', {
        'loss_A': np.random.random() + n_iter/50.0,
        'loss_B': np.random.random() * 0.5 + n_iter/50.0
    }, n_iter)
```

---

**2. 记录直方图 (Logging Histograms)**

用于可视化张量（如模型权重、偏置、梯度或激活值）的分布情况。

`writer.add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)`
*   `tag (str)`: 直方图的标识符。
*   `values (torch.Tensor, numpy.ndarray)`: 包含要绘制直方图的数据的张量。
*   `global_step (int, optional)`: 全局步骤。
*   `bins (str, optional)`: 生成直方图的分箱策略。`'tensorflow'` (默认), `'auto'`, `'fd'`, `'doane'`, `'scott'`, `'stone'`, `'rice'`, `'sturges'`, 或 `'sqrt'`。

```python
print("Logging histograms...")
# 模拟不同 epoch 的权重变化
for epoch in range(10):
    # 假设这是模型某一层权重的快照
    dummy_weights = torch.randn(1000) * (epoch + 1) * 0.1 + epoch
    writer.add_histogram('Model_Layer1/weights', dummy_weights, epoch)

    dummy_biases = torch.rand(100) - 0.5 + epoch * 0.1
    writer.add_histogram('Model_Layer1/biases', dummy_biases, epoch)
```

---

**3. 记录图像 (Logging Images)**

用于在 TensorBoard 中显示单个图像。

`writer.add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')`
*   `tag (str)`: 图像的标识符。
*   `img_tensor (torch.Tensor or numpy.ndarray)`: 图像数据。
    *   可以是 3D 张量 (CHW 或 HWC) 或 2D 张量 (HW - 灰度图)。
    *   CHW: Channel, Height, Width (PyTorch 默认)。
    *   HWC: Height, Width, Channel (Matplotlib, PIL 默认)。
    *   HW: Height, Width.
*   `global_step (int, optional)`: 全局步骤。
*   `dataformats (str, optional)`: 指定 `img_tensor` 的格式，如 `'CHW'`, `'HWC'`, `'HW'`。默认为 `'CHW'`。张量的值通常应归一化到 `[0, 1]` (浮点数) 或 `[0, 255]` (整数)。

```python
print("Logging a single image...")
# 创建一个随机图像作为示例
# CHW format: 3 channels (RGB), 64 height, 128 width
random_img_chw = torch.rand(3, 64, 128)
writer.add_image('Sample_Images/Random_CHW', random_img_chw, 0)

# HWC format: 32 height, 32 width, 3 channels (RGB)
random_img_hwc = torch.rand(32, 32, 3)
writer.add_image('Sample_Images/Random_HWC', random_img_hwc, 1, dataformats='HWC')

# Grayscale image (HW format)
random_img_hw = torch.rand(40, 40)
writer.add_image('Sample_Images/Random_Grayscale_HW', random_img_hw, 2, dataformats='HW')
```

---

**4. 记录一批图像 (Logging a Batch of Images)**

用于将一批图像作为一个网格（grid）显示。

`writer.add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')`
*   `tag (str)`: 图像批次的标识符。
*   `img_tensor (torch.Tensor or numpy.ndarray)`: 包含一批图像的张量。
    *   通常是 4D 张量。
*   `global_step (int, optional)`: 全局步骤。
*   `dataformats (str, optional)`: 指定 `img_tensor` 的格式，如 `'NCHW'` (Batch, Channel, Height, Width - PyTorch 默认) 或 `'NHWC'`。

```python
print("Logging a batch of images...")
# 创建一个随机图像批次 (4 images, 3 channels, 32x32)
random_batch_nchw = torch.rand(4, 3, 32, 32)
writer.add_images('Sample_Images_Batch/Random_Batch_NCHW', random_batch_nchw, 0)

# 也可以使用 torchvision.utils.make_grid 先手动创建网格，然后用 add_image
img_grid = torchvision.utils.make_grid(random_batch_nchw)
writer.add_image('Sample_Images_Batch/Random_Batch_Grid_Manual', img_grid, 1)
```

---

**5. 记录模型图 (Logging Model Graph)**

可视化模型的结构和操作流程。

`writer.add_graph(model, input_to_model=None, verbose=False, use_strict_trace=True)`
*   `model (torch.nn.Module)`: 你要可视化的 PyTorch 模型。
*   `input_to_model (torch.Tensor or tuple of torch.Tensor, optional)`: 一个符合模型 `forward` 方法输入的示例张量或张量元组。模型会用这个输入进行一次追踪（trace）以生成图。
*   `verbose (bool, optional)`: 是否打印图的调试信息。
*   `use_strict_trace (bool, optional)`: 是否使用严格的追踪模式。对于某些动态图模型，设置为 `False` 可能有助于生成图，但可能不完全准确。

```python
print("Logging model graph...")
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(10 * 12 * 12, 50) # Assuming input 1x28x28 for MNIST

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = x.view(-1, 10 * 12 * 12)
        x = self.fc(x)
        return x

dummy_model = SimpleNet()
# 示例输入：batch_size=1, channels=1, height=28, width=28 (e.g., MNIST)
dummy_input = torch.randn(1, 1, 28, 28)
try:
    writer.add_graph(dummy_model, dummy_input)
except Exception as e:
    print(f"Could not add graph (common issue with some PyTorch/ONNX versions, or complex inputs/control flow): {e}")
    print("Graph logging can be tricky. Ensure 'input_to_model' matches model's forward pass expectations exactly.")
```
**注意**: `add_graph` 对于包含复杂控制流（例如依赖于输入的 `if` 语句）的模型可能无法完美工作，因为它依赖于 JIT 追踪。

---

**6. 记录文本 (Logging Text)**

在 TensorBoard 中显示文本信息，如超参数配置、笔记等。

`writer.add_text(tag, text_string, global_step=None, walltime=None)`
*   `tag (str)`: 文本的标识符。
*   `text_string (str)`: 要显示的文本内容。支持 Markdown 格式。
*   `global_step (int, optional)`: 全局步骤。

```python
print("Logging text...")
hyperparameters_info = """
### Experiment Configuration
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Optimizer**: Adam
- **Epochs**: 50
"""
writer.add_text('Experiment_Setup/Hyperparameters', hyperparameters_info, 0)
writer.add_text('Notes', "This run uses a new data augmentation technique.", 0)
```

---

**7. 记录超参数 (Logging Hyperparameters - `hparams`)**

这是一个更结构化的方式来记录实验的超参数组合及其对应的最终性能指标，非常适合进行超参数调优和比较。

`writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)`
*   `hparam_dict (dict)`: 包含超参数名称和它们的值的字典，例如 `{'lr': 0.1, 'optimizer': 'Adam'}`。
*   `metric_dict (dict)`: 包含在这些超参数下达到的最终指标值的字典，例如 `{'hparam/accuracy': 0.95, 'hparam/loss': 0.12}`。TensorBoard 的 HParams 插件要求指标名称以 `hparam/` 开头，或者你可以不加前缀，它会自动处理。
*   `hparam_domain_discrete (dict, optional)`: 一个字典，键是超参数名称，值是该超参数所有可能取值的列表。例如 `{'optimizer': ['Adam', 'SGD']}`。这有助于 TensorBoard 知道超参数的范围。
*   `run_name (str, optional)`: 此特定超参数运行的名称。如果提供，它会用于创建此运行的子目录。

```python
print("Logging hyperparameters...")
# 模拟两次不同的超参数运行
hparams_run1 = {'lr': 0.01, 'batch_size': 32, 'optimizer': 'SGD'}
metrics_run1 = {'hparam/accuracy': 0.75, 'hparam/loss': 0.5}

hparams_run2 = {'lr': 0.001, 'batch_size': 64, 'optimizer': 'Adam'}
metrics_run2 = {'hparam/accuracy': 0.88, 'hparam/loss': 0.25}

# 注意：当使用 add_hparams 时，通常会为每个超参数组合创建一个新的 SummaryWriter
# 或者，如 PyTorch 官方教程所示，在单个 writer 的不同子目录中记录。
# 这里为了简化，我们尝试在同一个 writer 上记录，但这可能不是最佳实践。
# 更好的方法是为每个 hparam run 创建一个单独的 writer 实例，并给不同的 log_dir
# e.g., writer_hparam_run1 = SummaryWriter('runs/hparam_exp/run1')
# writer_hparam_run1.add_hparams(hparams_run1, metrics_run1)
# writer_hparam_run1.close()

try:
    # 记录第一个运行
    # 为了让 TensorBoard HParams 插件正确分组，通常你会为每次 hparam 运行使用单独的 writer
    # 或者，如果使用同一个 writer，确保 metric_dict 中的键对于不同的运行是唯一的，
    # 或者 add_hparams 会在内部处理。
    # 较新版本的 TensorBoard/PyTorch 处理得更好。
    writer.add_hparams(hparams_run1, metrics_run1, run_name="run1_sgd_lr0.01_bs32")
    writer.add_hparams(hparams_run2, metrics_run2, run_name="run2_adam_lr0.001_bs64")
except Exception as e:
    print(f"Could not log hparams (ensure PyTorch >= 1.3 and TensorBoard >= 2.2): {e}")
    print("HParams logging might require specific TensorBoard setup or newer versions.")
```
**提示**: 对于 `add_hparams`，最可靠的方式是为每个超参数组合创建一个 `SummaryWriter` 实例，并写入到不同的子目录中，例如 `runs/my_experiment/lr_0.01_bs_32`。

---

**8. 记录 Embeddings**

用于可视化高维数据（如词嵌入、图像特征嵌入）在低维空间（通常是2D或3D，通过PCA或t-SNE）的投影。

`writer.add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)`
*   `mat (torch.Tensor or numpy.ndarray)`: 形状为 `(N, D)` 的嵌入矩阵，其中 `N` 是数据点的数量，`D` 是嵌入的维度。
*   `metadata (list of str, optional)`: 长度为 `N` 的列表，包含每个数据点的标签或元数据。如果 `metadata_header` 被设置，那么 `metadata` 也可以是一个列表的列表，每行对应一个数据点，每列对应 `metadata_header` 中的一个头部。
*   `label_img (torch.Tensor, optional)`: 形状为 `(N, C, H, W)` 的张量，包含每个数据点对应的图像。这些图像将在 TensorBoard 的投影仪中显示在数据点旁边。
*   `global_step (int, optional)`: 全局步骤。
*   `tag (str, optional)`: 嵌入的名称。
*   `metadata_header (list of str, optional)`: 如果 `metadata` 是表格形式，则此列表包含每列的标题。

```python
print("Logging embeddings...")
# 假设我们有 50 个样本，每个样本是 20 维的特征向量
features_for_embedding = torch.randn(50, 20)
# 为每个样本生成标签 (例如，属于 5 个类别中的一个)
class_labels = [f'Class_{i // 10}' for i in range(50)]
# (可选) 为每个样本生成一个小的代表性图像 (这里用随机图像)
try:
    dummy_images_for_embedding = torch.rand(50, 1, 10, 10) # 50 images, 1 channel, 10x10 pixels
    writer.add_embedding(
        mat=features_for_embedding,
        metadata=class_labels,
        label_img=dummy_images_for_embedding,
        global_step=0,
        tag='My_Feature_Embeddings'
    )
except Exception as e:
    print(f"Could not log embeddings (check tensor shapes, versions, and installation of 'tensorboard-plugin-profile'): {e}")
    # Fallback without images if label_img causes issues
    try:
        writer.add_embedding(
            mat=features_for_embedding,
            metadata=class_labels,
            global_step=0,
            tag='My_Feature_Embeddings_no_img'
        )
    except Exception as e_no_img:
        print(f"Could not log embeddings (even without images): {e_no_img}")

```

---

**9. 记录 PR 曲线 (Precision-Recall Curve)**

用于可视化二分类模型的精确率-召回率曲线。

`writer.add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)`
*   `tag (str)`: PR 曲线的标识符。
*   `labels (torch.Tensor or numpy.ndarray)`: 布尔型或 0/1 的真实标签。
*   `predictions (torch.Tensor or numpy.ndarray)`: 模型对每个样本的预测概率（或置信度），范围通常在 [0, 1]。
*   `global_step (int, optional)`: 全局步骤。
*   `num_thresholds (int, optional)`: 用于生成曲线的阈值数量。

```python
print("Logging PR curve...")
# 模拟二分类任务的真实标签和预测概率
true_binary_labels = torch.randint(0, 2, (100,)) # 100 个 0 或 1 的真实标签
predicted_probabilities = torch.rand(100)       # 100 个预测概率 (0 到 1)

try:
    writer.add_pr_curve('Performance/PR_Curve', true_binary_labels, predicted_probabilities, 0)
except Exception as e:
    # 可能需要安装特定版本的 protobuf 或修复版本冲突
    print(f"Could not log PR curve: {e}. This sometimes requires `pip install protobuf==3.20.0` or similar fixes.")
```

---

**最后，关闭 `SummaryWriter`**

完成所有记录后，或者在程序退出前，确保关闭 `writer`。这会确保所有挂起的数据都被写入磁盘。

```python
writer.close()
print("\nAll data logged (or attempted). Writer closed.")
```

**启动 TensorBoard 查看结果**

1.  打开你的终端或命令行。
2.  导航到包含 `runs` 目录（或其他你指定的 `log_dir`）的父目录。
3.  运行命令:
    ```bash
    tensorboard --logdir=runs
    ```
    或者，如果你指定了不同的 `log_dir`，例如 `SummaryWriter('my_custom_logs')`，则运行：
    ```bash
    tensorboard --logdir=my_custom_logs
    ```
4.  TensorBoard 会输出一个 URL，通常是 `http://localhost:6006`。在你的浏览器中打开这个 URL 即可查看可视化界面。

```python
print("\nTo view results in TensorBoard, run the following command in your terminal:")
print("tensorboard --logdir=runs")
print("Then open http://localhost:6006 (or the URL shown in the terminal) in your browser.")
```
