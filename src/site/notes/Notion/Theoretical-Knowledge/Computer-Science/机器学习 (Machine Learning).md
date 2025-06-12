---
{"dg-publish":true,"permalink":"/notion/theoretical-knowledge/computer-science/machine-learning/"}
---

# 1. 基本概念 (Basic Concepts)

## 1.1 主要机器学习任务类型 (Major Types of Machine Learning Tasks)

机器学习算法可以根据其学习目标和所处理的数据类型大致分为以下几类：

1.  **监督学习 (Supervised Learning)**
    *   **目标**: 从带标签的训练数据（即每个数据点都有一个已知的“答案”或“目标输出”）中学习一个映射函数，以便对新的、未见过的数据进行预测。
    *   **子类型**:
        *   **回归 (Regression)**:
            *   **目标输出**: 连续的数值 (a continuous scalar value)。
            *   **例子**: 预测房价、股票价格、温度。
        ![Image/Machine Learning/1.png](/img/user/Image/Machine%20Learning/1.png)
        *   **分类 (Classification)**:
            *   **目标输出**: 离散的类别标签 (a discrete class label) from a predefined set.
            *   **例子**: 图像识别（猫/狗）、邮件分类（垃圾/非垃圾）、疾病诊断（有病/无病）。![Image/Machine Learning/2.png](/img/user/Image/Machine%20Learning/2.png)
            ![3.jpg](/img/user/Image/Machine%20Learning/3.jpg)

2.  **无监督学习 (Unsupervised Learning)**
    *   **目标**: 从未带标签的数据中发现隐藏的模式、结构或关系。算法自行探索数据。
    *   **子类型**:
        *   **聚类 (Clustering)**:
            *   **目标**: 将数据点分组成相似的集合（簇），使得同一簇内的数据点相似度高，不同簇之间的数据点相似度低。
            *   **例子**: 客户分群、异常检测。
        *   **降维 (Dimensionality Reduction)**:
            *   **目标**: 减少数据特征的数量，同时保留重要信息，以便于可视化、提高效率或减少噪声。
            *   **例子**: 主成分分析 (PCA)、t-SNE。
        *   **关联规则学习 (Association Rule Learning)**:
            *   **目标**: 发现数据项之间的有趣关系或关联。
            *   **例子**: 购物篮分析（“购买面包的人也倾向于购买牛奶”）。
        *   **(概率)结构学习 (Probabilistic Structure Learning / Graphical Model Learning)
        *   ***(部分属于此类)***:
            *   **目标**: 发现一组随机变量之间的概率依赖关系，并用图结构（如贝叶斯网络、马尔可夫网络）来表示这些关系。
            *   **特性**: 当没有预先指定变量间的关系，而是从数据中推断这些关系时，这通常被视为一种无监督的发现过程。它可以帮助理解数据的内在结构和生成机制。
            *   **例子**: 从基因表达数据中推断基因调控网络，从传感器数据中学习变量间的依赖。

3.  **强化学习 (Reinforcement Learning)**
    *   **目标**: 智能体 (agent) 通过与环境 (environment) 交互来学习如何做出决策，以最大化累积奖励 (cumulative reward)。智能体通过试错来学习最优策略。
    *   **例子**: 训练机器人行走、棋类游戏AI (AlphaGo)、自动驾驶策略。

## 1.2 机器学习的步骤 (Steps in Machine Learning)

### 1.2.1 定义一个带有未知参数的函数/模型 (Define a Function/Model with Unknown Parameters)

机器学习的核心任务之一是从数据中学习一个函数（或模型），该函数能够很好地描述输入和输出之间的关系，或者发现数据中的潜在结构。这个函数通常包含一些**未知参数 (unknown parameters)**，这些参数的值需要从训练数据中学习得到。

以一个简单的线性回归模型为例：
$$y = b + w x_1$$
这里：
*   $y$ 是我们想要预测的目标输出 (target output)。
*   $x_1$ 是一个输入特征 (input feature)。
*   $w$ (权重, weight) 和 $b$ (偏置, bias) 是模型的**未知参数**。我们的目标就是通过学习算法，利用训练数据来找到最优的 $w$ 和 $b$ 的值。

### 1.2.2 定义代价函数/损失函数以评估模型 (Define Cost/Loss Function to Evaluate the Model)

在定义了带有未知参数的模型之后，我们需要一种方法来**衡量模型的预测结果与真实目标值之间的差异**。这个衡量标准就是**代价函数 (Cost Function)** 或 **损失函数 (Loss Function)**。代价函数的值反映了当前模型参数的好坏：代价越小，模型对训练数据的拟合越好。

**代价函数是参数的函数 (Loss is a function of parameters):**
给定训练数据集，对于一组特定的模型参数（例如 $w$ 和 $b$），我们可以计算出模型在整个训练集上的总体表现。因此，代价函数 $L$ 可以看作是这些未知参数的函数。例如，对于参数 $w$ 和 $b$，代价函数可以表示为 $L(w, b)$。

$$L(w, b) = \frac{1}{N} \sum_{n=1}^{N} e_n$$
这里：
*   $N$ 是训练样本的总数。
*   $e_n$ 是模型在第 $n$ 个训练样本上的**误差 (error)** 或 **损失 (loss)**。
*   代价函数 $L(w, b)$ 是所有单个样本损失的平均值（或总和）。

**常见的单个样本损失计算方式 ($e_n$)**:
*   **平均绝对误差 (Mean Absolute Error, MAE)**:
    $$e_n = |\hat{y}^{(n)} - y^{(n)}|$$
*   **均方误差 (Mean Squared Error, MSE)**:
    $$e_n = (\hat{y}^{(n)} - y^{(n)})^2$$
*   **交叉熵 (Cross-Entropy)**:
    常用于分类问题，衡量预测概率分布与真实类别分布之间的差异。具体形式取决于二分类还是多分类。
    *   **二分类交叉熵 (Binary Cross-Entropy)**:
        $e_n = - [y^{(n)} \log(\hat{y}^{(n)}) + (1 - y^{(n)}) \log(1 - \hat{y}^{(n)})]$
        (其中 $y^{(n)} \in \{0, 1\}$, $\hat{y}^{(n)}$ 是预测为类别1的概率)
    *   **多分类交叉熵 (Categorical Cross-Entropy)**:
        $e_n = - \sum_{k=1}^{K} y_k^{(n)} \log(\hat{y}_k^{(n)})$
        (其中 $y^{(n)}$ 是one-hot编码的真实标签, $\hat{y}^{(n)}$ 是预测的概率分布, $K$ 是类别数)

**损失函数的作用 (The Role of the Loss Function):**
损失函数告诉我们**当前这组参数 $(w, b)$ 的表现有多好 (how good a set of values is)**。我们的目标是找到一组参数，使得这个损失函数的值最小。

**误差平面 (Error Surface):**
我们可以将损失函数 $L(w, b)$ 想象成一个多维空间中的曲面，其中参数（如 $w$ 和 $b$）是坐标轴，损失函数的值是高度。这个曲面被称为**误差平面 (Error Surface)** 或损失平面。
![4.png](/img/user/Image/Machine%20Learning/4.png) *(图示：损失值如何随着参数变化而变化，目标是找到曲面的最低点)*

### 1.2.3 参数优化 (Optimization)

一旦我们定义了模型和损失函数 $L(w, b)$，接下来的目标就是找到一组最优的参数 $w^*$ 和 $b^*$，使得损失函数的值最小。这个过程称为**优化 (Optimization)**。

数学上，我们可以表示为：
$$w^*, b^* = \arg\min_{w, b} L(w, b)$$
这意味着我们要寻找使损失函数 $L$ 达到最小值的参数 $w$ 和 $b$。

**主要方法：梯度下降 (Ways: Gradient Descent)**
梯度下降是一种广泛应用于机器学习和深度学习中的迭代优化算法，用于寻找函数的最小值。其基本思想是沿着损失函数梯度下降最快的方向逐步调整参数。

**梯度下降的步骤:**

1.  **初始化参数 (Pick an initial value)**:
    随机选择或根据某种策略设定参数的初始值，例如 $w^0, b^0$。

2.  **计算梯度 (Compute Gradient)**:
    计算损失函数 $L$ 在当前参数点（例如 $w^t, b^t$）关于每个参数的偏导数（即梯度）。
    *   对于参数 $w$: $\frac{\partial L}{\partial w} \Big|_{w=w^t, b=b^t}$
    *   对于参数 $b$: $\frac{\partial L}{\partial b} \Big|_{w=w^t, b=b^t}$
    梯度指明了在该点函数值增长最快的方向。

3.  **确定更新量 (Determine Update Amount)**:
    *   $\eta$ (eta) 代表**学习率 (Learning Rate)**，它是一个[[Notion/Theoretical-Knowledge/Computer-Science/Concept/超参数 (Hyperparameters)\|超参数 (Hyperparameters)]]，控制每次参数更新的步长。
    *   参数更新的量由学习率乘以梯度的负值决定（因为我们要向梯度反方向，即下降方向移动）：
        *   对 $w$ 的更新量: $-\eta \frac{\partial L}{\partial w} \Big|_{w=w^t, b=b^t}$
        *   对 $b$ 的更新量: $-\eta \frac{\partial L}{\partial b} \Big|_{w=w^t, b=b^t}$

4.  **更新参数 (Update Parameters)**:
    将当前参数值减去上一步计算的更新量，得到新的参数值：
    $$w^{t+1} = w^t - \eta \frac{\partial L}{\partial w} \Big|_{w=w^t, b=b^t}$$
    $$b^{t+1} = b^t - \eta \frac{\partial L}{\partial b} \Big|_{w=w^t, b=b^t}$$

5.  **迭代 (Update iteratively)**:
    重复步骤 2 到 4，直到损失函数收敛到足够小的值，或者达到预设的最大迭代次数，或者满足其他停止条件。

**梯度下降的直观理解**: 想象你在一个山上（误差平面），目标是走到山谷的最低点。在每一步，你都会观察当前位置哪个方向坡度最陡峭向下（梯度的反方向），然后朝着那个方向走一小步（步长由学习率控制）。

---

### 1.2.4 机器学习核心步骤与深度学习的关系 (Relationship of Core Machine Learning Steps to Deep Learning)

上述三个核心步骤——**1. 定义一个带有未知参数的模型**，**2. 定义一个损失函数来评估模型**，以及 **3. 通过优化算法寻找最优参数**——构成了监督式机器学习的完整流程，并且这一框架在**深度学习**中得到了直接的应用和显著的扩展：

1.  **参数化模型的核心思想一致 (Consistent Core Idea of Parameterized Models):**
    *   无论是简单的线性回归还是复杂的深度神经网络，其本质都是**参数化的函数/模型**，包含大量需要从数据中学习的未知参数（权重和偏置）。
    *   深度学习模型（如神经网络）通过多层非线性变换构建出表达能力极强的参数化函数。

2.  **损失函数作为统一的评估和优化目标 (Loss Function as a Unified Goal for Evaluation and Optimization):**
    *   对于任何参数化模型，都需要一个**损失函数**来量化其预测与真实目标之间的差距。
    *   在深度学习中，训练的目标同样是找到一组使损失函数最小化的参数。

3.  **优化算法作为学习的驱动力 (Optimization Algorithms as the Engine of Learning):**
    *   **梯度下降及其变体**是深度学习中最核心的优化工具。由于深度神经网络参数众多，损失函数的“误差平面”异常复杂，包含许多局部最小值、鞍点等。
    *   深度学习领域发展了许多先进的梯度下降变体，如：
        *   **随机梯度下降 (Stochastic Gradient Descent, SGD)**: 每次使用单个样本计算梯度并更新，速度快但波动大。
        *   **Mini-batch 梯度下降**: 每次使用一小批样本计算梯度，是 Batch GD 和 SGD 的折衷，也是目前最常用的方法。
        *   **带动量的优化器 (Optimizers with Momentum)**: 如 Momentum, Nesterov Accelerated Gradient (NAG)，通过引入动量项加速收敛并帮助越过小的局部极值点。
        *   **自适应学习率优化器 (Adaptive Learning Rate Optimizers)**: 如 AdaGrad, RMSProp, Adam, AdamW，它们能为每个参数自动调整学习率，通常能更快收敛且对初始学习率不那么敏感。
    *   **反向传播 (Backpropagation)**: 对于深度神经网络这样复杂的复合函数，高效计算梯度至关重要。反向传播算法利用链式法则，系统地计算损失函数关于网络中所有参数的梯度，为梯度下降提供“方向盘”。

4.  **深度学习的扩展与深化 (Extensions and Deepening in Deep Learning):**
    *   **模型复杂度与层级结构**: 深度学习通过构建**深层网络结构**极大地扩展了模型的复杂度，使其能够学习从低级到高级的层次化特征表示。
    *   **非线性能力**: 深度学习模型广泛使用**非线性激活函数**，使其能够学习高度非线性的映射关系。
    *   **特定任务的损失与模型**: 针对各种复杂任务（图像识别、自然语言处理等），深度学习发展了特定的网络架构（如CNNs, RNNs, Transformers）和相应的损失函数。
    *   **端到端学习 (End-to-End Learning)**: 深度学习常常实现“端到端”学习，即从原始输入直接学习到最终输出，整个过程由参数通过最小化损失函数自动学习得到。

**总结来说，"定义模型"、"定义损失函数" 和 "优化参数" 这三个步骤是机器学习（尤其是监督学习）的基本骨架。深度学习在这个骨架的基础上，通过构建更复杂强大的模型（神经网络）、采用合适的损失函数，并利用高效的优化算法（如基于反向传播的梯度下降及其变体），来解决更具挑战性的问题并取得了巨大成功。理解这三个基本步骤及其在深度学习中的具体实现，是掌握深度学习原理的关键。**
