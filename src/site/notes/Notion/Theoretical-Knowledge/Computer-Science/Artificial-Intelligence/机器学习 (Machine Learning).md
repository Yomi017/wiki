---
{"dg-publish":true,"permalink":"/notion/theoretical-knowledge/computer-science/artificial-intelligence/machine-learning/"}
---

# 1. 机器学习基本概念 (Basic Concepts)

**Video:** https://www.youtube.com/watch?v=Ye018rCVvOo

## 1.1 主要机器学习任务类型 (Major Types of Machine Learning Tasks)

机器学习算法可以根据其学习目标和所处理的数据类型大致分为以下几类：

1.  **监督学习 (Supervised Learning)**
    *   **目标**: 从带标签的训练数据（即每个数据点都有一个已知的“答案”或“目标输出”）中学习一个映射函数，以便对新的、未见过的数据进行预测。
    *   **子类型**:
        *   **回归 (Regression)**:
            *   **目标输出**: 连续的数值 (a continuous scalar value)。
            *   **例子**: 预测房价、股票价格、温度。
        ![Image/Computer-Science/Machine Learning/1.png](/img/user/Image/Computer-Science/Machine%20Learning/1.png)
        *   **分类 (Classification)**:
            *   **目标输出**: 离散的类别标签 (a discrete class label) from a predefined set.
            *   **例子**: 图像识别（猫/狗）、邮件分类（垃圾/非垃圾）、疾病诊断（有病/无病）。![Image/Computer-Science/Machine Learning/2.png](/img/user/Image/Computer-Science/Machine%20Learning/2.png)
            ![3.jpg](/img/user/Image/Computer-Science/Machine%20Learning/3.jpg)

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
![Image/Computer-Science/Machine Learning/4.png](/img/user/Image/Computer-Science/Machine%20Learning/4.png) *(图示：损失值如何随着参数变化而变化，目标是找到曲面的最低点)*

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
    *   $\eta$ (eta) 代表**学习率 (Learning Rate)**，它是一个[[Notion/Theoretical-Knowledge/Computer-Science/Artificial-Intelligence/Concept/超参数 (Hyperparameters)\|超参数 (Hyperparameters)]]，控制每次参数更新的步长。
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

---
## 2. 深度学习基本概念 (Basic Concepts of Deep Learning)

**Video:** https://www.youtube.com/watch?v=bHcJCp2Fyxs

## 2.1 步骤一：定义带有未知参数的函数/模型 (Step 1: Define a Function/Model with Unknown Parameters)

### 2.1.1 动机：线性模型的局限性 (Model Bias of Linear Models)

线性模型，如 $y = b + wx_1$ 或 $y = b + \sum_j w_j x_j$，在许多情况下非常有用且易于解释。然而，它们具有固有的**模型偏差 (Model Bias)**，这意味着它们只能表示输入和输出之间的线性关系。

*   **局限性 (Limitation):** 如果数据中的真实关系是非线性的，线性模型将无法很好地拟合数据，导致预测性能不佳。
    *   例如，在你的图示中（第一张图，红色曲线），如果 $y$ 和 $x_1$ 之间的关系是如图所示的折线，那么单一的直线（线性模型）无法准确地捕捉这种模式。

### 2.1.2 构建更灵活的模型：分段线性函数 (Building More Flexible Models: Piecewise Linear Functions)

为了克服线性模型的局限性，我们需要能够表示非线性关系的更灵活的模型。一种方法是使用**分段线性函数 (Piecewise Linear Curves)**。

*   **基本思想 (Core Idea):** 任何复杂的分段线性曲线（如图中的红色曲线）可以被看作是一个**常数 (constant)** 加上一系列更简单的“阶梯状”或“斜坡状”基础函数（如图中右上角示意的小蓝色函数）的**加权和 (sum of a set of blue functions)**。![Image/Computer-Science/Machine Learning/5.png](/img/user/Image/Computer-Science/Machine%20Learning/5.png)![Image/Computer-Science/Machine Learning/6.png](/img/user/Image/Computer-Science/Machine%20Learning/6.png)
        第一张图展示了一个目标分段线性函数（红色曲线）。
        第二张图展示了如何将这个红色曲线分解为多个基础的蓝色函数。
        每个蓝色函数在某个点改变其斜率。

*   **逼近任意连续曲线 (Approximating Arbitrary Continuous Curves):**
    *   即使目标函数不是严格的分段线性函数，我们也可以通过在其上选择足够多的点，并用连接这些点的折线（即分段线性函数）来近似它。
    *   **关键**: 要想获得好的近似效果，我们需要足够多的“片段”(pieces) 或基础蓝色函数。

*   **如何表示基础的“蓝色函数” (How to represent the blue function / Hard Sigmoid)?**
    *   这种在某个点改变斜率，从平坦变为有斜率再变为平坦的基础函数，可以被称为**硬 Sigmoid (Hard Sigmoid)** 或斜坡函数。
    *   更进一步，我们可以使用平滑的 **Sigmoid 函数**来近似这种硬 Sigmoid 的行为，或者直接用 Sigmoid 函数作为构建块。标准的 Sigmoid 函数定义为：
        $$ \text{sigmoid}(z) = \frac{1}{1+e^{-z}} $$
    *   通过调整 Sigmoid 函数的参数，我们可以控制这个“平滑阶梯”的形状：
        *   考虑一个经过缩放和平移的 Sigmoid 函数： $y = c \cdot \text{sigmoid}(b' + w'x_1)$ (这里为了与你笔记中的 $b_i, w_i$ 对应，我们暂时用 $b', w'$)
        *   $w'$ (权重, weight): 改变 Sigmoid 函数的**斜率 (slope)** 或陡峭程度。
        *   $b'$ (偏置, bias): 改变 Sigmoid 函数在 $x_1$ 轴上的**平移 (shift)** 位置。
        *   $c$ (系数, coefficient): 改变 Sigmoid 函数的**高度 (height)** 或幅度。

*   **用 Sigmoid 函数构建分段线性模型 (Building Piecewise Linear Models with Sigmoid Functions):**
    *   因此，原始的目标分段线性函数 $y$ (或其近似) 可以被表示为一系列 Sigmoid 函数的加权和，再加上一个整体的偏置：
        $$ y = b_{overall} + \sum_i c_i \cdot \text{sigmoid}(b_i + w_i x_1) $$
        这里：
        *   $b_{overall}$ 是整体的偏置项 (constant)。
        *   $c_i, b_i, w_i$ 分别是第 $i$ 个 Sigmoid 组件的高度、平移和斜率控制参数。
        *   **每个 $c_i \cdot \text{sigmoid}(b_i + w_i x_1)$ 就对应于一个“蓝色函数”组件。**

    *   **其他激活函数 (Other Activation Functions) - 例如 ReLU:**
        除了 Sigmoid 函数，还有许多其他的**激活函数 (Activation Functions)** 可以用来在神经网络中引入非线性，从而构建能够拟合复杂模式的模型。
        *   **ReLU (Rectified Linear Unit)** 是目前深度学习中最常用的一种激活函数。其定义为：
            $$ \text{ReLU}(z) = \max(0, z) $$
        *   如果使用 ReLU 作为激活函数，并且考虑到多特征输入（即每个 ReLU 单元的输入是 $\sum_j w_{ij}x_j + b_i$），那么模型可以表示为：
            $$ y = b_{output} + \sum_i c_i \cdot \text{ReLU}\left(b_{hidden,i} + \sum_j w_{ij} x_j\right) $$
            或者更简洁地写成：
            $$ y = b_{output} + \sum_i c_i \cdot \max\left(0, b_{hidden,i} + \sum_j w_{ij} x_j\right) $$
        *   **为什么 ReLU 常用？**
            *   **计算简单高效**: ReLU 的计算非常快（只是一个取最大值的操作）。
            *   **缓解梯度消失问题**: 对于正输入，ReLU 的梯度是1，这有助于在深层网络中更好地传播梯度，缓解了 Sigmoid 等函数在输入值很大或很小时梯度接近0（梯度消失）的问题。
            *   **稀疏性**: ReLU 会使一部分神经元的输出为0（当输入为负时），这可以带来一定的网络稀疏性，有时被认为有助于特征学习。
        *   当然，ReLU 也有其缺点，比如“Dying ReLU”问题（神经元可能永久失活）。

    *   **激活函数的角色**: [[Notion/Theoretical-Knowledge/Computer-Science/Artificial-Intelligence/深度学习 (Deep Learning)#^c0241b\|深度学习 (Deep Learning)#^c0241b]] 
        无论是 Sigmoid, ReLU, Tanh 还是其他激活函数，它们的核心作用都是在神经网络的每一层（或特定层）引入**非线性**，使得网络能够学习和表示输入与输出之间复杂的、非线性的映射关系。没有非线性激活函数，多层神经网络将退化为一个等效的单层线性模型。

    *   **哪个更好 (Which is better)?**
        *   激活函数的选择取决于具体的应用场景、网络结构和经验。
        *   在许多现代深度学习应用中，**ReLU 及其变体 (如 Leaky ReLU, PReLU, ELU 等) 通常是隐藏层的首选激活函数**，因为它们往往能带来更好的训练动态和性能。
        *   Sigmoid 和 Tanh 仍然在某些特定场景下使用，例如 Sigmoid 常用于二分类问题的输出层（输出概率），Tanh 有时用于隐藏层（其输出范围在-1到1之间，中心对称）。
### 2.1.3 推广到多特征输入：构建神经网络模型 (Generalizing to Multiple Features: Building a Neural Network Model)

上面的讨论是基于单个输入特征 $x_1$。现在，我们将这个思想推广到具有多个输入特征 $x_j$ 的情况。

*   **从简单线性模型到 Sigmoid 组合 (From Simple Linear Model to Sigmoid Combination):**
    *   对于单特征:
        $$ y = b + wx_1 \quad \Rightarrow \quad y = b_{overall} + \sum_i c_i \cdot \text{sigmoid}(b_i + w_i x_1) $$
    *   对于多特征，线性模型为: $y = b + \sum_j w_j x_j$
    *   类比地，我们将每个 Sigmoid 函数的输入从 $b_i + w_i x_1$ 推广为多个特征的线性组合 $b_i + \sum_j w_{ij} x_j$:
        $$ y = b_{overall} + \sum_i c_i \cdot \text{sigmoid}\left(b_i + \sum_j w_{ij} x_j\right) $$

*   **引入神经网络的表示方法 (Introducing Neural Network Notation):**
    让我们用更标准的神经网络符号来重写这个模型。
    *   **第一步：计算每个 Sigmoid 的输入 (通常称为加权输入或 pre-activation)**
        对于第 $i$ 个 Sigmoid 组件（可以看作是隐藏层的一个神经元），其输入 $r_i$ 是所有输入特征 $x_j$ 的加权和，再加上该组件的偏置 $b_i$:
        $$ r_i = b_i + \sum_j w_{ij} x_j $$
        其中 $w_{ij}$ 是连接第 $j$ 个输入特征 $x_j$ 到第 $i$ 个 Sigmoid 组件的权重。
        *   **矩阵形式 (Vectorized Form):**
            如果我们将所有输入特征表示为向量 $x = [x_1, x_2, \dots, x_N]^T$，所有偏置 $b_i$ 组成向量 $\mathbf{b} = [b_1, b_2, \dots, b_M]^T$，所有权重 $w_{ij}$ 组成权重矩阵 $W$ (其中 $W_{ij}$ 是 $w_{ij}$)，那么所有 $r_i$ 组成的向量 $\mathbf{r} = [r_1, r_2, \dots, r_M]^T$ 可以表示为：
            $$ \mathbf{r} = \mathbf{b} + W \mathbf{x} $$
             即：
             $$
             \begin{pmatrix}
             r_1^{(j+1)} \\
             r_2^{(j+1)} \\
             \vdots \\
             r_{N_{j+1}}^{(j+1)}
             \end{pmatrix}
             =
             \begin{pmatrix}
             b_1^{(j+1)} \\
             b_2^{(j+1)} \\
             \vdots \\
             b_{N_{j+1}}^{(j+1)}
             \end{pmatrix}+\begin{pmatrix}
             W_{1,1}^{(j+1)} & W_{1,2}^{(j+1)} & \cdots & W_{1,N_j}^{(j+1)} \\
             W_{2,1}^{(j+1)} & W_{2,2}^{(j+1)} & \cdots & W_{2,N_j}^{(j+1)} \\
             \vdots & \vdots & \ddots & \vdots \\
             W_{N_{j+1},1}^{(j+1)} & W_{N_{j+1},2}^{(j+1)} & \cdots & W_{N_{j+1},N_j}^{(j+1)} \\
             \end{pmatrix}
             \begin{pmatrix}
             a_1^{(j)} \\
             a_2^{(j)} \\
             \vdots \\
             a_{N_j}^{(j)}
             \end{pmatrix} $$

    *   **第二步：应用 Sigmoid 激活函数 (Apply Sigmoid Activation Function)**
        将每个 $r_i$ 通过 Sigmoid 函数得到激活值 $a_i$：
        $$ a_i = \text{sigmoid}(r_i) $$
        *   **矩阵形式 (Vectorized Form):**
            如果 $\sigma(\cdot)$ 表示逐元素应用 Sigmoid 函数，那么激活向量 $\mathbf{a} = [a_1, a_2, \dots, a_M]^T$ 可以表示为：
            $$ \mathbf{a} = \sigma(\mathbf{r}) = \sigma(\mathbf{b} + W \mathbf{x}) $$
        这些激活值 $a_i$ 构成了神经网络的**隐藏层 (Hidden Layer)** 的输出。

    *   **第三步：组合隐藏层输出得到最终预测 (Combine Hidden Layer Outputs for Final Prediction)
        最终的输出 $y$ 是这些隐藏层激活值 $a_i$ 的加权和，再加上一个最终的输出层偏置 $b_{output}$ (对应图 `7.png` 中的 $b$)。权重为 $c_i$。
        $$ y = b_{output} + \sum_i c_i a_i$$
        *   **矩阵形式 (Vectorized Form):**
            如果我们将权重 $c_i$ 组成一个行向量 $\mathbf{c}^T = [c_1, c_2, \dots, c_M]$ (或者列向量 $\mathbf{c}$ 然后取转置)，则：
            $$ y = b_{output} + \mathbf{c}^T \mathbf{a} $$

    *   **整合模型 (Putting It All Together):**
        将以上步骤整合，我们就得到了一个具有单隐藏层的神经网络模型：
        $$ y = b_{output} + \mathbf{c}^T \sigma(\mathbf{b}_{hidden} + W \mathbf{x}) $$
        这里：
        *   $\mathbf{x}$ 是输入特征向量。
        *   $W$ 是输入层到隐藏层的权重矩阵。
        *   $\mathbf{b}_{hidden}$ 是隐藏层的偏置向量。
        *   $\sigma(\cdot)$ 是 Sigmoid 激活函数（或其他非线性激活函数）。
        *   $\mathbf{c}^T$ 是隐藏层到输出层的权重向量。
        *   $b_{output}$ 是输出层的偏置。
        *   所有 $W, \mathbf{b}_{hidden}, \mathbf{c}^T, b_{output}$ 都是模型需要从数据中学习的**未知参数**。

    *   ![Image/Computer-Science/Machine Learning/7.png](/img/user/Image/Computer-Science/Machine%20Learning/7.png)
        这张图完美地展示了这个单隐藏层神经网络的结构：
        *   **输入层 (Input Layer)**: $x_1, x_2, x_3$。
        *   **隐藏层 (Hidden Layer)**:
            *   三个神经元（用黑色圆圈1, 2, 3表示）。
            *   每个神经元首先计算加权输入 $r_i = b_i + \sum_j w_{ij}x_j$ (图中 $b_1, b_2, b_3$ 是隐藏层偏置，通过连接值为1的绿色方块引入；$w_{11}, w_{12}, \dots$ 是权重)。
            *   然后通过 Sigmoid 激活函数（蓝色弯曲符号）得到激活值 $a_1, a_2, a_3$。
        *   **输出层 (Output Layer)**:
            *   计算 $y = b + c_1 a_1 + c_2 a_2 + c_3 a_3$ (图中 $b$ 是输出层偏置，通过连接值为1的绿色方块引入；$c_1, c_2, c_3$ 是隐藏层到输出层的权重)。

-   **随机性**: 当我们刚开始训练一个神经网络时，我们并不知道参数 $w,b,c$ (以及更复杂的网络中的所有权重和偏置) 应该是什么值。
-   因此，我们通常会用**小的随机数**来初始化这些参数。比如，从一个均值为0，方差很小的高斯分布中采样，或者在一个小的区间内均匀采样
*   所以最初，表达式 $c_i \cdot \text{sigmoid}(b_i + w_i x_1)$ （以及整个模型）是一个包含**未知参数**的**函数模板**或**函数蓝图**。
*   通过**训练数据和优化算法**（反向传播等操作），我们**学习（确定）了这些未知参数 $w_i, b_i, c_i$ 的具体数值。
*   一旦这些参数的数值被学习确定下来，整个模型就变成了一个**具体的、参数已知的函数**。
*   我们**使用这个参数已知的函数和学习到的参数值**来对新的输入数据进行**预测**。

#### 2.1.4 模型的未知参数集 ($\theta$) (The Set of Unknown Parameters in the Model, $\theta$)

在我们定义的神经网络模型中，例如单隐藏层网络：
$$ y = f(\mathbf{x}; W, \mathbf{b}_{hidden}, \mathbf{c}^T, b_{output}) = \mathbf{c}^T \sigma(W \mathbf{x} + \mathbf{b}_{hidden}) + b_{output} $$
存在一系列需要通过从数据中学习来确定的**未知参数 (Unknown parameters)**。

这些参数具体包括：

1.  **输入层到隐藏层的权重矩阵 ($W$)**:
    *   这是一个矩阵，其维度通常是 (隐藏层神经元数量 $N_1$) $\times$ (输入特征数量 $N_0$)。
    *   它包含了连接每个输入特征到每个隐藏层神经元的所有权重。

2.  **隐藏层的偏置向量 ($\mathbf{b}_{hidden}$)**:
    *   这是一个列向量，其维度是 (隐藏层神经元数量 $N_1$) $\times 1$。
    *   图 `8.png` 中用绿色的 `b` 表示（为了区分，我们这里用 $\mathbf{b}_{hidden}$）。
    *   它包含了每个隐藏层神经元的偏置项。

3.  **隐藏层到输出层的权重向量 ($\mathbf{c}^T$ 或 $\mathbf{c}$)**:
    *   如果输出 $y$ 是一个标量（如回归或二分类的logit），那么 $\mathbf{c}^T$ 是一个行向量，维度是 $1 \times$ (隐藏层神经元数量 $N_1$)。或者其转置 $\mathbf{c}$ 是一个列向量，维度是 ($N_1 \times 1$)。
    *   图 `8.png` 中用橙色的 `c^T` 表示。
    *   它包含了连接每个隐藏层神经元到输出单元的权重。

4.  **输出层的偏置 ($b_{output}$)**:
    *   这是一个标量值。
    *   图 `8.png` 中用灰色的 `b` 表示（为了区分，我们这里用 $b_{output}$）。
    *   它是输出单元的偏置项。

**将所有参数集合为单一向量 $\theta$:**

![Image/Computer-Science/Machine Learning/8.png](/img/user/Image/Computer-Science/Machine%20Learning/8.png)

为了在后续的优化过程中（例如使用梯度下降）更方便地处理这些不同形状和类型的参数，通常会将它们全部 **“展平 (flattened)”** 并按特定顺序串联起来，形成一个单一的、非常长的参数向量 $\theta$**。

如图 `8.png` 所示，这个参数向量 $\theta$ 可以这样构成：
*   首先是权重矩阵 $W$ 的所有元素（例如，可以逐行或逐列展开）。
*   然后是隐藏层偏置向量 $\mathbf{b}_{hidden}$ 的所有元素。
*   接着是输出层权重向量 $\mathbf{c}^T$ (或 $\mathbf{c}$) 的所有元素。
*   最后是输出层偏置 $b_{output}$。

所以，这个参数向量 $\theta$ 可以表示为：
$$ \theta = \begin{pmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \\ \vdots \\ \theta_K \end{pmatrix} $$
其中 $K$ 是模型中所有独立参数的总数量。例如，如果 $W$ 是 $N_1 \times N_0$，$b_{hidden}$ 是 $N_1 \times 1$，$\mathbf{c}^T$ 是 $1 \times N_1$，$b_{output}$ 是标量，则 $K = N_1 \cdot N_0 + N_1 + N_1 + 1$。

**模型函数以 $\theta$ 为参数:**
通过这种方式，我们可以将整个神经网络模型函数 $f$ 视为以输入 $\mathbf{x}$ 和这个单一的参数向量 $\theta$ 为参数的函数：
$$ y = f(\mathbf{x}; \theta) $$

**下一步：**
在明确了模型的函数形式 $f(\mathbf{x}; \theta)$ 并识别出所有未知参数 $\theta$ 之后，机器学习的下一步骤将是定义一个代价函数来评估使用特定 $\theta$ 时模型的表现，然后使用优化算法来找到使代价函数最小化的最优 $\theta^*$。

## 2.2 步骤二：定义代价函数/损失函数以评估模型 (Step 2: Define Cost/Loss Function to Evaluate the Model)

在**步骤一 (2.1)** 中，我们定义了一个带有未知参数 $\theta$ 的模型函数 $y = f(\mathbf{x}; \theta)$。现在，我们需要一种方法来衡量这个模型在使用特定参数集 $\theta$ 时，其预测结果与真实目标值之间的差异有多大。这个衡量标准就是**代价函数 (Cost Function)** 或 **损失函数 (Loss Function)**。

### 2.2.1 代价函数/损失函数 (Cost Function / Loss Function)

*   **定义 (Definition):**
    *   损失函数 $L(\theta)$ 是一个关于模型参数 $\theta$ 的函数。
    *   它量化了模型在使用当前参数 $\theta$ 时，在整个训练数据集上预测的好坏程度 (Loss means how good a set of value is / how bad the current set of parameters is)。
    *   损失值越小，表示模型的预测越接近真实值，即当前这组参数 $\theta$ 的表现越好。

*   **计算方法 (How it's Calculated):**
    1.  **单个样本的损失 (Loss for a single example):**
        对于训练集中的每一个样本 $(x^{(n)}, y_{true}^{(n)})$ (其中 $n$ 是样本索引)，我们首先计算模型在该样本上的预测值 $\hat{y}^{(n)} = f(x^{(n)}; \theta)$。
        然后，我们使用一个**损失度量 (loss metric)** $e_n$ 来计算预测值 $\hat{y}^{(n)}$ 与真实值 $y_{true}^{(n)}$ 之间的差异。常见的损失度量有：
        *   **均方误差 (Mean Squared Error, MSE)**: $e_n = (\hat{y}^{(n)} - y_{true}^{(n)})^2$
        *   **交叉熵 (Cross-Entropy)** (常用于分类): 具体形式取决于任务。

    2.  **整个训练集的总损失/平均损失 (Total/Average Loss over the Training Set):**
        代价函数 $L(\theta)$ 通常是所有训练样本损失的**平均值**（或总和）。如果训练集有 $N_{total}$ 个样本：
        $$ L(\theta) = \frac{1}{N_{total}} \sum_{n=1}^{N_{total}} e_n = \frac{1}{N_{total}} \sum_{n=1}^{N_{total}} \text{loss-metric}(f(x^{(n)}; \theta), y_{true}^{(n)}) $$

*   **损失函数的具体例子**: (这里可以链接到你之前整理的更详细的损失函数列表，或者简要重述)
    *   回归问题: 均方误差 (MSE)
    *   二分类问题: 二元交叉熵 (Binary Cross-Entropy)
    *   多分类问题: 分类交叉熵 (Categorical Cross-Entropy)

*   **使用 Mini-batch 计算损失 (Calculating Loss using Mini-batches):**
    在实际训练深度神经网络时，由于训练数据集通常非常大，一次性计算整个数据集上的损失 $L(\theta)$ 可能会非常耗时且占用大量内存。因此，我们通常采用 **Mini-batch 梯度下降** [[Notion/Theoretical-Knowledge/Computer-Science/Artificial-Intelligence/深度学习 (Deep Learning)#2.1. Mini-batch 梯度下降 (Mini-batch Gradient Descent)\|深度学习 (Deep Learning)#2.1. Mini-batch 梯度下降 (Mini-batch Gradient Descent)]]。
    *   **Mini-batch Loss**: 在每次迭代中，我们从训练数据中抽取一小批样本 (a mini-batch)，例如包含 $B$ 个样本。然后，我们计算模型在这 $B$ 个样本上的**平均损失**，并将其作为对整个训练集损失 $L(\theta)$ 的一个估计或代理 (proxy)。
        $$ L_{batch}(\theta) = \frac{1}{B} \sum_{k=1}^{B} \text{loss-metric}(f(x_{batch}^{(k)}; \theta), y_{true,batch}^{(k)}) $$
        优化算法将尝试最小化这个 $L_{batch}(\theta)$。
    *   **Epoch**: 当算法处理完训练数据集中所有的 mini-batches，即对整个训练数据集完整地过了一遍之后，称为完成了一个 **epoch** (1 epoch = see all the batches once)。

*   **损失函数的角色 (The Role of the Loss Function):**
    损失函数不仅告诉我们当前这组参数 $\theta$ 的表现如何，更重要的是，它的梯度将指导我们如何调整这些参数以改进模型。

## 2.3 步骤三：参数优化 (Step 3: Optimization)

一旦我们定义了模型 $f(\mathbf{x}; \theta)$ 和损失函数 $L(\theta)$，我们的目标就是找到一组最优的参数 $\theta^*$，使得损失函数 $L(\theta)$ 的值最小。这个寻找最优参数的过程称为**优化 (Optimization)**。

数学上，我们可以表示为：
$$ \theta^* = \arg\min_{\theta} L(\theta) $$
这意味着我们要寻找使损失函数 $L$ 达到最小值的参数向量 $\theta$。

### 2.3.1 梯度下降 (Gradient Descent)

梯度下降是一种广泛应用于机器学习和深度学习中的迭代优化算法，用于寻找函数的最小值。其基本思想是沿着损失函数梯度下降最快的方向逐步调整参数。

**梯度下降的步骤:**

1.  **初始化参数 (Initialize Parameters)**:
    *   随机选择或根据某种策略设定参数向量 $\theta$ 的初始值，记为 $\theta^0$。(Randomly) Pick initial values $\theta^0$.

2.  **迭代更新 (Iteratively Update):** 重复以下操作直到满足停止条件：
    *   a. **计算梯度 (Compute Gradient)**:
        *   计算损失函数 $L(\theta)$ 在当前参数点 $\theta^t$ (其中 $t$ 是迭代次数) 关于参数向量 $\theta$ 中每一个分量 $\theta_k$ 的偏导数。这些偏导数共同构成了损失函数在 $\theta^t$ 处的**梯度向量 (gradient vector)** $\mathbf{g}$ (或 $\nabla L(\theta^t)$)。
            $$ \mathbf{g} = \nabla L(\theta^t) = \begin{pmatrix}
            \frac{\partial L}{\partial \theta_1} \\
            \frac{\partial L}{\partial \theta_2} \\
            \vdots \\
            \frac{\partial L}{\partial \theta_K}
            \end{pmatrix}_{\theta=\theta^t} $$
        *   在神经网络中，这个梯度通常通过**反向传播 (Backpropagation)** 算法高效计算。
        *   当使用 mini-batch 时，计算的是 $L_{batch}(\theta)$ 的梯度。
           ![Image/Computer-Science/Machine Learning/9.png](/img/user/Image/Computer-Science/Machine%20Learning/9.png) 
        * (图示：梯度指向函数增加最快的方向)

    *   b. **更新参数 (Update Parameters)**:
        *   根据梯度信息，沿着梯度的**反方向**更新参数，以减小损失函数的值。
            $$ \theta^{t+1} \leftarrow \theta^t - \eta \cdot \mathbf{g} $$
            或者写成：
            $$ \theta^{t+1} \leftarrow \theta^t - \eta \nabla L(\theta^t) $$
        *   其中：
            *   $\theta^{t+1}$ 是更新后的参数向量。
            *   $\theta^t$ 是当前迭代的参数向量。
            *   $\eta$ (eta) 是**学习率 (Learning Rate)**，它是一个正的小值（超参数），控制每次参数更新的“步长”或幅度。学习率的选择对训练过程至关重要。
            *   $\mathbf{g}$ (或 $\nabla L(\theta^t)$) 是在 $\theta^t$ 处计算得到的梯度向量。

3.  **停止条件 (Stopping Condition)**:
    *   可以设定最大迭代次数 (或最大 epoch 数)。
    *   可以监控损失函数的值，当其变化很小或不再下降时停止。
    *   可以监控在验证集上的性能，当验证集性能不再提升（甚至开始下降，表明过拟合）时停止（早停法 Early Stopping）。
![Image/Computer-Science/Machine Learning/10.png](/img/user/Image/Computer-Science/Machine%20Learning/10.png)
**注：**
- 一个**神经元 (Neurou)**是一个基本的计算单元。
- 一个**激活函数**（如 Sigmoid）是神经元计算过程中的一个关键组成部分，它引入非线性。
- 当我们说“一个 Sigmoid 神经元”或在图中画一个 Sigmoid 符号时，我们通常指的就是一个以 Sigmoid 作为其激活函数的神经元/单元。
- **这整个体系，即由相互连接的“神经元”（或单元，每个单元执行加权求和与非线性激活函数如 Sigmoid、ReLU 等操作）组成的、分层的结构，就叫做神经网络 (Neural Network)。**
- 如果这个网络包含多个隐藏层，它就是一个**深度神经网络 (Deep Neural Network, DNN)**
**对应关系是：**
- 一个 Sigmoid (或其他激活函数) 是**一个神经元计算的一部分**。
- **多个神经元**可以组成**一个隐藏层 (Hidden Layer)**。
- **多个隐藏层**构成了**一个深度神经网络**

**拓展：**
**1. 为什么我们要的是 "Deep" 的 network 而不是 "Fat" 的？**
[[Notion/Theoretical-Knowledge/Computer-Science/Artificial-Intelligence/Question/Deep 而非 Fat 的 Neural Network\|Deep 而非 Fat 的 Neural Network]]
**2. 为什么我们不 "Deeper" ?**
[[Notion/Theoretical-Knowledge/Computer-Science/Artificial-Intelligence/Question/不一直 Deeper 的神经网络\|不一直 Deeper 的神经网络]]

#### 梯度下降的变体：批量大小的选择 (Variants of Gradient Descent: The Choice of Batch Size)

在实践中，我们很少一次性使用整个训练集来计算梯度并更新参数。相反，我们会根据每次更新所使用的样本数量，将梯度下降分为三种主要类型。**批量大小 (Batch Size)** 的选择是一个重要的超参数，它在**计算效率**和**模型性能**之间做出了权衡。

##### 1. 批量梯度下降 (Batch Gradient Descent / Full Batch)

*   **定义**: 批量大小 (Batch size) = 训练样本总数 N。
*   **过程**: 在每次参数更新前，**计算整个训练数据集**上的损失函数，然后根据这个总损失的梯度来更新一次参数。
*   **更新频率**: 每个 Epoch 只更新一次。
    *   (对应图中左侧: "Update after seeing all the 20 examples")
*   **优点 (Pros)**:
    *   **强大/稳定 (Powerful)**: 梯度的计算非常准确，因为它考虑了所有数据。优化路径非常平滑，直接朝向最小值点前进。
    *   理论上保证收敛到凸函数（convex）的全局最小值或非凸函数（non-convex）的局部最小值。
*   **缺点 (Cons)**:
    *   **耗时长 (Long time for cooldown)**: 当数据集非常大时，计算一次梯度的成本极高，导致训练过程非常缓慢。
    *   **内存需求高**: 需要将整个数据集加载到内存中。
    *   可能会陷入比较“尖锐”的局部最小值中。

##### 2. 随机梯度下降 (Stochastic Gradient Descent, SGD)

*   **定义**: 批量大小 (Batch size) = 1。
*   **过程**: **每看到一个训练样本**，就计算该单个样本的损失和梯度，并立即更新一次参数。
*   **更新频率**: 如果有 N 个样本，每个 Epoch 会更新 N 次。
    *   (对应图中右侧: "Update for each example. Update 20 times in an epoch")
*   **优点 (Pros)**:
    *   **更新快 (Short time for cooldown)**: 参数更新非常频繁，训练速度快。
    *   **有助于跳出局部最优**: 梯度的“噪声”性质（因为单个样本的梯度可能不代表全局方向）有时反而是一种优势，可以帮助优化过程跳出较差的局部最小值或鞍点，具有一定的正则化效果。
*   **缺点 (Cons)**:
    *   **噪声大 (Noisy)**: 每次更新的方向非常不稳定，导致损失函数下降的路径非常曲折（如上图右侧的锯齿状路径），收敛过程波动很大。
    *   无法充分利用现代计算硬件（如GPU）的并行计算优势。

##### 3. 小批量梯度下降 (Mini-batch Gradient Descent)

*   **定义**: 批量大小 (Batch size) 是一个介于 1 和 N 之间的值（例如 32, 64, 128）。
*   **过程**: 每次从训练集中抽取一小批（a mini-batch）样本，计算这批样本的平均损失和梯度，并更新一次参数。
*   **这是现代深度学习中应用最广泛的方法。**
*   **优点 (Pros)**:
    *   **完美折衷**: 平衡了批量梯度下降和随机梯度下降的优缺点。
    *   **高效利用硬件**: 可以充分利用GPU的并行计算能力，处理速度远快于SGD。
    *   **稳定且快速**: 相比SGD，梯度估计更准确，收敛过程更稳定；相比Batch GD，更新速度快得多。
    *   保留了适度的噪声，有助于模型泛化和逃离坏的局部极值。

| 特性       | 批量梯度下降 (Batch GD) | 随机梯度下降 (SGD) | 小批量梯度下降 (Mini-batch GD)    |
| :------- | :---------------- | :----------- | :------------------------- |
| **批量大小** | 整个数据集 (N)         | 1            | 介于 1 和 N 之间 (e.g., 32, 64) |
| **更新速度** | 非常慢               | 非常快          | 快                          |
| **收敛路径** | 平滑、直接             | 噪声大、曲折       | 相对平滑，有小幅波动                 |
| **内存占用** | 非常高               | 非常低          | 中等                         |
| **优点**   | 梯度准确，稳定           | 更新快，能跳出局部最优  | **综合了两者的优点，是实践首选**         |
| **缺点**   | 速度慢，易陷于尖锐极值       | 噪声大，收敛不稳定    | 需要额外设定 batch size 超参数      |

### 2.3.2 优化进阶：泰勒展开与损失曲面近似 (Advanced Optimization: Taylor Expansion & Loss Surface Approximation)

在梯度下降中，我们利用损失函数的一阶导数（梯度）来确定下降方向。为了更深入地理解优化过程，特别是更高级的优化算法（如牛顿法），我们可以使用**泰勒级数 (Taylor Series)** 来近似损失函数。

泰勒展开的核心思想是，在任意一个参数点 $\theta'$ 附近，我们可以用一个更简单的多项式函数来近似复杂的损失函数 $L(\theta)$。对于优化而言，通常使用到二阶的泰勒展开就足够了。

![Image/Computer-Science/Machine Learning/11.png](/img/user/Image/Computer-Science/Machine%20Learning/11.png)

如上图所示，在点 $\theta'$ 附近的二阶泰勒展开为：
$$ L(\theta) \approx L(\theta') + (\theta - \theta')^T g + \frac{1}{2} (\theta - \theta')^T H (\theta - \theta') $$

*   **$L(\theta')$**: 当前点 $\theta'$ 的损失值，这是一个常数。
*   **$(\theta - \theta')^T g$**: **一阶项 (First-order term)**。
    *   $\mathbf{g} = \nabla L(\theta')$ 是损失函数在点 $\theta'$ 的**梯度 (Gradient)**。
    *   这一项用一个线性函数来近似损失函数的变化。**梯度下降法只依赖于这一项**，它告诉我们在 $\theta'$ 附近，沿着负梯度 $-\mathbf{g}$ 的方向移动，损失值 $L(\theta)$ 会下降得最快。图中的绿色框和绿色直角三角形的高度就直观地表示了这一线性近似。
*   **$\frac{1}{2} (\theta - \theta')^T H (\theta - \theta')$**: **二阶项 (Second-order term)**。
    *   $H$ 是损失函数在点 $\theta'$ 的**海森矩阵 (Hessian Matrix)**，即损失函数的二阶偏导数矩阵 ($H_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}$)。
    *   这一项引入了关于损失曲面**曲率 (curvature)** 的信息。它用一个二次函数来更精确地描述损失曲面的形状（是“平坦”的碗还是“陡峭”的碗）。

#### 临界点分析 (Analysis at Critical Points)

当梯度下降进行到某一步时，如果梯度 $\mathbf{g} = \nabla L(\theta') = 0$，我们称 $\theta'$ 为一个**临界点 (Critical Point)**。在临界点，梯度下降会停止更新，因为它找不到下降的方向。此时，泰勒展开的一阶项为零，损失函数的行为完全由二阶项决定：
$$ L(\theta) \approx L(\theta') + \frac{1}{2} (\theta - \theta')^T H (\theta - \theta') $$

令 $\mathbf{v} = (\theta - \theta')$，我们可以通过海森矩阵 $H$ 的性质来判断这个临界点的类型：

*   如果对于所有非零向量 $\mathbf{v}$，都有 $\mathbf{v}^T H \mathbf{v} > 0$（即 $H$ 是正定矩阵），那么 $L(\theta) > L(\theta')$。这意味着 $\theta'$ 是一个**局部最小值点 (Local Minima)**。
*   如果对于所有非零向量 $\mathbf{v}$，都有 $\mathbf{v}^T H \mathbf{v} < 0$（即 $H$ 是负定矩阵），那么 $L(\theta) < L(\theta')$。这意味着 $\theta'$ 是一个**局部最大值点 (Local Maxima)**。
*   如果对于某些向量 $\mathbf{v}$，$\mathbf{v}^T H \mathbf{v} > 0$，而对于另一些向量 $\mathbf{v}$，$\mathbf{v}^T H \mathbf{v} < 0$（即 $H$ 是不定矩阵），那么 $\theta'$ 是一个**鞍点 (Saddle Point)**。

#### 如何逃离鞍点 (Don't be afraid of Saddle Point!)

![Image/Computer-Science/Machine Learning/12.png](/img/user/Image/Computer-Science/Machine%20Learning/12.png)

在高维空间中（例如深度神经网络的参数空间），鞍点远比局部最小值点更常见。那么，当梯度下降卡在鞍点时（因为梯度为零），我们该怎么办呢？**海森矩阵 $H$ 告诉了我们逃离的方向！**

*   **核心思想**: 在鞍点处，海森矩阵 $H$ 必然存在至少一个**负特征值 (negative eigenvalue)**。
*   **逃离步骤**:
    1.  假设我们当前在鞍点 $\theta'$，梯度为零。
    2.  我们计算海森矩阵 $H$ 的特征值和特征向量。
    3.  找到一个负特征值 $\lambda < 0$ 以及其对应的特征向量 $\mathbf{u}$。
    4.  根据特征向量的定义，我们有 $H\mathbf{u} = \lambda\mathbf{u}$。
    5.  现在，我们考虑沿着特征向量 $\mathbf{u}$ 的方向更新参数，即令更新步长 $\theta - \theta' = \mathbf{u}$。
    6.  代入泰勒展开式，计算新的损失值与当前损失值的差异：
        $$ L(\theta) - L(\theta') \approx \frac{1}{2} (\theta - \theta')^T H (\theta - \theta') = \frac{1}{2} \mathbf{u}^T H \mathbf{u} $$
        利用 $H\mathbf{u} = \lambda\mathbf{u}$，我们得到：
        $$ \frac{1}{2} \mathbf{u}^T (H \mathbf{u}) = \frac{1}{2} \mathbf{u}^T (\lambda \mathbf{u}) = \frac{1}{2} \lambda (\mathbf{u}^T \mathbf{u}) = \frac{1}{2} \lambda ||\mathbf{u}||^2 $$
    7.  因为 $\lambda < 0$ 且 $||\mathbf{u}||^2 > 0$，所以 $\frac{1}{2} \lambda ||\mathbf{u}||^2 < 0$。
    8.  这意味着 $L(\theta) - L(\theta') < 0$，即 $L(\theta) < L(\theta')$。

*   **结论**: 只要我们沿着海森矩阵**负特征值对应的特征向量方向**更新参数（例如，$\theta = \theta' + \alpha \mathbf{u}$，其中 $\alpha$ 是一个小的步长），我们就能有效地降低损失值，从而成功**逃离鞍点**。

**实际应用中的意义:**

虽然在大型神经网络中显式地计算整个海森矩阵及其特征向量的成本极高，但这个理论非常重要：
1.  它解释了为什么鞍点在理论上不是一个根本性的障碍。总有“下山”的路可走。
2.  许多先进的优化算法，如带动量的SGD、Adam等，虽然没有直接计算海森矩阵，但它们引入的机制（如动量）在实践中能帮助模型“冲过”平坦区域和鞍点。
3.  一些二阶优化算法的变体（如Hessian-Free优化）会尝试用更高效的方法来近似海森矩阵与某个向量的乘积（即 $H\mathbf{v}$），从而利用曲率信息来逃离鞍点和加速收敛。

**与优化的关系:**

1.  **梯度下降 (Gradient Descent)**: 只考虑一阶信息（梯度 `g`），可以看作是在一个线性的近似下寻找下降方向。它简单高效，但不知道“走多远”最合适（需要手动设置学习率 $\eta$），也无法很好地处理不同方向上曲率差异很大的情况（例如狭长的山谷）。

2.  **牛顿法 (Newton's Method)**: 同时考虑一阶（梯度 `g`）和二阶信息（海森矩阵 `H`）。它通过找到上述二次近似函数的最小值点来确定下一步的更新方向和步长。更新规则为 $\theta^{t+1} = \theta^t - H^{-1}g$。
    *   **优点**: 收敛速度通常比梯度下降快得多，因为它利用了曲率信息，能够更直接地跳向最小值点。
    *   **缺点**: 计算和存储海森矩阵 $H$ 以及其逆矩阵 $H^{-1}$ 的开销巨大。对于有数百万参数的深度神经网络来说，这是不现实的。

因此，在深度学习中，虽然我们不直接使用标准的牛顿法，但许多先进的优化器（如 Adam、RMSProp 等）都受到了“利用曲率信息来调整步长”这一思想的启发，它们通过各种方式来近似海森矩阵的信息，以实现比朴素梯度下降更快的收敛。泰勒展开为理解这些高级优化算法提供了坚实的理论基础。

#### “最小比率”：一个用于分析损失曲面几何的指标 (The "Minimum Ratio": A Metric for Analyzing Loss Surface Geometry)

**重要澄清**: 此处讨论的 "Minimum ratio" 是一个用于分析海森矩阵特性的指标，与线性规划单纯形法 (Simplex Method) 中的 "Minimum Ratio Test" **完全无关**。

在深度学习的优化研究中，研究者们有时会使用一个比率来量化损失函数在某个特定点 $\theta$ 附近的几何形状，特别是判断它有多像一个真正的“谷底”（局部最小值）。你提供的图片中就定义了这样一个比率：

$$ \text{Minimum ratio} = \frac{\text{Number of Positive Eigenvalues}}{\text{Total Number of Eigenvalues}} $$

**这个指标的意义是什么？**

这个比率衡量了在当前点，损失曲面在所有主轴方向中，有多少个方向是“向上弯曲”的（即具有正曲率）。

*   **分母 (Total Number of Eigenvalues)**: 等于模型参数的总数量。它代表了参数空间的总维度。
*   **分子 (Number of Positive Eigenvalues)**: 代表了海森矩阵 `H` 的正特征值的数量。我们知道，每个正特征值对应一个向上弯曲的（类似山谷的）方向。

**如何解读这个比率的值？**

1.  **当 Ratio ≈ 1.0 时**:
    *   这意味着几乎所有的特征值都是正的。
    *   海森矩阵 `H` 近似于一个**正定矩阵**。
    *   当前点 $\theta$ 的几何形状非常像一个**真正的局部最小值**，即一个在所有方向上都向上弯曲的“碗”或“盆地”。在优化过程中，我们希望找到的就是这样的点。

2.  **当 Ratio ≈ 0.5 时**:
    *   这意味着大约一半的特征值是正的，另一半是负的。
    *   这是**鞍点 (Saddle Point)** 的一个典型特征。损失曲面在很多方向上向上弯，但在同样多的方向上向下弯。

3.  **当 Ratio ≈ 0.0 时**:
    *   这意味着几乎所有的特征值都是负的。
    *   海森矩阵 `H` 近似于一个**负定矩阵**。
    *   当前点 $\theta$ 的几何形状非常像一个**局部最大值**，即一个在所有方向上都向下弯曲的“山峰”。

**为什么叫 "Minimum ratio"？**

这个名字可能有些令人困惑。一个可能的解释是，这个比率是用来**分析和表征损失函数的最小值点 (loss minimum)** 的。一个“好的”最小值点，其“Minimum ratio”应该接近1。因此，这个比率可以看作是衡量一个临界点“有多好”或者“有多像一个真正的最小值”的指标。在文献中，你可能也会看到它被称为**正特征值比率 (Positive Eigenvalue Ratio)** 或**局部凸性比率 (Local Convexity Ratio)**，这些名字可能更具描述性。

在训练过程中，模型参数从一个随机初始点开始，可能会经过许多“Minimum ratio”较低的区域（鞍点），最终收敛到一个“Minimum ratio”接近1的区域（一个好的局部最小值）。