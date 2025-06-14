---
{"dg-publish":true,"permalink":"/notion/theoretical-knowledge/computer-science/artificial-intelligence/machine-learning/"}
---

# 1. 机器学习基本概念 (Basic Concepts)

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

---
## 2. 深度学习基本概念 (Basic Concepts of Deep Learning)

## 2.1 引入：从线性模型到更灵活的模型 (Introduction: From Linear Models to More Flexible Models)

### 2.1.1 线性模型的局限性 (Model Bias of Linear Models)

线性模型，如 $y = b + wx_1$ 或 $y = b + \sum_j w_j x_j$，在许多情况下非常有用且易于解释。然而，它们具有固有的**模型偏差 (Model Bias)**，这意味着它们只能表示输入和输出之间的线性关系。

*   **局限性 (Limitation):** 如果数据中的真实关系是非线性的，线性模型将无法很好地拟合数据，导致预测性能不佳。
    *   例如，在你的图示中（第一张图，红色曲线），如果 $y$ 和 $x_1$ 之间的关系是如图所示的折线，那么单一的直线（线性模型）无法准确地捕捉这种模式。

### 2.1.2 构建更灵活的模型：分段线性函数 (Building More Flexible Models: Piecewise Linear Functions)

为了克服线性模型的局限性，我们需要能够表示非线性关系的更灵活的模型。一种方法是使用**分段线性函数 (Piecewise Linear Curves)**。

*   **基本思想 (Core Idea):** 任何复杂的分段线性曲线（如图中的红色曲线）可以被看作是一个**常数 (constant)** 加上一系列更简单的“阶梯状”或“斜坡状”基础函数（如图中右上角示意的小蓝色函数）的**加权和 (sum of a set of blue functions)**。![5.png](/img/user/Image/Machine%20Learning/5.png)![6.png](/img/user/Image/Machine%20Learning/6.png)
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

    *   ![7.png](/img/user/Image/Machine%20Learning/7.png)
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