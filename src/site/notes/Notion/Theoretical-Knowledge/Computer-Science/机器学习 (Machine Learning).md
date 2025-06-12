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

**(与深度学习的关系 Relationship to Deep Learning):**

这个基本思想——**定义一个带有未知参数的函数，并通过数据来学习这些参数——是机器学习的核心，并且在深度学习中得到了极大的扩展和深化。**

*   **相似性 (Similarity):**
    *   **参数化模型 (Parameterized Model):** 无论是简单的线性回归还是复杂的深度神经网络，它们都是参数化的模型。它们都包含需要从数据中学习的权重和偏置（或其他类型的参数）。
    *   **学习过程 (Learning Process):** 学习这些参数的过程通常都涉及定义一个损失函数 (loss function) 来衡量模型预测与真实值之间的差异，然后使用优化算法 (optimization algorithm, 如梯度下降) 来调整参数，以最小化这个损失。

*   **深度学习的扩展 (How Deep Learning Extends This):**
    *   **模型复杂度与层级结构 (Model Complexity and Hierarchy):** 深度学习模型（如神经网络）通常比上述简单线性模型复杂得多。它们由多个层 (layers) 组成，每一层都包含许多神经元和相应的参数。这种层级结构使得模型能够学习数据中更复杂、更抽象的特征表示。
    *   **非线性 (Non-linearity):** 深度学习模型通过引入非线性激活函数 (non-linear activation functions)，能够学习输入和输出之间高度非线性的关系，而不仅仅是线性关系。
    *   **参数数量 (Number of Parameters):** 深度学习模型通常拥有数百万甚至数十亿的参数，而简单的线性模型参数数量很少。

**总结来说，将模型看作一个“带有未知参数的函数”是理解许多机器学习方法（包括深度学习）的基石。深度学习可以被视为这种思想的一种更强大、更灵活、更具表现力的实现，它通过构建深层、复杂的参数化函数来解决更具挑战性的问题。**

