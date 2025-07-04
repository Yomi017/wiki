---
{"dg-publish":true,"permalink":"/notion/","tags":["gardenEntry"]}
---

这是一个基于Obsidian以及Vercel的个人学习知识库

### 1. 梯度下降关心的是什么？

首先，梯度下降算法本身并不关心损失函数是什么形式。它的工作流程非常“机械”：
1.  给我一个可微的损失函数 $L(\theta)$。
2.  我计算出 $L(\theta)$ 关于参数 $\theta$ 的梯度 $\nabla L(\theta)$。
3.  我沿着负梯度方向更新 $\theta$。

所以，无论是**均方误差 (Mean Squared Error, MSE)** 还是**负对数似然 (Negative Log-Likelihood, NLL)**，只要它们是可微的，梯度下降都能用它们来优化模型。

### 2. 均方误差 vs. 负对数似然

我们来仔细看看这两个损失函数：

#### 均方误差 (MSE)

$$ L_{\text{MSE}}(\theta) = \frac{1}{N} \sum_{n=1}^{N} (y_n - f(\mathbf{x}_n, \theta))^2 $$
*   $f(\mathbf{x}_n, \theta)$ 是模型对输入 $\mathbf{x}_n$ 的预测值。
*   $(y_n - f(\mathbf{x}_n, \theta))$ 是预测误差（残差）。
*   **物理意义**: 它直接衡量了**预测值与真实值之间的几何距离（的平方）**。你说得对，它的“量纲”是目标值 $y$ 量纲的平方。这非常直观。

#### 负对数似然 (NLL)

$$ L_{\text{NLL}}(\theta) = -\sum_{n=1}^{N} \log p(y_n | \mathbf{x}_n, \theta) $$
*   **物理意义**: 它衡量的是**模型对观测数据出现（被预测出来）的概率的“惊讶程度”**。如果模型认为观测到的真实数据 $y_n$ 出现的概率很低（$p$很小），那么 $-\log p$ 就会很大，损失就很大。它的“量纲”是信息论中的“比特”或“奈特”，是无量纲的。

### 3. 两者何时等价？【关键点】

现在到了最关键的部分。**为什么看起来完全不同的两个东西，都能用来做回归任务？**

答案是：**当我们假设模型的预测误差服从高斯分布（正态分布）时，最小化均方误差就等价于最大化对数似然！**

让我们来推导一下：

1.  **做出假设**: 我们假设对于一个给定的输入 $\mathbf{x}_n$，模型的预测值 $f(\mathbf{x}_n, \theta)$ 是真实值 $y_n$ 加上一个服从高斯分布的噪声 $\epsilon$。
    $$ y_n = f(\mathbf{x}_n, \theta) + \epsilon $$
    其中，$\epsilon \sim \mathcal{N}(0, \sigma^2)$，即噪声的均值为0，方差为 $\sigma^2$。

2.  **写出似然函数**: 根据这个假设，给定 $\mathbf{x}_n$ 和 $\theta$， $y_n$ 的条件概率分布就是一个高斯分布，其均值为模型的预测值 $f(\mathbf{x}_n, \theta)$：
    $$ p(y_n | \mathbf{x}_n, \theta) = \mathcal{N}(y_n | f(\mathbf{x}_n, \theta), \sigma^2) $$
    根据高斯分布的概率密度函数公式：
    $$ p(y_n | \mathbf{x}_n, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_n - f(\mathbf{x}_n, \theta))^2}{2\sigma^2}\right) $$

3.  **计算负对数似然 (NLL)**:
    $$ L_n(\theta) = -\log p(y_n | \mathbf{x}_n, \theta) $$
    $$ = -\log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_n - f(\mathbf{x}_n, \theta))^2}{2\sigma^2}\right) \right) $$
    利用对数性质 $\log(ab) = \log a + \log b$ 和 $\log(\exp(x)) = x$：
    $$ = -\left[ \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) - \frac{(y_n - f(\mathbf{x}_n, \theta))^2}{2\sigma^2} \right] $$
    $$ = \frac{1}{2\sigma^2}(y_n - f(\mathbf{x}_n, \theta))^2 + \log(\sqrt{2\pi\sigma^2}) $$

4.  **计算总损失**:
    $$ L_{\text{NLL}}(\theta) = \sum_{n=1}^{N} L_n(\theta) = \sum_{n=1}^{N} \left[ \frac{1}{2\sigma^2}(y_n - f(\mathbf{x}_n, \theta))^2 + \log(\sqrt{2\pi\sigma^2}) \right] $$
    $$ = \frac{1}{2\sigma^2} \sum_{n=1}^{N} (y_n - f(\mathbf{x}_n, \theta))^2 + N \log(\sqrt{2\pi\sigma^2}) $$

5.  **进行优化**: 我们的目标是找到使 $L_{\text{NLL}}(\theta)$ 最小的参数 $\theta$。观察上式，第二项 $N \log(\sqrt{2\pi\sigma^2})$ 是一个与 $\theta$ 无关的常数，在求导和优化过程中可以忽略。第一项前面的系数 $\frac{1}{2\sigma^2}$ 也是一个正的常数。
    因此，**最小化 $L_{\text{NLL}}(\theta)$ 就等价于最小化**：
    $$ \sum_{n=1}^{N} (y_n - f(\mathbf{x}_n, \theta))^2 $$
    这正是**平方误差（或均方误差，只差一个常数因子 $1/N$）**！

### 结论

*   **是的，梯度下降时，损失函数可以是负对数似然形式。** 这在处理概率模型时非常自然和普遍（例如，分类问题中的交叉熵损失就是NLL的一种）。
*   **“量纲”问题是个好问题，但它不影响优化。** 梯度下降只关心损失函数对参数的相对变化率（梯度），而不在乎损失函数的绝对值和单位。
*   **两者是统一的。** 均方误差损失可以被看作是**负对数似然损失在“误差服从高斯分布”这个特定假设下的一个特例**。
*   **选择哪个？**
    *   **均方误差**：更直观，从几何距离出发。
    *   **负对数似然**：更具普遍性，从概率和信息论出发。它不仅能推导出均方误差，还能在不同假设下推导出其他损失函数（如分类问题的交叉熵损失，当假设服从伯努利分布时）。这为我们提供了一个统一的框架来理解各种损失函数。

所以，你的疑惑非常有价值，它揭示了机器学习中不同方法背后深层的统计学联系。