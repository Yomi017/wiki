---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/machine-learning/machine-learning/"}
---

# Lecture 1: Logistics and introduction

## Definition:
A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in **T**, as measured by **P**, improves with **experience E**.
*   **Experience E (data)**: games played by the program or human
*   **Performance measure P**: winning rate
*   **Task T**: to win

## Taxonomy of Machine Learning (A Simplistic View)

### 1. Supervised Learning
*   **Core idea**: Learns from **labeled data**.
*   **Example Tasks**:
    *   **Regression**: The prediction result is a **continuous** variable.
        *   e.g., price prediction
        *   (x = Area) → (y = Price)?
    *   **Classification**: The prediction result is a **discrete** variable.
        *   e.g., type prediction
        *   (x = Area, y = Price) → (z = Type)?

### 2. Unsupervised Learning
*   **Core idea**: Learns from **unlabeled data**.
*   **Example Tasks**:
    *   **Clustering**:
        *   Given a dataset containing *n* samples:
            (x⁽¹⁾, y⁽¹⁾), (x⁽²⁾, y⁽²⁾), (x⁽³⁾, y⁽³⁾), ..., (x⁽ⁿ⁾, y⁽ⁿ⁾)
        *   **Task (vague)**: find interesting structures in the data.

### 3. Semi-supervised Learning
*   **Core idea**: Learns from a mix of labeled and unlabeled data.

### 4. Reinforcement Learning
*   **Core idea**: Learns from **environment feedback** (rewards/penalties).
*   **Example Task**: Multi-armed bandit problem (MAB)
    *   it involves a feedback loop between an **Agent** and an **Environment**:
        1.  The **Agent** takes an **Action** (e.g., pull an arm).
        2.  The **Environment** returns **Feedback** (e.g., a corresponding reward).
        3.  The agent learns from this feedback to make better decisions in the future.

### Learning Modes (When do we collect data?)
*   **Offline learning**: The model is trained on a static dataset before deployment.
*   **Online learning**: The model is trained incrementally as new data becomes available.

## Mathematical Tools

[[Wiki/Notion/Class/Artificial-Intelligence/Mathematics for AI/Mathematics for AI\|Mathematics for Ai]]

[[Wiki/Notion/Class/Artificial-Intelligence/Mathematics for AI/Mathematics for AI (Chinese)\|Mathematics for Ai (Chinese)]]

### Parameter Estimation: Maximum Likelihood Estimation (MLE)
A foundational method for estimating the parameters of a statistical model from data.

*   **Core Principle**: Find the parameter values ($θ$) that maximize the **likelihood function**. in other words, we find the parameters that make the observed data most probable.
    $\hat{θ_{ML}} = \displaystyle\arg \max_θ P(Data | θ)$

*   **Example: MLE for Bernoulli Distribution**
    *   **Scenario**: We have a dataset $D = {X₁, ..., Xₘ}$ from a Bernoulli distribution (e.g., $m$ coin flips), where $X_i$ is 1 (heads) or 0 (tails).
    *   **Goal**: Estimate the probability of heads, $θ = P(X=1)$.
    *   **Derivation Steps**:
        1.  **Likelihood Function**: Assuming data points are independent and identically distributed (i.i.d.), the likelihood of observing the entire dataset is the product of individual probabilities:
            $L(θ) = \prod_i p(X_i; θ)$
        2.  **Log-Likelihood Function**: To simplify calculation (turning products into sums) and for numerical stability, we maximize the log-likelihood, which is equivalent.
            $\log L(θ) = \sum_i \log p(X_i; θ)$
        3.  **Substitute Bernoulli PMF**: The probability mass function for a single $Xi$ can be written as $p(X_i; θ) = θ{(X_i)} * (1-θ){(1-X_i)}$. its logarithm is $\log p(Xi; θ) = Xi \log(θ) + (1-X_i) \log(1-θ)$. The log-likelihood becomes:
            $\log L(θ) = \sum_i [X_i log(θ) + (1-X_i) log(1-θ)]$
        4.  **Simplify**: Let $M₁ = \sum_i X_i$ be the total count of 1s (heads). The total count of 0s is $m - M₁$. The expression simplifies to:
            $\log L(θ) = M₁ \log(θ) + (m - M₁) \log(1-θ)$
    *   **Result**: To find the maximum, we take the derivative of this final expression with respect to $θ$, set it to 0, and solve. The resulting estimate is highly intuitive:
        **$\hat{θ_{ML}} = \dfrac {M_1} m$** (The proportion of 1s in the data)

# Lecture 3: Supervised Learning: Regression and Classification II

### **Ridge Regression: Study Notes**

#### **1. Core Idea & Philosophy**

Ridge Regression is an enhancement of Ordinary Least Squares (OLS). Its core idea is to address the problem of **overfitting** by **sacrificing a small amount of bias to achieve a significant reduction in model variance**.

**Analogy:**
*   **OLS** is like a "novice detective" who tries to create a complex theory that perfectly explains 100% of the current evidence. This theory is often fragile and performs poorly on new cases (high variance).
*   **Ridge Regression** is like a "veteran detective" who knows that evidence contains noise and coincidences. He seeks a **simpler, more general theory** that might not explain every tiny detail perfectly but is more robust and predictive for new cases (low variance).

---

#### **2. Motivation: Why is Ridge Regression Necessary?**

Ridge Regression is primarily designed to solve critical failures of OLS that occur in a specific, common scenario.

**The Problem Scenario: High-Dimension, Low-Sample Data (d >> m)**
*   The number of **features $d$** is much larger than the number of **samples $m$**.
*   This is common in modern datasets like genetics, finance, and text analysis.

**Problems for OLS in this Scenario:**

**A. Conceptual Level: Severe Overfitting**
*   With more features than samples, the model has enough flexibility to "memorize" the training data, including its noise and random fluctuations.
*   This results in learned model weights ($w$) that are **absurdly large**, assigning huge importance to potentially irrelevant features.
*   The model performs perfectly on training data but fails to generalize to unseen test data.

**B. Mathematical Level: The Matrix $XᵀX$ is Non-Invertible (Singular)**
*   The analytical solution for OLS is: $w* = (XᵀX)⁻¹Xᵀy$.
*   This solution requires the matrix $XᵀX$ to be invertible.
*   **$XᵀX$ is often non-invertible or ill-conditioned (nearly non-invertible) in two cases:**
    1.  **When $m < d+1$ (Fewer Samples than Features):** This is the main reason. By linear algebra, $rank(XᵀX) = rank(X) ≤ min(m, d+1)$. If $m < d+1$, the rank of $XᵀX$ is less than its dimension ($d+1$), meaning it is not full rank and is therefore **guaranteed to be singular (non-invertible)**.
    2.  **Multicollinearity:** When features are highly correlated (e.g., including both "house size in sq. meters" and "house size in sq. feet"). This makes the columns of $X$ linearly dependent, which in turn makes $XᵀX$ singular.

---

#### **3. The Solution: How Ridge Regression Works**

Ridge Regression introduces a **penalty term** into the objective function to constrain the size of the model's weights.

**Step 1: Modify the Objective Function**

*   **Ordinary Least Squares (OLS) Objective:**
    *   Minimize the **Sum of Squared Errors (SSE)**
    *   $J_{OLS}(w) = ||y - Xw||₂²$

*   **Ridge Regression Objective:**
    *   Minimize **SSE + λ * Penalty on Model Complexity**
    *   $J_{Ridge}(w) = ||y - Xw||₂² + λ||w||₂²$

*   **Dissecting the Penalty Term:**
    *   $||w||₂² = Σ(wᵢ)²$: This is the squared **L2-Norm** of the weight vector. It is the sum of the squares of all weights. A large value implies a complex model with large weights.
    *   **$λ$ (Lambda)**: The **regularization parameter**. This is a hyperparameter that we set to control the strength of the penalty.
        *   **Large $λ$**: Strong penalty. The model is forced to shrink the weights towards zero to avoid a large penalty.
        *   **Small $λ$**: Weak penalty. The model behaves more like OLS.
        *   **$λ = 0$**: No penalty. Ridge Regression becomes identical to OLS.

**Step 2: Derive the New Analytical Solution**

By taking the derivative of the new objective function with respect to $w$ and setting it to zero, we get the analytical solution for Ridge Regression:

> $w^*_{ridge} = (XᵀX + λI)⁻¹ Xᵀy$

*   $I$ is the **Identity Matrix**.
*   Compared to the OLS solution, the only difference is the addition of the $+ λI$ term. This single term is what solves the non-invertibility problem.

---

#### **4. The Core Insight: Why $+ λI$ is the "Special Medicine"**

This term works by altering the **eigenvalues** of the matrix to guarantee its invertibility.

1.  **Invertibility and Eigenvalues**: A matrix is **singular (non-invertible)** if and only if it has **at least one eigenvalue that is 0**.

2.  **Eigenvalues of $XᵀX$**: $XᵀX$ is a positive semi-definite matrix, which means all its eigenvalues $μᵢ$ are **non-negative ($μᵢ ≥ 0$)**. When it's singular, it has at least one eigenvalue $μᵢ = 0$.

3.  **The Effect of $+ λI$**: When we add $λI$ to $XᵀX$, the eigenvalues of the new matrix $(XᵀX + λI)$ become $(μᵢ + λ)$.

4.  **The Result**:
    *   We choose $λ$ to be a **small positive number ($λ > 0$)**.
    *   The original eigenvalues were $μᵢ ≥ 0$.
    *   The new eigenvalues are $μᵢ + λ$.
    *   Therefore, **all new eigenvalues are strictly positive ($> 0$)**.
    *   A matrix whose eigenvalues are all greater than zero is **guaranteed to be invertible**.

**Conclusion:** The $λI$ term acts as a "stabilizer" by shifting all eigenvalues of $XᵀX$ up by a positive amount $λ$, ensuring that none are zero and thus making the matrix $(XᵀX + λI)$ invertible.

---

#### **5. Summary & Comparison**

| Aspect | Ordinary Least Squares (OLS) | Ridge Regression |
| :--- | :--- | :--- |
| **Objective Function** | Minimize SSE | Minimize [SSE + λ * L2-Norm of weights] |
| **Model Complexity** | Unconstrained, weights can be very large | Constrained by L2 penalty, forcing weights to be smaller |
| **Handling $d > m$** | $XᵀX$ is singular; no stable solution exists | $(XᵀX + λI)$ is always invertible; provides a stable solution |
| **Analytical Solution** | $(XᵀX)⁻¹Xᵀy$ | $(XᵀX + λI)⁻¹Xᵀy$ |
| **Key Property** | Unbiased estimate, but can have very high variance | Biased estimate, but with significantly lower variance |
| **Best Use Case** | Low-dimensional data, no multicollinearity | High-dimensional data (esp. $d>m$), multicollinearity is present |

---

#### **6. How to Choose $λ$?**

$λ$ is a critical hyperparameter that controls the trade-off between bias and variance. It is not learned from the training data. The optimal value of $λ$ is typically found using **Cross-Validation**, a technique that evaluates the model's performance on unseen data for different $λ$ values and selects the one that generalizes best.


# Lecture 5: Supervised Learning: Regression and Classification IV

### **主题一：梯度下降 (Gradient Descent - GD)**

梯度下降是一种基础且强大的迭代优化算法，其目标是找到一个函数（通常是成本或损失函数）的局部最小值。

*   **核心思想**:
    想象一下你站在一座山上，想要最快地走到山谷。最直接的方法就是沿着当前位置最陡峭的下坡方向走一小步，然后重复这个过程。在数学中，函数在某一点的梯度（`∇`）指向该点函数值增长最快的方向。因此，梯度的反方向（负梯度）就是函数值下降最快的方向。梯度下降算法正是利用了这一原理。

*   **算法流程**:
    算法从一个随机的参数点 $$ w_0 $$ 开始，然后通过多次迭代来更新参数 $$ w $$，每一次迭代都沿着负梯度的方向移动一小步。

*   **核心更新公式**:
    $$
    w_{k+1} \leftarrow w_k - \eta \nabla_w C(w_k)
    $$
    *   $$ w_k $$: 参数在第 $$ k $$ 次迭代时的值。
    *   $$ w_{k+1} $$: 参数在下一次迭代时的更新值。
    *   $$ \nabla_w C(w_k) $$: 成本函数 $$ C(w) $$ 在点 $$ w_k $$ 处的梯度。它是一个向量，指明了函数值增长最快的方向。
    *   $$ \eta $$ (eta): 学习率（Learning Rate），也叫步长。它是一个超参数，控制着每一步更新的幅度。如果 $$ \eta $$ 太小，收敛会非常慢；如果 $$ \eta $$ 太大，可能会在最小值附近震荡甚至发散，无法收敛。

*   **梯度下降的变体 (Variations of GD)**:
    为了解决标准梯度下降的一些问题（如收敛慢、学习率难以选择），研究者们提出了多种变体。
    1.  **Adagrad (Adaptive Gradient Algorithm)**:
        Adagrad 能够为不同的参数自适应地调整学习率。它对更新频繁的参数使用较小的学习率，对更新不频繁的参数使用较大的学习率。
        **更新公式**:
        $$
        w_{k+1} \leftarrow w_k - \eta(M_k + \epsilon I)^{-1} \nabla_w C(w_k)
        $$
        其中，$$ M_k $$ 是一个对角矩阵，其对角线上的第 $$ i $$ 个元素 $$ M_{ii,k} $$ 是历史梯度在该维度上的平方和：
        $$
        M_{ii,k} = \sum_{s=0}^{k} [\nabla_w C(w_s)]_i^2
        $$
        $$ \epsilon $$ 是一个很小的常数，用于防止分母为零。
    2.  **动量法 (Momentum-based GD)**:
        该方法引入了一个“动量”项，模拟物理学中的惯性。它累积了过去梯度的方向，使得更新方向不仅取决于当前梯度，还受历史方向的影响，有助于加速收敛并冲出局部极小值。
        **更新公式**:
        $$
        v_k \leftarrow \beta v_{k-1} + (1-\beta) \nabla_w C(w_k)
        $$
        $$
        w_{k+1} \leftarrow w_k - \eta v_k
        $$
        *   $$ v_k $$: 第 $$ k $$ 步的动量项，是过去梯度的指数移动平均。
        *   $$ \beta $$: 动量因子，通常取值接近 1（如 0.9）。
    3.  **Nesterov 加速梯度 (NAG)**:
        NAG 是动量法的一种改进，它通过“预见性”来优化更新。它首先根据之前的动量预估一个未来的位置，然后在这个预估位置计算梯度来进行修正。
        **更新公式**:
        $$
        v_t \leftarrow \beta v_{t-1} + \eta \nabla_w C(w_k - \beta v_{t-1})
        $$
        $$
        w_{k+1} \leftarrow w_k - v_k
        $$
        关键区别在于，梯度 $$ \nabla_w C(\cdot) $$ 是在“预估的未来位置” $$ (w_k - \beta v_{t-1}) $$ 上计算的，而不是当前位置 $$ w_k $$。

---

### **主题二：支持向量机 (Support Vector Machine - SVM)**

SVM 是一种强大的二分类模型，其目标是找到一个**最大间隔超平面 (Maximum Margin Hyperplane)**，该超平面能以最大的“缓冲区”将两类数据点分开。

*   **间隔 (Margin)**:
    1.  **函数间隔 (Functional Margin)**: 对于一个样本 $$(\mathbf{x}_t, y_t)$$, 其函数间隔定义为：
        $$
        \text{Functional Margin} = y_t(\theta^\top \mathbf{x}_t + \theta_0)
        $$
        它的大小不仅反映了分类的正确性（正值表示正确），还反映了分类的确信度。但它的一个缺点是，如果我们把 $$ \theta $$ 和 $$ \theta_0 $$ 同时放大，函数间隔也会成比例放大，但决策边界并未改变。
    2.  **几何间隔 (Geometric Margin)**: 为了消除上述缩放效应，我们引入几何间隔，它表示数据点到超平面的真实欧几里得距离。
        $$
        \gamma_{\text{geom}} = \frac{y_t(\theta^\top \mathbf{x}_t + \theta_0)}{\|\theta\|}
        $$
        SVM 的核心目标就是最大化这个几何间隔。

*   **最大间隔分类器 (硬间隔 SVM)**:
    对于线性可分的数据，SVM 寻找最大几何间隔。通过设置函数间隔的最小值为1（通过缩放 $$ \theta $$），最大化几何间隔 $$ \frac{1}{\|\theta\|} $$ 就等价于最小化 $$ \|\theta\| $$，或者等价于最小化 $$ \frac{1}{2}\|\theta\|^2 $$（为了计算方便）。这就构成了 SVM 的**原始问题 (Primal Form)**：
    $$
    \min_{\theta, \theta_0} \frac{1}{2} \|\theta\|^2 \quad \text{subject to} \quad y_t(\theta^\top \mathbf{x}_t + \theta_0) \ge 1, \quad \forall t = 1, \dots, n
    $$
    *   **目标函数**: $$ \min \frac{1}{2} \|\theta\|^2 $$，旨在最大化间隔。
    *   **约束条件**: $$ y_t(\theta^\top \mathbf{x}_t + \theta_0) \ge 1 $$，确保所有点都被正确分类，并且离决策边界至少有1的函数间隔。

*   **支持向量 (Support Vectors)**:
    在最终的模型中，只有那些恰好落在间隔边界上的数据点（即满足 $$ y_t(\theta^\top \mathbf{x}_t + \theta_0) = 1 $$ 的点）对决策边界的位置起决定性作用。这些点被称为**支持向量**。

*   **软间隔 SVM (Soft Margin SVM)**:
    当数据不是线性可分时，硬间隔 SVM 无法找到解。软间隔 SVM 通过引入**松弛变量 $$ \xi_t \ge 0 $$** 来允许某些点违反约束（即进入间隔区甚至被错分）。
    **软间隔原始问题**:
    $$
    \min_{\theta, \theta_0, \xi} \frac{1}{2} \|\theta\|^2 + C \sum_{t=1}^n \xi_t
    $$
    **subject to**:
    $$
    y_t(\theta^\top \mathbf{x}_t + \theta_0) \ge 1 - \xi_t, \quad \text{and} \quad \xi_t \ge 0, \quad \forall t
    $$
    *   $$ \xi_t $$: 松弛变量，表示第 $$ t $$ 个样本偏离其正确间隔边界的程度。如果 $$ \xi_t > 0 $$，说明该点违反了硬间隔约束。
    *   $$ C $$: 惩罚参数，是一个超参数，用于权衡“最大化间隔”（小 $$ \|\theta\|^2 $$）和“最小化分类错误”（小 $$ \sum \xi_t $$）之间的关系。$$ C $$ 越大，对误分类的惩罚越重。

---

### **主题三：特征工程 (Feature Engineering)**

特征工程是将原始数据转化为能更好地表达问题本质的特征，从而提升模型性能的过程。

*   **多项式变换 (Polynomial Transformation)**:
    通过创建特征的交互项（如 $$ x_i x_j $$）或高次项（如 $$ x_i^2 $$），可以让线性模型学习非线性关系。一个广义的线性判别函数可以写成：
    $$
    f_\mathbf{w}(\mathbf{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{1\le i \le j \le d} w_{i,j} x_i x_j + \sum_{1\le i \le j \le k \le d} w_{i,j,k} x_i x_j x_k + \dots
    $$

*   **名义变量 (Nominal/Categorical Values)**:
    对于没有顺序关系的类别（如“苹果”、“香蕉”），常用的处理方法是**独热编码 (One-Hot Encoding)**。
    **公式**:
    $$
    \phi(x) = [\mathbb{1}(x=\text{apple}), \mathbb{1}(x=\text{orange}), \mathbb{1}(x=\text{banana})]
    $$
    其中 $$ \mathbb{1}(\text{statement}) $$ 是**指示函数 (indicator function)**，当条件为真时取值为1，否则为0。例如，如果输入是 "orange"，其编码就是 $$ $$。

---

### **主题四：特征选择 (Feature Selection)**

特征选择是从已有特征中选出对目标变量最相关的一个子集，以简化模型、避免过拟合和降低计算复杂度。

*   **过滤法 (Filter Methods)**:
    过滤法独立于任何模型，它们使用统计指标来为特征打分，然后根据分数进行筛选。
    1.  **皮尔逊相关系数 (Pearson Correlation)**:
        衡量两个**连续变量**之间的线性关系强度。取值范围为 [-1, 1]。
        $$
        r^{(i)} = \frac{\sum_{t=1}^n (x_{t,i} - \bar{x}^{(i)})(y_t - \bar{y})}{\sqrt{\sum_{t=1}^n (x_{t,i} - \bar{x}^{(i)})^2} \sqrt{\sum_{t=1}^n (y_t - \bar{y})^2}}
        $$
        *   $$ r^{(i)} $$: 第 $$ i $$ 个特征与目标变量 $$ y $$ 的相关系数。
        *   $$ \bar{x}^{(i)} $$ 和 $$ \bar{y} $$: 分别是第 $$ i $$ 个特征和目标变量的均值。
    2.  **卡方检验 (Chi-squared Test)**:
        用于检验两个**分类变量**之间是否独立。它通过比较观测频数和期望频数的差异来工作。
        $$
        \chi^2_{(i)} = \sum_{j,k} \frac{(O_{jk} - E_{jk})^2}{E_{jk}}
        $$
        *   $$ O_{jk} $$: 第 $$ i $$ 个特征取第 $$ j $$ 个值且目标变量取第 $$ k $$ 个值的观测频数。
        *   $$ E_{jk} $$: 在独立假设下的期望频数。
        卡方值越大，表明特征与目标变量的关联性越强。
    3.  **互信息 (Mutual Information)**:
        衡量一个随机变量中包含的关于另一个随机变量的信息量，可以捕捉线性和非线性关系。
        对于离散变量：
        $$
        I(X;Y) = \sum_{y \in \mathcal{Y}} \sum_{x \in \mathcal{X}} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)
        $$
        互信息越大，表明特征 $$ X $$ 对目标 $$ Y $$ 的信息增益越多，关联性越强。