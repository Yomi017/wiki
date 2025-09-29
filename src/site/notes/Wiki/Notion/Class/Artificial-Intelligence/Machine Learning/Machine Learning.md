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
### **梯度下降 (Gradient Descent - GD) 算法：动机**

*   **核心目标**: 在机器学习中，我们通常会定义一个**成本函数** (Cost Function) $C(w)$，它衡量了模型在当前参数 $w$ 下的“糟糕程度”。我们的目标就是找到一组参数 $w$ 来最小化这个函数。这里的 $w$ 是一个包含所有模型参数（如权重）的向量，$w = [w_1, w_2, \dots, w_d]^\top$。

*   **梯度的角色**:
    为了找到最小值，我们需要知道应该朝哪个方向调整参数。**梯度 (Gradient)** 就是这个问题的答案。函数 $C(w)$ 在某一点的梯度，记作 $\nabla_w C(w)$，是一个向量，它指向该点函数值**上升最快**的方向。
    
    这个梯度向量由函数对每个参数的偏导数构成：
    $$
    \nabla_w C(w) = \begin{pmatrix} \frac{\partial C}{\partial w_1} \\ \frac{\partial C}{\partial w_2} \\ \vdots \\ \frac{\partial C}{\partial w_d} \end{pmatrix}
    $$

*   **下降最快的方向**:
    既然梯度指向“上坡”最陡的方向，那么它的反方向，即**负梯度** $-\nabla_w C(w)$，就必然指向“下坡”最陡的方向。因此，如果我们想让函数值 $C(w)$ 减小得最快，就应该沿着负梯度的方向去更新参数 $w$。这就是梯度下降算法的核心思想。

### **梯度下降 (GD) 算法：基础**

*   **算法流程**:
    梯度下降是一个迭代过程。我们从一个随机的初始参数点 $w_0$ 开始，然后反复执行一个简单的更新步骤，直到满足某个停止条件。

*   **核心更新公式**:
    每一步迭代都遵循以下公式：
    $$
    w_{k+1} \leftarrow w_k - \eta \nabla_w C(w_k)
    $$
    *   $w_k$ 是参数在第 $k$ 次迭代时的位置。
    *   $w_{k+1}$ 是参数更新后的新位置。
    *   $\nabla_w C(w_k)$ 是在当前位置 $w_k$ 计算出的梯度。
    *   $\eta$ (eta) 是**学习率 (Learning Rate)**，一个正的小常数，它决定了我们每一步“走多远”。

*   **理论保证**:
    根据微积分原理，只要学习率 $\eta$ 被设定在一个合理的范围内（不能太大），那么每执行一次更新，成本函数的值都会减小：$C(w_{k+1}) < C(w_k)$。通过不断重复这个过程，我们就能逐步逼近函数的（局部）最小值。

*   **收敛准则 (Convergence Criteria)**:
    我们不能让算法无限地运行下去，必须定义一个停止条件。常见的停止条件有：
    1.  **最大迭代次数**: 预先设定一个迭代上限，例如1000次，达到后强制停止。
    2.  **目标函数变化阈值**: 当函数值的下降变得非常缓慢，例如 $|C(w_{k+1}) - C(w_k)|$ 小于一个很小的数 $\epsilon$，就认为已经收敛。
    3.  **参数变化阈值**: 当参数向量本身几乎不再变化，例如 $\|w_{k+1} - w_k\|$ 小于一个很小的数 $\epsilon$，也认为已经收敛。

*   **局限性：局部最小值**:
    梯度下降算法的一个关键特性是它只会沿着下坡方向走。如果一个函数的图像有多个山谷（多个局部最小值），梯度下降只能保证找到它出发点所在的那个山谷的底部，而不能保证找到全局最低的那个山谷。一旦到达任何一个梯度为零的点（局部最小值或鞍点），$\nabla_w C(w)$ 变为零向量，更新就会停止。

*   **学习率 $\eta$ 的影响**:
    *   **$\eta$ 太小**: 更新步长太小，就像小碎步下山，虽然稳妥但速度极慢。
    *   **$\eta$ 太大**: 更新步长太大，可能会在山谷底部来回“跨越”，导致震荡而无法精确到达最低点，甚至可能一步跨到对面更高的山坡上，导致函数值不降反升，即**发散**。

### **Adagrad (Adaptive Gradient Algorithm)**

*   **动机**:
    标准梯度下降对所有参数都使用同一个学习率 $\eta$。但在实际问题中，不同参数的重要性可能不同，有些参数可能需要更快的更新，有些则需要更慢、更稳健的更新。Adagrad 旨在为**每一个参数**自动地、自适应地调整其学习率。

*   **核心思想**:
    对于那些在训练过程中梯度一直很大的参数（意味着它们被频繁、大幅度地更新），我们应该对其保持谨慎，减小其学习率；反之，对于梯度一直很小的参数，我们应该给予其更大的学习率以鼓励其更新。

*   **更新公式**:
    Adagrad 对标准GD公式中的学习率 $\eta$ 做了一个修正：
    $$
    w_{k+1, i} \leftarrow w_{k, i} - \frac{\eta}{\sqrt{M_{ii,k} + \epsilon}} [\nabla_w C(w_k)]_i
    $$
    *   这个更新是针对第 $i$ 个参数 $w_i$ 的。
    *   $[\nabla_w C(w_k)]_i$ 是梯度向量的第 $i$ 个分量。
    *   $M_{ii,k}$ 是一个关键项，它累积了第 $i$ 个参数从训练开始到第 $k$ 步**所有历史梯度的平方和**：
        $$
        M_{ii,k} = \sum_{s=0}^{k} ([\nabla_w C(w_s)]_i)^2
        $$
    *   $\epsilon$ 是一个极小的正数（如 $10^{-8}$），用于防止分母为零。

*   **优缺点**:
    *   **优点**: 自动调整学习率，无需手动设置。
    *   **缺点**: 由于 $M_{ii,k}$ 是一个持续累加的正数，它会单调递增。这导致分母不断变大，学习率会持续衰减，最终可能变得过小，使得模型在后期无法有效学习。

### **动量法 (Momentum-based GD)**

*   **动机**:
    在梯度下降的过程中，如果损失函数的等高线是狭长的椭圆形，梯度方向会垂直于等高线，导致更新路径呈“之”字形，收敛缓慢。动量法旨在缓解这个问题并加速收敛。

*   **核心思想**:
    引入物理学中的“惯性”或“动量”概念。参数的更新方向不仅取决于当前所在位置的梯度，还受到其历史更新方向的影响。

*   **更新公式**:
    动量法引入一个“速度”向量 $v$，它累积了历史梯度的信息：
    $$
    v_k \leftarrow \beta v_{k-1} + (1-\beta) \nabla_w C(w_k)
    $$
    $$
    w_{k+1} \leftarrow w_k - \eta v_k
    $$
    *   $v_k$ 是第 $k$ 步的速度向量，它是所有历史梯度的**指数加权移动平均**。
    *   $\beta$ 是动量因子，通常取值接近1（如0.9），它决定了历史速度在多大程度上被保留。当梯度方向变化时，动量可以平滑更新，抑制震荡；当梯度方向一致时，动量会累积，使得更新加速。

### **Nesterov 加速梯度 (NAG)**

*   **动机**:
    动量法虽然有效，但有一个小缺陷：它是先计算当前位置的梯度，再结合历史速度进行“盲目”的更新。NAG 通过引入“预判”机制，使其更加智能。

*   **核心思想**:
    “先跳再修正”。NAG 不在当前位置 $w_k$ 计算梯度，而是先用历史速度 $v_{t-1}$ “预估”一下下一步大致会跳到哪里（一个临时位置 $w_k - \beta v_{t-1}$），然后在这个**未来**的位置计算梯度，用这个更具前瞻性的梯度来修正最终的更新方向。

*   **更新公式**:
    $$
    v_t \leftarrow \beta v_{t-1} + \eta \nabla_w C(w_k - \beta v_{t-1})
    $$
    $$
    w_{k+1} \leftarrow w_k - v_k
    $$
    *   关键区别在于梯度计算点 $\nabla_w C(\cdot)$ 是在临时更新点 $w_k - \beta v_{t-1}$，而不是当前点 $w_k$。这使得 NAG 能够更早地感知到函数曲率的变化，从而提前减速，减少震荡，收敛更快。

---

### **分类器的间隔 (Margin of Classifier)**

*   **正确分类的数学表达**:
    对于一个线性分类器，其决策边界由 $\theta^\top\mathbf{x} = 0$ 定义。我们通常使用符号函数 $\text{sign}(\theta^\top\mathbf{x})$ 来进行预测。如果一个样本 $(\mathbf{x}, y)$ 被正确分类，那么预测的符号必须和真实标签 $y$ (取值为+1或-1) 的符号一致。这可以简洁地写成一个统一的表达式：
    $$
    y (\theta^\top\mathbf{x}) > 0
    $$

*   **定义 1.1: 函数间隔 (Functional Margin)**:
    *   **定义**: 样本 $(\mathbf{x}, y)$ 关于分类器 $\theta$ 的函数间隔就是表达式 $y(\theta^\top\mathbf{x})$ 的值。
    *   **解释**:
        *   **符号**: 函数间隔的**正负号**代表了分类是否正确。正数表示正确，负数表示错误。
        *   **大小**: 函数间隔的**绝对值** $|y(\theta^\top\mathbf{x})|$ 代表了分类的**置信度**。值越大，表示数据点离决策边界越远，我们对这个分类结果就越有信心。

### **线性可分 (Linearly Separable)**

*   **定义 1.2**:
    如果存在一个参数 $\theta$，使得对于数据集中的**所有**样本 $(\mathbf{x}_t, y_t)$，它们的函数间隔都为正，即：
    $$
    y_t(\theta^\top\mathbf{x}_t) > 0, \quad \forall t = 1, \dots, n
    $$
    那么这个数据集就被称为**线性可分的**。
*   **几何意义**: 这意味着，我们可以找到一个超平面，能够像一把刀切蛋糕一样，完美地将两类数据点分到超平面的两侧，没有任何一个点被切错。

### **γ-线性可分 (γ-linearly separable)**

*   **定义 1.3**:
    这是一个比线性可分更强的条件。如果存在一个参数 $\theta$ 和一个正数 $\gamma > 0$，使得对于数据集中的**所有**样本，它们的函数间隔都**不小于** $\gamma$，即：
    $$
    y_t(\langle\theta, \mathbf{x}_t\rangle) \ge \gamma, \quad \forall t = 1, \dots, n
    $$
*   **核心思想**: 这不仅要求数据能被分开，还要求在两类数据之间必须存在一条“安全地带”或“缓冲区”，而这个缓冲区的最小“函数宽度”就是 $\gamma$。所有的数据点都必须离决策边界至少有这么远（以函数间隔衡量）。

### **几何间隔 (Geometric Margin)**

*   **动机**:
    函数间隔 $y(\theta^\top\mathbf{x})$ 存在一个问题：如果我们把 $\theta$ 替换为 $2\theta$，决策边界 $\theta^\top\mathbf{x}=0$ 并未改变，但所有点的函数间隔都翻了一倍。这意味着函数间隔的大小是相对的，无法作为衡量真实距离的绝对标准。

*   **定义**:
    几何间隔是数据点到超平面的**真实欧几里得距离**，它是通过对函数间隔进行**归一化**得到的：
    $$
    \gamma_{\text{geom}} = \frac{y(\theta^\top\mathbf{x})}{\|\theta\|} = \frac{\text{函数间隔}}{\text{参数向量的长度}}
    $$
*   **意义**: 几何间隔是一个**绝对的、不受缩放影响的**度量。它真实地反映了分类边界的“稳健性”。几何间隔越大，分类器对新数据的泛化能力通常越好。**支持向量机(SVM)的核心目标，就是找到那个能使最小几何间隔最大化的决策超平面。**

### **最大间隔线性分类器**

*   **SVM 优化问题的推导**:
    1.  **目标**: 最大化数据集中所有样本的最小几何间隔：
        $$
        \max_{\theta, \theta_0} \left( \min_t \frac{y_t(\mathbf{x}_t^\top \theta + \theta_0)}{\|\theta\|} \right)
        $$
    2.  **简化**: 我们可以通过缩放 $\theta$ 和 $\theta_0$ 来任意设定最小的**函数间隔**。为了方便，我们进行**规范化**，令 $\min_t y_t(\mathbf{x}_t^\top \theta + \theta_0) = 1$。
    3.  **转化**: 在这个规范化条件下，最大化 $\frac{1}{\|\theta\|}$ 就等价于最小化 $\|\theta\|$，进一步等价于最小化 $\frac{1}{2}\|\theta\|^2$（这是一个二次函数，更易于优化）。
    4.  **约束**: 规范化条件 $\min_t y_t(\mathbf{x}_t^\top \theta + \theta_0) = 1$ 意味着对于所有点，都必须满足 $y_t(\mathbf{x}_t^\top \theta + \theta_0) \ge 1$。

*   **SVM 的原始形式 (Primal form of SVM)**:
    综合以上推导，我们得到了**硬间隔 SVM** 的标准优化问题：
    $$
    \min_{\theta, \theta_0} \frac{1}{2} \|\theta\|^2 \quad \text{subject to} \quad y_t(\mathbf{x}_t^\top \theta + \theta_0) \ge 1, \quad \forall t
    $$

### **Primal-SVM：唯一解证明**

*   **结论**: 硬间隔SVM的优化问题有且仅有一个全局最优解。
*   **证明方法**: 反证法 (Proof by Contradiction)。
*   **证明步骤**:
    1.  **假设**: 存在两个不同的最优解 $\theta_1$ 和 $\theta_2$ ($\theta_1 \neq \theta_2$)。
    2.  **性质**: 因为它们都是最优解，所以它们的目标函数值必须相等，即 $\frac{1}{2}\|\theta_1\|^2 = \frac{1}{2}\|\theta_2\|^2 \implies \|\theta_1\| = \|\theta_2\|$。
    3.  **构造新解**: 考虑它们的中点 $\bar{\theta} = \frac{\theta_1 + \theta_2}{2}$。
    4.  **验证可行性**: 我们可以证明 $\bar{\theta}$ 仍然满足约束条件 $y_t(\mathbf{x}_t^\top \bar{\theta}) \ge 1$。因为 $y_t(\mathbf{x}_t^\top \theta_1) \ge 1$ 和 $y_t(\mathbf{x}_t^\top \theta_2) \ge 1$，两者相加再除以2，不等式依然成立。
    5.  **导出矛盾**: 根据向量的**严格三角不等式**，只要 $\theta_1$ 和 $\theta_2$ 不是同向的（在这里因为它们不同但长度相等，所以必然不同向），那么它们和的长度必然小于它们长度的和。即 $\|\theta_1 + \theta_2\| < \|\theta_1\| + \|\theta_2\|$。因此：
        $$
        \|\bar{\theta}\| = \left\|\frac{\theta_1 + \theta_2}{2}\right\| < \frac{\|\theta_1\| + \|\theta_2\|}{2} = \|\theta_1\|
        $$
    6.  **结论**: 我们找到了一个新解 $\bar{\theta}$，它的范数比假设的最优解 $\theta_1$ 的范数还要小，这与“$\theta_1$ 是最优解”的前提相矛盾。因此，最初的假设是错误的，最优解必须是唯一的。

### **软间隔 SVM (Soft Margin SVM)**

*   **动机**:
    在现实数据中，数据往往不是完全线性可分的，或者存在一些异常点。硬间隔 SVM 要求所有点都必须被严格分开，这使得它在这些情况下无法找到解。

*   **核心思想**:
    放宽硬性约束，允许一些样本点“犯规”。具体来说，允许某些点：
    1.  进入间隔区内（分类正确，但离边界太近）。
    2.  被错误地分到另一边。
    但是，我们要为这些“犯规”的行为付出一定的**代价**。

*   **软间隔 SVM 的原始形式**:
    我们为每个样本 $(\mathbf{x}_t, y_t)$ 引入一个**松弛变量 (slack variable)** $\xi_t \ge 0$。
    $$
    \min_{\theta, \theta_0, \xi} \frac{1}{2} \|\theta\|^2 + C \sum_{t=1}^n \xi_t
    $$
    约束条件变为：
    $$
    y_t(\mathbf{x}_t^\top \theta + \theta_0) \ge 1 - \xi_t, \quad \forall t
    $$
*   **公式解释**:
    *   $\xi_t$ (xi): 衡量了第 $t$ 个样本“犯规”的程度。
        *   如果 $\xi_t = 0$，说明该点满足硬间隔要求。
        *   如果 $0 < \xi_t \le 1$，说明该点在间隔区内，但分类仍正确。
        *   如果 $\xi_t > 1$，说明该点已经被错误分类。
    *   $\sum \xi_t$: 所有样本的总“犯规”量。
    *   $C$: **惩罚参数**，是一个非常重要的超参数。它控制了我们对“犯规”的容忍程度。
        *   **$C$ 很大**: 我们对犯规的惩罚很重，模型会尽力减小 $\sum \xi_t$，趋向于找到一个尽可能将所有点都正确分类的解，即使这意味着间隔很窄（容易过拟合）。
        *   **$C$ 很小**: 我们对犯规的容忍度很高，模型更愿意为了获得一个更宽的间隔（减小 $\|\theta\|^2$）而容忍一些点的犯规（允许 $\sum \xi_t$ 较大），通常泛化能力更好。
*   **Hinge Loss 形式 (Slide 83)**:
    软间隔SVM的优化问题可以被等价地写成一个无约束的优化问题。从约束 $y_t(\dots) \ge 1 - \xi_t$ 和 $\xi_t \ge 0$ 可以推导出，对于每个样本，我们希望其 $\xi_t$ 尽可能小，所以 $\xi_t = \max(0, 1 - y_t(\mathbf{x}_t^\top \theta + \theta_0))$。代入目标函数，得到：
    $$
    \min_{\theta, \theta_0} \underbrace{\frac{1}{2} \|\theta\|^2}_{\text{正则项/最大化间隔}} + C \sum_{t=1}^n \underbrace{\max(0, 1 - y_t(\mathbf{x}_t^\top \theta + \theta_0))}_{\text{Hinge Loss / 经验风险}}
    $$