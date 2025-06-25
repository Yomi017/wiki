---
{"dg-publish":true,"permalink":"/notion/class/proof/projection-matrix/"}
---

### 故事的起点：我们的目标

我们的目标非常明确：给定一个向量 $\mathbf{x}$ 和一个子空间 $U$（由一组基向量 $B$ 定义），我们要找到 $U$ 里面那个离 $\mathbf{x}$ 最近的点 $\mathbf{p}$。

这个“最近”就是我们的线索，它的数学语言是：**误差向量 $(\mathbf{x}-\mathbf{p})$ 必须与整个子空间 $U$ 垂直**。

### 推理过程：公式的先后顺序和逻辑

#### 第1步：将目标具体化（找到未知数）

我们不知道 $\mathbf{p}$ 是什么，但我们知道它在子空间 $U$ 里。任何在 $U$ 里的向量，都可以被 $U$ 的基 $B = [\mathbf{b}_1, \dots, \mathbf{b}_k]$ 线性表示。

所以，一定存在一组“坐标”或“权重” $\boldsymbol{\lambda} = [\lambda_1, \dots, \lambda_k]^T$，使得：
$$ \mathbf{p} = \lambda_1\mathbf{b}_1 + \dots + \lambda_k\mathbf{b}_k $$
用矩阵形式写出来，就是我们的第一个公式：
$$ \mathbf{p} = B\boldsymbol{\lambda} \quad \cdots \quad (1) $$
**逻辑**：这一步把一个未知**向量** $\mathbf{p}$ 的问题，转化为了一个未知**坐标** $\boldsymbol{\lambda}$ 的问题。未知数从一个向量变成了一组数字，问题变得更具体了。

#### 第2步：利用核心线索建立方程

我们的核心线索是：$(\mathbf{x} - \mathbf{p}) \perp U$。
这意味着 $(\mathbf{x} - \mathbf{p})$ 必须与 $U$ 的所有基向量都垂直。
$$
\begin{cases}
    \mathbf{b}_1^T (\mathbf{x} - \mathbf{p}) = 0 \\
    \vdots \\
    \mathbf{b}_k^T (\mathbf{x} - \mathbf{p}) = 0
\end{cases}
$$
把这 $k$ 个方程合并成一个矩阵方程，就得到了：
$$ B^T(\mathbf{x} - \mathbf{p}) = \mathbf{0} \quad \cdots \quad (2) $$
**逻辑**：这一步把几何上的“垂直”关系，转化为了代数上的方程组。

#### 第3步：解方程，找到未知坐标 $\boldsymbol{\lambda}$

现在我们有两个方程，(1) 和 (2)。把 (1) 代入 (2) 中，消去我们不直接关心的中间变量 $\mathbf{p}$：
$$ B^T(\mathbf{x} - B\boldsymbol{\lambda}) = \mathbf{0} $$
展开并整理，我们就得到了大名鼎鼎的**正规方程 (Normal Equation)**：
$$ (B^TB)\boldsymbol{\lambda} = B^T\mathbf{x} \quad \cdots \quad (3) $$
**逻辑**：这是整个故事的高潮！我们建立了一个只包含**已知量**（$B$ 和 $\mathbf{x}$）和我们最终想求的**未知量**（$\boldsymbol{\lambda}$）的方程。解出这个方程，就能得到坐标 $\boldsymbol{\lambda}$。

至此，一个具体问题的求解流程已经完整了：
> **求解流程：** `给定 B, x` -> `用(3)解出 λ` -> `用(1)算出 p`

这个流程回答了“如何计算一个特定向量的投影”。

#### 第4步：从具体到抽象，推导通用公式（投影矩阵）

现在我们想更进一步。我们不想每次都解方程，我们想要一个“万能公式”——一个矩阵 $P$，只要把它乘到**任何**向量 $\mathbf{x}$ 上，就能直接得到它的投影 $\mathbf{p}$。即 $\mathbf{p} = P\mathbf{x}$。

这个 $P$ 就是**投影矩阵**。我们怎么得到它呢？

从正规方程 (3) 出发，我们可以从理论上“解”出 $\boldsymbol{\lambda}$（通过在两边左乘 $(B^TB)^{-1}$）：
$$ \boldsymbol{\lambda} = (B^TB)^{-1}B^T\mathbf{x} $$
现在我们有了坐标 $\boldsymbol{\lambda}$ 的通用表达式。再把它代回到公式 (1) $\mathbf{p} = B\boldsymbol{\lambda}$ 中：
$$ \mathbf{p} = B \underbrace{ \left( (B^TB)^{-1}B^T\mathbf{x} \right) }_{\boldsymbol{\lambda}} $$
利用矩阵乘法的结合律，重新组合括号：
$$ \mathbf{p} = \underbrace{ \left( B(B^TB)^{-1}B^T \right) }_{P} \mathbf{x} $$
我们就得到了投影矩阵 $P$ 的公式：
$$ P = B(B^TB)^{-1}B^T \quad \cdots \quad (4) $$
**逻辑**：这一步是对求解过程的“封装”和“抽象化”。我们把求解特定 $\mathbf{x}$ 的投影的**过程**，提炼成了一个可以作用于**任何** $\mathbf{x}$ 的**算子**（矩阵 $P$）。

### 总结：公式的逻辑链条

1.  **`p = Bλ`**: **问题转化**。将寻找向量 `p` 的问题转化为寻找坐标 `λ`。
2.  **`(BᵀB)λ = Bᵀx`**: **建立方程**。利用几何定义（垂直），建立一个关于未知坐标 `λ` 的方程。这是求解的核心。
3.  **`P = B(BᵀB)⁻¹Bᵀ`**: **公式化/抽象化**。从求解方程的过程中，提炼出一个通用的投影算子 `P`，它代表了整个投影操作。

所以，它们的**逻辑先后性**是：
> 为了找到 **`p`** -> 我们引入了 **`λ`** -> 为了求解 **`λ`** 我们建立了**正规方程** -> 为了得到一个通用的投影方法，我们从正规方程的解中推导出了**投影矩阵 `P`**。

这个流程清晰地展示了从一个具体的几何问题，如何一步步通过代数手段，最终抽象出一个普适的数学工具（投影矩阵）的过程。