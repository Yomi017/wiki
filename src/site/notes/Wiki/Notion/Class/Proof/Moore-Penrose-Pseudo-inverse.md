---
{"dg-publish":true,"permalink":"/wiki/notion/class/proof/moore-penrose-pseudo-inverse/"}
---

我们要证明：对于线性系统 $Ax=b$，其中 $A \in \mathbb{R}^{m \times n}$ 且 $m > n$ (过定系统)，如果 $A$ 具有列满秩（即 $A$ 的列是线性独立的），则向量 $x = (A^T A)^{-1} A^T b$ 是使 $\|Ax - b\|_2^2$ 最小化的唯一解。

**证明：**

我们的目标是找到 $x$ 使得残差的平方和 $\|Ax - b\|_2^2$ 最小。
我们知道欧几里得范数的平方定义为向量与其自身的转置相乘：
$$ \|v\|_2^2 = v^T v $$
所以，我们要最小化：
$$ f(x) = \|Ax - b\|_2^2 = (Ax - b)^T (Ax - b) $$

首先，展开这个表达式：
$$ f(x) = (x^T A^T - b^T)(Ax - b) $$
$$ f(x) = x^T A^T Ax - x^T A^T b - b^T Ax + b^T b $$
注意到 $x^T A^T b$ 是一个标量（$1 \times 1$ 矩阵），它的转置就是它本身，即 $(x^T A^T b)^T = b^T A x$。
因此，$x^T A^T b = b^T A x$。
所以，$f(x)$ 可以简化为：
$$ f(x) = x^T A^T Ax - 2x^T A^T b + b^T b $$

为了找到 $f(x)$ 的最小值，我们对 $x$ 求导，并令导数为零。这类似于微积分中求一元函数极值的方法，但在向量微积分中，我们使用梯度。

对 $f(x)$ 关于向量 $x$ 求梯度 $\nabla_x f(x)$ 并令其为零：

回忆一些向量微积分的导数规则：
*   $\nabla_x (x^T C x) = (C + C^T)x$
*   $\nabla_x (d^T x) = d$

应用这些规则：
*   对于 $x^T A^T Ax$，令 $C = A^T A$。
    则 $\nabla_x (x^T A^T Ax) = ((A^T A) + (A^T A)^T)x = (A^T A + A^T A)x = 2A^T Ax$。
    （因为 $(A^T A)^T = A^T (A^T)^T = A^T A$，所以 $A^T A$ 是对称矩阵）
*   对于 $-2x^T A^T b$，令 $d = A^T b$。
    则 $\nabla_x (-2x^T A^T b) = -2 \nabla_x (x^T (A^T b)) = -2(A^T b)$。
*   对于 $b^T b$，它是一个常数（不含 $x$），所以梯度为零。

将这些项组合起来，令梯度为零：
$$ \nabla_x f(x) = 2A^T Ax - 2A^T b = 0 $$
$$ 2A^T Ax = 2A^T b $$
$$ A^T Ax = A^T b $$

这是一个被称为**正规方程 (Normal Equations)** 的线性系统。

现在，我们需要证明 $(A^T A)^{-1}$ 存在，以便求解 $x$。
*   **A 的列是线性独立的**（这是假设条件，即 $A$ 具有列满秩）。
*   **定理：** 如果一个矩阵 $A$ 具有线性独立的列，那么矩阵 $A^T A$ 是可逆的。
    *   **证明 $(A^T A)^{-1}$ 存在：** 要证明 $A^T A$ 可逆，我们只需证明 $A^T A \mathbf{v} = \mathbf{0}$ 的唯一解是 $\mathbf{v} = \mathbf{0}$。
        假设 $A^T A \mathbf{v} = \mathbf{0}$。
        左乘 $\mathbf{v}^T$: $\mathbf{v}^T A^T A \mathbf{v} = \mathbf{v}^T \mathbf{0}$
        $$ (A\mathbf{v})^T (A\mathbf{v}) = 0 $$
        $$ \|A\mathbf{v}\|_2^2 = 0 $$
        这意味着 $A\mathbf{v} = \mathbf{0}$。
        由于 $A$ 的列是线性独立的（即 $A$ 的零空间只有零向量），因此 $A\mathbf{v} = \mathbf{0}$ 仅当 $\mathbf{v} = \mathbf{0}$ 时成立。
        所以，$A^T A$ 的零空间只有零向量，这意味着 $A^T A$ 是可逆的。

既然 $(A^T A)^{-1}$ 存在，我们可以从左侧乘以它来求解 $x$：
$$ (A^T A)^{-1} (A^T A) x = (A^T A)^{-1} A^T b $$
$$ I x = (A^T A)^{-1} A^T b $$
$$ x = (A^T A)^{-1} A^T b $$

这个解就是使 $\|Ax - b\|_2^2$ 最小化的唯一向量 $x$。它被称为**最小二乘解**。
由于 $A$ 的列是线性独立的，该解也是唯一的。

**总结：**

伪逆法的公式 $x = (A^T A)^{-1} A^T b$ 是通过对残差的平方和 $\|Ax - b\|_2^2$ 求梯度并使其为零推导出来的。在 $A$ 具有列满秩的条件下，$A^T A$ 保证是可逆的，从而使得这个最小二乘解是唯一且可计算的。这个解能够“最佳拟合”过定系统，即使没有精确解，也能找到使误差最小化的近似解。