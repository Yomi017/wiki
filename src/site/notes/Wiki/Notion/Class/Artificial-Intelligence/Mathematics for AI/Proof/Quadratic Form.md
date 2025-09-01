---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/mathematics-for-ai/proof/quadratic-form/"}
---

### 1. 核心定义：它是什么？

简单来说，矩阵的二次型是一个**函数**，它接受一个向量作为输入，然后输出一个**标量（一个数）**。这个函数由一个固定的方阵 $A$ 和一个变量向量 $\mathbf{x}$ 定义。

其数学表达式为：
$$ Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} $$
其中：
*   $\mathbf{x}$ 是一个列向量，例如 $\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$。
*   $A$ 是一个 $n \times n$ 的方阵。
*   $\mathbf{x}^T$ 是 $\mathbf{x}$ 的转置，即一个行向量 $\begin{pmatrix} x_1 & x_2 & \cdots & x_n \end{pmatrix}$。

**一句话概括：二次型就是将一个向量“夹”在一个矩阵的两侧进行运算，最终得到一个标量的过程。**

### 2. “二次型”这个名字的由来

为什么叫“二次”型？我们来看一个具体的例子。

假设 $\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$ 并且 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$。

我们来计算 $\mathbf{x}^T A \mathbf{x}$：
$$
\begin{align*}
\mathbf{x}^T A \mathbf{x} &= \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} a & b \\ c & d \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \\
&= \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} ax_1 + bx_2 \\ cx_1 + dx_2 \end{pmatrix} \\
&= x_1(ax_1 + bx_2) + x_2(cx_1 + dx_2) \\
&= ax_1^2 + bx_1x_2 + cx_2x_1 + dx_2^2 \\
&= ax_1^2 + (b+c)x_1x_2 + dx_2^2
\end{align*}
$$
观察最终得到的这个多项式：$ax_1^2 + (b+c)x_1x_2 + dx_2^2$。
它的**每一个项**（$x_1^2, x_1x_2, x_2^2$）的变量次数之和都是**2**。这就是它被称为**二次**型（Quadratic Form）的原因。

### 3. 与对称矩阵的关系（非常重要）

在上面的例子中，我们看到混合项的系数是 $(b+c)$。现在思考一个问题：
如果有一个非对称矩阵 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$，它的二次型是：
$Q(\mathbf{x}) = x_1^2 + (2+3)x_1x_2 + 4x_2^2 = x_1^2 + 5x_1x_2 + 4x_2^2$。

现在我们看另一个**对称**矩阵 $A_{sym} = \begin{pmatrix} 1 & 2.5 \\ 2.5 & 4 \end{pmatrix}$，它的二次型是：
$Q_{sym}(\mathbf{x}) = x_1^2 + (2.5+2.5)x_1x_2 + 4x_2^2 = x_1^2 + 5x_1x_2 + 4x_2^2$。

**结论**：非对称矩阵 $A$ 和它的“对称化”版本 $A_{sym} = \frac{1}{2}(A + A^T)$ 产生的二次型是**完全相同**的。

正因为如此，在讨论二次型时，我们**通常只考虑对称矩阵**。因为任何非对称矩阵的二次型都可以由一个唯一的对称矩阵来表示，这样可以大大简化理论和计算。从现在开始，我们默认二次型中的矩阵 $A$ 是对称的。

对于对称矩阵 $A = \begin{pmatrix} a & b \\ b & d \end{pmatrix}$，二次型就有一个更简洁的形式：
$$ Q(\mathbf{x}) = ax_1^2 + 2bx_1x_2 + dx_2^2 $$
这里，对角线元素 $a, d$ 对应平方项的系数，非对角线元素 $b$ 对应混合项系数的一半。

### 4. 几何意义：二次型代表什么？

二次型的几何意义非常直观。方程 $\mathbf{x}^T A \mathbf{x} = c$（其中 $c$ 是一个常数）在二维或三维空间中定义了**二次曲线或二次曲面**。

*   **如果 $A$ 是单位矩阵 $I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$**：
    $\mathbf{x}^T I \mathbf{x} = x_1^2 + x_2^2 = c$。这是一个**圆**。

*   **如果 $A$ 是对角矩阵 $A = \begin{pmatrix} a & 0 \\ 0 & d \end{pmatrix}$ (且 $a,d > 0$)**：
    $\mathbf{x}^T A \mathbf{x} = ax_1^2 + dx_2^2 = c$。这是一个**椭圆**，其主轴沿着坐标轴方向。

*   **如果 $A$ 是一个通用的对称矩阵**：
    $\mathbf{x}^T A \mathbf{x} = c$ 定义的图形仍然是二次曲线（如椭圆、双曲线），但它们的主轴方向是 $A$ 的**特征向量**方向，轴的“拉伸”程度由 $A$ 的**特征值**决定。

**核心思想：二次型描述了一个空间中的“碗”状或“鞍”状的曲面。**

### 5. 正定性：二次型的分类

二次型最重要的性质是它的**符号**。对于任何非零向量 $\mathbf{x} \ne \mathbf{0}$，$\mathbf{x}^T A \mathbf{x}$ 的值是正、是负、还是都有可能？这引出了对矩阵（及其二次型）的分类，即**正定性 (Definiteness)**。

| 类别 | 定义 (对所有 $\mathbf{x} \ne \mathbf{0}$) | 几何形状 (想象一个碗) | 关联的特征值 |
| :--- | :--- | :--- | :--- |
| **正定 (Positive Definite)** | $\mathbf{x}^T A \mathbf{x} > 0$ | 一个朝上开口的碗，最小值在原点 | 全部为正 ($>0$) |
| **负定 (Negative Definite)** | $\mathbf{x}^T A \mathbf{x} < 0$ | 一个朝下开口的碗，最大值在原点 | 全部为负 ($<0$) |
| **半正定 (Positive Semidefinite)** | $\mathbf{x}^T A \mathbf{x} \ge 0$ | 一个平底碗，可能有“零点”线/面 | 全部非负 ($\ge 0$) |
| **半负定 (Negative Semidefinite)** | $\mathbf{x}^T A \mathbf{x} \le 0$ | 一个平顶碗，可能有“零点”线/面 | 全部非正 ($\le 0$) |
| **不定 (Indefinite)** | $\mathbf{x}^T A \mathbf{x}$ 的值有正有负 | 一个马鞍的形状 | 有正有负 |

### 6. 应用：为什么二次型如此重要？

*   **最优化理论**：在微积分中，函数的二阶导数（Hessian 矩阵）决定了一个临界点是局部最小值、最大值还是鞍点。Hessian 矩阵的正定性就是通过其二次型来判断的。正定意味着局部最小值。
*   **物理学**：动能、势能等很多物理量都以二次型的形式出现。例如，一个系统的势能必须是正定的，以保证其稳定性。
*   **机器学习与统计学**：
    *   **协方差矩阵**：它是一个半正定矩阵，其二次型与数据的方差有关。
    *   **核方法 (Kernel Methods)**：在支持向量机 (SVM) 等算法中，核矩阵必须是半正定的，这保证了映射后的空间距离是有效的。
    *   **距离定义**：$\mathbf{x}^T A \mathbf{x}$（当 $A$ 正定时）可以用来定义一种广义的距离，称为马氏距离的平方。

总而言之，二次型是连接**矩阵代数**、**多项式函数**和**几何**三者的桥梁，是理解许多高级应用的关键。