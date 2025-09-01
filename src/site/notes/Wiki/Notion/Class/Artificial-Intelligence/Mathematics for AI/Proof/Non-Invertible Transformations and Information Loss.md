---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/mathematics-for-ai/proof/non-invertible-transformations-and-information-loss/"}
---

信息丢失主要体现在两个方面，它们互为表里，都是由矩阵的**秩 (Rank)** 小于其维度造成的。

1.  **输入信息的“混淆” (非单射 / Not Injective):** 多个不同的输入被映射到了同一个输出。
2.  **输出空间的“降维” (非满射 / Not Surjective):** 整个输入空间被“压扁”到一个更低维度的子空间中。

让我们通过一个非常直观的例子来理解这一切。

---

### **例子：投影 (Projection)**

想象一个从二维空间 $\mathbb{R}^2$ 到其自身的线性映射 $\Phi$，它的作用是把任何一个向量**投影**到 X 轴上。

*   **映射规则:** $\Phi((x, y)) = (x, 0)$
*   **输入空间:** 整个二维平面 ($\mathbb{R}^2$)
*   **输出空间:** 也是二维平面 ($\mathbb{R}^2$)

#### **1. 构建变换矩阵**

我们使用标准基 $B = C = (\mathbf{e}_1, \mathbf{e}_2)$ 来构建变换矩阵 $A_\Phi$。

*   **第一列:** 计算 $\Phi(\mathbf{e}_1)$ 的坐标
    *   $\Phi(\mathbf{e}_1) = \Phi((1, 0)) = (1, 0) = 1\mathbf{e}_1 + 0\mathbf{e}_2$
    *   坐标是 $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$
*   **第二列:** 计算 $\Phi(\mathbf{e}_2)$ 的坐标
    *   $\Phi(\mathbf{e}_2) = \Phi((0, 1)) = (0, 0) = 0\mathbf{e}_1 + 0\mathbf{e}_2$
    *   坐标是 $\begin{pmatrix} 0 \\ 0 \end{pmatrix}$

所以，变换矩阵是：
$$ A_\Phi = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} $$

#### **2. 分析矩阵**

*   这是一个 $2 \times 2$ 的**方阵**。
*   它的行列式 $\det(A_\Phi) = 1 \cdot 0 - 0 \cdot 0 = 0$。
*   因为行列式为0，所以这个矩阵是**不可逆的 (non-invertible)**。
*   它的秩 $\text{rk}(A_\Phi) = 1$，小于它的维度 $n=2$。

现在，我们来看看信息丢失发生在哪里。

---

### **信息丢失的体现**

#### **方面一：输入信息的混淆 (非单射 / Many-to-One)**

我们取两个**不同**的输入向量：
*   $\mathbf{v}_1 = \begin{pmatrix} 3 \\ \mathbf{2} \end{pmatrix}$
*   $\mathbf{v}_2 = \begin{pmatrix} 3 \\ \mathbf{5} \end{pmatrix}$

用矩阵 $A_\Phi$ 对它们进行变换：
*   $A_\Phi \mathbf{v}_1 = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix}$
*   $A_\Phi \mathbf{v}_2 = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 3 \\ 5 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix}$

**看到了吗？** 两个完全不同的输入向量 $\mathbf{v}_1$ 和 $\mathbf{v}_2$ 被映射到了**完全相同**的输出向量 $\begin{pmatrix} 3 \\ 0 \end{pmatrix}$。

**这就是信息丢失！**
如果你只拿到了输出结果 $\begin{pmatrix} 3 \\ 0 \end{pmatrix}$，你**无法唯一地确定**原始输入是 $\begin{pmatrix} 3 \\ 2 \end{pmatrix}$ 还是 $\begin{pmatrix} 3 \\ 5 \end{pmatrix}$，或者是任何形如 $\begin{pmatrix} 3 \\ y \end{pmatrix}$ 的向量。原始向量的 **y 分量信息** 在变换过程中被完全丢失了。

*   **理论连接：** 这种情况发生是因为矩阵的**零空间 (Null Space)** 不只有零向量。对于矩阵 $A_\Phi$，它的零空间是所有形如 $\begin{pmatrix} 0 \\ y \end{pmatrix}$ 的向量（即 Y 轴）。任何在零空间中的向量都被“湮灭”成零。

#### **方面二：输出空间的降维 (非满射 / Not Onto)**

观察所有可能的输出结果，它们都是形如 $\begin{pmatrix} x \\ 0 \end{pmatrix}$ 的向量。

*   **所有的输出**都落在了 **X 轴**上。
*   整个二维的输入平面，被“压扁”成了一条一维的直线（X 轴）。

**这就是信息丢失！**
你永远无法得到一个不在 X 轴上的输出。例如，向量 $\mathbf{w} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$ **永远不可能**成为这个映射的输出。无论你输入什么，输出的 y 分量永远是 0。整个空间的“维度”信息丢失了。

*   **理论连接：** 这种情况发生是因为矩阵的**列空间 (Column Space)** 的维度（即秩）小于目标空间的维度。对于 $A_\Phi$，它的列空间是由 $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ 和 $\begin{pmatrix} 0 \\ 0 \end{pmatrix}$ 张成的，其维度为 1，小于目标空间 $\mathbb{R}^2$ 的维度 2。

### **总结**

| | 可逆方阵 (Isomorphism) | 不可逆方阵 (Information Loss) |
| :--- | :--- | :--- |
| **映射性质** | 双射 (Bijective)，一一对应 | 非单射 (Many-to-one)，非满射 (Not onto) |
| **输入** | 一个输入只对应一个输出 | 多个输入可能对应一个输出 |
| **输出** | 覆盖整个目标空间 | 只覆盖目标空间的一个子空间（线、面等） |
| **信息** | **无损**，可完全恢复输入 | **有损**，无法唯一确定输入 |
| **几何变换** | 旋转、缩放、错切 (可逆操作) | **投影**、压扁 (不可逆操作) |
| **零空间** | 只有零向量 | 包含非零向量 |
| **秩** | `rank = n` (满秩) | `rank < n` (秩亏) |