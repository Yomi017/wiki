---
{"dg-publish":true,"permalink":"/notion/class/proof/geometric-transformation-of-symmetric-matrices/"}
---

**一个线性变换是纯粹的、无扭曲的拉伸（可以被一组正交的特征向量和实数特征值完全描述），当且仅当其对应的矩阵是实对称矩阵**。

这个证明分为两个部分，这正对应了**谱定理**的两个方向。

1.  **方向一：如果矩阵 `A` 是实对称的，那么它代表的变换是纯粹的拉伸。**
2.  **方向二：如果一个变换是纯粹的拉伸，那么它的矩阵 `A` 必然是实对称的。**

---

### 证明方向一：实对称矩阵 ⇒ 纯粹拉伸

这部分其实就是**谱定理的内容**。谱定理告诉我们，如果一个矩阵 `A` 是实对称的（`A = Aᵀ`），那么它一定可以被正交对角化：

$$ A = Q \Lambda Q^\top $$

其中：
*   $Q$ 是一个**正交矩阵**，它的列 $\boldsymbol{q}_1, \boldsymbol{q}_2, \dots, \boldsymbol{q}_n$ 是一组标准正交的特征向量。
*   $\Lambda$ 是一个**对角矩阵**，对角线上的元素是对应的实数特征值 $\lambda_1, \lambda_2, \dots, \lambda_n$。

现在我们来解释这个分解如何证明“纯粹拉伸”的几何行为。考虑对任意向量 $\boldsymbol{x}$ 进行变换 $A\boldsymbol{x}$：

$$ A\boldsymbol{x} = (Q \Lambda Q^\top) \boldsymbol{x} $$

我们可以把这个变换过程分解成三步：

1.  **第一步：$Q^\top \boldsymbol{x}$ (坐标系旋转)**
    *   由于 $Q$ 的列是标准正交基 $\{\boldsymbol{q}_i\}$，所以 $Q^\top$ ($=Q^{-1}$) 的作用是将向量 $\boldsymbol{x}$ 从标准坐标系（基为 $\{\boldsymbol{e}_i\}$）**旋转**到以特征向量为基的新坐标系中。
    *   得到的新坐标向量 $\boldsymbol{y} = Q^\top \boldsymbol{x}$，它的分量 $y_i$ 就是 $\boldsymbol{x}$ 在特征向量 $\boldsymbol{q}_i$ 方向上的投影长度。

2.  **第二步：$\Lambda \boldsymbol{y}$ (在新坐标系下拉伸)**
    *   $\Lambda$ 是一个对角矩阵。它的作用非常简单：将新坐标向量 $\boldsymbol{y}$ 的每个分量 $y_i$ 乘以对应的特征值 $\lambda_i$。
    *   这正是在新的、相互正交的基（主轴）方向上，**独立地进行纯粹的拉伸或压缩**。没有任何剪切或旋转发生。

3.  **第三步：$Q (\Lambda \boldsymbol{y})$ (旋转回原坐标系)**
    *   $Q$ 的作用是将经过拉伸后的新坐标向量，从特征向量坐标系**旋转回**原来的标准坐标系。

**结论**：
整个变换 $A\boldsymbol{x}$ 的过程可以被完美地解释为：**旋转到主轴坐标系 → 沿着主轴纯粹地拉伸 → 旋转回原坐标系**。

这个过程没有内在的“扭曲”或“剪切”。所有的形变都发生在第二步，而那一步是在一个正交坐标系下的纯粹缩放。因此，**任何实对称矩阵所代表的变换，其本质都是纯粹的、沿着一组正交方向的拉伸**。证明完毕。

---

### 证明方向二：纯粹拉伸 ⇒ 实对称矩阵

现在我们反过来证明。假设一个线性变换 $T$ 是“纯粹的拉伸”。这在数学上意味着，存在一组**标准正交基** $\{\boldsymbol{q}_1, \boldsymbol{q}_2, \dots, \boldsymbol{q}_n\}$，使得变换 $T$ 在这些基向量上的作用仅仅是进行缩放。

也就是说，这组标准正交基 $\{\boldsymbol{q}_i\}$ 同时也是变换 $T$ 的**特征向量**，并且对应的特征值 $\lambda_i$ 是**实数**（因为我们讨论的是实空间中的拉伸/压缩）。

$$ T(\boldsymbol{q}_i) = \lambda_i \boldsymbol{q}_i $$

现在，我们来推导这个变换 $T$ 在标准坐标系下的矩阵 $A$ 是什么样子的。

根据我们上面的假设，我们可以写出：
*   一个由标准正交特征向量组成的**正交矩阵** $Q = [\boldsymbol{q}_1 | \boldsymbol{q}_2 | \dots | \boldsymbol{q}_n]$。
*   一个由实数特征值组成的**对角矩阵** $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$。

我们知道，矩阵 $A$ 乘以它的任何一个特征向量 $\boldsymbol{q}_i$，结果等于 $\lambda_i \boldsymbol{q}_i$：
$$ A \boldsymbol{q}_i = \lambda_i \boldsymbol{q}_i $$

我们可以把这 `n` 个方程用矩阵形式合在一起写：
$$ A [\boldsymbol{q}_1 | \dots | \boldsymbol{q}_n] = [\lambda_1 \boldsymbol{q}_1 | \dots | \lambda_n \boldsymbol{q}_n] $$
$$ A Q = Q \Lambda $$

因为 $Q$ 是正交矩阵，所以 $Q^{-1} = Q^\top$。我们在上式右边乘以 $Q^\top$：
$$ A Q Q^\top = Q \Lambda Q^\top $$
$$ A I = Q \Lambda Q^\top $$
$$ A = Q \Lambda Q^\top $$

现在我们来验证这个矩阵 $A$ 是否是对称的。我们计算它的转置 $A^\top$：
$$ A^\top = (Q \Lambda Q^\top)^\top $$
利用转置的性质 $(XYZ)^\top = Z^\top Y^\top X^\top$：
$$ A^\top = (Q^\top)^\top \Lambda^\top Q^\top $$

*   $(Q^\top)^\top = Q$
*   $\Lambda$ 是对角矩阵，所以它本身就是对称的，$\Lambda^\top = \Lambda$。

代入回去：
$$ A^\top = Q \Lambda Q^\top $$

我们发现，$A^\top$ 和 $A$ 的表达式完全一样！
$$ A^\top = A $$

**结论**：
一个代表纯粹拉伸（即拥有n个标准正交特征向量和实数特征值）的线性变换，其对应的矩阵**必然是实对称矩阵**。证明完毕。

---

### **总证明总结**

我们已经从两个方向证明了：

**一个线性变换的矩阵是实对称的 `(A = Aᵀ)`  ⟺  该变换是沿着一组正交主轴的纯粹拉伸（可被 `A = QΛQᵀ` 分解）。**

这完美地解释了为什么对称矩阵有如此“美好”的性质——它的代数形式和几何行为是同一枚硬币的两面，共同描述了一种无扭曲、可分解的纯粹变换。