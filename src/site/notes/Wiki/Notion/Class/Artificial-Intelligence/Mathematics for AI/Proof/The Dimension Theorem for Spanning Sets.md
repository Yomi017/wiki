---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/mathematics-for-ai/proof/the-dimension-theorem-for-spanning-sets/"}
---


# 维度 ≤ 生成向量数

**您的困惑：** “因为秩是列空间的维度，而列空间是由 $n$ 个列向量生成的，所以它的维度不可能超过 $n$。” 这个性质本身是如何从最基本的公理证明的？

**回答：** 好的，我们来严格地从最基本的定义出发，证明这个“维度定理”的一个基础版本。

**目标：** 证明一个由 $n$ 个向量生成的向量空间，其维度不会超过 $n$。

**所需的基本公理和定义 (Axioms and Definitions):**

1.  **生成空间 (Span):** 一个向量集合 $S = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ 的生成空间 $\text{Span}(S)$，是这些向量所有可能的线性组合的集合。
2.  **线性无关 (Linear Independence):** 一个向量集合 $U = \{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$ 是线性无关的，当且仅当方程 $c_1\mathbf{u}_1 + \cdots + c_k\mathbf{u}_k = \mathbf{0}$ 的唯一解是所有系数 $c_i=0$。
3.  **基 (Basis):** 一个向量空间 $V$ 的基是一个**线性无关**的**生成集**。
4.  **维度 (Dimension):** 一个向量空间 $V$ 的维度 $\text{dim}(V)$ 是其任何一个基中向量的数量。

**证明（使用反证法 Proof by Contradiction）：**

1.  **前提 (Premise):**
    *   我们有一个向量空间 $V$。
    *   这个空间是由 $n$ 个向量 $S = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ 生成的，即 $V = \text{Span}(S)$。

2.  **为反证法做出假设 (Assumption for Contradiction):**
    *   我们**假设**这个空间的维度**大于** $n$。比如说，我们假设 $\text{dim}(V) = k$，其中 $k > n$。

3.  **从假设推导后果 (Consequences of the Assumption):**
    *   根据维度的定义，如果 $\text{dim}(V) = k$，那么必然存在一个由 $k$ 个向量组成的**基**。我们称这个基为 $B = \{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$。
    *   根据基的定义，这个集合 $B$ 中的 $k$ 个向量是**线性无关的**。

4.  **找到矛盾 (Finding the Contradiction):**
    *   现在我们有两个关于空间 $V$ 的事实：
        a.  $V$ 由集合 $S$ 中的 $n$ 个向量生成。
        b.  集合 $B$ 中的 $k$ 个向量位于空间 $V$ 中，并且它们是线性无关的。
    *   因为 $V = \text{Span}(S)$，所以空间 $V$ 中的**每一个**向量（包括基向量 $\mathbf{u}_i$）都可以被表示为 $S$ 中向量的线性组合。
    *   这意味着，我们有 $k$ 个线性无关的向量（集合 $B$），而这 $k$ 个向量中的每一个都是由另外 $n$ 个向量（集合 $S$）线性组合而成的。
    *   这直接触发了我们之前证明过的一个结论：如果你用 $n$ 个向量去构造 $k$ 个新向量，而 $k > n$，那么这 $k$ 个新向量**必然是线性相关的**。
    *   因此，集合 $B = \{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$ **必须是线性相关的**。

5.  **得出结论 (Conclusion):**
    *   我们的推导结果（集合 $B$ 是线性相关的）与我们从假设中得到的事实（集合 $B$ 是一个基，因此是线性无关的）产生了直接的**矛盾**。
    *   这个矛盾的根源在于我们最初的**假设**：“空间的维度大于 $n$”。
    *   因此，这个假设是错误的。

**所以，一个由 $n$ 个向量生成的向量空间的维度，必然小于或等于 $n$。证明完毕。**