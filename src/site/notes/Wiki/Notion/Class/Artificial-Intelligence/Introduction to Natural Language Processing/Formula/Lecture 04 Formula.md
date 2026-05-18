---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-04-formula/"}
---



# Lecture 4 Formula: Forward and Backward Algorithms

## Core Formula Map

- Forward/Backward 用 DP 边缘化所有 hidden state paths。
- Forward 看过去，Backward 看未来，两者可在任意时间点合成 $P(O)$。

## Formula Details

### Forward Definition

**Formula**

$$
\alpha_t(j)=P(o_{1:t},q_t=j)
$$

**Meaning**：$\alpha_t(j)$ 是到时间 $t$ 为止观测前缀且当前 tag 为 $j$ 的联合概率。

**Intuition**：把所有以 $j$ 结尾的历史路径压缩成一个数。

**When to Use**：HMM inference 计算句子概率。


### Forward Recurrence

**Formula**

$$
\alpha_1(j)=\pi_jb_j(o_1),\qquad \alpha_t(j)=\sum_i\alpha_{t-1}(i)a_{ij}b_j(o_t)
$$

**Meaning**：$i$ 枚举前一状态，$j$ 是当前状态。

**Intuition**：到达 $j$ 的路径可能来自任意前一 tag，所以要 sum。

**When to Use**：Forward algorithm trace。

**Derivation / Proof**：按前一状态 $q_{t-1}=i$ 做 total probability expansion。


### Backward Definition and Recurrence

**Formula**

$$
\beta_t(i)=P(o_{t+1:T}|q_t=i),\qquad \beta_T(i)=1,\qquad \beta_t(i)=\sum_j a_{ij}b_j(o_{t+1})\beta_{t+1}(j)
$$

**Meaning**：$\beta_t(i)$ 是从状态 $i$ 出发生成未来观测的概率。

**Intuition**：从句尾往前看未来；空未来概率为 1。

**When to Use**：Backward algorithm trace、EM soft counts。


### Forward-Backward Identity

**Formula**

$$
P(O)=\sum_j\alpha_T(j)=\sum_j\alpha_t(j)\beta_t(j)
$$

**Meaning**：任意时间点 $t$ 都可把过去和未来拼回完整观测概率。

**Intuition**：Forward 覆盖 past/current，Backward 覆盖 future。

**When to Use**：证明题、posterior 计算。

**Derivation / Proof**：完整证明见 [[Wiki/Notion/Class/Artificial-Intelligence/Introduction to Natural Language Processing/Proof/Forward Backward Identity\|Forward Backward Identity]]。

**Exam / Homework Trap**：$\sum_j\beta_1(j)$ 本身通常不等于 $P(O)$，还缺初始概率和第一个 emission。


## Derivation / Proof Notes

Forward recurrence 是 sum-product；Viterbi 会把 sum 换成 max。

Forward recurrence 的推导：

$$
\alpha_t(j)=P(o_{1:t},q_t=j)
$$

按前一状态 $q_{t-1}=i$ 分解：

$$
\alpha_t(j)=\sum_iP(o_{1:t},q_{t-1}=i,q_t=j)
$$

利用 HMM 条件独立性：

$$
P(o_{1:t},q_{t-1}=i,q_t=j)
=P(o_{1:t-1},q_{t-1}=i)P(q_t=j|q_{t-1}=i)P(o_t|q_t=j)
$$

因此：

$$
\alpha_t(j)=\sum_i\alpha_{t-1}(i)a_{ij}b_j(o_t)
$$

Backward recurrence 的推导从未来开始：

$$
\beta_t(i)=P(o_{t+1:T}|q_t=i)
=\sum_j P(q_{t+1}=j,o_{t+1:T}|q_t=i)
$$

拆出下一步 transition 和 emission：

$$
\beta_t(i)=\sum_j a_{ij}b_j(o_{t+1})\beta_{t+1}(j)
$$

做数值题时，Forward table 每个 cell 都是“到这里为止的总概率”；Backward table 每个 cell 是“从这里出发生成未来的总概率”。两者都不是 posterior，posterior 要再除以 $P(O)$。

## Exam / Homework Traps

- Forward 是 summation，不是 max。
- $\beta_T(i)=1$ 是空序列概率，不代表状态概率。
- 复杂度是 $O(TN^2)$。
