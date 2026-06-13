---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-07-formula/"}
---



# Lecture 7 Formula: PCFG, Inside/Outside, Best Parse

## Core Formula Map

- PCFG 给 CFG rules 加概率。
- Inside 算某个 non-terminal 生成 span 的总概率；Viterbi PCFG 找最大概率 parse。

## Formula Details

### PCFG Rule Probability

**Formula**

$$
P(t,w|G)=\prod_{r\in t}P(r)
$$

**Meaning**：$t$ 是 parse tree；每条 rule application 的概率相乘。

**Intuition**：context-free 假设下，树概率分解成规则概率乘积。

**When to Use**：计算 parse tree probability。


### Sentence Probability under PCFG

**Formula**

$$
P(w|G)=\sum_t P(t,w|G)
$$

**Meaning**：对所有能生成句子的 parse trees 求和。

**Intuition**：句子可能有多棵 parse tree，概率要 marginalize。

**When to Use**：区分 best parse 和 sentence probability。


### Inside Recurrence

**Formula**

$$
\beta_A(i,j)=\sum_{A\to BC}\sum_{k=i}^{j-1}P(A\to BC)\beta_B(i,k)\beta_C(k+1,j)
$$

**Meaning**：$\beta_A(i,j)$ 是 $A$ 生成 span $i..j$ 的概率。

**Intuition**：和 CYK 类似，但 boolean 变成概率求和。

**When to Use**：Inside algorithm trace。

**Derivation / Proof**：完整证明见 [[Wiki/Notion/Class/Artificial-Intelligence/Introduction to Natural Language Processing/Proof/Inside Algorithm Derivation\|Inside Algorithm Derivation]]。


### Outside and Posterior Span

**Formula**

$$
P(A,i,j|w)\propto \alpha_A(i,j)\beta_A(i,j)
$$

**Meaning**：$\alpha$ 是 span 外部生成概率；$\beta$ 是 span 内部生成概率。

**Intuition**：inside × outside 给出某个 constituent 在整句 parse 中的贡献。

**When to Use**：理解 Outside algorithm、PCFG EM。


### Viterbi PCFG

**Formula**

$$
\delta_A(i,j)=\max_{A\to BC,k}P(A\to BC)\delta_B(i,k)\delta_C(k+1,j)
$$

**Meaning**：$\delta$ 存最大概率，backpointer 存来源。

**Intuition**：把 Inside 的 sum 换成 max，得到最优 parse tree。

**When to Use**：最佳解析树题。


## Derivation / Proof Notes

Inside 是 sum-product over parse trees；Viterbi PCFG 是 max-product。

PCFG rule probability 要按同一个 left-hand side 归一化：

$$
\sum_{\alpha:A\to\alpha}P(A\to\alpha)=1
$$

所以一个 parse tree 的概率是每条规则概率的乘积：

$$
P(t,w|G)=\prod_{r\in t}P(r)
$$

如果一句话有多棵 parse tree，句子概率不是选最大的那棵，而是求和：

$$
P(w|G)=\sum_{t\in Trees(w)}P(t,w|G)
$$

Inside recurrence 是 CYK 的概率版。对每个 rule 和 split，把左子树概率、右子树概率、当前 rule 概率相乘；再对所有可能 rule/split 求和：

$$
\beta_A(i,j)=\sum_{A\to BC}\sum_kP(A\to BC)\beta_B(i,k)\beta_C(k+1,j)
$$

Viterbi PCFG 把求和换成最大值：

$$
\delta_A(i,j)=\max_{A\to BC,k}P(A\to BC)\delta_B(i,k)\delta_C(k+1,j)
$$

所以做题时先问清楚目标：如果问 sentence probability，用 Inside/sum；如果问 most likely parse，用 Viterbi/max。

## Exam / Homework Traps

- PCFG rule 概率通常按同一 LHS 归一化。
- CYK 找可解析性；Inside 算总概率；Viterbi PCFG 找最优树。
