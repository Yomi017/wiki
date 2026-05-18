---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-11-formula/"}
---



# Lecture 11 Formula: Transformer and Attention

## Core Formula Map

- Transformer 用 self-attention 并行建模 token 间关系。
- 核心公式是 scaled dot-product attention、multi-head shape、LayerNorm 和 causal mask。

## Formula Details

### Scaled Dot-Product Attention

**Formula**

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Meaning**：$Q$ queries，$K$ keys，$V$ values，$d_k$ 是 key dimension。

**Intuition**：query 和 key 点积给 attention logits，softmax 后对 values 加权平均。

**When to Use**：Transformer attention 公式、代码 shape tracing。

**Exam / Homework Trap**：缩放 $\sqrt{d_k}$ 用来避免 dot product 随维度变大导致 softmax 饱和。


### Multi-head Shapes

**Formula**

$$
q=xW_Q,\ k=xW_K,\ v=xW_V,\qquad q\in\mathbb{R}^{B\times nh\times T\times hs}
$$

**Meaning**：$B$ batch，$T$ sequence length，$nh$ heads，$hs$ head size。

**Intuition**：每个 head 在子空间里独立做 attention。

**When to Use**：`q @ k.transpose(-2,-1)` shape tracing。


### Positional Embedding

**Formula**

$$
x_i=e(w_i)+p_i
$$

**Meaning**：$e(w_i)$ 是 token embedding；$p_i$ 是第 $i$ 个位置的位置向量。

**Intuition**：self-attention 本身对输入顺序不敏感，必须给 token 注入位置信息。否则两个相同 token 在不同位置会有相同初始表示。

**When to Use**：解释 “Trust is what builds trust” 中两个 trust 为什么需要不同表示；判断 Transformer 架构组件。

**Exam / Homework Trap**：positional embedding/encoding 不是为了增加词义，而是为了让模型知道顺序和相对/绝对位置。


### Causal Mask

**Formula**

$$
att_{i,j}=-\infty\quad \text{for }j>i
$$

**Meaning**：未来位置 logits 被设为 $-\infty$。

**Intuition**：softmax 后未来位置概率为 0，保证 autoregressive。

**When to Use**：decoder-only Transformer training/inference。


### LayerNorm

**Formula**

$$
LayerNorm(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

**Meaning**：$\mu,\sigma^2$ 在 hidden dimension 内计算。

**Intuition**：稳定每个 token hidden vector 的尺度。

**When to Use**：LayerNorm computational question。


### Residual Update

**Formula**

$$
x_{l+1}=x_l+F(LN(x_l))
$$

**Meaning**：$F$ 可以是 attention block 或 MLP block。

**Intuition**：残差让信息和梯度更容易跨层传播。

**When to Use**：Transformer architecture component 判断题。


## Derivation / Proof Notes

Attention shape 推导和 causal mask 是代码理解高频点。

假设输入 hidden states：

$$
x\in\mathbb{R}^{B\times T\times C}
$$

设 $C=nh\cdot hs$，线性投影后 reshape/transposition：

$$
q,k,v\in\mathbb{R}^{B\times nh\times T\times hs}
$$

代码里的核心乘法：

$$
q @ k.transpose(-2,-1)
$$

shape 为：

$$
[B,nh,T,hs] @ [B,nh,hs,T]\to [B,nh,T,T]
$$

这就是 attention score matrix：每个 head 中，每个 query position 对所有 key positions 的打分。

缩放项的直觉：如果 $q,k$ 的各维方差近似为 1，则 dot product 的方差会随 $d_k$ 增大：

$$
Var(q^Tk)\approx d_k
$$

除以 $\sqrt{d_k}$ 后 variance 回到稳定量级，softmax 不容易因为 logits 太大而饱和。

Causal mask 的代码直觉：

$$
score_{i,j}=-\infty\quad(j>i)
$$

因为：

$$
softmax(-\infty)=0
$$

所以位置 $i$ 不能 attend to 未来位置 $j>i$。训练时可并行算整个 $T\times T$ matrix，但 mask 保证信息流仍是 autoregressive。

## Exam / Homework Traps

- training 可并行算所有位置，但 causal mask 禁止看未来；inference 仍 token-by-token。
- self-attention 本身无序，需 positional encoding。
