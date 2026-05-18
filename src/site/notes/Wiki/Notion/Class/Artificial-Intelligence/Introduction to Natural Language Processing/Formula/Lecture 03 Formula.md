---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-03-formula/"}
---



# Lecture 3 Formula: POS Tagging and HMM Modeling

## Core Formula Map

- HMM 把词序列作为 observations，把 POS tags 作为 hidden states。
- 两类核心概率：tag-to-tag transition 与 tag-to-word emission。
- Bayes decoding 把 $P(Q|O)$ 转成最大化 $P(O|Q)P(Q)$。

## Formula Details

### HMM Joint Factorization

**Formula**

$$
P(O,Q)=P(Q)P(O|Q)=\pi_{q_1}b_{q_1}(o_1)\prod_{t=2}^T a_{q_{t-1},q_t}b_{q_t}(o_t)
$$

**Meaning**：$O$ 是词序列，$Q$ 是 tag 序列，$a$ 是 transition，$b$ 是 emission。

**Intuition**：每一步先从上一个 tag 转移到当前 tag，再由当前 tag 发射当前词。

**When to Use**：计算给定 tag path 和 word sequence 的 joint probability。


### Transition Probability

**Formula**

$$
a_{ij}=P(q_t=j|q_{t-1}=i)=\frac{C(i\to j)}{C(i)}
$$

**Meaning**：$i,j$ 是 POS tags；$C(i\to j)$ 是 tag bigram count。

**Intuition**：HMM 在 hidden tag 序列上做 bigram Markov assumption。

**When to Use**：估计 tag transition matrix。

**Exam / Homework Trap**：$a_{ij}$ 不等于 $a_{ji}$。


### Emission Probability

**Formula**

$$
b_i(o)=P(o_t=o|q_t=i)=\frac{C(i\to o)}{C(i)}
$$

**Meaning**：$i$ 是 tag，$o$ 是 word。

**Intuition**：给定词性 tag 后，当前词从该 tag 的词分布中生成。

**When to Use**：给定 tag sequence 计算 words emitted probability。

**Exam / Homework Trap**：n-gram 的 $P(w_i|w_{i-1})$ 不是 HMM emission；emission 是 $P(word|tag)$。


### MAP Decoding Objective

**Formula**

$$
Q^*=\arg\max_Q P(Q|O)=\arg\max_Q P(O|Q)P(Q)
$$

**Meaning**：$P(O)$ 对所有候选 $Q$ 相同，所以 argmax 中可省略。

**Intuition**：同时考虑词与 tag 的匹配程度和 tag 序列自身是否合理。

**When to Use**：解释为什么不能逐词选 most frequent tag。


## Derivation / Proof Notes

Lecture 3 主要证明是 Bayes decoding：$P(Q|O)=P(O|Q)P(Q)/P(O)$，因 $O$ 固定所以省略 $P(O)$。

给定 tag sequence $Q=q_1,\ldots,q_T$ 和 word sequence $O=o_1,\ldots,o_T$，HMM 使用两个独立性假设：

$$
P(q_t|q_{1:t-1})\approx P(q_t|q_{t-1})
$$

$$
P(o_t|q_{1:T},o_{1:t-1})\approx P(o_t|q_t)
$$

所以 joint probability 展开为：

$$
P(O,Q)=P(q_1)P(o_1|q_1)\prod_{t=2}^{T}P(q_t|q_{t-1})P(o_t|q_t)
$$

Supervised MLE 时，如果训练语料已经给出 POS tags，就把 hard counts 放进 categorical MLE：

$$
\hat{a}_{ij}=\frac{C(q_{t-1}=i,q_t=j)}{C(q_{t-1}=i)},\qquad
\hat{b}_i(o)=\frac{C(q_t=i,o_t=o)}{C(q_t=i)}
$$

做题时最容易混淆的是：transition 的条件是前一个 tag；emission 的条件是当前 tag；n-gram 的条件才是前一个 word。

## Exam / Homework Traps

- HMM emission 是 tag 发射词，不是词预测词。
- transition matrix 行和为 1；emission matrix 每个 tag 下对词表归一化。
