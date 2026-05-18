---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-09-formula/"}
---



# Lecture 9 Formula: Machine Translation and IBM Model 1

## Core Formula Map

- MT 可写成 noisy channel：fluency × faithfulness。
- IBM Model 1 用 uniform alignment 和 lexical translation probability 建模。

## Formula Details

### Noisy Channel MT

**Formula**

$$
E^*=\arg\max_E P(E|F)=\arg\max_E P(F|E)P(E)
$$

**Meaning**：$E$ 是目标英文，$F$ 是源语言句子。

**Intuition**：$P(E)$ 管 fluency，$P(F|E)$ 管 faithfulness。

**When to Use**：MT objective、信达雅解释。

**Derivation / Proof**：由 Bayes rule 得到，$P(F)$ 与候选 $E$ 无关所以省略。


### IBM Model 1 Joint

**Formula**

$$
P(F,A|E)=P(J|I)P(A|I,J)\prod_{j=1}^J P(f_j|e_{a_j})
$$

**Meaning**：$A$ 是 alignment，$a_j$ 指第 $j$ 个 foreign word 对齐到哪个 English word。

**Intuition**：先生成长度和 alignment，再由对齐英文词生成每个 foreign word。

**When to Use**：IBM Model 1 probability calculation。


### Marginal Translation Probability

**Formula**

$$
P(F|E)=\sum_A P(F,A|E)
$$

**Meaning**：对所有可能 alignment 求和。

**Intuition**：alignment 未观测，所以要 marginalize。

**When to Use**：理解 EM/alignment uncertainty。


### Uniform Alignment

**Formula**

$$
P(A|I,J)=\frac{1}{(I+1)^J}
$$

**Meaning**：$I+1$ 包含 NULL；每个 foreign word 有 $I+1$ 个对齐选择。

**Intuition**：IBM Model 1 不建模位置偏好，alignment 均匀。

**When to Use**：Homework IBM Model 1 计算。


## Derivation / Proof Notes

Noisy channel 来自 Bayes rule；IBM Model 1 的主要简化是词独立翻译和 uniform alignment。

Noisy channel MT 从 Bayes rule 来：

$$
P(E|F)=\frac{P(F|E)P(E)}{P(F)}
$$

decode 时 $F$ 已经固定，所以：

$$
E^*=\arg\max_EP(F|E)P(E)
$$

这里 $P(E)$ 是 language model，负责 fluency；$P(F|E)$ 是 translation model，负责 fidelity。

IBM Model 1 对每个 foreign word $f_j$ 引入 alignment $a_j$：

$$
a_j\in\{0,1,\ldots,I\}
$$

其中 0 常表示 NULL。若 alignment prior uniform：

$$
P(A|I,J)=\prod_{j=1}^{J}\frac{1}{I+1}=\frac{1}{(I+1)^J}
$$

translation probability 要对所有 alignments 求和：

$$
P(F|E)=\sum_A P(F,A|E)
$$

做题若给定 alignment，就算 joint；若没给 alignment，就要 marginalize 或说明需要 sum over alignments。

## Exam / Homework Traps

- $P(E)$ 是 fluency，不是 $P(E|F)$。
- IBM Model 1 允许 many-to-one，但不直接建模 phrase/many-to-many。
