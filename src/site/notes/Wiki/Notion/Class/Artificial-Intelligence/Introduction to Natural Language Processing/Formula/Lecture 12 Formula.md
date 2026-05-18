---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-12-formula/"}
---



# Lecture 12 Formula: Pretraining, BERT, GPT

## Core Formula Map

- Pretraining 学通用语言分布；BERT 用 MLM，GPT 用 autoregressive LM。
- 公式重点是 AR loss 和 MLM mask-only loss。

## Formula Details

### Autoregressive Pretraining

**Formula**

$$
L_{AR}(\theta)=-\sum_{D}\sum_t\log p_\theta(w_t|w_{<t})
$$

**Meaning**：$w_{<t}$ 是左侧上下文。

**Intuition**：训练过程与生成过程一致：从左到右预测下一个 token。

**When to Use**：GPT pretraining objective。


### Masked Language Modeling

**Formula**

$$
L_{MLM}(\theta)=-\sum_{t\in M}\log p_\theta(w_t|x_{\setminus M})
$$

**Meaning**：$M$ 是 masked positions。

**Intuition**：只对被 mask 的位置计算 loss，可使用双向上下文。

**When to Use**：BERT objective、masking ratio 题。


### GPT Batch Slicing

**Formula**

$$
x=tokens[i:i+B],\qquad y=tokens[i+1:i+B+1]
$$

**Meaning**：$x$ 是输入窗口，$y$ 是右移一位的 target。

**Intuition**：每个位置都预测下一个 token。

**When to Use**：code understanding: autoregressive dataset loader。


## Derivation / Proof Notes

BERT/GPT 区别核心是 objective 和可见上下文方向。

Autoregressive objective 的训练样本通常来自同一段 token 序列右移一位：

$$
x=[w_1,\ldots,w_{T-1}],\qquad y=[w_2,\ldots,w_T]
$$

loss 是每个位置的 next-token NLL：

$$
L_{AR}=-\sum_{t=1}^{T-1}\log p_\theta(y_t|x_{\le t})
$$

MLM 只在 mask positions 上计 loss：

$$
L_{MLM}=-\sum_{t\in M}\log p_\theta(w_t|x_{\setminus M})
$$

这意味着 BERT 可以使用左右两边上下文做表示学习，但不自然适合从左到右生成；GPT 训练和生成一致，适合 autoregressive decoding。

Pretraining / mid-training / post-training 的公式区别通常不是网络结构，而是数据分布和 loss mask：pretraining 用 raw corpus，mid-training 用领域语料，post-training 用 instruction/preference 数据。

## Exam / Homework Traps

- BERT MLM 可看双向上下文；GPT 不能看未来。
- pretraining 不等于 alignment，后面还要 SFT/RLHF/DPO。
