---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-08-formula/"}
---



# Lecture 8 Formula: Neural Networks and RNN LM

## Core Formula Map

- 神经网络用可微函数替代 count table。
- RNN 用 hidden state 递归压缩历史。
- BPTT 解释长程梯度消失/爆炸。

## Formula Details

### Neuron and BCE

**Formula**

$$
a=\sigma\left(\sum_i w_ix_i+b\right),\qquad \ell(a,y)=-\log(a^y(1-a)^{1-y})
$$

**Meaning**：$a$ 是 sigmoid 输出，$y$ 是二分类标签。

**Intuition**：线性打分 + 非线性得到概率，BCE 惩罚错误概率。

**When to Use**：神经网络基础、binary classifier。


### MLP Layer

**Formula**

$$
z^{[1]}=W^{[1]}x+b^{[1]},\qquad a^{[1]}=\sigma(z^{[1]})
$$

**Meaning**：$W,b$ 是参数，$\sigma$ 是非线性。

**Intuition**：矩阵乘法一次计算一层所有 hidden units。

**When to Use**：shape tracing、vectorization。


### RNN Recurrence

**Formula**

$$
h_t=\tanh(Wh_{t-1}+Ux_t+b),\qquad o_t=Vh_t+c,\qquad \hat{y}_t=softmax(o_t)
$$

**Meaning**：$h_t$ 保存历史摘要，$x_t$ 是当前输入 embedding。

**Intuition**：同一组参数在每个时间步复用，逐步吸收序列信息。

**When to Use**：RNN language model、POS tagging RNN。


### LM Loss

**Formula**

$$
L=-\sum_{t=1}^T\log p_\theta(x_{t+1}|x_{\le t})
$$

**Meaning**：每个位置预测下一个 token。

**Intuition**：把序列训练转成多个 next-token prediction。

**When to Use**：RNN/Transformer LM loss。


### BPTT Gradient Chain

**Formula**

$$
\frac{\partial L}{\partial h_t}=\frac{\partial L}{\partial h_T}\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}
$$

**Meaning**：梯度要穿过许多 recurrent Jacobians。

**Intuition**：重复矩阵乘法可能让梯度指数级缩小或放大。

**When to Use**：解释 vanishing/exploding gradients。

**Exam / Homework Trap**：$\tanh$ 限制 activation 范围，但不能彻底解决长链梯度问题。


## Derivation / Proof Notes

RNN 的核心推导是展开时间图后做 backprop；长链乘法决定梯度稳定性。

RNN language model 的一步预测通常是：

$$
h_t=\tanh(Wh_{t-1}+Ux_t+b)
$$

$$
o_t=Vh_t+c,\qquad p_t=softmax(o_t)
$$

如果目标是下一个 token $x_{t+1}$，cross entropy loss 为：

$$
\ell_t=-\log p_t[x_{t+1}]
$$

整句 loss：

$$
L=\sum_t\ell_t
$$

BPTT 的关键不是背完整矩阵求导，而是理解梯度要穿过时间链：

$$
\frac{\partial L}{\partial h_t}
=\sum_{s\ge t}\frac{\partial \ell_s}{\partial h_s}
\frac{\partial h_s}{\partial h_t}
$$

其中：

$$
\frac{\partial h_s}{\partial h_t}
=\prod_{k=t+1}^{s}\frac{\partial h_k}{\partial h_{k-1}}
$$

如果这些 Jacobian 的谱半径长期小于 1，梯度消失；长期大于 1，梯度爆炸。这就是为什么长距离依赖对 vanilla RNN 困难。

代码理解题常问 shape：若 $x_t\in\mathbb{R}^{d_x}$，$h_t\in\mathbb{R}^{d_h}$，则：

$$
U\in\mathbb{R}^{d_h\times d_x},\quad
W\in\mathbb{R}^{d_h\times d_h},\quad
V\in\mathbb{R}^{|Vocab|\times d_h}
$$

## Exam / Homework Traps

- 没有非线性，多层 MLP 会坍缩成一个线性层。
- RNN 参数共享 across time。
