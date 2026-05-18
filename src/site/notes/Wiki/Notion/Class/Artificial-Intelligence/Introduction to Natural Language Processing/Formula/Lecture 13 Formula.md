---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-13-formula/"}
---



# Lecture 13 Formula: SFT and Alignment Data

## Core Formula Map

- SFT 是在 prompt-response 数据上继续做 next-token prediction。
- 关键是 loss mask：通常只训练 response，不训练 prompt。

## Formula Details

### SFT Loss

**Formula**

$$
L_{SFT}(\theta)=-\sum_t m_t\log p_\theta(y_t|x,y_{<t})
$$

**Meaning**：$m_t$ 是 loss mask；prompt 常为 0，response 常为 1。

**Intuition**：模型学习如何回答，而不是复述用户 prompt。

**When to Use**：SFT loss mask code question。


### Masked Average Loss

**Formula**

$$
L=\frac{\sum_t m_t\ell_t}{\sum_t m_t}
$$

**Meaning**：$\ell_t$ 是 token loss，$m_t$ 决定是否计入。

**Intuition**：不同样本 response 长度不同，用 mask average 保持尺度稳定。

**When to Use**：计算 masked_loss。


## Derivation / Proof Notes

SFT 数学上仍是 NLL，但数据分布从 raw text 变成 instruction-response。

对一条 prompt-response 样本：

$$
x=(prompt),\qquad y=(response)
$$

拼接后做 next-token prediction，但用 mask 控制哪些 token 进入 loss：

$$
L=\frac{\sum_t m_t[-\log p_\theta(z_t|z_{<t})]}{\sum_t m_t}
$$

其中 $z$ 是拼接后的完整序列。常见设置：

$$
m_t=0\quad \text{for prompt tokens},\qquad m_t=1\quad \text{for response tokens}
$$

若 prompt loss weight 设为小正数，例如 0.1，则：

$$
m_t=
\begin{cases}
0.1,& prompt\\
1,& response
\end{cases}
$$

直觉是：prompt 权重为 0 时，模型专注学习如何回答；给 prompt 一点权重可作为 regularization，降低遗忘风险。

## Exam / Homework Traps

- prompt loss weight 不一定必须为 0，但常见设置是只训 response。
- SFT 数据质量和轨迹信息会影响 coding-agent 能力。
