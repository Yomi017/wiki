---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-22-formula/"}
---



# Lecture 22 Formula: Mixture of Experts

## Core Formula Map

- MoE 增加总参数，但每个 token 只激活少量 experts。
- 公式重点是 router probability、top-k、load balancing。

## Formula Details

### Router Probability

**Formula**

$$
p_i=softmax(W_rh)_i
$$

**Meaning**：$h$ 是 token hidden state，$p_i$ 是路由到 expert $i$ 的概率。

**Intuition**：router 为每个 token 选择最合适 expert。

**When to Use**：MoE routing calculation。


### Top-k MoE Output

**Formula**

$$
y=\sum_{i\in TopK(p)} \tilde{p}_i E_i(h)
$$

**Meaning**：$E_i$ 是 expert，$\tilde p_i$ 是 top-k 内重归一化权重。

**Intuition**：只激活少数 experts，节省 compute。

**When to Use**：Top-1/Top-2 routing 题。


### Load Balancing Loss

**Formula**

$$
L_{aux}\propto \sum_i f_iP_i
$$

**Meaning**：$f_i$ 是实际 routed token fraction，$P_i$ 是平均 router probability。

**Intuition**：鼓励 token 不要全部挤到少数 expert。

**When to Use**：load-balance code understanding。

**Exam / Homework Trap**：$P_i$ 可微；$f_i$ 来自 hard routing，通常当常量或近似处理。


### Active Parameters

**Formula**

$$
ActiveParams\ll TotalParams
$$

**Meaning**：每个 token 只用被选中的 experts。

**Intuition**：MoE 用更大总容量，但控制每 token 计算量。

**When to Use**：MoE scaling conceptual question。


## Derivation / Proof Notes

routing collapse 是 MoE 主要风险；load-balance loss 用来缓解。

MoE FFN 可以写成：

$$
y=\sum_{i\in TopK(p)}\tilde{p}_iE_i(h)
$$

其中：

$$
p=softmax(W_rh)
$$

Top-k 之后要在被选中的 experts 内重新归一化：

$$
\tilde{p}_i=\frac{p_i}{\sum_{j\in TopK(p)}p_j}
$$

Load balancing loss 里常出现两个量：

$$
f_i=\frac{\#\text{tokens routed to expert }i}{\#\text{tokens}}
$$

$$
P_i=\frac{1}{T}\sum_{t=1}^{T}p_i(x_t)
$$

$f_i$ 来自 hard routing，通常不可微或近似处理；$P_i$ 来自 router softmax，可微。辅助 loss 鼓励二者不要集中在少数 experts。

MoE quiz 常见陷阱：某个 expert 当前 token 没被选中，不代表它未来 token 不能被选中；routing 是逐 token 动态决定的。

## Exam / Homework Traps

- Top-2 routing 更稳但通信/计算更贵。
- MoE 通常替换 Transformer FFN block。
