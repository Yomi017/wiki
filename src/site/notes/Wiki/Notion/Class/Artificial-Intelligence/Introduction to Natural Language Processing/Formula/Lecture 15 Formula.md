---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-15-formula/"}
---



# Lecture 15 Formula: RLHF, Reward Model, Bradley-Terry

## Core Formula Map

- RLHF 用 preference data 训练 reward model，再优化 policy。
- Bradley-Terry 把两个回答的 reward 差转成 preference probability。

## Formula Details

### Bradley-Terry Preference

**Formula**

$$
P(y_w\succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))
$$

**Meaning**：$y_w$ winner，$y_l$ loser，$r$ 是 reward score。

**Intuition**：只看分数差；winner 比 loser 高越多，偏好概率越大。

**When to Use**：reward model probability calculation。


### Reward Model Loss

**Formula**

$$
L_{RM}=-\log\sigma(r_w-r_l)
$$

**Meaning**：$r_w,r_l$ 是 winner/loser rewards。

**Intuition**：最大化人工偏好 pair 的概率。

**When to Use**：RLHF reward model training。


### Policy Gradient

**Formula**

$$
\nabla_\theta \mathbb{E}_{y\sim\pi_\theta}[r(y)]=\mathbb{E}_{y\sim\pi_\theta}[r(y)\nabla_\theta\log\pi_\theta(y)]
$$

**Meaning**：$\pi_\theta$ 是 policy，$r$ 是 reward。

**Intuition**：log-derivative trick 把不可微 sampling 的梯度转为 logprob 梯度。

**When to Use**：policy gradient derivation。


### KL-Regularized RLHF

**Formula**

$$
\max_\theta \mathbb{E}[r(x,y)]-\beta KL(\pi_\theta(\cdot|x)\|\pi_{ref}(\cdot|x))
$$

**Meaning**：$\pi_{ref}$ 是 reference policy。

**Intuition**：既追求高 reward，也避免偏离原模型太远导致 reward hacking。

**When to Use**：RLHF objective conceptual question。


## Derivation / Proof Notes

Policy gradient 证明用 $\nabla p=p\nabla\log p$。

Policy gradient 的核心恒等式：

$$
\nabla_\theta \mathbb{E}_{y\sim\pi_\theta}[r(y)]
=\nabla_\theta\sum_y\pi_\theta(y)r(y)
$$

把梯度移入求和：

$$
=\sum_y r(y)\nabla_\theta\pi_\theta(y)
$$

使用 log-derivative trick：

$$
\nabla_\theta\pi_\theta(y)=\pi_\theta(y)\nabla_\theta\log\pi_\theta(y)
$$

所以：

$$
\nabla_\theta \mathbb{E}[r(y)]
=\sum_y\pi_\theta(y)r(y)\nabla_\theta\log\pi_\theta(y)
=\mathbb{E}_{y\sim\pi_\theta}[r(y)\nabla_\theta\log\pi_\theta(y)]
$$

RLHF 中常加入 KL penalty：

$$
r'(x,y)=r(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

直觉是：reward model 鼓励更讨人喜欢的输出，KL 项阻止模型为了 reward 过度偏离 reference policy。

## Exam / Homework Traps

- reward model 分数可为任意实数；偏好概率经 sigmoid 落在 $[0,1]$。
- KL penalty 控制过度优化 reward。
