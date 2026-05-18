---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-17-formula/"}
---



# Lecture 17 Formula: DPO, GRPO, Preference Optimization

## Core Formula Map

- DPO 直接用 preference pairs 优化 policy/reference log ratio。
- GRPO 用同一 prompt 下多回答 reward 的组内 baseline。

## Formula Details

### DPO Logit and Loss

**Formula**

$$
z=\beta\left[(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x))-(\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))\right],\qquad L_{DPO}=-\log\sigma(z)
$$

**Meaning**：$\beta$ 控制偏离 reference 的强度。

**Intuition**：提高 winner 相对 reference 的 logprob，同时降低 loser 相对 reference 的 logprob。

**When to Use**：DPO computation/code question。

**Derivation / Proof**：完整证明见 [[Wiki/Notion/Class/Artificial-Intelligence/Introduction to Natural Language Processing/Proof/DPO Derivation\|DPO Derivation]]。


### Implicit Reward

**Formula**

$$
r_\theta(x,y)=\beta\left(\log\pi_\theta(y|x)-\log\pi_{ref}(y|x)\right)+\beta\log Z(x)
$$

**Meaning**：$Z(x)$ 是 partition function。

**Intuition**：pairwise difference 中同一 prompt 的 $\log Z(x)$ 会抵消。

**When to Use**：解释 DPO 为什么不需要显式 reward model/partition function。


### GRPO Group Advantage

**Formula**

$$
A_i=\frac{r_i-\bar r}{s_r},\qquad \bar r=\frac{1}{G}\sum_{i=1}^G r_i
$$

**Meaning**：$G$ 是同一 prompt 采样回答数；$\bar r$ 是组内 baseline。

**Intuition**：用同组平均替代 critic，比较同一问题下回答优劣。

**When to Use**：GRPO advantage calculation。


## Derivation / Proof Notes

DPO 从 KL-regularized RLHF optimal policy 与 Bradley-Terry preference 推出。

DPO 的核心 logit 可以拆成两组 log ratio：

$$
z=\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
$$

如果 $z$ 增大，$\sigma(z)$ 增大，loss $-\log\sigma(z)$ 下降。因此 DPO 更新方向是：

1. 提高 winner 相对于 reference 的 log probability。
2. 降低 loser 相对于 reference 的 log probability。
3. 强度由 $\beta$ 控制。

GRPO 对同一个 prompt 采样 $G$ 个 answers，计算 rewards：

$$
r_1,\ldots,r_G
$$

组内标准化 advantage：

$$
A_i=\frac{r_i-\bar r}{s_r}
$$

这里 $\bar r$ 是同组平均，$s_r$ 是同组标准差或尺度项。直觉：同一个问题下，回答之间可直接比较；这减少对额外 value/critic model 的依赖。

KTO/SimPO/DAPO 这些扩展不用死背公式，但要知道它们在处理 DPO/GRPO 的局限：paired data 稀缺、长度偏好、reference 依赖、exploration/entropy collapse。

## Exam / Homework Traps

- DPO 不需要 PPO rollout。
- GRPO baseline 来自同一 prompt 的 group，不是无关样本。
