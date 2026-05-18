---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-16-formula/"}
---



# Lecture 16 Formula: PPO, KL, TRPO

## Core Formula Map

- PPO 用 clipped ratio 稳定 policy update。
- TRPO 用 KL trust region 约束更新步长。

## Formula Details

### Probability Ratio

**Formula**

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

**Meaning**：$r_t$ 衡量新旧 policy 对同一 action 的概率变化。

**Intuition**：ratio 大于 1 表示新 policy 更喜欢该 action。

**When to Use**：PPO clipping calculation。


### PPO Clipped Objective

**Formula**

$$
L^{CLIP}(\theta)=\mathbb{E}\left[\min(r_t(\theta)A_t,\ clip(r_t(\theta),1-\epsilon,1+\epsilon)A_t)\right]
$$

**Meaning**：$A_t$ 是 advantage。

**Intuition**：限制 policy 一次更新不要把概率改得太激进。

**When to Use**：PPO numerical question。

**Exam / Homework Trap**：$A_t$ 正负会影响 min/clip 的解释。


### KL Trust Region

**Formula**

$$
\max_\theta g^T\Delta\theta\quad \text{s.t.}\quad \frac12\Delta\theta^TF\Delta\theta\le \delta
$$

**Meaning**：$F$ 是 Fisher Information Matrix。

**Intuition**：KL 在局部近似为二次型，约束更新步长。

**When to Use**：TRPO conceptual derivation。


### Natural Gradient Direction

**Formula**

$$
\Delta\theta\propto F^{-1}g
$$

**Meaning**：$g$ 是普通 policy gradient。

**Intuition**：在 KL 几何下修正梯度方向。

**When to Use**：TRPO/natural gradient short answer。


## Derivation / Proof Notes

PPO 是 TRPO 思想的实用近似，用 clipping 替代复杂 constrained optimization。

PPO ratio：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

若 $A_t>0$，说明这个 action 比 baseline 好，希望增加概率，但不要增加太多：

$$
\min(r_tA_t,clip(r_t,1-\epsilon,1+\epsilon)A_t)
$$

当 $r_t>1+\epsilon$ 时，clipped term 固定在 $(1+\epsilon)A_t$，继续增大 action 概率不会增加 objective。

若 $A_t<0$，说明这个 action 不好，希望降低概率，但也不要降太猛。由于乘以负数，min 的行为会翻转，这是 PPO 数值题最常见陷阱。

TRPO 的 trust region 用 KL 约束：

$$
KL(\pi_{\theta_{old}}\|\pi_\theta)\le \delta
$$

局部二阶近似可写成：

$$
\frac12\Delta\theta^TF\Delta\theta\le\delta
$$

于是自然梯度方向出现：

$$
\Delta\theta\propto F^{-1}g
$$

考试短答可以这样说：PPO 用 clipping 近似 trust region，避免每一步 policy 更新过大。

## Exam / Homework Traps

- clip 不是简单截断 loss，而是截断 ratio 后取 min。
- KL 小不代表输出完全一样，只表示分布距离受控。
