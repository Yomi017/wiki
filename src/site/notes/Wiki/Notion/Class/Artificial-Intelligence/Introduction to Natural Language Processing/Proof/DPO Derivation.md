---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/proof/dpo-derivation/"}
---


### 故事的起点：DPO 为什么不需要显式 reward model

KL-regularized RLHF 的最优 policy 可写成：

$$
\pi^*(y|x)\propto \pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

整理得到隐式 reward：

$$
r(x,y)=\beta\left(\log\pi^*(y|x)-\log\pi_{ref}(y|x)\right)+\beta\log Z(x)
$$

### 推理过程

Preference model 用 Bradley-Terry：

$$
P(y_w\succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))
$$

把隐式 reward 代入差值。因为 winner 和 loser 来自同一个 prompt $x$，所以 $\beta\log Z(x)$ 抵消：

$$
r_w-r_l=\beta\left[(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x))-(\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))\right]
$$

于是 DPO loss 为：

$$
L_{DPO}=-\log\sigma(r_w-r_l)
$$

### Exam Focus

DPO 的关键是 pairwise difference 让 partition function cancel；因此不需要显式计算 $Z(x)$。

### 梯度直觉

DPO 的 logit 可写成：

$$
z=\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
$$

loss：

$$
L_{DPO}=-\log\sigma(z)
$$

若 winner 相对 reference 的 log probability 增大，$z$ 增大，loss 下降；若 loser 相对 reference 的 log probability 增大，$z$ 减小，loss 上升。

因此可以把 DPO 记成一句话：**increase the policy/reference ratio of the chosen answer and decrease that of the rejected answer**。

### 常见易错点

DPO 不是普通 SFT。SFT 只最大化 reference answer 的 likelihood；DPO 明确比较 winner 和 loser，并且比较的是 policy 相对 reference policy 的变化。
