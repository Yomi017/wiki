---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-24-formula/"}
---



# Lecture 24 Formula: Score-Based and Text Diffusion

## Core Formula Map

- Score function 指向 log density 增长最快方向。
- Text diffusion 可在离散 token、embedding/latent 或 block level 上做。

## Formula Details

### Score Function

**Formula**

$$
s(x)=\nabla_x\log p(x)
$$

**Meaning**：$s(x)$ 是数据密度 log-prob 对输入的梯度。

**Intuition**：指向更高概率区域的方向。

**When to Use**：score matching、Langevin dynamics。


### Gaussian Conditional Score

**Formula**

$$
\nabla_{x_t}\log q(x_t|x_0)=-\frac{x_t-x_0}{\sigma^2}
$$

**Meaning**：$q(x_t|x_0)=\mathcal{N}(x_0,\sigma^2I)$。

**Intuition**：离 clean point 越远，score 越强地拉回去。

**When to Use**：score derivation question。


### Langevin Dynamics

**Formula**

$$
x_{k+1}=x_k+\frac{\eta}{2}s_\theta(x_k)+\sqrt{\eta}z_k,\qquad z_k\sim\mathcal{N}(0,I)
$$

**Meaning**：$\eta$ 是 step size，$z_k$ 是噪声。

**Intuition**：沿 score 往高密度移动，同时加噪保持采样多样性。

**When to Use**：score-based inference。


### Classifier Guidance

**Formula**

$$
\nabla_x\log p(x|c)=\nabla_x\log p(x)+\nabla_x\log p(c|x)
$$

**Meaning**：$c$ 是控制条件。

**Intuition**：生成模型 score 加上分类器条件梯度，引导样本满足属性。

**When to Use**：controlled generation。


## Derivation / Proof Notes

Score matching 避免直接知道 normalized $p(x)$；只学 gradient direction。

Score function：

$$
s(x)=\nabla_x\log p(x)
$$

它不是概率本身，而是“往哪里走会让 log density 增大”的方向。

对 Gaussian conditional：

$$
q(x_t|x_0)=\mathcal{N}(x_0,\sigma^2I)
$$

log density 忽略常数项：

$$
\log q(x_t|x_0)=-\frac{1}{2\sigma^2}\|x_t-x_0\|^2+C
$$

对 $x_t$ 求梯度：

$$
\nabla_{x_t}\log q(x_t|x_0)=-\frac{x_t-x_0}{\sigma^2}
$$

所以 noisy point 离 clean point 越远，score 拉回去的方向越强。

Text diffusion 的额外难点：token 是离散 ID，不天然有连续距离。常见路线包括 discrete mask diffusion、embedding/latent space diffusion、以及 block diffusion。Block diffusion 试图在 autoregressive 的可扩展推理和 diffusion 的 self-correction 之间折中。

## Exam / Homework Traps

- 离散文本不能直接套连续 Gaussian，需要 mask/embedding/latent/block 等处理。
