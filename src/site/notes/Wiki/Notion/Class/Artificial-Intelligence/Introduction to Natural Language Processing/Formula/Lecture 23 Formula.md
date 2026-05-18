---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-23-formula/"}
---



# Lecture 23 Formula: DDPM and Diffusion

## Core Formula Map

- Diffusion forward process 逐步加噪；reverse process 学会去噪生成。
- DDPM 常用噪声预测 MSE loss。

## Formula Details

### Forward Noising

**Formula**

$$
q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
$$

**Meaning**：$\bar{\alpha}_t=\prod_{s=1}^t\alpha_s$。

**Intuition**：任意时间步 $x_t$ 可直接由 clean data 和 Gaussian noise 构造。

**When to Use**：DDPM forward calculation。

**Derivation / Proof**：完整证明见 [[Wiki/Notion/Class/Artificial-Intelligence/Introduction to Natural Language Processing/Proof/DDPM Forward Process Derivation\|DDPM Forward Process Derivation]]。


### Reparameterized Forward Sample

**Formula**

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

**Meaning**：$\epsilon$ 是标准高斯噪声。

**Intuition**：把采样写成 clean signal + noise，方便训练噪声预测网络。

**When to Use**：Homework DDPM numerical question。


### DDPM Noise Prediction Loss

**Formula**

$$
L=\mathbb{E}_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|^2
$$

**Meaning**：$\epsilon_\theta$ 预测加入的噪声。

**Intuition**：如果能预测噪声，就能从 noisy sample 往 clean sample 走。

**When to Use**：DDPM objective question。


### Reverse Sampling Step

**Formula**

$$
p_\theta(x_{t-1}|x_t)=\mathcal{N}(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$

**Meaning**：模型学习反向去噪分布。

**Intuition**：生成从 Gaussian noise 开始，逐步反向采样。

**When to Use**：diffusion algorithm steps。


## Derivation / Proof Notes

训练知道 $x_0$，所以 $q(x_{t-1}|x_t,x_0)$ tractable；推理不知道 $x_0$，需模型近似 reverse。

Forward process 每一步：

$$
q(x_t|x_{t-1})=\mathcal{N}(\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$

令：

$$
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
$$

反复展开后：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

训练时可以随机采样 $t$ 和 $\epsilon$，直接构造 $x_t$，然后训练网络预测噪声：

$$
L=\mathbb{E}_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|^2
$$

为什么预测噪声有用：如果模型知道噪声 $\epsilon$，就能从 noisy sample 中估计 clean signal 的方向。采样时从 $x_T\sim\mathcal{N}(0,I)$ 开始，重复应用 reverse step：

$$
x_T\to x_{T-1}\to\cdots\to x_0
$$

考试要分清：forward process 是预定义的、无需训练；reverse process 是学习出来的。

## Exam / Homework Traps

- inference 不知道真实 $x_0$。
- sampling starts from Gaussian noise。
