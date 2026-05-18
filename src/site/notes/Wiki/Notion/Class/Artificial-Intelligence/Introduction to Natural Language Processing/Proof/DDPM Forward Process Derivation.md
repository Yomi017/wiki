---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/proof/ddpm-forward-process-derivation/"}
---


### 故事的起点：为什么可以一步采样 $x_t$

DDPM forward process 每步加一点 Gaussian noise：

$$
q(x_t|x_{t-1})=\mathcal{N}(\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$

定义：

$$
\bar{\alpha}_t=\prod_{s=1}^t\alpha_s
$$

### 推理过程

Gaussian 线性组合仍是 Gaussian。反复展开可得到：

$$
q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
$$

因此可重参数化为：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

### Exam Focus

训练时知道 $x_0$，所以能直接构造任意 $x_t$；推理时不知道 $x_0$，必须从 noise 开始逐步 reverse sampling。

### 为什么方差是 $1-\bar{\alpha}_t$

可以把一步 noising 写成重参数化：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

若 $x_{t-1}$ 的 signal 系数是 $\sqrt{\bar{\alpha}_{t-1}}$，代入后 signal 系数变为：

$$
\sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}=\sqrt{\bar{\alpha}_t}
$$

noise 部分是多个独立 Gaussian 的线性组合。独立 Gaussian 方差相加，最终总 noise variance 变成：

$$
1-\bar{\alpha}_t
$$

所以 $x_t$ 可以直接写成 clean signal 加单个标准 Gaussian noise：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

这就是训练时可以随机抽一个 $t$，一步构造 $x_t$ 的原因。
