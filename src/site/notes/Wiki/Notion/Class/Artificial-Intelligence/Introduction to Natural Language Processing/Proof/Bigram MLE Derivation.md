---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/proof/bigram-mle-derivation/"}
---


### 故事的起点：我们的目标

在 [[Wiki/Notion/Class/Artificial-Intelligence/Introduction to Natural Language Processing/Introduction to NLP\|Introduction to NLP]] 的 n-gram language model 中，bigram 模型把完整历史近似为只看前一个词：

$$
P(w_i|w_1,\ldots,w_{i-1})\approx P(w_i|w_{i-1})
$$

课件中直接给出 MLE 估计：

$$
\hat{P}(w_i|w_{i-1})
=\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
$$

我们要证明的是：这个 count ratio 不是凭直觉硬写出来的，而是从 **maximum likelihood estimation (MLE)** 推出来的。

### 推理过程：把 bigram 估计变成一个条件多项分布问题

#### 第1步：固定前一个词，把问题局部化

先固定前一个词为 $v$。现在我们只关心一件事：

> 在已经看到 $v$ 的条件下，下一个词是每个 $w\in V$ 的概率是多少？

定义参数：

$$
\theta_w=P(w|v)
$$

对同一个前词 $v$ 来说，所有可能后继词的概率构成一个 categorical distribution，所以它们必须满足：

$$
\sum_{w\in V}\theta_w=1,\qquad \theta_w\ge 0
$$

**逻辑**：我们把整套 bigram 参数拆成很多个独立的小问题。每个前词 $v$ 对应一行条件概率分布；这一行内部的概率和必须等于 1。

#### 第2步：写出训练语料的 likelihood

设训练语料中 bigram $(v,w)$ 出现的次数为：

$$
c(v,w)
$$

如果 $v$ 后面接 $w$ 出现一次，模型给这次观察的概率就是 $\theta_w$。如果它出现 $c(v,w)$ 次，就贡献：

$$
\theta_w^{c(v,w)}
$$

把所有可能后继词的贡献乘起来，得到 likelihood：

$$
L(\theta)=\prod_{w\in V}\theta_w^{c(v,w)}
$$

**逻辑**：MLE 的目标是让训练语料最可能出现。出现次数越多的后继词，对 likelihood 的影响越大。

#### 第3步：取 log，把乘法变成加法

直接最大化乘积不方便，所以取 log-likelihood：

$$
\ell(\theta)
=\log L(\theta)
=\sum_{w\in V}c(v,w)\log\theta_w
$$

因为 $\log$ 是单调递增函数，最大化 $L(\theta)$ 等价于最大化 $\ell(\theta)$。

**逻辑**：log 不改变最大值位置，只是让推导从乘法变成加法，方便求导。

#### 第4步：加入概率和为 1 的约束

我们不能随便让每个 $\theta_w$ 变大，因为它们必须满足：

$$
\sum_{w\in V}\theta_w=1
$$

用 Lagrange multiplier $\lambda$ 写出带约束目标：

$$
\mathcal{J}(\theta,\lambda)
=
\sum_{w\in V}c(v,w)\log\theta_w
+\lambda\left(1-\sum_{w\in V}\theta_w\right)
$$

**逻辑**：这一项的作用是强迫参数仍然是一组合法概率。没有这个约束，likelihood 会倾向于把概率无限推大，问题就不成立。

#### 第5步：对每个参数求偏导

对每个 $\theta_w$ 求偏导，并令其为 0：

$$
\frac{\partial \mathcal{J}}{\partial \theta_w}
=
\frac{c(v,w)}{\theta_w}-\lambda
=0
$$

因此：

$$
\frac{c(v,w)}{\theta_w}=\lambda
$$

整理得到：

$$
\theta_w=\frac{c(v,w)}{\lambda}
$$

**逻辑**：最优解里，每个词的概率 $\theta_w$ 和它的出现次数 $c(v,w)$ 成正比；出现越多，MLE 给它的概率越大。

#### 第6步：用归一化约束求出 $\lambda$

把 $\theta_w=\frac{c(v,w)}{\lambda}$ 代回概率和为 1 的约束：

$$
\sum_{w\in V}\theta_w
=
\sum_{w\in V}\frac{c(v,w)}{\lambda}
=
\frac{1}{\lambda}\sum_{w\in V}c(v,w)
=1
$$

所以：

$$
\lambda=\sum_{w\in V}c(v,w)
$$

而 $\sum_{w\in V}c(v,w)$ 正是所有以 $v$ 为前一个词的 bigram 总数，也就是：

$$
Count(v)
$$

因此：

$$
\lambda=Count(v)
$$

**逻辑**：$\lambda$ 在这里扮演归一化常数。它把所有后继词的 count 压回一组和为 1 的概率。

#### 第7步：得到最终 MLE

代回 $\theta_w=\frac{c(v,w)}{\lambda}$：

$$
\hat{\theta}_w
=
\frac{c(v,w)}{Count(v)}
$$

也就是：

$$
\hat{P}(w|v)
=
\frac{Count(v,w)}{Count(v)}
$$

把 $v$ 换回 $w_{i-1}$，把后继词 $w$ 换回 $w_i$，就得到 bigram 的 MLE：

$$
\hat{P}(w_i|w_{i-1})
=
\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
$$

### 总结：公式的逻辑链条

1. **固定前词 $v$**：把 bigram estimation 拆成“给定 $v$ 后，下一个词的分布”。
2. **写 likelihood**：每个观察到的 bigram $(v,w)$ 贡献一次 $\theta_w$。
3. **取 log-likelihood**：把乘积转成求和，方便求导。
4. **加归一化约束**：保证 $\sum_w P(w|v)=1$。
5. **求导并归一化**：得到概率正比于 count，归一化后就是 count ratio。

所以，bigram MLE 的本质是：

$$
\text{probability}
=
\frac{\text{how many times this event happened}}
{\text{how many times this condition happened}}
$$

### Exam Focus

* 分子 $Count(w_{i-1},w_i)$ 是这个 bigram 出现的次数。
* 分母 $Count(w_{i-1})$ 是所有以前词 $w_{i-1}$ 开头的 bigram 总数。
* 分母不是整个语料 token 总数。
* 如果 $Count(w_{i-1},w_i)=0$，MLE 会给 0 概率，这正是后面需要 Laplace smoothing 的原因。
