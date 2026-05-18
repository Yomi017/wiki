---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/proof/categorical-mle-derivation/"}
---


### 故事的起点：为什么 MLE 会变成 count ratio

给定 iid observations $D=\{x_1,\ldots,x_n\}$，每个 $x$ 属于 $K$ 个类别之一。令第 $i$ 类出现 $c_i$ 次，参数为：

$$
\theta_i=P(X=i),\qquad \sum_{i=1}^K\theta_i=1
$$

目标是证明：

$$
\hat{\theta}_i=\frac{c_i}{n}
$$

### 推理过程

Likelihood 为：

$$
L(\theta)=\prod_{i=1}^K\theta_i^{c_i}
$$

取 log：

$$
\ell(\theta)=\sum_i c_i\log\theta_i
$$

加入约束：

$$
\mathcal{J}(\theta,\lambda)=\sum_i c_i\log\theta_i+\lambda\left(1-\sum_i\theta_i\right)
$$

求偏导：

$$
\frac{\partial\mathcal{J}}{\partial\theta_i}=\frac{c_i}{\theta_i}-\lambda=0
$$

所以：

$$
\theta_i=\frac{c_i}{\lambda}
$$

代回归一化约束：

$$
\sum_i\theta_i=\frac{1}{\lambda}\sum_i c_i=\frac{n}{\lambda}=1
$$

因此 $\lambda=n$，得到：

$$
\hat{\theta}_i=\frac{c_i}{n}
$$

### Exam Focus

MLE 的 count ratio 来自带约束优化；约束 $\sum_i\theta_i=1$ 必须写出来。

### 和 NLP 公式的连接

Unigram MLE 是 categorical MLE 的直接应用：类别是词表中的 token，$c_i=c(w_i)$，$n=N$，所以：

$$
\hat{P}(w)=\frac{c(w)}{N}
$$

Bigram MLE 是“固定历史后的 categorical MLE”：给定前词 $w_1$，候选后继词 $w_2\in V$ 构成一个 categorical distribution：

$$
\hat{P}(w_2|w_1)=\frac{c(w_1,w_2)}{\sum_{w'\in V}c(w_1,w')}=\frac{c(w_1,w_2)}{c(w_1)}
$$

HMM transition/emission 也一样：

$$
\hat{a}_{ij}=\frac{C(i\to j)}{C(i)},\qquad
\hat{b}_{i}(o)=\frac{C(i\ emits\ o)}{C(i)}
$$

所以这条证明是 Lecture 1 后所有 count-based probability formula 的母版。
