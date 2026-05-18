---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-01-formula/"}
---



# Lecture 1 Formula: Probability Foundations and MLE

## Core Formula Map

- 概率公理、条件概率与 Bayes rule 是后面所有概率模型的基础。
- 期望、方差、categorical/multinomial 把文本计数问题写成统计估计问题。
- MLE 用 likelihood 最大化解释 unigram、bigram、HMM 参数估计。

## Formula Details

### Probability Axioms

**Formula**

$$
P(\Omega)=1,\qquad 0\le P(A)\le 1,\qquad P(A\cup B)=P(A)+P(B)-P(A\cap B)
$$

**Meaning**：$\Omega$ 是样本空间，$A,B$ 是事件。

**Intuition**：概率质量总和为 1；事件概率不能为负；并集要减掉交集，避免重复计算。

**When to Use**：判断概率表达是否合法，或解释 union/intersection 概率。

**Exam / Homework Trap**：如果 $A,B$ mutually exclusive，才有 $P(A\cap B)=0$，否则不能直接相加。


### Conditional Probability and Bayes Rule

**Formula**

$$
P(A|B)=\frac{P(A\cap B)}{P(B)},\qquad P(B|A)=\frac{P(A|B)P(B)}{P(A)}
$$

**Meaning**：$P(A|B)$ 是观察到 $B$ 后 $A$ 的概率；Bayes rule 把 posterior 写成 likelihood、prior、marginal。

**Intuition**：条件概率是“缩小样本空间”；Bayes rule 是用 evidence 更新 belief。

**When to Use**：Naive Bayes、HMM decoding、MT noisy channel、reward/preference 概率解释。

**Derivation / Proof**：由 $P(A\cap B)=P(A|B)P(B)=P(B|A)P(A)$ 两边相等直接得到 Bayes rule。


### Expectation and Variance

**Formula**

$$
\mathbb{E}[X]=\sum_x p(X=x)x,\qquad Var[X]=\mathbb{E}[(X-\mathbb{E}[X])^2]
$$

**Meaning**：$X$ 是随机变量；$p(X=x)$ 是取值概率。

**Intuition**：期望是加权平均；方差衡量随机变量围绕期望的波动。

**When to Use**：理解 reward expectation、policy gradient、sampling variance。

### Bernoulli and Binomial

**Formula**

$$
X\sim Bernoulli(p),\qquad P(X=1)=p,\quad P(X=0)=1-p
$$

$$
Y=\sum_{i=1}^{n}X_i,\quad X_i\overset{iid}{\sim}Bernoulli(p),\qquad P(Y=k)=\binom{n}{k}p^k(1-p)^{n-k}
$$

**Meaning**：$p$ 是一次试验成功概率；$Y$ 是 $n$ 次独立 Bernoulli 中成功次数；$\binom{n}{k}$ 统计哪 $k$ 次成功。

**Intuition**：Bernoulli 是“一次二选一”，Binomial 是“重复 $n$ 次后数成功次数”。NLP 中二分类标签、reward 是否通过、某 token 是否被 mask 都可先抽象成 Bernoulli。

**When to Use**：判断二分类概率模型、理解 binary classifier 的 likelihood、解释 independent trials。

**Derivation / Proof**：某个固定成功/失败序列的概率是 $p^k(1-p)^{n-k}$；能产生 $k$ 次成功的位置组合有 $\binom{n}{k}$ 种，所以相乘得到 binomial probability。

**Exam / Homework Trap**：Binomial 要求试验独立同分布；如果每次成功概率不同，就不能直接用 $\binom{n}{k}p^k(1-p)^{n-k}$。


### Categorical and Multinomial

**Formula**

$$
X\sim Categorical(\theta_1,\ldots,\theta_K),\qquad P(X=i)=\theta_i,\qquad \sum_{i=1}^K\theta_i=1
$$

$$
P(c_1,\ldots,c_K)=\frac{n!}{\prod_i c_i!}\prod_{i=1}^{K}\theta_i^{c_i},\qquad \sum_i c_i=n
$$

**Meaning**：Categorical 是一次 $K$ 类选择；Multinomial 是 $n$ 次 categorical 后的 count vector。

**Intuition**：词表上的 next-token prediction 本质上就是在巨大词表上输出 categorical distribution；训练语料里的 token counts 则对应 multinomial observations。

**When to Use**：unigram MLE、softmax 输出解释、HMM transition/emission count table。

**Exam / Homework Trap**：Categorical 是一次抽样，Multinomial 是多次抽样后的计数；两者参数都是同一组 $\theta_i$，但随机变量形式不同。


### Categorical MLE

**Formula**

$$
\hat{\theta}_i=\frac{c_i}{n}
$$

**Meaning**：$\theta_i=P(X=i)$，$c_i$ 是第 $i$ 类观察次数，$n=\sum_i c_i$。

**Intuition**：最可能解释观察数据的 categorical 参数就是经验频率。

**When to Use**：unigram MLE、tag transition/emission MLE、count-based 概率估计。

**Derivation / Proof**：完整证明见 [[Wiki/Notion/Class/Artificial-Intelligence/Introduction to Natural Language Processing/Proof/Categorical MLE Derivation\|Categorical MLE Derivation]]。

**Exam / Homework Trap**：必须满足 $\sum_i\theta_i=1$；不能对每个类别单独最大化。


## Derivation / Proof Notes

MLE 的核心逻辑：写 likelihood，取 log，加概率和为 1 的约束，用 Lagrange multiplier 求解。

对 categorical MLE，设第 $i$ 类出现 $c_i$ 次：

$$
L(\theta)=\prod_i\theta_i^{c_i},\qquad \ell(\theta)=\sum_i c_i\log\theta_i
$$

加约束 $\sum_i\theta_i=1$：

$$
\mathcal{J}(\theta,\lambda)=\sum_i c_i\log\theta_i+\lambda(1-\sum_i\theta_i)
$$

一阶条件给出：

$$
\frac{\partial\mathcal{J}}{\partial\theta_i}=\frac{c_i}{\theta_i}-\lambda=0
\quad\Rightarrow\quad
\theta_i=\frac{c_i}{\lambda}
$$

再代回归一化约束：

$$
\sum_i\theta_i=\frac{\sum_i c_i}{\lambda}=\frac{n}{\lambda}=1
\quad\Rightarrow\quad
\lambda=n
$$

所以 $\hat{\theta}_i=c_i/n$。这就是为什么后面所有 count-based MLE 都长得像“某事件次数 / 条件总次数”。

## Exam / Homework Traps

- posterior / prior / likelihood / marginal 的角色要分清。
- categorical 是一次多类抽样；multinomial 是多次抽样后的 count vector。
- MLE 是让数据概率最大，不是带 prior 的 MAP。
