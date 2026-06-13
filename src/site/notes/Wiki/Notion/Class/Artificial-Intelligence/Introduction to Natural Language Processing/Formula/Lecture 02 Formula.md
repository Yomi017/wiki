---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-02-formula/"}
---



# Lecture 2 Formula: n-gram, PCA, Word2Vec, GloVe, FastText

## Core Formula Map

- n-gram 用短历史近似完整历史，核心是 chain rule + Markov approximation。
- Laplace smoothing 修复零概率，但会重新分配概率质量。
- Word2Vec/GloVe/FastText 把词从 symbolic ID 变成可学习向量。

## Formula Details

### Language Model Chain Rule

**Formula**

$$
P(w_1,\ldots,w_n)=\prod_{i=1}^n P(w_i|w_1,\ldots,w_{i-1})
$$

**Meaning**：$w_i$ 是第 $i$ 个 token；右侧逐步预测每个 token。

**Intuition**：句子概率可分解成一串 next-token 条件概率。

**When to Use**：从完整 LM 过渡到 n-gram、RNN、Transformer 自回归训练。

**Exam / Homework Trap**：chain rule 是精确等式；n-gram approximation 才是假设。


### Bigram MLE

**Formula**

$$
\hat{P}(w_i|w_{i-1})=\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
$$

**Meaning**：$Count(w_{i-1},w_i)$ 是 bigram 次数；$Count(w_{i-1})$ 是以前词开头的 bigram 总数。

**Intuition**：在固定前词的条件分布里，MLE 等于条件频率。

**When to Use**：Homework bigram probability、零概率判断。

**Derivation / Proof**：完整证明见 [[Wiki/Notion/Class/Artificial-Intelligence/Introduction to Natural Language Processing/Proof/Bigram MLE Derivation\|Bigram MLE Derivation]]。

**Exam / Homework Trap**：`like math` 和 `likes math` 是不同 bigram；不做 stemming 时不能合并。


### n-gram Markov Approximation

**Formula**

$$
P(w_i|w_1,\ldots,w_{i-1})\approx P(w_i|w_{i-n+1},\ldots,w_{i-1})
$$

$$
P(w_i|w_{i-2},w_{i-1})=\frac{Count(w_{i-2},w_{i-1},w_i)}{Count(w_{i-2},w_{i-1})}
$$

**Meaning**：第一行是一般 n-gram 假设；第二行是 trigram MLE。

**Intuition**：完整历史太长，几乎每种历史都很稀疏；n-gram 只保留最近 $n-1$ 个词，把无法估计的问题变成 count table。

**When to Use**：判断 bigram/trigram 条件历史、从句子中数 n-gram、解释参数稀疏。

**Derivation / Proof**：固定历史 $h=(w_{i-n+1},\ldots,w_{i-1})$ 后，所有候选后继词形成一个 categorical distribution。对这个条件分布做 MLE，就得到 $\hat{P}(w|h)=Count(h,w)/Count(h)$。

**Exam / Homework Trap**：bigram 的历史长度是 1，trigram 的历史长度是 2；不要把 $Count(w_{i-2},w_{i-1},w_i)$ 除以 $Count(w_{i-1})$。


### Laplace Smoothing

**Formula**

$$
P_L(w_2|w_1)=\frac{c(w_1,w_2)+1}{c(w_1)+|V|}
$$

**Meaning**：$V$ 是可能被预测的词表；$|V|$ 是候选后继词数量。

**Intuition**：给每个可能后继词加 1 个虚拟计数，所以总分母加 $|V|$。

**When to Use**：Homework Q2 smoothing、避免整句概率被 unseen bigram 变成 0。

**Derivation / Proof**：$\sum_{w_2\in V}(c(w_1,w_2)+1)=c(w_1)+|V|$。

**Exam / Homework Trap**：通常 `<S>` 只作为条件，不作为预测目标；`</S>` 会作为预测目标。


### Unigram Laplace Smoothing

**Formula**

$$
P_L(w)=\frac{c(w)+1}{N+|V|}
$$

**Meaning**：$N$ 是训练语料中被预测 token 的总数；$|V|$ 是候选词表大小。

**Intuition**：unigram 没有条件历史，所以分母不是某个前词的 count，而是整个语料 token 总数。

**When to Use**：Homework 中如果问 unigram probability 或 sentence probability under unigram model，用这个公式。

**Derivation / Proof**：给每个词表项加 1 后，总计数从 $N=\sum_{w\in V}c(w)$ 变成 $\sum_{w\in V}(c(w)+1)=N+|V|$。

**Exam / Homework Trap**：unigram smoothing 和 bigram smoothing 的分母不同：前者是 $N+|V|$，后者是 $c(w_1)+|V|$。


### PCA Projection

**Formula**

$$
X_c=X-\mu,\qquad C=\frac{1}{n-1}X_c^TX_c,\qquad Z=X_cW_k
$$

**Meaning**：$X_c$ 是中心化数据；$C$ 是协方差矩阵；$W_k$ 是前 $k$ 个 principal components。

**Intuition**：找最大方差方向，把高维词向量投影到低维可视化空间。

**When to Use**：词向量可视化、解释 semantic clusters。

**Derivation / Proof**：求 $Cv_j=\lambda_jv_j$，按 $\lambda$ 从大到小选 $v_1,\ldots,v_k$。


### Word2Vec Softmax

**Formula**

$$
P(w_o|w_c)=\frac{\exp(u_o^Tv_c)}{\sum_{w\in V}\exp(u_w^Tv_c)}
$$

**Meaning**：$v_c$ 是 center vector；$u_o$ 是真实 context vector；分母遍历所有候选上下文词。

**Intuition**：点积是匹配分数，softmax 把分数归一化成概率。

**When to Use**：Skip-gram 建模、理解为什么 full softmax 昂贵。

**Exam / Homework Trap**：每个词有两套向量 $v_w,u_w$；不要把 center/context 向量混为一谈。


### Word2Vec NLL and Gradient

**Formula**

$$
L=-u_o^Tv_c+\log\sum_{w\in V}\exp(u_w^Tv_c),\qquad \frac{\partial L}{\partial v_c}=-u_o+\sum_{w\in V}P(w|w_c)u_w
$$

**Meaning**：第一项来自真实上下文；第二项是模型预测分布下的期望上下文向量。

**Intuition**：梯度下降会拉近真实 pair，同时推离模型高估的平均上下文方向。

**When to Use**：推导题、解释 Word2Vec 更新方向。

**Derivation / Proof**：由 $-\log$ softmax 展开得到 NLL；log-sum-exp 求导给出 softmax-weighted expectation。


### Negative Sampling

**Formula**

$$
L_{NS}=-\log\sigma(u_o^Tv_c)-\sum_{k=1}^K\log\sigma(-u_{w_k^-}^Tv_c)
$$

**Meaning**：$w_k^-$ 是负样本；$\sigma$ 是 sigmoid。

**Intuition**：把大词表多分类问题近似成正样本 vs 少量负样本的二分类问题。

**When to Use**：Word2Vec 优化效率、判断 negative sampling 是否等于 full softmax。

**Exam / Homework Trap**：negative sampling 不是精确 softmax，输出不严格是归一化概率。

**Derivation / Proof**：full softmax 的 NLL 对一个正样本是

$$
L=-\log\frac{\exp(u_o^Tv_c)}{\sum_{w\in V}\exp(u_w^Tv_c)}
=-u_o^Tv_c+\log\sum_{w\in V}\exp(u_w^Tv_c)
$$

对 $v_c$ 求导：

$$
\frac{\partial L}{\partial v_c}
=-u_o+
\frac{\sum_{w\in V}\exp(u_w^Tv_c)u_w}{\sum_{w\in V}\exp(u_w^Tv_c)}
=-u_o+\sum_{w\in V}P(w|w_c)u_w
$$

所以梯度下降更新 $v_c\leftarrow v_c-\eta\partial L/\partial v_c$ 会把 $v_c$ 往真实 $u_o$ 的方向推，同时远离模型当前平均预测的上下文方向。


### GloVe Objective

**Formula**

$$
J=\sum_{i,j=1}^{|V|}f(X_{ij})\left(w_i^T\tilde{w}_j+b_i+\tilde{b}_j-\log X_{ij}\right)^2
$$

**Meaning**：$X_{ij}$ 是共现次数；$f$ 是权重函数；$b_i,\tilde b_j$ 吸收频率偏置。

**Intuition**：用向量点积拟合全局共现矩阵的 log count。

**When to Use**：区分 Word2Vec 局部预测训练与 GloVe 全局矩阵拟合。

**Exam / Homework Trap**：GloVe 不是 softmax loss，也不是 negative sampling。


### FastText Subword Composition

**Formula**

$$
v_w=\sum_{g\in G_w}z_g\quad \text{or}\quad v_w=\frac{1}{|G_w|}\sum_{g\in G_w}z_g
$$

**Meaning**：$G_w$ 是词 $w$ 的字符 n-gram 集合；$z_g$ 是子词向量。

**Intuition**：词向量由子词向量组合，形态相近词共享参数。

**When to Use**：解释 morphology、OOV、rare word 表示。


## Derivation / Proof Notes

核心证明包括 Bigram MLE、Laplace 分母、Word2Vec NLL 展开与梯度推导。

Bigram MLE 的证明路线可以记成一句话：**固定前词 $w_1$ 后，后继词 $w_2$ 是一个 categorical distribution**。因此：

$$
\hat{P}(w_2|w_1)=\frac{c(w_1,w_2)}{\sum_{w'\in V}c(w_1,w')}=\frac{c(w_1,w_2)}{c(w_1)}
$$

Laplace smoothing 的分母不是随便加 1，而是给每一个候选后继词都加 1：

$$
\sum_{w_2\in V}(c(w_1,w_2)+1)=\sum_{w_2\in V}c(w_1,w_2)+\sum_{w_2\in V}1=c(w_1)+|V|
$$

PCA 做题步骤：

1. 把每个样本向量减去均值 $\mu$。
2. 计算 covariance matrix $C$。
3. 求 eigenvectors/eigenvalues。
4. 选最大 eigenvalues 对应的前 $k$ 个方向组成 $W_k$。
5. 用 $Z=X_cW_k$ 得到低维表示。

Word2Vec full softmax 的计算瓶颈来自分母：

$$
\sum_{w\in V}\exp(u_w^Tv_c)
$$

如果 $|V|$ 很大，每个正样本更新都要扫描整个词表，所以 negative sampling 用 $K$ 个负样本把成本从 $O(|V|)$ 降成近似 $O(K)$。

## Exam / Homework Traps

- unigram 分母是 token 总数 $N$；bigram 分母是前词 count。
- Laplace 中 $|V|$ 是可预测词表大小，不是句子长度。
- Word2Vec full softmax 分母遍历整个词表；negative sampling 只采少量负例。
