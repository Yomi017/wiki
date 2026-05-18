---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-10-formula/"}
---



# Lecture 10 Formula: HMM Alignment, Search, BLEU, Seq2Seq Attention

## Core Formula Map

- HMM alignment 加入 locality/jump model。
- Decoding search 用 beam/A* 等近似找高分翻译。
- Attention 用 decoder state 查询 encoder states。

## Formula Details

### HMM Alignment

**Formula**

$$
P(F,A|E)=P(J|I)\prod_j P(a_j|a_{j-1},I)P(f_j|e_{a_j})
$$

**Meaning**：$P(a_j|a_{j-1},I)$ 是 jump/locality model。

**Intuition**：相邻 foreign words 通常对齐到相近 English positions。

**When to Use**：HMM alignment vs IBM Model 1。


### A* Search Score

**Formula**

$$
f(p)=g(p)+h(p)
$$

**Meaning**：$g(p)$ 是 partial translation 已得分，$h(p)$ 是未来估计。

**Intuition**：已完成质量 + 未完成部分乐观估计。

**When to Use**：search 比较题。


### BLEU Modified Precision

**Formula**

$$
p_n=\frac{\sum_{\text{ngram}}\min(Count_{cand},Count_{ref})}{\sum_{\text{ngram}}Count_{cand}}
$$

**Meaning**：candidate n-gram 命中被 reference count clip。

**Intuition**：防止重复输出高频词虚增 precision。

**When to Use**：BLEU 计算、短句陷阱。

**Exam / Homework Trap**：只输出 `the` 可能 unigram precision 高，但翻译很差。


### BLEU Geometric Mean and Brevity Penalty

**Formula**

$$
BLEU=BP\cdot \exp\left(\sum_{n=1}^{N}w_n\log p_n\right)
$$

$$
BP=
\begin{cases}
1,& c>r\\
\exp(1-r/c),& c\le r
\end{cases}
$$

**Meaning**：$p_n$ 是 modified n-gram precision；$w_n$ 常取平均权重；$c$ 是 candidate length，$r$ 是 reference length。

**Intuition**：modified precision 防止重复词刷分；brevity penalty 防止输出极短句子拿高 precision。

**When to Use**：BLEU 计算题、解释短候选为什么会被惩罚。

**Exam / Homework Trap**：BLEU 主要看 precision，不直接看 recall；短句惩罚是对这个缺陷的补丁。


### Seq2Seq Attention Context

**Formula**

$$
e_{t,i}=score(s_t,h_i),\qquad \alpha_{t,i}=softmax(e_{t,i}),\qquad c_t=\sum_i\alpha_{t,i}h_i
$$

**Meaning**：$s_t$ 是 decoder state，$h_i$ 是 encoder hidden state。

**Intuition**：decoder 每一步动态选择 source positions。

**When to Use**：attention shape/meaning 题。


## Derivation / Proof Notes

BLEU modified precision 的 clipping 是常考细节；attention 是 query-key-value 直觉的前身。

Seq2Seq attention 的计算顺序：

1. decoder 当前 state $s_t$ 与每个 encoder state $h_i$ 打分：

$$
e_{t,i}=score(s_t,h_i)
$$

2. 对 source positions 做 softmax：

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_j\exp(e_{t,j})}
$$

3. 用权重求 context vector：

$$
c_t=\sum_i\alpha_{t,i}h_i
$$

4. decoder 用 $c_t$ 和 $s_t$ 预测下一个 target token。

BLEU modified precision 的核心是 clipping：

$$
Count_{clip}(g)=\min(Count_{cand}(g),Count_{ref}(g))
$$

如果 candidate 反复输出同一个高频词，未 clipping 的 precision 会虚高；clipping 后该词最多贡献 reference 中出现的次数。

## Exam / Homework Traps

- beam 太小会 prune 掉正确候选。
- attention 不消除 encoder 计算成本。
