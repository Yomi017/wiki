---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-19-formula/"}
---



# Lecture 19 Formula: Scaling Laws and FLOPs

## Core Formula Map

- Scaling laws 连接 model size、data size、compute。
- FLOPs 估算是系统题基础。

## Formula Details

### Training FLOPs Approximation

**Formula**

$$
C\approx 6ND
$$

**Meaning**：$N$ 是参数量，$D$ 是训练 token 数。

**Intuition**：每个 token 训练约涉及 forward/backward，粗略为每参数 6 次操作。

**When to Use**：计算训练 FLOPs。


### Matrix-Vector FLOPs

**Formula**

$$
FLOPs\approx 2mn
$$

**Meaning**：$A\in\mathbb{R}^{m\times n}$ 乘向量。

**Intuition**：乘法和加法各算一次操作。

**When to Use**：FLOPs 基础计算。


### Matrix-Matrix FLOPs

**Formula**

$$
FLOPs\approx 2mnp
$$

**Meaning**：$A\in\mathbb{R}^{m\times n}$，$B\in\mathbb{R}^{n\times p}$。

**Intuition**：输出有 $mp$ 个元素，每个元素做 $n$ 次乘加。

**When to Use**：矩阵乘法计算题。


### Chinchilla-style Compute Tradeoff

**Formula**

$$
C\approx 6ND,\qquad N,D\ \text{must be balanced under fixed }C
$$

**Meaning**：固定 compute 下，模型参数和数据 token 此消彼长。

**Intuition**：过大模型/过少数据或过小模型/过多数据都不 compute-optimal。

**When to Use**：scaling law conceptual question。


## Derivation / Proof Notes

FLOPs 是估算，不等于 wall-clock；通信、内存带宽、利用率都会影响实际速度。

矩阵乘法 FLOPs 推导：

$$
A\in\mathbb{R}^{m\times n},\quad B\in\mathbb{R}^{n\times p}
$$

输出 $C=AB$ 有 $mp$ 个元素。每个元素：

$$
C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}
$$

需要约 $n$ 次乘法和 $n$ 次加法，所以：

$$
FLOPs\approx 2mnp
$$

Training compute 常用粗略估算：

$$
C\approx 6ND
$$

其中 $N$ 是参数量，$D$ 是训练 token 数。这个公式把 forward/backward 的主要 dense matmul 成本揉成一个经验常数，适合数量级估算，不适合精确 profile。

Scaling law 题常考 tradeoff：固定 compute $C$ 下，不能只增大参数 $N$ 或只增大数据 $D$；需要二者平衡，否则会出现 under-trained large model 或 over-trained small model。

## Exam / Homework Traps

- Scaling law 不保证所有 downstream tasks 单调变好。
- Data quality 会改变 token value。
