---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-21-formula/"}
---



# Lecture 21 Formula: Compression: Quantization, Pruning, Distillation

## Core Formula Map

- Compression 用更少内存/算力保留模型能力。
- 公式重点是 quantization scale、pruning score、distillation loss。

## Formula Details

### Uniform Quantization

**Formula**

$$
q=round\left(\frac{x}{S}\right)+Z,\qquad x\approx S(q-Z)
$$

**Meaning**：$S$ 是 scale，$Z$ 是 zero-point。

**Intuition**：把连续权重映射到低 bit integer，再近似还原。

**When to Use**：quantization calculation。


### Symmetric Scale

**Formula**

$$
S=\frac{\max |x|}{2^{b-1}-1}
$$

**Meaning**：$b$ 是 bit width。

**Intuition**：signed integer 最大值对齐到权重绝对最大值。

**When to Use**：INT8/INT4 scale 题。


### WANDA-style Pruning Score

**Formula**

$$
score_{ij}=|W_{ij}|\cdot \|X_j\|
$$

**Meaning**：$W_{ij}$ 是权重，$X_j$ 是对应 activation。

**Intuition**：不仅看权重大小，也看 activation 重要性。

**When to Use**：pruning reasoning question。


### Distillation Loss

**Formula**

$$
L=\alpha L_{task}+(1-\alpha)T^2 KL(p_T^{teacher}\|p_T^{student})
$$

**Meaning**：$T$ 是 temperature。

**Intuition**：student 同时学标签和 teacher softened distribution。

**When to Use**：distillation conceptual formula。


## Derivation / Proof Notes

量化误差、剪枝重要性、蒸馏目标分别对应三类压缩策略。

Uniform quantization 的两步：

$$
q=round\left(\frac{x}{S}\right)+Z
$$

dequantization：

$$
\hat{x}=S(q-Z)
$$

量化误差：

$$
e=x-\hat{x}
$$

一般 $S$ 越大，动态范围越广但分辨率越粗；$S$ 越小，分辨率细但容易 overflow/clipping。

Pruning 题要区分 magnitude pruning 和 activation-aware pruning。只看 $|W_{ij}|$ 可能误删小但常被大 activation 放大的权重；WANDA-style score 用：

$$
score_{ij}=|W_{ij}|\|X_j\|
$$

蒸馏中的 temperature softmax：

$$
p_T(i)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

$T$ 越大，分布越平滑，student 能学到 teacher 对错误类别的相对偏好，而不只是 hard label。

## Exam / Homework Traps

- 小权重不一定不重要；activation 大时贡献可能更大。
- quantization 会节省内存，但可能带来精度损失。
