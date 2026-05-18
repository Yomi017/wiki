---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-14-formula/"}
---



# Lecture 14 Formula: LoRA and QLoRA

## Core Formula Map

- PEFT 假设适配更新位于低秩子空间。
- LoRA 冻结 base weight，只训练低秩 adapter；QLoRA 进一步量化 base model。

## Formula Details

### LoRA Update

**Formula**

$$
W'=W+\Delta W,\qquad \Delta W=BA,\qquad B\in\mathbb{R}^{k\times r},\ A\in\mathbb{R}^{r\times d}
$$

**Meaning**：$r\ll \min(d,k)$ 是 rank。

**Intuition**：用低秩矩阵近似完整权重更新，减少训练参数。

**When to Use**：LoRA 参数量计算。


### LoRA Parameter Count

**Formula**

$$
\#params=r(k+d)\quad \text{vs.}\quad kd
$$

**Meaning**：full fine-tuning 训练 $kd$ 个参数，LoRA 只训练 $r(k+d)$。

**Intuition**：当 $r$ 很小，训练参数大幅下降。

**When to Use**：memory / trainable parameter calculation。


### QLoRA Memory Estimate

**Formula**

$$
Memory\approx \frac{N\times bits}{8}
$$

**Meaning**：$N$ 是参数量，bits 是每参数存储/训练开销估计。

**Intuition**：把 frozen base model 量化，adapter 仍可训练。

**When to Use**：65B model memory comparison。


## Derivation / Proof Notes

LoRA 初始常让一个矩阵为 0，使初始 $\Delta W=0$，不破坏 base model。

如果原始线性层是：

$$
y=Wx,\qquad W\in\mathbb{R}^{k\times d}
$$

full fine-tuning 要训练 $kd$ 个权重。LoRA 冻结 $W$，只训练：

$$
\Delta W=BA,\qquad B\in\mathbb{R}^{k\times r},\quad A\in\mathbb{R}^{r\times d}
$$

前向计算变为：

$$
y=(W+BA)x=Wx+B(Ax)
$$

trainable parameters：

$$
kr+rd=r(k+d)
$$

当 $r\ll \min(k,d)$ 时，$r(k+d)\ll kd$。例如 $k=d=4096,r=8$：

$$
kd=16,777,216,\qquad r(k+d)=65,536
$$

QLoRA 的内存题常用近似：

$$
Memory_{base}\approx \frac{N\cdot bits}{8}
$$

但完整训练显存还包括 optimizer states、gradients、activations；所以这个公式主要用于估算 frozen quantized base model 的权重存储。

## Exam / Homework Traps

- LoRA 可合并进权重用于推理，不一定增加推理延迟。
- QLoRA 是量化 frozen base + 训练 adapter。
