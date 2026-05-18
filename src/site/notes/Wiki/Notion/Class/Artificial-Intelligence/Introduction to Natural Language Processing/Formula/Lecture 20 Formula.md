---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-20-formula/"}
---



# Lecture 20 Formula: LLM Inference, KV Cache, FlashAttention

## Core Formula Map

- Inference 分 prefill 和 decode；decode 常受 KV cache memory bandwidth 限制。
- FlashAttention 用 tiling 和 online softmax 降低 HBM 访问。

## Formula Details

### KV Cache Size

**Formula**

$$
Size=Batch\times SeqLen\times 2\times Layers\times KVHeads\times HeadDim\times Bytes
$$

**Meaning**：2 表示 K 和 V；Bytes 是每个数的字节数。

**Intuition**：每生成/缓存一个 token，都要为每层保存 key/value。

**When to Use**：KV cache memory calculation。

**Exam / Homework Trap**：MQA/GQA 会减少 KVHeads，不一定等于 query heads。


### Incremental Attention with KV Cache

**Formula**

$$
score_t=\frac{q_tK_{1:t}^T}{\sqrt{d_k}},\qquad y_t=softmax(score_t)V_{1:t}
$$

**Meaning**：$q_t$ 是最新 token 的 query；$K_{1:t},V_{1:t}$ 是历史和当前 token 的 cached keys/values。

**Intuition**：decode 第 $t$ 步只需要新 token 的 query，但要读取所有历史 KV。KV cache 用 memory 换 computation，避免每一步重算历史 K/V。

**When to Use**：解释 naive decoding 为什么重复计算、KV cache 为什么让每步从重算全部历史变成追加一个 token。

**Exam / Homework Trap**：KV cache 降低重复计算，但 decode 仍需读历史 KV，所以长上下文会 memory-bandwidth bound。


### Prefill vs Decode Cost

**Formula**

$$
TTFT\approx T_{prefill},\qquad TPOT\approx T_{decode\ per\ token}
$$

**Meaning**：TTFT 是首 token 延迟，TPOT 是每输出 token 时间。

**Intuition**：prefill 可并行，decode 逐 token 且反复读 KV cache。

**When to Use**：inference systems conceptual question。


### Online Softmax Rescale

**Formula**

$$
m_{new}=\max(m_{old},m_{block}),\qquad d_{new}=d_{old}e^{m_{old}-m_{new}}+d_{block}e^{m_{block}-m_{new}}
$$

**Meaning**：$m$ 是 running max，$d$ 是 softmax denominator。

**Intuition**：不同 block 的 exponentials 必须放到同一个 max 尺度下合并。

**When to Use**：FlashAttention code understanding。


### PagedAttention Block Mapping

**Formula**

$$
logical\ blocks\rightarrow physical\ KV\ blocks
$$

**Meaning**：用 block table 映射逻辑序列块到物理显存块。

**Intuition**：减少 KV cache 内外部碎片，支持动态 batch。

**When to Use**：PagedAttention short answer。


## Derivation / Proof Notes

KV cache 大小公式要看 KV heads；FlashAttention 的核心是少存 attention matrix、分块 online softmax。

KV cache size 单位换算模板：

$$
Size_{bytes}=B\times L\times 2\times N_{layers}\times N_{kvheads}\times d_{head}\times Bytes
$$

如果要转成 GiB：

$$
Size_{GiB}=\frac{Size_{bytes}}{1024^3}
$$

其中 $2$ 是 K 和 V 两份缓存；FP16/BF16 通常 $Bytes=2$。MQA/GQA 的效果体现在 $N_{kvheads}$ 变小，而不是 query heads 一定变小。

FlashAttention online softmax 的必要性来自稳定 softmax：

$$
softmax(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_jx_j
$$

分块计算时每个 block 有自己的 $m_{block}$ 和 denominator $d_{block}$。合并时必须重缩放到新的 running max：

$$
m_{new}=\max(m_{old},m_{block})
$$

$$
d_{new}=d_{old}e^{m_{old}-m_{new}}+d_{block}e^{m_{block}-m_{new}}
$$

所以 quiz 里 “online softmax 必须近似” 是错的：可以 exact online computation，只要维护 running max 和 rescaled denominator。

PagedAttention 的公式感不强，但题目会问“为什么有用”：它把连续 KV cache 拆成 fixed-size blocks，用 block table 映射 logical blocks 到 physical blocks，减少 fragmentation 并支持动态 batching。

## Exam / Homework Traps

- LayerNorm 参数量不影响 KV cache size。
- StreamingLLM 保留 attention sinks + recent tokens。
