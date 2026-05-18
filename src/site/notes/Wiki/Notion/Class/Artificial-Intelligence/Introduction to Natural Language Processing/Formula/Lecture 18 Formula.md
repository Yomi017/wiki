---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-18-formula/"}
---



# Lecture 18 Formula: Synthetic Data Metrics

## Core Formula Map

- Synthetic data 公式少，但评估模板很重要。
- 核心是 correctness、complexity、diversity、fidelity 的判定。

## Formula Details

### Synthetic Data Filtering Score

**Formula**

$$
Score(x,y)=\alpha Correctness+\beta Complexity+\gamma Diversity+\delta Fidelity
$$

**Meaning**：$\alpha,\beta,\gamma,\delta$ 是加权策略，不一定课件固定给出。

**Intuition**：把多个质量维度合成可排序/过滤的评分。

**When to Use**：设计 synthetic data pipeline。


### Fidelity Check

**Formula**

$$
Generated(y)\subseteq Evidence(x)
$$

**Meaning**：生成回答中的事实应被源证据支持。

**Intuition**：防止从 KG/triple/retrieved text 外扩出未提供事实。

**When to Use**：判断 faithful vs hallucinated synthetic answer。

**Exam / Homework Trap**：正确但未被源证据支持的内容，对 fidelity 来说仍可能算不忠实。


## Derivation / Proof Notes

Lecture 18 更偏概念：公式文件用判定模板服务考试短答/设计题。

Synthetic data pipeline 可以写成一个过滤/重采样过程：

$$
D_{syn}=\{(x,y): Score(x,y)\ge \tau\}
$$

其中 $\tau$ 是保留阈值。常见质量维度：

$$
Score=\alpha Correctness+\beta Complexity+\gamma Diversity+\delta Fidelity
$$

四个维度的直觉：

- **Correctness**：答案是否对。
- **Complexity**：样本是否足够难，能提供训练信号。
- **Diversity**：题型、领域、表达是否多样，避免 mode collapse。
- **Fidelity**：如果有 evidence/context，答案是否被证据支持。

概念题中要区分 correctness 和 fidelity：一个答案可能事实正确，但如果它没有被给定 evidence 支持，在 grounded QA/RAG setting 下仍然 fidelity 差。

## Exam / Homework Traps

- synthetic data 可能 mode collapse、data leakage、bias amplification。
- LLM-as-judge 也会出错，不能当绝对真值。
