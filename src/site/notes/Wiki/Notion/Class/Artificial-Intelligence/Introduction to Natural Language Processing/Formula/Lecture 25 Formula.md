---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-25-formula/"}
---



# Lecture 25 Formula: Agentic Systems, RAG, MCP

## Core Formula Map

- Agentic system 是 LLM + memory/planning/tools/reflection 的循环。
- RAG 公式化为 retrieve + condition + generate。
- MCP 统一工具接口，减少集成复杂度。

## Formula Details

### RAG Decomposition

**Formula**

$$
p(y|x)\approx \sum_{d\in TopK(R(x))}p(y|x,d)p(d|x)
$$

**Meaning**：$R(x)$ 是 retriever，$d$ 是检索文档。

**Intuition**：先找相关证据，再条件生成答案。

**When to Use**：RAG pipeline failure analysis。


### Agent Loop

**Formula**

$$
state_t=(context,memory,tools,goal),\qquad action_t\sim \pi(a|state_t)
$$

**Meaning**：agent 根据当前状态选择工具调用、回答或反思动作。

**Intuition**：普通 LLM call 变成多步决策系统。

**When to Use**：agentic system short answer。


### Reflection / Judge Score

**Formula**

$$
score=Judge(x,y,criteria)
$$

**Meaning**：LLM-as-judge 按 criteria 给回答评分或建议修正。

**Intuition**：反思能改进输出，但 judge 本身也可能偏差/幻觉。

**When to Use**：reflection evaluation question。


### MCP Integration Count

**Formula**

$$
\text{without protocol}=N\times M,\qquad \text{with MCP-style interface}=N+M
$$

**Meaning**：$N$ 是 models，$M$ 是 tools/services。

**Intuition**：统一协议把每个模型到每个工具的两两适配变成两侧接入协议。

**When to Use**：MCP computational question。


## Derivation / Proof Notes

Lecture 25 公式少，重点是把系统组件关系写成可检查的流程和复杂度。

RAG 可以拆成三个评分/概率阶段：

$$
d_{1:k}=R(x)
$$

$$
context=BuildContext(x,d_{1:k})
$$

$$
y\sim p_\theta(y|x,context)
$$

如果答案错，排错也按这三段拆：

1. retrieval 没拿到正确证据；
2. context construction 截断、排序或拼接失败；
3. generator 忽略证据或 hallucinate。

Agent loop 可以写成：

$$
state_t=(goal,context,memory,tool\ results)
$$

$$
action_t\sim\pi(a|state_t)
$$

$$
state_{t+1}=Update(state_t,action_t,observation_t)
$$

MCP 的复杂度直觉：

$$
N\times M
$$

表示每个 model 都要单独适配每个 tool/service；统一协议后，每个 model 和 tool 只要接入协议：

$$
N+M
$$

这不是模型能力公式，而是系统集成复杂度公式。

## Exam / Homework Traps

- RAG 失败可能来自 retrieval、context construction 或 generation。
- tools 要有验证步骤；current information 需要检索。
