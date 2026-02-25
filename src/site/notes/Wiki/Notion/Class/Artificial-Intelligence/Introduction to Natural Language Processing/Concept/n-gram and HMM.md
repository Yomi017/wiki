---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/concept/n-gram-and-hmm/"}
---


## Part I: n-gram 模型

- **定义 (Definition):** 一种基于局部统计规律来预测序列中下一个单词的模型，属于对语言**浅层句法 (Shallow Syntax)** 的建模。
- **核心机制 (Mechanism):**
    - **马尔可夫性 (Markovian):** 预测当前词时仅使用邻近的历史信息（Near history），通常仅依赖于前一个或前几个词。
    - **局部结构捕捉:** 擅长捕捉语言中的局部搭配规律。例如：在英语中，冠词 "_the_" 之后极大概率会紧跟一个名词。
- **直觉构建 (Intuition):** 这是在深度学习流行前，NLP 研究者用来处理词序和流利度的主要工具。它通过计算词对或词组在海量文本中的出现频率来模拟语言规则。

---

## Part II: 隐马尔可夫模型 (HMM)

- **定义 (Definition):** 在 NLP 中常用于词性标注 (POS-tagging) 和简单的句法分析，同样属于处理**浅层句法**的线性序列模型。
- **核心机制 (Mechanism):**
    - **序列预测:** 与 n-gram 类似，HMM 也是**马尔可夫式 (Markovian)** 的，依赖于直接的前驱状态来预测当前状态。
    - **语法一致性:** 常用于处理简单的**主谓一致 (Subject-Verb agreement)** 问题。例如：模型可以识别出 "_he_" 后面应该接动词的单三形式 "_drinks_"，而 "_they_" 后面接 "_drink_"。
- **地位演变:** 虽然现代大语言模型 (LLM) 通过先进的架构可以跳过这些中间处理步骤，但 HMM 提供的“状态转移”直觉对于理解语言的线性组织逻辑依然非常重要。

---

## Part III: Common Limitations

- **无法处理长程依赖 (Cannot handle long-range dependencies):** 这是 n-gram 和 HMM 共同的致命缺陷。由于它们只看“邻近历史”，当句子结构变得复杂时就会失效。
    - **失败案例:** "_The **books** that I bought yesterday **are** expensive._"
    - **分析:** 在这个句子中，复数动词 "_are_" 严格依赖于较远位置的主语 "_books_"。由于中间隔了定语从句，主语和谓语之间的“物理距离”超出了 n-gram 和 HMM 的捕捉范围，这类**深层句法 (Deep syntax)** 问题需要更高级的模型（如 CFG 或现代 Transformer）来解决。
