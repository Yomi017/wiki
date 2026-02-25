---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/introduction-to-nlp/"}
---

好的，我理解了。你需要一份完整涵盖整个 PDF 内容（从语言学层级到 CYK 算法再到现代视角），并且严格遵守你给出的**层级结构**（Part I, Part II...）以及使用 `$$` 包裹 LaTeX 公式的笔记。

以下是为你重头整理的详细笔记：

# Lecture 6: Introduction to NLP: Grammar, Syntax, and Context-Free Grammars

## Part I: Linguistic Hierarchy & Syntax Basics

*   **Levels of Languages (语言层级):**
    语言处理存在一个从声音到隐含意义的垂直层级。
    *   **Phonology (语音学):** 研究所有声音 (All sounds)。
    *   **Morphology (形态学):** 研究单词的内部结构。例如：*unhappiness* 拆解为 _un-happi-ness_。
    *   **Syntax (句法学):** 研究单词如何组织成短语和句子（Organizing words）。
    *   **Semantics (语义学):** 研究字面含义（Literal meaning）。
    *   **Pragmatics (语用学):** 研究隐含意义，如隐喻、讽刺、幽默和沉默。

*   **Syntax Definition (句法定义):**
    句法是定义单词如何组织成更大单元的规则。
    *   **Legal vs. Illegal:** 规则区分了语言中的合法与非法句子。
    *   **Fluency:** 衡量对句法规则的熟悉程度。婴儿通过习得（Acquisition）掌握，二语学习者通过规则（Rules）显式学习。

*   **Shallow vs. Deep Syntax (浅层与深层句法):**
    *   **Shallow Syntax:** 关注语言的局部结构。
        *   **Examples:** the 后面接名词；主谓一致（*he drinks* vs. *they drink*）。
        *   **Modeling:** [[n-grams and HMM\|n-grams and HMM]] 是马尔可夫模型（Markovian），仅利用近距离历史，无法处理长程依赖。
    *   **Deep Syntax:** 关注全局和长程依赖。
        *   **Example:** "*The **books** that I bought yesterday **are** expensive.*"
        *   **Analysis:** are 取决于复数名词 books，两者之间的距离超出了 n-gram 处理范围。

*   **Constituent (语法成分):**
    作为一个单一语法单元行动的一组词。
    *   **Property 1:** 可以在句中不同位置出现而不改变语义（Position flexibility）。
    *   **Property 2:** 不能在维持语义的情况下被进一步拆解（Indivisibility）。

---

## Part II: Context-Free Grammar (CFG)

*   **Formal Definition (形式化定义):**
    一个上下文无关文法由四个元素组成：
    1.  $N$: 非终结符集合（Non-terminal symbols），如 NP, VP。
    2.  $\Sigma$: 终结符集合（Terminal symbols），即具体的单词。
    3.  $R$: 规则/产生式集合，形式为 $A \rightarrow \beta$，其中 $A \in N$。
    4.  $S$: 起始符号，$S \in N$。

*   **核心短语范畴 (Phrasal Categories)**，这些符号代表由多个词组成的短语单位：
     **$NP$ (Noun Phrase): 名词短语**。句子的主体，通常指代人、事、物。例如：*a flight*（一个航班）, *the big blue bus*（那辆蓝色大巴士）。
    
     **$VP$ (Verb Phrase): 动词短语**。描述动作或状态。例如：*prefer a morning flight*（倾向于早班机）, *disappear*（消失）。
    
     **$PP$ (Prepositional Phrase): 介词短语**。由介词开头，表示地点、时间、方式等。例如：*from Atlanta*（来自亚特兰大）, *in the morning*（在早上）。
    
     **$AP$ (Adjective Phrase): 形容词短语**。以形容词为核心。例如：*least expensive*（最便宜的）。

 *  **核心句子与词类范畴 (Clausal & Word Categories)**
      **$S$ (Sentence): 句子**。文法的最高层级，通常由 $NP$ 和 $VP$ 组成。
      
     **$Aux$ (Auxiliary): 助动词**。协助主要动词构成时态、语态或疑问。例如：*do, does, can, will, have*。在是非问句（Yes-no Questions）中常用到它（如：*Do any of these flights...*）。
     
     **$Det$ (Determiner): 限定词**。放在名词前限定其范围。例如：*a, an, the, this, that*。
     
     **$Nominal$ (名词性成分):** 介于名词和名词短语之间的成分。它比单个名词信息更多，但还没加上限定词。例如：*morning flight* 是一个 $Nominal$，加上 *a* 之后变成 $NP$ (*a morning flight*)。
     
     **$Wh-NP$: 疑问名词短语**。引导疑问句的特殊名词短语，通常包含 *who, what, which* 等。例如：*What airlines*。

*   **Sentence Structures (句子结构规则):**
    *   **Declarative (陈述句):** $$S \rightarrow NP\ VP$$
    *   **Imperative (祈使句):** $$S \rightarrow VP$$
    *   **Yes-no Questions (是非问句):** $$S \rightarrow Aux\ NP\ VP$$
    *   **Wh-structures:** $$S \rightarrow Wh-NP\ VP$$ 或 $$S \rightarrow Wh-NP\ Aux\ NP\ VP\ PP$$

*   **Noun Phrase & Nominals (名词短语与名词性成分):**
    *   **NP Rule:** $$NP \rightarrow (Det) (Card) (Ord) (Quant) (AP) Nominal$$ (括号内为可选成分)。
    *   **Recursion (递归):** 通过限定词规则 $$Det \rightarrow NP's$$ 可以实现深层嵌套，如 "*Denver’s mayor’s mother’s canceled flight*"。
    *   **Nominal Forms:**
        *   $$Nominal \rightarrow Noun$$ (基础形式)
        *   $$Nominal \rightarrow Nominal\ Noun$$ (*morning flight*)
        *   $$Nominal \rightarrow Nominal\ PP$$ (*flight to Boston*)
        *   $$Nominal \rightarrow relative-clause$$ (*flight that serves breakfast*)

*   **Verb Phrase (VP):**
    *   $VP \rightarrow Verb$ (*disappear*)
    *   $VP \rightarrow Verb\ NP$ (*prefer a morning flight*)
    *   $VP \rightarrow Verb\ S$ (含有句法补足语 Sentential complement)

*   **Data Source:** **Penn Treebank** 是常用的解析树标注语料库，用于派生 CFG 或训练解析器。

---

## Part III: Syntactic Parsing & Ambiguity

*   **The Goal of Parsing:**
    给定一个 CFG，为句子分配有效的解析树。即寻找根节点为 $S$、叶子节点为句子单词的树。

*   **Search Strategies (搜索策略):**
    *   **Top-down Search (自顶向下):** 从起始符 $S$ 开始向下扩展，直到匹配单词。
    *   **Bottom-up Search (自底向上):** 从单词开始向上构建非终结符，直到到达 $S$。
    *   **Common Issue:** 存在大量的**重复子问题**。解析树的某些部分（如特定的 NP）会被反复构建，即使最终会导致解析失败（$Dead-end$）。

*   **Parsing Ambiguity (分析歧义性):**
    *   **Attachment Ambiguity (附件歧义):** 修饰成分的归属不明确。
        *   *Example:* "I saw the Grand Canyon flying to New York." (谁在飞？我还是大峡谷？)
    *   **Coordination Ambiguity (并列歧义):** 连词连接的范围不明确。
        *   *Example:* "old men and women" (是“老的老头和老太太”，还是“老头和所有女人”？)
![Wiki/Image/Class/Introdution to NLP/1.png](/img/user/Wiki/Image/Class/Introdution%20to%20NLP/1.png)
---

## Part IV: CYK Algorithm & CNF

*   **Motivation (动机):**
    利用动态规划（Dynamic Programming）缓存子问题的解，避免重复解析。

*   **Chomsky Normal Form (CNF):**
    CYK 算法要求文法必须转换为 CNF 形式：
    1.  $$A \rightarrow BC$$ (右侧为两个非终结符)
    2.  $$A \rightarrow a$$ (右侧为一个终结符)
    *   **Conversion Algorithm (转换算法):**
        *   处理混合右侧：$$A \rightarrow Bc \implies A \rightarrow BC, C \rightarrow c$$
        *   消除单位产生式：$$A \rightarrow B$$
        *   处理长右侧：$$A \rightarrow BCD \implies A \rightarrow XD, X \rightarrow BC$$

*   **CYK Algorithm Mechanics (算法原理):**
    *   使用一个 $(n+1) \times (n+1)$ 的矩阵，其中 $n$ 是单词数。
    *   单元格 $(i, j)$ 包含所有能生成从位置 $i$ 到 $j$ 的成分的非终结符。
    *   **Logic:** 由于满足 CNF，每个成分 $(i, j)$ 都可以被切分为两部分：$(i, k)$ 和 $(k, j)$，其中 $i < k < j$。
    *   **Formula:** 如果 $B \in table[i, k]$ 且 $C \in table[k, j]$，且存在规则 $A \rightarrow BC$，则将 $A$ 加入 $table[i, j]$。

---

## Part V: The Bitter Lesson (现代视角)

*   **Core Concept (Rich Sutton):**
    在 NLP 发展史上，关于“构建人类知识（如句法规则）”与“利用大规模计算发现知识”一直存在争论。
*   **Key Argument:**
    从长远来看，利用大规模数据和可扩展计算的方法（$Data-driven$）总是优于硬编码人类知识的方法。
*   **Shift in Paradigm:**
    *   **Past:** 专注于复杂的句法解析步骤（$Parsing$）。
    *   **Present (LLM):** 通过**下一标记预测**（$Next\ token\ prediction$），模型可以直接学习语言规律。
*   **Conclusion:**
    虽然在 LLM 时代显式构建语法树可能显得不再必要，但学习这些概念对于理解语言结构和建立 NLP 直觉仍然至关重要。