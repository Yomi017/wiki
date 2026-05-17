---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/introduction-to-nlp/"}
---


# AIAA 4051 Introduction to Natural Language Processing

> 详细主题版笔记。来源为 Lecture 1-25 课件逐页知识集、现有 Obsidian 笔记和 Homework 复习点；不保留逐页课件标题，但每页知识点都被吸收进对应主题。

## How to Use This Note

* 先读每讲 `Part I: Lecture Map` 把握路线。
* 再读 `Part II` 的详细解释，理解定义、直觉、公式、推导和例子。
* 最后用 `Part III: Concept Coverage from Lecture Materials` 对照课件查漏补缺。
* Homework 相关内容集中写在 Lecture 2-7 的 `Homework / Exam Connection`。

# Lecture 1: 课程介绍、数学基础、概率基础与 MLE

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；为什么学习 NLP：自然语言的重要性；为什么学习 NLP：生成式 AI 与跨学科影响；为什么学习 NLP：职业与能力投资；先修要求；后续可以做什么；课程 logistics：成绩与课堂规则；课程 logistics：Office hour 与 GPU
  * NLP 大图景；微积分回顾：多元函数、导数与梯度；线性代数回顾：向量与矩阵；线性代数回顾：范数、角度与点积；概率回顾：样本空间、事件、随机变量；概率公理；联合分布；概率定理：补集、全概率、条件概率
  * Bayes Rule；Bayes Rule 的展开形式与条件版本；期望与方差；Bernoulli 与 Binomial 分布；Categorical 与 Multinomial 分布；独立性假设；最大似然估计 MLE 的直觉；MLE 在 AI/NLP 中的地位
  * Categorical 分布的 MLE 形式化；NLP 模型训练的一般流程；扩展阅读；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Foundations

### 1. NLP 的统一视角：把语言问题写成概率问题

NLP 课程里出现的模型看起来很多：n-gram、word vector、HMM、CFG、PCFG、RNN、Transformer、RLHF、diffusion、agent。它们的共同点是：都试图把语言对象转化成可计算的概率、向量或结构。

最基础的抽象是：

1. 输入是一段文本、一个 token 序列、一个 prompt，或者一个多模态/工具环境状态。
2. 模型内部有参数 $\theta$。
3. 模型输出一个概率分布，比如下一个词、POS tag、parse tree、translation、answer、action。
4. 训练就是调节 $\theta$，让真实数据在模型下概率更高。

因此，NLP 的核心不是“背模型名字”，而是要能看懂每个模型如何回答三个问题：

* **Random variable 是什么？** 词、tag、tree、alignment、action 还是 reward。
* **Probability 怎么分解？** 独立性假设、Markov 假设、context-free 假设、attention factorization。
* **Parameter 怎么学？** MLE、gradient descent、EM、SFT、RL、pretraining。

### 2. 梯度、向量和矩阵为什么是 NLP 的地基

在现代 NLP 中，文本最终会被转成向量。词向量、hidden state、attention score、logit、reward 都是向量或矩阵。训练时，损失函数通常是多变量函数：

$$
L(\theta)
$$

梯度告诉我们参数往哪个方向改能让 loss 下降：

$$
\nabla_\theta L
=\left(\frac{\partial L}{\partial \theta_1},\ldots,\frac{\partial L}{\partial \theta_n}\right)
$$

矩阵乘法是神经模型高效计算的核心。一个线性层可以写成：

$$
z=Wx+b
$$

attention 的核心也依赖矩阵：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这就是为什么 Lecture 1 复习 calculus、linear algebra、probability：后面每个 lecture 都在复用这些工具。

### 3. 概率分布、MLE 与语言模型训练

如果 $X$ 是一个 categorical random variable，表示下一个 token，那么模型要估计：

$$
P_\theta(X=w)
$$

给定观测计数 $c_i$，categorical distribution 的 MLE 是：

$$
\hat{\theta_i}=\frac{c_i}{n}
$$

这就是 unigram / bigram 模型中“数频率”的数学来源。复杂模型不能直接数所有可能组合，于是改用神经网络输出概率，再最大化 log-likelihood：

$$
\max_\theta \sum_{(x,y)\in D}\log P_\theta(y|x)
$$

或者最小化 negative log-likelihood：

$$
\min_\theta -\sum_{(x,y)\in D}\log P_\theta(y|x)
$$

### 4. Exam Focus

Lecture 1 的考试点通常不是复杂计算，而是概念区分：

* categorical distribution 是一次多类抽样；multinomial 是多次抽样后的计数。
* MLE 的直觉是“让观测数据最可能”，不是随便取平均。
* 独立性是假设，不是真理；它用表达能力换可估计性。
* Bayes rule 中 posterior、prior、likelihood、marginal 的角色要分清。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 1，课程教师与联系方式。
* **直觉：** 这一页本身没有技术内容，但确认课程主题是自然语言处理（NLP），后续所有模型都围绕“如何把文本建模为概率分布并进行预测”展开。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. 为什么学习 NLP：自然语言的重要性

* **定义 / 内容：** 自然语言可以类比为人类社会操作系统的“编程语言”。它承担三类功能：交流接口、保存知识与文化、支撑人类思维/计算。
* **直觉：** NLP 的对象不是普通字符串，而是人类社会中承载意义、知识和推理的语言系统。因此 NLP 既是工程问题，也是认知、文化和社会问题。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 3. 为什么学习 NLP：生成式 AI 与跨学科影响

* **定义 / 内容：** NLP 的影响在生成式 AI 时代迅速扩大；语言本身具有生成性；NLP 改变多个学科，同时带来正面与负面影响；课程连接前沿研究。
* **直觉：** 生成式 AI 的核心能力之一就是根据上下文生成语言，所以 NLP 已经从传统“文本分类/标注任务”扩展到推理、对话、工具调用、多学科研究等方向。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 4. 为什么学习 NLP：职业与能力投资

* **定义 / 内容：** NLP 是时间投资价值较高的方向：AI 公司需求高、跨领域应用多、能深化对概率、优化、编程、机器学习等知识的理解。
* **直觉：** NLP 技术不是孤立技能，它需要数学、数据、算法和工程能力。学好 NLP 能把大学里学的基础知识整合到真实 AI 系统中。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 5. 先修要求

* **定义 / 内容：** 需要微积分、概率统计、Python 编程、算法与数据结构、LLM prompting，以及主动好奇的学习心态。
* **直觉：** NLP 模型通常包含概率建模、梯度优化、矩阵运算和动态规划等内容。课程强调“不需要一开始就精通”，但要能在课堂中主动补齐。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 6. 后续可以做什么

* **定义 / 内容：** 可继续学习多模态 AI、具身 AI、Responsible AI、AI+X；也可做研究、构造数据集、训练评估模型、写论文、做 NLP 应用或创业。
* **直觉：** 这页给出课程目标的上层路线：从掌握 NLP 基础，到能参与研究和构建应用。后续每个模型都可以理解为进入这些方向的工具。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 7. 课程 logistics：成绩与课堂规则

* **定义 / 内容：** 课程信息在 Canvas；随堂 quiz 占 24%；quiz 闭卷短测；有期末考试、组队研究项目和 poster session；有反馈机制。
* **直觉：** quiz 占比很高，说明每节课的核心概念、公式和算法步骤都要即时掌握。后续知识集中保留 quiz 页也是为了帮助复习考试重点。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 8. 课程 logistics：Office hour 与 GPU

* **定义 / 内容：** Office hours 见 Canvas/大纲；课程提供有限但足够的 GPU 计算 credits；优秀项目可继续获得建议和算力支持。
* **直觉：** NLP 尤其是神经网络和 LLM 训练/推理需要算力。课程项目不仅是作业，也可能发展为研究结果。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 9. NLP 大图景

* **定义 / 内容：** NLP 模型包括 n-gram、word vectors、text classifier、POS tagger、parser、entity/relation extractor、translator、chatbot 等。它们大多是带参数 θ 的统计模型，用来定义文本上的概率分布。
* **直觉：** 统一视角是：数据 → 拟合模型 → 输出预测。预测结果可以是类别、POS tag、parse tree、BIO tag、关系、翻译或对话。优化算法与神经网络结构是训练这些模型的关键。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. 微积分回顾：多元函数、导数与梯度

* **定义 / 内容：** NLP/ML 中处理的是多变量函数；梯度 $∇f$ 是由各变量偏导数组成的向量。
* **直觉：** 模型训练就是调参数，使损失函数下降。若参数是高维向量/矩阵，就需要梯度告诉我们每个参数方向上如何更新。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. 线性代数回顾：向量与矩阵

* **定义 / 内容：** 向量、矩阵、张量是 NLP 表示文本和模型参数的基本数据结构。
* **直觉：** 词向量、隐藏状态、注意力矩阵、神经网络权重都用线性代数表示。GPU/TPU 擅长矩阵运算，因此 NLP 模型要转化为矩阵运算才能高效训练和推理。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 12. 线性代数回顾：范数、角度与点积

* **定义 / 内容：** 向量范数衡量大小，常见有 $L_1$、$L_2$、$L_∞$；点积与夹角相关，可衡量两个向量方向是否相近。
* **直觉：** 在词向量和 Transformer 中，相似词或相关 token 通常被表示为方向接近的向量。点积越大，模型越倾向认为二者相关；范数会影响相似度与训练稳定性。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 13. 概率回顾：样本空间、事件、随机变量

* **定义 / 内容：** 样本空间 $Ω$ 包含所有可能对象；事件是 $Ω$ 的子集；随机变量把对象映射到数字；概率分布描述随机变量取值的概率。
* **直觉：** 在 NLP 中，$Ω$ 可以是词表，事件可以是一个文档中的词集合，随机变量可以是“词对应的编号”或“情感得分”。这套语言让文本预测可以被写成概率问题。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 14. 概率公理

* **定义 / 内容：** 概率满足 $P(Ω)=1$，$0≤P(A)≤1$，以及 $P(A∪B)=P(A)+P(B)-P(A∩B)$。
* **直觉：** 例如“正面情感词”和“经济相关词”可以有交集。求并集概率时要减掉交集，否则交集被重复计算。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 15. 联合分布

* **定义 / 内容：** 联合分布列出多个随机变量所有取值组合及其概率；它包含计算边缘概率、条件概率等所需的全部信息。
* **直觉：** 两个二元变量 A、B 的联合分布有 4 种组合，但实际 NLP 中变量数量极大，完整联合分布通常不可得、不可存储、不可精确估计，所以必须做独立性或 Markov 假设。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. 概率定理：补集、全概率、条件概率

* **定义 / 内容：** $P(A)+P(A^c)=1$；全概率可写为 $P(A)=P(A∩B)+P(A∩B^c)$；条件概率 $P(A|B)=P(A∩B)/P(B)$。
* **直觉：** 条件概率表示“已知 B 发生后 A 的概率”。例如先限定在“经济词”中，再看其中有多少是“正面词”。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 17. Bayes Rule

* **定义 / 内容：** 贝叶斯公式：$P(B|A)=P(A|B)P(B)/P(A)$。其中 posterior 是看到 A 后对 B 的信念，prior 是看到 A 前对 B 的信念，likelihood 是 B 为真时看到 A 的概率，marginal 是归一化项。
* **直觉：** 例子是比较 $P("Great")$ 与 $P("Great"|"Economic")$。同一个词在不同上下文中概率不同，Bayes Rule 是从证据更新信念的工具。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Bayes Rule 的展开形式与条件版本

* **定义 / 内容：** 分母可由全概率展开：$P(A)=P(A|B)P(B)+P(A|B^c)P(B^c)$；也可以处理带额外条件的形式，如 $P(B|A∩C)$。
* **直觉：** 机器学习中常常省略与优化变量无关的归一化项，但理解分母的归一化作用很重要：它保证 posterior 是合法概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 19. 期望与方差

* **定义 / 内容：** 期望是随机变量取值按概率加权的平均：$E[X]=Σ_x p(X=x)x$；方差是到期望的平方距离的期望：$Var[X]=E[(X-E[X])^2]$。
* **直觉：** 期望衡量“中心趋势”，方差衡量“不确定性/分散度”。模型训练中常用期望损失，统计估计中也常关心方差。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 20. Bernoulli 与 Binomial 分布

* **定义 / 内容：** Bernoulli 分布用于二分类/两种结果，比如 coin flip；只需一个参数如正类概率。n 个 i.i.d. Bernoulli 变量之和形成 Binomial 分布。
* **直觉：** 后续 logistic regression 会预测 Bernoulli 的正类概率；文本二分类如正/负情感也可抽象为 Bernoulli 随机变量。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. Categorical 与 Multinomial 分布

* **定义 / 内容：** Categorical 分布用于多分类结果，例如骰子的 6 个面；k 类分布只需 k-1 个自由参数。n 个 i.i.d. categorical 变量的计数形成 multinomial 分布。
* **直觉：** 语言模型的“预测下一个 token”就是在词表上的 categorical 分布；多次 token 生成的计数可以看作 multinomial。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. 独立性假设

* **定义 / 内容：** 若 $P(X,Y)=P(X)P(Y)$ 或 $P(X|Y)=P(X)$，则 X 与 Y 独立。NLP 中独立性通常是建模假设，不是真实事实。
* **直觉：** 独立性可显著降低参数数量。两个 Bernoulli 变量若相关，需要 3 个自由参数；若独立，只需每个变量各 1 个参数。但假设过强会牺牲表达能力。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 23. 最大似然估计 MLE 的直觉

* **定义 / 内容：** 用观察频率估计未知概率。例如掷骰子很多次后，数字 1 出现次数/总次数就是 $P(1)$ 的 MLE。
* **直觉：** MLE 的思想是找一组参数，使实际观察到的数据最可能出现。帽子符号（如 $\hat{θ}$）表示估计量，MLE 是估计方法之一。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 24. MLE 在 AI/NLP 中的地位

* **定义 / 内容：** 多数 AI 模型通过 MLE 训练：预训练预测 next/masked token；SFT 用 query 预测人工答案；语言模型是在巨大词表上估计高维 categorical 分布。
* **直觉：** n-gram、embedding、parse tree、Transformer、GPT 都是在定义“如何计算文本概率分布”。复杂模型的区别主要在结构和参数化方式。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. Categorical 分布的 MLE 形式化

* **定义 / 内容：** 令 $θ_i=P(X=i)$，观测计数为 $c_i$，似然 $L(θ)=∏_i θ_i^{c_i}$，约束 $Σ_i θ_i=1$。MLE 为 $θ_i=c_i/n$。
* **直觉：** 通过 Lagrangian 可以推导出频率估计。这个结果是很多计数模型的基础，例如 unigram 词概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 26. NLP 模型训练的一般流程

* **定义 / 内容：** 步骤：收集文本数据；构造概率模型；写出似然函数；最大化 log-likelihood 得到参数估计；复杂模型通常用 SGD、AdaGrad 等数值优化。
* **直觉：** 真实 NLP 模型很少能像骰子例子那样直接令梯度为 0 求解，通常要用自动微分和优化器迭代训练。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 27. 扩展阅读

* **定义 / 内容：** 推荐 LLM survey、Wikipedia NLP、Goodfellow《Deep Learning》基础章节、SLP3 token/word 章节，以及用 LLM 辅助学习 Python/PyTorch。
* **直觉：** 这页告诉你补基础的方向：数学基础、Python 工程、LLM 概览、token 概念会贯穿整个课程。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 28. Demo

* **定义 / 内容：** 用 LLM 生成训练语料；用 MLE 估计词概率；练习 Python 读文件、tokenize、构造矩阵向量、画图。
* **直觉：** 这是把概率公式落到工程实现：语料 → token → count → probability → visualization。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 29. Conclusion

* **定义 / 内容：** NLP 研究机会多；课程信息在 Canvas；需要线代、微积分、概率统计、Python；下一讲进入 language models。
* **直觉：** Lecture 1 的核心是课程定位与数学地基，真正技术线从 Lecture 2 的 n-gram、embedding 开始。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 30. Quiz

* **定义 / 内容：** MLE 可用数值优化求解；独立性判断；categorical distribution 不等同于 multinomial distribution。
* **直觉：** categorical 是一次多类取样的分布；multinomial 是多次 categorical 取样后各类别计数的分布。这个区分后续语言建模会反复出现。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 2: Tokenization、n-gram、语义表示、Word2Vec/GloVe/FastText

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；NLP 的语言层级；Word tokenization；词的拆解：复合词、命名风格、数字；n-gram language models：句子概率；Unigram 模型与 MLE；Bigram、Trigram 与零概率问题；n-gram smoothing：Laplace 平滑
  * Symbolic vs. Semantics：符号与意义；为什么计算机天然是符号系统；Word semantics：字典与 WordNet；One-hot word vector；Distributional word vector；词向量可视化与 PCA；从文本学习词向量：distributional hypothesis；Word2Vec Skip-gram 直觉
  * Word2Vec 建模；Word2Vec 优化；Word2Vec 梯度推导；梯度下降解释；Word2Vec 优化效率：Negative Sampling 与 Mini-batch SGD；GloVe 直觉；GloVe 建模与目标函数；FastText：subword 表示
  * 扩展阅读；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Language Model Notes

### 1. Tokenization：模型看到的不是“句子”，而是 token 序列

Tokenization 是 NLP 的第一步。原始文本是字符串，但统计模型和神经模型都需要离散单位。最简单的方法是按空格切分，但真实文本会出现标点、复合词、大小写、数字、emoji、代码符号等复杂情况。

例如：

* `world,` 如果只用空格切，会把逗号粘在词上。
* `state-of-the-art` 可拆成多个语义成分。
* `COVID19` 可以拆成字母部分和数字部分。

tokenization 的选择会影响 vocabulary size、OOV、subword sharing 和模型泛化。现代 LLM 常用 BPE / SentencePiece 等 subword tokenizer，本质上就是在“词”和“字符”之间找折中。

### 2. n-gram Language Model：用短历史近似完整历史

完整句子概率是：

$$
P(w_1,\ldots,w_n)
$$

根据 chain rule：

$$
P(w_1,\ldots,w_n)=\prod_{i=1}^n P(w_i|w_1,\ldots,w_{i-1})
$$

问题是完整历史太长，参数不可估计。n-gram 用 Markov-style approximation：

$$
P(w_i|w_1,\ldots,w_{i-1})\approx P(w_i|w_{i-n+1},\ldots,w_{i-1})
$$

bigram 特例：

$$
P(w_i|w_1,\ldots,w_{i-1})\approx P(w_i|w_{i-1})
$$

MLE 估计为：

$$
P(w_i|w_{i-1})=\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
$$

### 3. Laplace Smoothing：解决零概率，但会重新分配概率质量

如果某个 bigram 在训练集没出现，MLE 会给它概率 0。只要句子中出现一个概率 0 的 bigram，整句概率就变成 0，这太极端。

Laplace smoothing 给每个候选词加 1：

$$
P_L(w_2|w_1)=\frac{c(w_1,w_2)+1}{c(w_1)+|V|}
$$

直觉是：每个可能事件都先给一个“虚拟计数”。这样未见事件不再是 0，但代价是高频事件的概率会被压低。

### 4. Word Semantics：从 symbolic 到 distributional

one-hot vector 只能表示“词 ID”，不能表示词义。`cat` 和 `tiger` 在 one-hot 中正交，模型看不出它们都和动物有关。

distributional hypothesis 说：

> You shall know a word by the company it keeps.

如果两个词出现在相似上下文，它们就应该有相似语义。Word2Vec、GloVe、FastText 都是在不同方式下实现这个想法。

### 5. Word2Vec Skip-gram 的训练直觉

Skip-gram 用中心词预测上下文词：

$$
P(w_o|w_c)=\frac{\exp(u_o^Tv_c)}{\sum_w \exp(u_w^Tv_c)}
$$

loss 是：

$$
L=-u_o^Tv_c+\log\sum_w\exp(u_w^Tv_c)
$$

梯度：

$$
\frac{\partial L}{\partial v_c}
=-u_o+\sum_w P(w|w_c)u_w
$$

第一项把中心词向量拉向真实上下文；第二项把它推离模型当前认为可能的平均上下文。softmax 分母太贵，所以 negative sampling 只采少数负样本。

### Homework / Exam Connection

Bigram 题的解题固定流程：

1. 列出所有训练句子，包括 `<S>` 和 `</S>`。
2. 统计前一个词的总次数 $Count(w_{i-1})$。
3. 统计 bigram 次数 $Count(w_{i-1},w_i)$。
4. 无 smoothing 用 MLE；有 Laplace 就套：

$$
\frac{c(w_1,w_2)+1}{c(w_1)+|V|}
$$

易错点：

* 题目如果给了 vocabulary，就必须只在该 vocabulary 上平滑。
* 判断 next word 是看条件概率最大的词，不是看全局词频最高的词。
* 算整句概率时要按 bigram 连乘；如果题目没有要求 `</S>`，不要擅自乘上结束符概率。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** Lecture 2 开始进入语言模型与词向量。
* **直觉：** 本讲从“语言的基本单位”出发，逐步过渡到用概率和向量表示词义。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. NLP 的语言层级

* **定义 / 内容：** 语言层级包括 phonology、morphology、syntax、semantics、pragmatics。
* **直觉：** 从声音到词形、句法、字面意义、语用含义，NLP 可处理不同层次的问题。本讲主要关注 token、word semantics 和 word vector。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 3. Word tokenization

* **定义 / 内容：** tokenization 是把文本切成更小单位；方法包括简单 `split()`、正则、NLTK。标点是否保留取决于任务。
* **直觉：** `split()` 会把 `world,` 和逗号粘在一起，不理想；正则可去掉标点；NLTK 可把标点作为独立 token。标点在情感、句法和生成任务中可能有用。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 4. 词的拆解：复合词、命名风格、数字

* **定义 / 内容：** 复合 token 可拆解，如 `state-of-the-art → state/of/the/art`，`camelCaseWord → camel/Case/Word`，`COVID19 → COVID/19`。
* **直觉：** 真实文本里词不总是空格分隔。正确拆解可减少 OOV，提高模型对形态和组合结构的泛化能力。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. n-gram language models：句子概率

* **定义 / 内容：** 语言模型定义句子概率 $P(w_1,...,w_n)$；可用于 spell-checking 和 next-word prediction。完整联合分布参数太多，因此用 unigram/bigram 等分解。
* **直觉：** 例如 $P("the book") >> P("book the")$，$P("Obama"|"President") >> P("book"|"President")$。n-gram 的核心是用短历史近似长历史。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Unigram 模型与 MLE

* **定义 / 内容：** unigram 假设词彼此独立：$P(w_1,...,w_n)=∏_i P(w_i)$；语料似然可写为 $∏_{w∈V} P(w)^{c(w)}$；MLE 为 $P(w)=c(w)/N$。
* **直觉：** 这就是 Lecture 1 的 categorical MLE 用到词表上。缺点是完全忽略词序和上下文。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Bigram、Trigram 与零概率问题

* **定义 / 内容：** bigram：$P(w|v)=Count(v,w)/Count(v)$；trigram：$P(w|u,v)=Count(u,v,w)/Count(u,v)$。问题：未出现组合概率会被估为 0。
* **直觉：** 在小语料 `I am Sam...` 中，$P(eggs|brown)$ 可能因为没见过而为 0，但这不代表真实语言中不可能。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. n-gram smoothing：Laplace 平滑

* **定义 / 内容：** unigram Laplace：$P_L(w)=(c(w)+1)/(N+|V|)$；bigram Laplace：$P_L(w_2|w_1)=(c(w_1,w_2)+1)/(c(w_1)+|V|)$。
* **直觉：** 给所有词/词对加 1，避免未见事件概率为 0。代价是会把概率质量从高频事件挪给低频/未见事件。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Symbolic vs. Semantics：符号与意义

* **定义 / 内容：** 二进制序列本身没有固定意义；按 ASCII 可解码成 “Hello”，按 RGB 是颜色，按波形是声音。Chinese Room 思想实验说明符号操作不等于理解语义。
* **直觉：** NLP 的根本难题之一是：计算机操作的是符号/数字，而人类关心的是意义。词向量试图把语义结构映射到数值空间。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. 为什么计算机天然是符号系统

* **定义 / 内容：** 计算机用电压等低层物理信号组合成高层概念，但电压本身无语义；人类通过多感官与世界交互形成语义。
* **直觉：** 从计算机角度，`cat` 只是二进制序列；从人类角度，它对应“毛茸茸的动物”。这解释了为什么 grounding 和语义理解很难。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. Word semantics：字典与 WordNet

* **定义 / 内容：** 字典以 lemma 为条目，每个 lemma 可有多个 sense；WordNet 以 word sense 为基本单位，有 synset、hypernym、hyponym 等关系。
* **直觉：** WordNet 优点是编码人类知识和语义关系；缺点是静态、更新慢、难覆盖新词，本质仍偏 symbolic。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. One-hot word vector

* **定义 / 内容：** 词表中每个 token 用一个唯一 one-hot 向量表示；优点是唯一 ID；缺点是语义相关词相似度为 0，且无法表达多个语义方面。
* **直觉：** `cat` 和 `tiger` 在 one-hot 中正交，模型无法从向量本身知道它们都是动物。one-hot 适合作为索引，不适合作为语义表示。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 13. Distributional word vector

* **定义 / 内容：** word embedding 是低维稠密向量，通常 50–300 维；每个维度可理解为某种语义关联强度；向量支持线性代数操作。
* **直觉：** 例如 $king - man + woman ≈ queen$ 这类类比展示了语义变换可由向量运算近似，这是 GPU 友好的语义表示方式。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 14. 词向量可视化与 PCA

* **定义 / 内容：** 高维词向量可通过 PCA 降到 2 维可视化；PCA 尽量保留数据方差。
* **直觉：** 可视化能观察相似词是否聚在一起，但二维图只是高维结构的投影，不能完全代表真实语义空间。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 15. 从文本学习词向量：distributional hypothesis

* **定义 / 内容：** “You shall know a word by the company it keeps.” 词的上下文可定义为窗口、句子或文档内附近词。
* **直觉：** 如果国家名经常与总统、首都等词共现，模型就能学到国家相关语义。窗口大小决定语义粒度：小窗口偏句法/局部关系，大窗口偏主题。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. Word2Vec Skip-gram 直觉

* **定义 / 内容：** 在滑动窗口中构造 `(center, context)` 正样本；希望中心词向量能高概率预测附近上下文词。
* **直觉：** 以 `green` 为中心，`like`、`eggs` 比远处或窗口外词概率更高。Skip-gram 是用中心词预测上下文；CBOW 则是用上下文预测中心词。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 17. Word2Vec 建模

* **定义 / 内容：** 每个词有两个向量：中心词向量 $v_w$ 和上下文词向量 $u_w$。用 softmax 得到 $P(w_o|w_c)=exp(u_o^T v_c)/Σ_w exp(u_w^T v_c)$。
* **直觉：** 点积衡量中心词和上下文词的匹配程度；指数保证正值并保序；分母把所有词归一化为词表上的概率分布。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Word2Vec 优化

* **定义 / 内容：** 目标是最大化正样本 $(w_c,w_o)$ 的 $P(w_o|w_c)$，等价于最小化 NLL：$L=-u_o^T v_c + log Σ_w exp(u_w^T v_c)$。
* **直觉：** 梯度会把中心词向量拉向真实上下文词，同时推离模型错误认为可能的词。更新可用 gradient descent。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Word2Vec 梯度推导

* **定义 / 内容：** 对 $v_c$ 求导得到 $∂L/∂v_c = -u_o + Σ_w P(w|w_c)u_w$。
* **直觉：** 第一项是 ground truth 的吸引力，第二项是模型预测分布下的期望上下文向量。训练的目标是让真实上下文比其他词更匹配。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 20. 梯度下降解释

* **定义 / 内容：** Ground Truth 项把中心向量朝真实上下文词方向移动；Expectation 项减去模型当前认为可能的平均词向量。
* **直觉：** 若模型已对正确词给 100% 概率，两项抵消，梯度为 0。否则模型会继续调整，让容易混淆的相关词分开。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 21. Word2Vec 优化效率：Negative Sampling 与 Mini-batch SGD

* **定义 / 内容：** softmax 分母要遍历整个词表，代价高；negative sampling 只采 K 个负样本。Mini-batch SGD 每步只用小批量正样本。
* **直觉：** 词在多个窗口中反复出现，通过与许多词“互动”学习语义。negative sampling 是 Word2Vec 实用化的关键。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 22. GloVe 直觉

* **定义 / 内容：** Word2Vec 用局部窗口增量学习；GloVe 显式利用全局共现统计。关键直觉是共现概率的“比值”比单个概率更能表达语义差异。
* **直觉：** 概率比值能消除一些无关尺度信息，突出某个词对某类上下文的相对偏好。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 23. GloVe 建模与目标函数

* **定义 / 内容：** GloVe 令 $w_i^T \tilde{w}_j + b_i + \tilde b_j ≈ log X_{ij}$，并最小化加权平方误差 $Σ f(X_{ij})(...-log X_{ij})^2$。
* **直觉：** $X_{ij}$ 是词 i 与 j 的共现频次；权重函数 $f$ 会限制超高频词（如 the）支配损失。它本质上是带权线性回归。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. FastText：subword 表示

* **定义 / 内容：** Word2Vec/GloVe 把词当原子 ID，忽略 morphology 且无法处理 OOV。FastText 用字符 n-gram 表示词，例如 `apple` 可由 `<ap, app, ppl, ple, le>` 和 `<apple>` 组成。
* **直觉：** 词向量由子词向量组合而成，因此 `run/running` 可共享部分参数，新词也可由已有 n-gram 组合出表示。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. 扩展阅读

* **定义 / 内容：** SLP3 tokenization/n-gram；Chinese Room；Goodfellow mini-batch SGD；Word2Vec、GloVe、FastText 原论文/项目。
* **直觉：** 本讲内容横跨符号处理、概率语言模型和神经词向量，阅读方向也对应这三条线。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 26. Demo

* **定义 / 内容：** 生成小语料，构造词表，训练 skip-gram，用 PCA 可视化词向量。
* **直觉：** 这个 demo 的重点是观察相关词是否在嵌入空间中靠近，从而理解分布式语义学习。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 27. Conclusion

* **定义 / 内容：** 总结 tokenization、symbolic vs semantics、n-gram、Word2Vec/GloVe/FastText；词嵌入后来成为 LLM 输入层基础；现代 LLM 多采用 BPE 等 subword 方法。
* **直觉：** 本讲从 count-based 概率模型过渡到 embedding，为后续 POS tagging、HMM 和神经网络打基础。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 28. Quiz

* **定义 / 内容：** negative sampling 解决 softmax 分母遍历词表过贵的问题；GloVe 先计算全局共现矩阵；FastText 用字符 n-gram 与词本身组合表示词。
* **直觉：** 三个题分别对应 Word2Vec 训练效率、GloVe 与 Word2Vec 的区别、FastText 处理 morphology/OOV 的核心机制。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 3: POS tagging 与 Hidden Markov Model 建模

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；English POS 类别；POS running example；POS 的重要性：句法信息与拼写纠错；POS 的重要性：语义信息与下游任务；POS tagging 为什么难；Buffalo 问题；社交媒体与 OOV
  * HMM 动机：用上下文消除 POS 歧义；HMM 形式化；HMM 与 Bayes Rule；n-gram 理论：两个概率都太复杂；Markov assumption；用 Markov assumption 简化 tag 序列；Transition probability matrix；Emission probability
  * Emission probability matrix；完整 HMM 与三个任务；扩展阅读；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed HMM Modeling Notes

### 1. POS Tagging 是结构化预测的入门任务

POS tagging 的输入是词序列：

$$
O=[o_1,\ldots,o_T]
$$

输出是同样长度的 tag 序列：

$$
Q=[q_1,\ldots,q_T]
$$

它不是逐词独立分类，因为每个 tag 会影响邻近 tag。例如 determiner 后常接 adjective 或 noun，pronoun 后常接 verb。HMM 把 POS tag 当作 hidden states，把 words 当作 observations。

### 2. HMM 的两个概率：transition 和 emission

HMM 参数包括：

* 初始概率：

$$
\pi_i=P(q_1=i)
$$

* 转移概率：

$$
a_{ij}=P(q_t=j|q_{t-1}=i)
$$

* 发射概率：

$$
b_i(o)=P(o_t=o|q_t=i)
$$

transition 描述语法结构，emission 描述某个 tag 生成某个词的可能性。

### 3. HMM 的两个独立性假设

**First-order Markov assumption:**

$$
P(q_t|q_1,\ldots,q_{t-1})\approx P(q_t|q_{t-1})
$$

**Emission independence:**

$$
P(o_t|q_1,\ldots,q_T,o_1,\ldots,o_{t-1})
\approx P(o_t|q_t)
$$

这两个假设很强，但它们把指数级问题变成可计算问题。

### 4. MAP Decoding

预测目标是：

$$
Q^*=\arg\max_Q P(Q|O)
$$

用 Bayes rule：

$$
P(Q|O)=\frac{P(O|Q)P(Q)}{P(O)}
$$

因为 $P(O)$ 与 $Q$ 无关：

$$
Q^*=\arg\max_Q P(O|Q)P(Q)
$$

HMM decoding 不是只选每个词最常见 tag，而是选整体最可能 tag sequence。

### Homework / Exam Connection

常见选择题陷阱：

* 5-gram 不是依赖前 1 个词，而是依赖前 4 个词。
* HMM 一阶假设是 tag 依赖前一个 tag，不是 word 依赖前一个 word。
* n-gram 是 language model，不是 POS tagger。
* HMM prediction 是 MAP sequence prediction。
* n-gram probability 不等于 HMM emission probability。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** Lecture 3 聚焦词性标注（POS tagging）和 HMM 的建模动机。
* **直觉：** POS tagging 是典型序列标注任务，也是学习 HMM、动态规划和结构化预测的入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. English POS 类别

* **定义 / 内容：** POS 是词类/语法类别；同一 POS 的词有相似语法属性。主要 tag 包括 NN、NNP、VB、JJ、RB、DT、IN、CC 等。
* **直觉：** POS 不只是词义标签，它反映词在句子结构中的功能。后续 HMM 会把 POS tag 当作隐藏状态。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 3. POS running example

* **定义 / 内容：** 例句中 `He(PRP) slowly(RB) looked(VBD) over(RP) ...` 展示了不同 POS 的作用。名词、动词、形容词是“meat”；副词、介词/particle、determiner 是“glue”。
* **直觉：** 内容词承载主要语义，功能词组织句法和细节。POS tag 能把句子拆成语法角色，辅助后续任务。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. POS 的重要性：句法信息与拼写纠错

* **定义 / 内容：** POS 提供词序模式，如 noun-verb、determiner-noun、adjective-noun、verb-adverb、preposition-noun。POS 可辅助纠错，如 there/their、passed/past、effect/affect、loose/lose。
* **直觉：** 单词看起来相似时，POS 上下文能判断哪个词合理。例如 “his horse” 中 possessive pronoun 后接 noun 更合理。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. POS 的重要性：语义信息与下游任务

* **定义 / 内容：** POS 可帮助机器翻译、关系抽取、事件抽取和实体抽取。例如 `building` 是动词还是名词会影响翻译；`Bill Gates` 中 Gates 需识别为专名。
* **直觉：** 同一字符串在不同 POS 下语义不同。POS 是把 token 连接到语义解释的中间层。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. POS tagging 为什么难

* **定义 / 内容：** 难点是 lexical ambiguity。多数 tag 类型不歧义，但歧义词在 running text 中很常见；most-frequent-tag baseline 约 92%，人类/模型约 97%。
* **直觉：** `back` 可为 RB/NN/JJ/VB，`like` 可为 IN/VB/JJ，`fast` 可为 JJ/RB/NN。模型必须利用上下文消歧。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Buffalo 问题

* **定义 / 内容：** 因为 POS 歧义，`Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo` 在英语中语法正确。
* **直觉：** Buffalo 可表示地名、动物、动词“欺负”。这说明词性与句法结构能让同一字符串承担多重角色。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. 社交媒体与 OOV

* **定义 / 内容：** 现代社交媒体需要新 tag，如 hashtag、@mention、RT、lol/emoji。POS 在 LLM 时代不再是必需中间任务，但仍有助于建立 NLP 直觉。
* **直觉：** OOV 可以部分通过 subword 模型缓解。传统 NLP 中人工标签和规则很重要；现代 LLM 可通过大规模学习弱化中间标注。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. HMM 动机：用上下文消除 POS 歧义

* **定义 / 内容：** HMM 使用 emission probability 与 transition probability。$P("back"|NN)$ 衡量 NN 发射 back 的可能；$P(NN|PRP\$)$ 衡量 possessive pronoun 后接 noun 的可能。
* **直觉：** 词本身只告诉我们候选 POS，上一个 tag 会进一步缩小范围。HMM 的核心就是同时建模“tag 如何转移”和“tag 如何生成词”。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. HMM 形式化

* **定义 / 内容：** 词表 $V$，POS 集合 $S={s_1,...,s_N}$；观测句子 $O=[o_1,...,o_T]$；隐藏状态 $Q=[q_1,...,q_T]$；目标是 $Q*=argmax_Q P(Q|O)$。
* **直觉：** 直接枚举所有 tag 序列复杂度指数级，因为每个位置有 N 种 tag，共 $N^T$ 种组合。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. HMM 与 Bayes Rule

* **定义 / 内容：** $P(Q|O)=P(O|Q)P(Q)/P(O)$；由于 O 固定，优化可转为最大化 $P(O|Q)P(Q)$。
* **直觉：** 不能只看 $P(O|Q)$，否则可能选出局部最像词的 tag 但 tag 序列不合理；也不能只看 $P(Q)$，否则会得到常见 tag 序列但与实际词无关。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 12. n-gram 理论：两个概率都太复杂

* **定义 / 内容：** 需要建模 tag 序列概率 $P(Q)$ 和词发射概率 $P(O|Q)$；完整条件链都需要指数多参数。
* **直觉：** 这促使我们引入 Markov assumption 和 emission independence，把不可计算的联合分布简化成可估计的局部条件概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 13. Markov assumption

* **定义 / 内容：** 下一步只依赖当前状态，不依赖更早历史。例子：看当前红绿灯决定开车；Markov 曾通过统计 Pushkin 小说中字母转移研究语言。
* **直觉：** Markov 假设不是说历史完全无关，而是用当前状态作为历史摘要，从而降低模型复杂度。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 14. 用 Markov assumption 简化 tag 序列

* **定义 / 内容：** $P(Q)=P(q_1)∏_{t=2}^T P(q_t|q_{t-1})$；可用图模型表示依赖关系，边代表条件依赖，无边代表条件独立。
* **直觉：** 从完整历史条件概率变成一阶转移概率，参数量从指数级降到 $|S|^2$ 级。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 15. Transition probability matrix

* **定义 / 内容：** 转移矩阵 $A$ 中 $a_ij=P(q_t=j|q_{t-1}=i)$；每一行和为 1。例子包括 $D→A/N$、$A→N$、$N→V$、$V→N/D$。
* **直觉：** 矩阵中高概率转移对应英语常见语法模式，如 determiner 后常接 adjective 或 noun。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. Emission probability

* **定义 / 内容：** 一个词由当前 POS tag 发射；假设当前词不依赖前词、过去 POS 或未来 POS，只依赖当前 tag。
* **直觉：** 这是强简化：真实语言中词当然受上下文影响，但 HMM 用这个假设换取可估计性和可推理性。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 17. Emission probability matrix

* **定义 / 内容：** 发射矩阵 $B$ 中 $b_{j,o}=P(o_t=o|q_t=j)$；每个 tag 对不同词有不同发射概率。
* **直觉：** 例如 D 高概率发射 `the/a`，A 高概率发射 `big/red`，N 高概率发射 `dog/cat/car`，V 高概率发射 `run/eat/see/go`。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 18. 完整 HMM 与三个任务

* **定义 / 内容：** HMM 参数包括起始概率 $π$、转移矩阵 $A$、发射矩阵 $B$。核心任务：estimation（估参数）、inference（算句子概率）、prediction（预测 POS tag）。
* **直觉：** 后续 Lecture 4 解决 inference，Lecture 5 解决 prediction 和 estimation。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. 扩展阅读

* **定义 / 内容：** SLP3 POS tagging 与 HMM 建模章节。
* **直觉：** 这页是本讲理论来源，可用于补充 tagset 和 HMM 细节。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 20. Demo

* **定义 / 内容：** 生成带 POS tag 的语料；指定 tag 集合；用相对频率估计 HMM 参数；可视化 transition matrix。
* **直觉：** 相对频率估计就是 supervised MLE：有 tag 标注时直接数数即可。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 21. Conclusion

* **定义 / 内容：** 下一讲讲 forward/backward algorithms；建议复习动态规划的 optimal substructure 和 overlapping subproblems。
* **直觉：** HMM 的推理看似指数级，但链式结构允许动态规划高效求解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. Quiz

* **定义 / 内容：** POS tag 对应：Verb→VB，Adverb→RB；transition matrix 不一定对称；emission matrix 空间复杂度为 $|S|×|V|$。
* **直觉：** $A_ij$ 与 $A_ji$ 含义不同，如 noun→verb 和 verb→noun 的概率没有理由相等。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 4: HMM Forward / Backward Algorithm

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；Inference 问题定义；Forward algorithm 与动态规划；Forward probability 定义；Forward base case；Forward 第二步；Forward example：Fly High；Forward general case 与证明
  * 展开递推与复杂度；句子概率与矩阵化；Backward algorithm 定义；Backward base case；Backward example：fly high；Backward general case 与证明；Backward 算法流程与矩阵化；扩展阅读
  * Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Forward / Backward Notes

### 1. Inference 目标：算观测序列概率

HMM inference 问的是：

$$
P(O|\lambda)
$$

其中 $\lambda=(A,B,\pi)$。因为 hidden tag sequence 不可见：

$$
P(O)=\sum_Q P(O,Q)
$$

直接枚举 $Q$ 有 $N^T$ 条路径，复杂度指数级。Forward/Backward 用 DP 把所有路径“压缩”进局部概率。

### 2. Forward Probability

定义：

$$
\alpha_t(j)=P(o_1,\ldots,o_t,q_t=j)
$$

它是一个 joint probability：到时间 $t$ 为止观测到前 $t$ 个词，并且当前状态是 $j$。

初始化：

$$
\alpha_1(j)=\pi_j b_j(o_1)
$$

递推：

$$
\alpha_t(j)=\sum_i \alpha_{t-1}(i)a_{ij}b_j(o_t)
$$

最后：

$$
P(O)=\sum_j\alpha_T(j)
$$

### 3. Backward Probability

定义：

$$
\beta_t(i)=P(o_{t+1},\ldots,o_T|q_t=i)
$$

它是 conditional probability：已知当前状态是 $i$，未来观测出现的概率。

初始化：

$$
\beta_T(i)=1
$$

递推：

$$
\beta_t(i)=\sum_j a_{ij}b_j(o_{t+1})\beta_{t+1}(j)
$$

### 4. 任意中间时刻也可以恢复整句概率

Forward 和 Backward 在任意 $t$ 可以拼起来：

$$
P(O)=\sum_j \alpha_t(j)\beta_t(j)
$$

这是因为 $\alpha_t(j)$ 覆盖过去和当前，$\beta_t(j)$ 覆盖未来；二者乘积覆盖完整 observation sequence。

### Homework / Exam Connection

易错点：

* $\alpha_t(j)$ 是 joint probability。
* $\beta_t(j)$ 是 conditional probability。
* 所以 $P(O)=\sum_j\alpha_T(j)$，但不能直接写成 $\sum_j\beta_1(j)$。
* Forward 是求和 DP，不是 greedy search。
* Forward 和 Backward 复杂度都是 $O(TN^2)$。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** 本讲解决 HMM 的 inference：给定观测句子和 HMM 参数，计算句子概率。
* **直觉：** 这是 HMM 三大任务中的第二个：不直接预测 tag，而是把所有可能 hidden state 序列边缘化。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. Inference 问题定义

* **定义 / 内容：** 给定观测 $O=[o_1,...,o_T]$ 和参数 $(A,B,π)$，求 $P(O|A,B,π)$。根据全概率：$P(O)=Σ_Q P(O,Q)$；暴力枚举复杂度 $O(|S|^T)$。
* **直觉：** 句子概率可用于评估句子在模型下是否“合理”。难点是 hidden tag 序列不可见且数量指数级。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. Forward algorithm 与动态规划

* **定义 / 内容：** Forward algorithm 不暴力枚举，而是用动态规划缓存子问题。DP 依赖 overlapping subproblems 与 optimal/subproblem structure。
* **直觉：** 就像最短路可以复用子路径，HMM 中到达某个状态的概率也可以复用之前时间步的结果。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 4. Forward probability 定义

* **定义 / 内容：** 定义 $α_t(j)=P(o_1,...,o_t,q_t=j)$，表示到时间 t 为止、当前状态为 j 的联合概率。
* **直觉：** $α$ 是“history tracer”，把过去所有可能路径压缩成一个数。求和与乘法可交换，这是递推的代数基础。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 5. Forward base case

* **定义 / 内容：** $α_1(i)=π_i b_i(o_1)$，即初始为 tag i 且发射第一个词的联合概率。
* **直觉：** 第一步没有前驱状态，所以只需起始概率乘以发射概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 6. Forward 第二步

* **定义 / 内容：** $α_2(j)=Σ_i α_1(i)a_{ij}b_j(o_2)$。对所有可能前一状态 i 求和，并复用 $α_1(i)$。
* **直觉：** 这里的求和是 marginalization：我们不关心前一状态具体是什么，只关心所有可能路径对当前状态的总贡献。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 7. Forward example：Fly High

* **定义 / 内容：** 对两个 tag（如 N/V）分别算 $α_1$，再用所有前一 tag 的贡献计算 $α_2(N)$ 和 $α_2(V)$。
* **直觉：** 同一个词序列可对应不同 tag 路径；Forward 累加所有路径概率，而不是选最大路径。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 8. Forward general case 与证明

* **定义 / 内容：** 通用递推：$α_t(j)=Σ_{k=1}^N α_{t-1}(k)a_{kj}b_j(o_t)$。
* **直觉：** 证明思路是把 $P(o_{1:t},q_t=j)$ 按前一状态 $q_{t-1}=k$ 分解，再使用 HMM 条件独立假设。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 9. 展开递推与复杂度

* **定义 / 内容：** 若递归完全展开，会回到对所有状态路径求和；动态规划通过缓存把重复计算合并。
* **直觉：** 每个时间步对每个当前状态枚举 N 个前一状态，复杂度 $O(TN^2)$，远低于 $O(N^T)$。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 10. 句子概率与矩阵化

* **定义 / 内容：** 句子概率 $P(O)=Σ_j α_T(j)$。Forward 算法流程：初始化 $α_1$；迭代计算 $α_t$；最后求和。
* **直觉：** 可用向量/矩阵形式消除内层循环，提高实现效率：本质是前向概率向量乘 transition matrix，再按 emission 概率逐元素缩放。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. Backward algorithm 定义

* **定义 / 内容：** Backward algorithm 也通过 DP 做边缘化。定义 $β_t(i)=P(o_{t+1},...,o_T|q_t=i)$。
* **直觉：** $β$ 是“fortune teller”，表示从当前状态 i 出发，未来观测出现的概率。它从句尾往句首递推。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 12. Backward base case

* **定义 / 内容：** $β_T(i)=1$，表示句尾之后的空未来序列概率为 1。
* **直觉：** 1 是乘法的中性元素，便于从最后一个状态向前累积未来概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 13. Backward example：fly high

* **定义 / 内容：** 先设最后一步 $β_T(N)=β_T(V)=1$，再计算 $β_{T-1}(N)=Σ_k a_{N,k}b_k(high)β_T(k)$，V 同理。
* **直觉：** Backward 对“下一状态”求和；它问的是从当前状态转移到哪个未来状态都可能，所有可能都要加起来。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 14. Backward general case 与证明

* **定义 / 内容：** 通用递推：$β_t(i)=Σ_j a_{ij}b_j(o_{t+1})β_{t+1}(j)$。
* **直觉：** 证明同样依赖 HMM 条件独立：未来观测只通过下一状态与当前状态连接。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 15. Backward 算法流程与矩阵化

* **定义 / 内容：** 初始化 $β_T(i)=1$；从 $t=T-1$ 到 1 递推；可用转置矩阵和 emission 向量做向量化。
* **直觉：** Forward 从过去到未来，Backward 从未来到过去；二者都可计算同一个 $P(O)$，也会在 EM/Baum-Welch 中配合使用。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. 扩展阅读

* **定义 / 内容：** SLP3 forward/backward；动态规划；概率图模型；message passing；HMM 学习。
* **直觉：** Forward/Backward 可看作链式图模型上的 message passing，是更一般概率图模型推理的特例。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 17. Demo

* **定义 / 内容：** 用 Lecture 3 的 POS-tag 文本构建 HMM；固定 ground-truth 参数；运行 forward 和 backward。
* **直觉：** demo 的重点是验证两个方向的递推能得到一致的句子概率，并理解矩阵实现。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 18. Conclusion

* **定义 / 内容：** Forward/Backward 都用于计算观测句子的 marginal probability，都是动态规划算法。下一步是学习 HMM 参数。
* **直觉：** 本讲解决 $P(O)$，下一讲会用这些概率来估计未知 tag 的 soft counts。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 19. Quiz

* **定义 / 内容：** 每个时间步的 forward 复杂度是 $O(N^2)$；forward/backward 都用 DP；句尾 backward probability 初始化为 1。
* **直觉：** 如果句长是 T，总复杂度为 $O(TN^2)$；N 是 POS tag 数量。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 5: HMM Viterbi Decoding 与 EM / Baum-Welch

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；HMM prediction 问题；Running example：Time flies like an arrow；Structured prediction；Viterbi algorithm 概念；Viterbi base case；Viterbi t=2 情况；Viterbi general case
  * Backpointer 记录最优路径；Viterbi algorithm 流程；Running example：They base；Viterbi 趣闻；Supervised HMM 参数估计；EM algorithm 基本思想；EM 例子与性质；HMM 转移矩阵的 soft counts
  * Soft label 的计算；发射矩阵的 soft counts；Baum-Welch Algorithm；扩展阅读；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Viterbi and EM Notes

### 1. Viterbi：把 Forward 的 sum 换成 max

Forward 算所有路径总概率；Viterbi 找最可能的一条路径。

定义：

$$
v_t(j)=\max_{q_1,\ldots,q_{t-1}}
P(q_1,\ldots,q_{t-1},q_t=j,o_1,\ldots,o_t)
$$

初始化：

$$
v_1(j)=\pi_j b_j(o_1)
$$

递推：

$$
v_t(j)=\max_i v_{t-1}(i)a_{ij}b_j(o_t)
$$

要恢复路径，需要 backpointer：

$$
p_t(j)=\arg\max_i v_{t-1}(i)a_{ij}b_j(o_t)
$$

### 2. Example: They base

题目给：

$$
v_1(NNP)=0.5\cdot0.8=0.4
$$

$$
v_1(N)=0,\quad v_1(V)=0
$$

计算：

$$
v_2(V)=\max_i v_1(i)a_{i,V}b_V(base)
$$

最大来自 $NNP\rightarrow V$：

$$
v_2(V)=0.4\cdot0.7\cdot0.4=0.112
$$

而：

$$
v_2(N)=0.4\cdot0.1\cdot0.6=0.024
$$

所以最优路径：

$$
Q^*=[NNP,V]
$$

### 3. Supervised HMM MLE

如果 tag 已知，可以直接数：

$$
a_{ij}=\frac{C(i\rightarrow j)}{C(i)}
$$

$$
b_i(o)=\frac{C(i\rightarrow o)}{C(i)}
$$

$$
\pi_i=\frac{C(q_1=i)}{\text{number of sentences}}
$$

### 4. EM / Baum-Welch

如果 tag 不知道，不能直接数 hard counts。EM 用 soft counts。

状态 posterior：

$$
\gamma_t(i)=P(q_t=i|O,\lambda)
=\frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}
$$

转移 posterior：

$$
\xi_t(i,j)=P(q_t=i,q_{t+1}=j|O,\lambda)
=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}
{P(O|\lambda)}
$$

E-step：用 Forward/Backward 算 $\gamma,\xi$。  
M-step：用 soft counts 更新 $\pi,A,B$。

### 5. Conditional Independence Proof for γ

从定义：

$$
\gamma_t(i)=P(q_t=i|O,\lambda)
=\frac{P(q_t=i,O|\lambda)}{P(O|\lambda)}
$$

把 $O$ 拆成：

$$
O=(o_{1:t},o_{t+1:T})
$$

HMM 条件独立给出：

$$
P(o_{t+1:T}|o_{1:t},q_t=i,\lambda)
=P(o_{t+1:T}|q_t=i,\lambda)
$$

所以：

$$
P(q_t=i,O|\lambda)
=P(o_{1:t},q_t=i|\lambda)
P(o_{t+1:T}|q_t=i,\lambda)
=\alpha_t(i)\beta_t(i)
$$

得到：

$$
\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}
$$

### Homework / Exam Connection

* Viterbi 要保存 backpointer；只保存最大概率不能恢复路径。
* EM 对初始化敏感，通常只能保证局部最优。
* Baum-Welch 的 E-step 需要 Forward 和 Backward。
* $\gamma_t(i)$ 不是只由 $\alpha_t(i)$ 决定；它必须乘 $\beta_t(i)$。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** 本讲处理 HMM prediction 和 parameter estimation。
* **直觉：** Lecture 4 算的是所有路径总概率；Lecture 5 要找最优路径，并在隐状态未知时学习参数。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. HMM prediction 问题

* **定义 / 内容：** 给定句子、tag 集合和 HMM 参数，求 $Q*=argmax_Q P(Q|O;θ)$。难点仍是可能 tag 序列指数多。
* **直觉：** 这是 POS tagging 的核心：不是求句子总体概率，而是找最可能的隐藏 tag 序列。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 3. Running example：Time flies like an arrow

* **定义 / 内容：** 逐词独立选择 most frequent tag 会得到不自然序列 `Noun Verb Verb DT Noun`；局部 $P(Verb|like)$ 不考虑邻居 tag。
* **直觉：** tag 之间相互影响，不能逐位置独立预测；这引出 structured prediction。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Structured prediction

* **定义 / 内容：** 应同时预测整个 tag 序列，并评价整体质量；这个过程称为 decoding。目标序列同时考虑 emission probability 与 transition probability。
* **直觉：** 结构化预测把输出看作有依赖结构的对象，而不是一组独立标签。暴力搜索仍是 $O(|S|^T)$。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Viterbi algorithm 概念

* **定义 / 内容：** Viterbi 利用链式结构做 DP，定义 $v_t(j)=max_{q_1,...,q_{t-1}} P(q_1,...,q_{t-1},q_t=j,o_1,...,o_t)$。
* **直觉：** $v_t(j)$ 是到时间 t、以状态 j 结尾的最佳路径概率。它类似 forward 的 $α_t(j)$，但把求和换成取最大。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 6. Viterbi base case

* **定义 / 内容：** $v_1(j)=π_j b_j(o_1)$。
* **直觉：** 第一个 token 没有历史路径，最佳路径就是以 j 开始并发射 $o_1$ 的概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 7. Viterbi t=2 情况

* **定义 / 内容：** $v_2(j)=max_i v_1(i)a_{ij}b_j(o_2)$。
* **直觉：** 对于当前状态 j，只需要知道前一步到每个 i 的最佳概率，不需要保存所有完整路径。这就是 DP 的压缩。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 8. Viterbi general case

* **定义 / 内容：** 通用递推：$v_t(j)=max_k v_{t-1}(k)a_{kj}b_j(o_t)$。
* **直觉：** HMM decomposition 让当前最优路径由“某个前一状态的最优路径 + 一次转移 + 一次发射”组成。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 9. Backpointer 记录最优路径

* **定义 / 内容：** 只保存最大值不够，还要保存 $p_t(j)=argmax_k v_{t-1}(k)a_{kj}b_j(o_t)$。最后从 $q_T*=argmax_k v_T(k)$ 开始回溯。
* **直觉：** backpointer 告诉我们每个最优局部结果来自哪个前一状态，最终才能重构完整 tag 序列。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Viterbi algorithm 流程

* **定义 / 内容：** 初始化 $v_1$；对每个时间步和每个状态计算最大概率并记录 backpointer；回溯找 $Q^*$；返回最大概率。
* **直觉：** 复杂度 $O(TN^2)$，空间可为 $O(TN)$ 用于保存 backpointers。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. Running example：They base

* **定义 / 内容：** 通过给定 $π$、$A$、$B$ 计算 `They base` 的最佳 tag。`They` 只可能发射自 NNP/Pronoun，`base` 可为 noun 或 verb，因此 V 和 N 竞争转移与发射概率。
* **直觉：** Viterbi 会自动平衡“词本身像什么 tag”和“tag 序列是否合理”。空/零概率项可以跳过。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. Viterbi 趣闻

* **定义 / 内容：** Viterbi 算法由 Andrew Viterbi 于 1967 年为通信问题提出，后来广泛用于手机通信、空间通信、语音识别、数据记录、搜索、DNA sequencing 等。
* **直觉：** 这说明 NLP 中的动态规划算法往往来自更一般的序列解码问题。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. Supervised HMM 参数估计

* **定义 / 内容：** 若 hidden states 已知，可用 MLE：$a_ij=C(i→j)/C(i)$，$b_{i,o}=C(i→o)/C(i)$，$π_i=C(q_1=i)/m$。
* **直觉：** 有标注 POS 语料时，估计 HMM 参数就是统计起始 tag、tag 转移和 tag 发射词的相对频率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. EM algorithm 基本思想

* **定义 / 内容：** EM 用于带 unlabeled data 的参数估计。E-step 估计标签概率（pseudo labels/soft labels）；M-step 用这些 soft labels 做 MLE；循环迭代。
* **直觉：** 当 tag 不可见时，不能直接数数；EM 先用当前模型“猜”每个位置/转移的概率，再用概率计数更新参数。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 15. EM 例子与性质

* **定义 / 内容：** EM 可用于从未标注数据估计 Gaussian mixture 的均值和协方差；每轮 EM 保证数据 likelihood 不下降。
* **直觉：** EM 通常收敛到局部最优而非全局最优，因此初始化很重要。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. HMM 转移矩阵的 soft counts

* **定义 / 内容：** 不知道真实 tag 序列时，用 soft count 估计 tag i 在位置 t 转移到 tag j 的概率，即 $P(q_t=i,q_{t+1}=j|O,λ)$。
* **直觉：** hard label 是“这一处确实 i→j”；soft label 是“这一处有多大概率 i→j”。软计数是所有位置概率的累积。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. Soft label 的计算

* **定义 / 内容：** $ξ_t(i,j)=P(q_t=i,q_{t+1}=j|O,λ)=α_t(i)a_{ij}b_j(o_{t+1})β_{t+1}(j)/P(O)$。
* **直觉：** forward 给过去概率，transition/emission 给当前跨步概率，backward 给未来概率，除以 $P(O)$ 做归一化。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 18. 发射矩阵的 soft counts

* **定义 / 内容：** 估计 $b_{i,o}$ 需要 $P(q_t=i,o_t=o|O,λ)$，通常用 $γ_t(i)=P(q_t=i|O,λ)=α_t(i)β_t(i)/P(O)$。
* **直觉：** 若某位置观测词就是 o，就把该位置属于 tag i 的概率计入 `i emits o` 的 soft count。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Baum-Welch Algorithm

* **定义 / 内容：** HMM 的 EM 叫 Baum-Welch。流程：初始化 $(A,B,π)$；E-step 运行 forward/backward 得到 soft counts；M-step 更新 $A,B,π$；直到收敛。
* **直觉：** 注意 EM 只保证局部最优；坏初始化会导致坏结果。少量监督标签可帮助模型找到更好解。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 20. 扩展阅读

* **定义 / 内容：** 结构化预测、EM、beam search 等。
* **直觉：** 本讲的 Viterbi 是结构化预测的基础；后续机器翻译和生成也会用 decoding/search。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 21. Demo

* **定义 / 内容：** Lecture 4 demo 中包含 Viterbi 算法。
* **直觉：** 实现 Viterbi 时要特别检查：初始化、递推、backpointer、回溯顺序。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 22. Conclusion

* **定义 / 内容：** Viterbi 用 DP 找最优 POS tag 序列；EM 用于 HMM 参数估计，在未标注数据上同时估计 tag 分布与参数。
* **直觉：** HMM 三大任务至此闭环：inference、prediction、estimation。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 23. Quiz

* **定义 / 内容：** Viterbi 复杂度是 $O(TN^2)$ 而不是 $O(TN)$；$v_t(j)$ 的递推/定义；EM 用 DP 与 MLE。
* **直觉：** EM 的 E-step 用 forward/backward（DP），M-step 用 soft counts 做 MLE。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 6: Syntax、CFG、CYK Parsing

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；Grammar and syntax；Syntax 定义；n-gram/HMM 建模 shallow syntax；Constituent；Context-free grammar 概念；CFG 四元组、推导、语言；CFG 示例与 parse tree
  * CFG：句子类型规则；CFG：Noun Phrase；CFG：Nominal；CFG：Verb Phrase；CFG 来源：Penn Treebank；The bitter lesson；Syntactic parsing 目标；Top-down parsing search
  * Bottom-up parsing search；Parsing ambiguity：attachment ambiguity；Parsing ambiguity：coordination ambiguity；Repeated subproblems；CYK parsing algorithm；CNF CFG；CNF 转换算法；CYK 矩阵结构
  * CYK algorithm 伪代码；CYK parsing examples；扩展阅读；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed CFG / CYK Notes

### 1. Constituent 和 CFG 的意义

句法分析不是只看相邻词，而是要识别哪些词组成一个整体。这个整体叫 **constituent (语法成分)**。例如：

* `with cameras` 是 PP。
* `scenes with cameras` 可以是 NP。
* `observe scenes` 可以是 VP。

CFG 用规则描述这些组合：

$$
G=(N,\Sigma,R,S)
$$

其中 $N$ 是 non-terminals，$\Sigma$ 是 terminals，$R$ 是 rules，$S$ 是 start symbol。

### 2. CNF 与 derivation step 数量

CNF 只允许：

$$
A\rightarrow BC
$$

或：

$$
A\rightarrow a
$$

如果一个 CNF parse tree 生成 $n$ 个词，那么：

* 叶子层需要 $n$ 个 lexical rules。
* 二叉树内部需要 $n-1$ 个 binary rules。

所以总 derivation steps：

$$
n+(n-1)=2n-1
$$

这是 Homework 选择题常考点。

### 3. CYK Table

CYK 用 span boundary，而不是 word index。长度为 $n$ 的句子有 gap index $0,\ldots,n$。cell $(i,j)$ 表示从 gap $i$ 到 gap $j$ 的子串。

如果：

$$
B\in table[i,k]
$$

$$
C\in table[k,j]
$$

并且有规则：

$$
A\rightarrow BC
$$

那么：

$$
A\in table[i,j]
$$

### 4. Attachment Ambiguity Example

句子：

`agents observe scenes with cameras`

词级初始化：

$$
[0,1]=NP,\quad [1,2]=V,\quad [2,3]=NP,\quad [3,4]=P,\quad [4,5]=NP
$$

先得到：

$$
[3,5]=PP
$$

因为：

$$
PP\rightarrow P\ NP
$$

路径 1：PP attach 到 NP

$$
[2,5]=NP
$$

因为：

$$
NP\rightarrow NP\ PP
$$

然后：

$$
[1,5]=VP
$$

因为：

$$
VP\rightarrow V\ NP
$$

含义是：`observe [scenes with cameras]`。

![Wiki/Image/Class/Introdution to NLP/1.png](/img/user/Wiki/Image/Class/Introdution%20to%20NLP/1.png)

路径 2：PP attach 到 VP

$$
[1,3]=VP
$$

因为：

$$
VP\rightarrow V\ NP
$$

然后：

$$
[1,5]=VP
$$

因为：

$$
VP\rightarrow VP\ PP
$$

含义是：`[observe scenes] with cameras`。

### 5. CYK Complexity Proof

CYK 的循环结构：

1. span length 有 $O(n)$ 种。
2. 每个 span length 下 start index 有 $O(n)$ 种。
3. 每个 cell 枚举 split point $k$，有 $O(n)$ 种。
4. 每个 split 需要检查 grammar rules，最坏 $O(|R|)$。

所以：

$$
O(n)\cdot O(n)\cdot O(n)\cdot O(|R|)
=O(n^3|R|)
$$

### Homework / Exam Connection

* CYK 是 bottom-up，但 cell 中出现的 non-terminal 不保证一定进入最终 parse tree。
* 标准 CYK 不处理概率，也不会自动选最可能树。
* attachment ambiguity 是一个修饰语可以接到不同 parent node。
* CNF 下 $n$ 个词需要 $2n-1$ 个 rule applications。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** 本讲进入 syntax 与 syntactic parsing。
* **直觉：** 从局部序列模型转向句法树模型，关注词如何组成更大的结构。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. Grammar and syntax

* **定义 / 内容：** syntax 位于 morphology 之上、semantics 之下，研究词如何组织成句子。传统 NLP 曾大量依赖这些中间结构，LLM 时代可部分绕过。
* **直觉：** 学习 syntax 不是因为现代 LLM 必须显式解析，而是为了理解语言结构和传统 NLP 思路。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 3. Syntax 定义

* **定义 / 内容：** syntax 是定义词如何组织成更大单位的规则，可区分合法/非法句子；fluency 是使用句法规则的熟练程度。
* **直觉：** 母语者通常通过使用隐式学会句法，二语学习者常显式学习规则。NLP 模型要么显式编码规则，要么从数据中学习。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 4. n-gram/HMM 建模 shallow syntax

* **定义 / 内容：** shallow syntax 是局部结构，如 `the + noun`、subject-verb agreement；n-gram/HMM 只用近邻历史，难处理 long-range dependencies。deep syntax 涉及全局依赖，如 `books ... are`。
* **直觉：** `The books that I bought yesterday are expensive` 中 `are` 依赖远处的 `books`，局部模型很难捕捉。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Constituent

* **定义 / 内容：** constituent 是作为单一语法单位的一组词；可在句中移动而语义基本不变；不能随意拆开仍保持语义。
* **直觉：** 例如时间短语 `On September seventeenth` 可以移动位置，但拆开后句子语义/语法会坏掉。constituent 是 parse tree 的基本单元。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 6. Context-free grammar 概念

* **定义 / 内容：** CFG 用数学规则建模 constituent structure。context 是 constituent 外部元素；context-free 表示 constituent 的角色/语义不随外部位置改变。
* **直觉：** CFG 假设短语一旦形成，其内部结构可独立于外部环境处理，这使解析可以递归分解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. CFG 四元组、推导、语言

* **定义 / 内容：** CFG $G=(N,Σ,R,S)$：非终结符集合 N、终结符集合 Σ、产生式 R、起始符号 S。若 $A→β$，则 $αAγ⇒αβγ$。语言 $L_G={w∈Σ* | S⇒*w}$。
* **直觉：** 非终结符如 NP/VP，终结符是实际词。推导从 S 开始，通过规则最终生成词序列。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. CFG 示例与 parse tree

* **定义 / 内容：** 产生式可分为生成非终结符的规则和生成终结符的规则；parse tree 表示多步推导。
* **直觉：** 底层终结符规则类似确定性 POS tagging；上层规则组合短语结构，如 S→NP VP。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 9. CFG：句子类型规则

* **定义 / 内容：** Declarative：$S→NP\ VP$；Imperative：$S→VP$；Yes-no question：$S→Aux\ NP\ VP$；Wh-structure：$S→Wh\text{-}NP\ VP$ 或 $S→Wh\text{-}NP\ Aux\ NP\ VP\ PP$。
* **直觉：** 不同句型可由不同的 S 产生式覆盖。CFG 可显式描述英语句法模式。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. CFG：Noun Phrase

* **定义 / 内容：** $NP→(Det)(Card)(Ord)(Quant)(AP)Nominal$；括号表示可选；$AP→(RB)JJ$；Det 可为简单限定词或 possessive `NP's`，并可递归。
* **直觉：** `Denver’s mayor’s mother’s canceled flight` 展示了递归结构：NP 可嵌套在 Det 中。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. CFG：Nominal

* **定义 / 内容：** 简单 nominal：$Nominal→Noun$；复杂 nominal 包括 $Nominal→Nominal\ Noun$、$Nominal→Nominal\ PP$、$Nominal→Nominal\ Gerundive\text{-}VP$、$Nominal→Nominal\ ed\text{-}VP$、$Nominal→Nominal\ infinitive$、relative clause。
* **直觉：** Nominal 是名词短语内部的核心结构，可不断附加修饰成分，如 `flight to Boston`、`flight leaving before 10`。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. CFG：Verb Phrase

* **定义 / 内容：** VP 规则包括 $VP→Verb$、$VP→Verb\ NP$、$VP→Verb\ NP\ PP$、$VP→Verb\ PP$、$VP→Verb\ VP$、$VP→Verb\ S$；$S$ 可作 sentential complement。
* **直觉：** 动词短语结构表达动作及其宾语、介词短语、补语等信息。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. CFG 来源：Penn Treebank

* **定义 / 内容：** Penn Treebank 是带 parse tree 标注的语料，可从中提取 CFG，也可作为 parser 训练数据。
* **直觉：** 人工标注树库把语言学知识转成数据，传统 parser 常依赖这类资源。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 14. The bitter lesson

* **定义 / 内容：** 问题：投入精力构建人类知识规则，还是构建可扩展模型让数据自发现知识？LLM 时代语法规则作为显式人类知识似乎不再必要。
* **直觉：** 这页强调传统 NLP 和现代深度学习的范式差异：手写结构 vs 可扩展数据驱动。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 15. Syntactic parsing 目标

* **定义 / 内容：** 给定 CFG，为句子分配合法 parse tree；寻找 root 为 S、leaves 为句子词的树。Parsing 是用 CFG 生成句子的反过程。
* **直觉：** 生成是从 S 到词；解析是从词反推可能的 S 树。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. Top-down parsing search

* **定义 / 内容：** Top-down 从根 S 开始，扩展未完成树中的非终结符；若无法匹配输入词则剪枝；匹配完整输入后停止。
* **直觉：** 优点是目标导向；缺点是可能展开大量与输入词不匹配的树。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 17. Bottom-up parsing search

* **定义 / 内容：** Bottom-up 从输入词作为 leaves 开始，用 CFG 右侧匹配生成非终结符；若找不到 RHS 匹配则剪枝；最终到达 S。
* **直觉：** 优点是直接基于输入；缺点是可能构造许多最终无法到达 S 的局部结构。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 18. Parsing ambiguity：attachment ambiguity

* **定义 / 内容：** `I saw the Grand Canyon flying to New York` 有 attachment ambiguity：`flying to New York` 可修饰 saw 的动作，也可错误地修饰 Grand Canyon。
* **直觉：** 同一句子可能有多棵 parse tree；解析不仅要找合法树，还要选语义上合理的树。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Parsing ambiguity：coordination ambiguity

* **定义 / 内容：** `old men and women` 可解释为 old 修饰 men and women，也可解释为 old 只修饰 men。
* **直觉：** 并列结构的范围不明确，是 syntactic ambiguity 的典型来源。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 20. Repeated subproblems

* **定义 / 内容：** 搜索中会反复构造相同局部树，如 $Det→that$、$Noun→flight$、$NP→Det\ NOM$。这些重复子问题浪费计算。
* **直觉：** 这正是动态规划能优化 parsing 的原因：同一 span 的解析结果应缓存复用。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 21. CYK parsing algorithm

* **定义 / 内容：** CYK 是 DP 算法：子问题是解析句子子片段；成功解析大 constituent 必须成功解析其子部分；缓存成功 parse。
* **直觉：** CYK 避免反复解析同一短语，只保留能生成某个 span 的非终结符。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 22. CNF CFG

* **定义 / 内容：** Chomsky Normal Form 限制规则为 $A→BC$ 或 $A→a$。可通过转换处理 mixed RHS、unit production、long RHS，表达能力不丢失。
* **直觉：** CNF 让每个非叶节点都有两个非终结符孩子，便于用二维表枚举左右切分点。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. CNF 转换算法

* **定义 / 内容：** 转换步骤：处理 mixed RHS；处理 unit productions；处理 long RHS。
* **直觉：** 例如 $A→Bc$ 可变成 $A→BC,\ C→c$；$A→BCD$ 可引入中间符号 $X$ 变成 $A→XD,\ X→BC$。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 24. CYK 矩阵结构

* **定义 / 内容：** 句长 n 使用 $(n+1)×(n+1)$ 矩阵；cell $(i,j)$ 存能生成 span $i..j$ 的非终结符；对 $j>i+1$ 枚举切分 $(i,k)+(k,j)$。
* **直觉：** 矩阵左侧/下方的子 span 已经求出，可用于组合更长 span。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. CYK algorithm 伪代码

* **定义 / 内容：** 先对每个词做词法规则匹配；再按 span 长度递增，枚举起点 i、终点 j、切分点 k、规则 $A→BC$。
* **直觉：** 若 $B$ 在 $(i,k)$，$C$ 在 $(k,j)$，且有规则 $A→BC$，就把 A 加入 $(i,j)$。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 26. CYK parsing examples

* **定义 / 内容：** 若最终 cell $[0,n]$ 包含起始符号 S，则找到完整 parse tree；若包含多个 S/结构，则有多棵 parse tree。
* **直觉：** CYK 不只判断可解析性，也可为后续概率解析保存多种可能结构。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 27. 扩展阅读

* **定义 / 内容：** SLP3 CFG、parsing ambiguity、CNF/CYK；Bitter Lesson。
* **直觉：** 建议把 CFG 与 CYK 当作动态规划在树结构问题上的应用来复习。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 28. Demo

* **定义 / 内容：** 生成小型 CNF CFG；随机采样句子；实现 CYK；可视化 parse tree。
* **直觉：** demo 强调从规则到句子，再从句子回到规则树的闭环。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 29. Conclusion

* **定义 / 内容：** syntax 研究词组织方式；区分 shallow/deep syntax；解析可用 search 或 DP。
* **直觉：** 本讲从 HMM 的线性序列结构扩展到树状句法结构。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 30. Quiz

* **定义 / 内容：** 判断 CFG 是否为 CNF；写出生成 `the big dog eats a fish` 的 derivations；CYK 中 $(0,2)$ 无内容，因为 `the big` 不能由给定 CFG 生成。
* **直觉：** 这页考查 CNF、推导过程和 CYK cell 的 span 含义。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 7: Probabilistic CFG、Inside/Outside、最优解析树

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；PCFG 动机；PCFG 定义；Parse tree 概率与句子概率；PCFG 三个假设；PCFG 假设示例；PCFG 的三个算法任务；Inside / Outside probability
  * Inside algorithm；Inside probability example；Outside probability；Outside algorithm base case；Outside DP 依赖 inside probability；用 inside/outside 得到句子概率；寻找最优 parse tree；最优树 running example
  * 扩展阅读；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed PCFG / Inside Notes

### 1. 从 CFG 到 PCFG

CFG 只能回答“这个句子是否合法 / 这棵树是否可能”。PCFG 给每条 rule 加概率，能比较不同 parse tree 的可能性。

对同一个 LHS，所有规则概率和为 1：

$$
\sum_{\alpha}P(A\rightarrow \alpha)=1
$$

一棵 parse tree 的概率是所有 rule 概率的乘积：

$$
P(t,w|G)=\prod_{r\in t}P(r)
$$

句子概率是所有 parse tree 概率求和：

$$
P(w|G)=\sum_t P(t,w|G)
$$

### 2. Inside Probability

Inside probability 表示一个 non-terminal 生成某个 span 的概率。

Base case：

$$
\beta_A(i,i)=P(A\rightarrow w_i)
$$

CNF 递推：

$$
\beta_A(i,j)=
\sum_{A\rightarrow BC}
\sum_{k=i}^{j-1}
P(A\rightarrow BC)\beta_B(i,k)\beta_C(k+1,j)
$$

它和 CYK 很像，只是 CYK 存“能不能生成”，Inside 存“生成概率是多少”。

### 3. Homework Example: cats catch mice

Grammar：

$$
S\rightarrow NP\ VP,\quad P=1
$$

$$
VP\rightarrow V\ NP,\quad P=1
$$

Lexicon：

$$
NP\rightarrow cats,\ P=0.5
$$

$$
NP\rightarrow mice,\ P=0.5
$$

$$
V\rightarrow catch,\ P=1
$$

Base：

$$
\beta_{NP}(1,1)=0.5
$$

$$
\beta_V(2,2)=1
$$

$$
\beta_{NP}(3,3)=0.5
$$

VP：

$$
\beta_{VP}(2,3)
=P(VP\rightarrow V\ NP)\beta_V(2,2)\beta_{NP}(3,3)
=1\cdot1\cdot0.5=0.5
$$

Sentence：

$$
\beta_S(1,3)
=P(S\rightarrow NP\ VP)\beta_{NP}(1,1)\beta_{VP}(2,3)
=1\cdot0.5\cdot0.5=0.25
$$

### 4. Outside and Viterbi-style PCFG

Outside probability 表示 span 外部的上下文概率。Inside + Outside 可用于估计某个 rule 或 constituent 在整句 parse 中出现的 posterior。

如果要找最优 parse tree，而不是句子总概率，就把 Inside 的求和换成 max，并记录 backpointer。这就是 PCFG 的 Viterbi-style parsing。

## Part II-B: Homework / Exam Connection

* Inside Algorithm homework 对应本讲：base case 是 lexical probability，recursive case 是 rule probability 乘左右 inside probability。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** 本讲将 CFG 概率化，学习 PCFG 与对应 DP 算法。
* **直觉：** CFG 只能告诉我们哪些树合法；PCFG 可以比较树的概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. PCFG 动机

* **定义 / 内容：** CYK 找所有 parse trees，但不能评价树概率、比较哪棵更可能、找最优树。`book the dinner flight` 可有多种解析。
* **直觉：** 多义句需要概率模型决定更合理结构。每个子树都是 constituent，最小树含一个 terminal。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 3. PCFG 定义

* **定义 / 内容：** 给 CFG 的每条 production 分配概率；同一 left-hand-side 的规则概率和为 1。目标可为任意 derivation 概率或句子概率。
* **直觉：** 例如 VP 的不同展开方式有不同概率，parse tree 概率是树中所有 rule 概率的乘积。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Parse tree 概率与句子概率

* **定义 / 内容：** $P(t,w_1...w_m|G)$ 是一棵树生成句子的概率；句子概率 $P(w_1...w_m|G)=Σ_t P(t,w_1...w_m|G)$。
* **直觉：** 这是对 parse tree 做 marginalization。合法树数量可能指数多，所以需要 DP。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 5. PCFG 三个假设

* **定义 / 内容：** place invariance、context-free、ancestor-free。即规则概率不依赖子树位置、外部上下文或祖先节点。
* **直觉：** 这些假设让局部 rule 概率可以独立相乘，解析概率可分解并用 DP 计算。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 6. PCFG 假设示例

* **定义 / 内容：** 通过 `the man snores` 的树展示条件概率如何因 context-free、ancestor-free、place invariance 简化为 rule 概率乘积。
* **直觉：** 下标/上标仅标记位置与重复非终结符；最终规则概率不依赖这些具体位置。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. PCFG 的三个算法任务

* **定义 / 内容：** 求句子概率；求最可能 parse tree；从训练语料用 MLE 学 PCFG。
* **直觉：** 对应 HMM 的 inference、decoding、estimation。PCFG 是树结构上的概率模型。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 8. Inside / Outside probability

* **定义 / 内容：** inside probability 类似 HMM forward；outside probability 类似 HMM backward。$N^j_{pq}$ 表示非终结符 j 推导从 p 到 q 的词。
* **直觉：** inside 看子树内部生成某 span 的概率；outside 看该 span 外部上下文与祖先结构的概率。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 9. Inside algorithm

* **定义 / 内容：** 句子概率为起始符号生成全句的 inside probability。Base：$β_j(k,k)=P(N^j→w_k)$；递推在 CNF 下枚举左右孩子和切分点。
* **直觉：** 公式：$β_j(p,q)=Σ_{r,s,d} P(N^j→N^rN^s)β_r(p,d)β_s(d+1,q)$。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 10. Inside probability example

* **定义 / 内容：** 计算 `saw stars`、`with ears`、`saw stars with ears` 等 span 的 inside probability。
* **直觉：** 长 span 的概率由所有可能切分和规则贡献相加；例如 VP 可由 $V\ NP$ 或 $VP\ PP$ 两种方式生成，要累加。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. Outside probability

* **定义 / 内容：** outside probability $α_j(p,q)$ 表示某个非终结符覆盖 span $(p,q)$ 时，生成该 span 外部词和上层结构的概率。
* **直觉：** inside 从叶子往上算，outside 从根和外部结构往下/向内传递。二者相乘可得到某个节点参与整句解析的概率贡献。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 12. Outside algorithm base case

* **定义 / 内容：** 起始符号覆盖整句的 outside probability 为 1；其他符号覆盖整句为 0。递推中目标节点可能是父节点的左孩子或右孩子。
* **直觉：** 若目标是左孩子，需要父节点 outside 与右 sibling inside；若目标是右孩子，需要父 outside 与左 sibling inside。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 13. Outside DP 依赖 inside probability

* **定义 / 内容：** outside 递推需要已解的 parent outside 和 sibling inside。条件包括目标是左/右 child，以及 sibling 的起止位置。
* **直觉：** 这就是树结构中的 message passing：一个节点的外部概率由其所有可能父结构和兄弟结构贡献累加。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 14. 用 inside/outside 得到句子概率

* **定义 / 内容：** 若非终结符 j 覆盖 span $[p,q]$，整句概率中包含 $α_j(p,q)β_j(p,q)$。对某 span 上所有可能非终结符求和，可得含该 span 的句子概率贡献。
* **直觉：** 这也是后续估计 PCFG soft counts 的基础：某规则/节点出现的 posterior 可由 inside/outside 组合得到。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. 寻找最优 parse tree

* **定义 / 内容：** 最优树 decoding 类似 Viterbi；CYK-like algorithm 把 inside 的求和换成 max，并记录 backpointers。
* **直觉：** 在 cell $(p,q)$ 中保存最大概率 $δ_j(p,q)$ 和来源 $ψ_j(p,q)$，最后从右上角回溯重构最优树。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. 最优树 running example

* **定义 / 内容：** `saw stars with ears` 的 VP 最优解析比较两种候选：$V\ NP$ 与 $VP\ PP$；取最大 $0.009072$，backpointer 记录 $(V,NP,2)$。
* **直觉：** 与 inside algorithm 相比，decoding 不累加所有树，而是保留最大概率树的结构来源。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. 扩展阅读

* **定义 / 内容：** FSNLP PCFG 与 parsing with PCFG。
* **直觉：** 可补充 PCFG 参数估计、树库训练与概率解析器细节。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 18. Demo

* **定义 / 内容：** 用 toy PCFG 演示 Inside 和 Outside algorithm。
* **直觉：** demo 应重点观察表格填充方向：inside 自底向上，outside 自顶向下结合 sibling。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 19. Conclusion

* **定义 / 内容：** PCFG 是概率版 CFG；可在多棵树中找最优树；Inside/Outside 算句子概率；CYK-like 算最优解析树。
* **直觉：** 本讲把树结构的合法性问题变为概率推理和概率解码问题。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 20. Quiz

* **定义 / 内容：** 同一 LHS 的 PCFG 规则概率应和为 1；inside 是 DP；outside 需要 inside 结果；用 CYK 找最优树要求 PCFG 为 CNF。
* **直觉：** 关键易错点是 outside 并非完全独立，它需要 sibling 的 inside probability。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 8: Neural Network 与 RNN Language Modeling

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；n-gram 的动机问题；n-gram 三个问题；神经网络作为解决方案；Logistic regression 是神经网络；从 logistic regression 到 MLP；神经网络是堆叠的 logistic 模型；Vectorization
  * RNN 基本思想；RNN 结构、参数与训练数据；RNN 计算公式；RNN training 与 BPTT；为什么需要 tanh 非线性；扩展阅读；Demo；Conclusion
  * Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Neural LM Notes

### 1. 为什么 n-gram 不够：稀疏、复杂度、固定窗口

n-gram language model 的概率来自计数。它能工作，是因为短局部上下文里常有强统计规律；它的问题也正来自这里：当上下文稍微变长，可能的组合数量会指数级增加。

例如要估计：

$$
P(\text{professor}|\text{and asked the})
$$

训练集中必须足够多次出现 `and asked the professor` 这类片段，否则概率估计会不稳定；如果完全没见过，MLE 甚至会给 0。增大 $n$ 看似能捕捉更长上下文，但也会让 data sparsity 和 model complexity 更严重。

神经网络语言模型换了一个思路：不再显式记住每个短语的频率，而是学习一个共享参数的函数：

$$
P_\theta(w_t|context)
$$

这样相似上下文可以共享表示，未见过的组合也可以通过 embedding 和 hidden state 泛化。

### 2. Logistic Regression 是最小的神经网络

二分类 logistic regression 可以写成：

$$
z=w^Tx+b
$$

$$
a=\sigma(z)=\frac{1}{1+\exp(-z)}
$$

binary cross-entropy loss：

$$
\ell(a,y)=-\log(a^y(1-a)^{1-y})
=-y\log a-(1-y)\log(1-a)
$$

这个模型已经具备神经网络的基本结构：输入经过线性变换，接一个非线性函数，再用 loss 衡量输出和标签的差距。更深的网络只是把这种可微计算图堆叠起来。

### 3. MLP：隐藏层和非线性让模型表达复杂模式

一个隐藏层 MLP 可写成：

$$
z^{[1]}=W^{[1]}x+b^{[1]}
$$

$$
a^{[1]}=g(z^{[1]})
$$

$$
z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}
$$

隐藏层把原始输入变成中间特征；非线性 $g$ 很关键。如果没有非线性，多层线性变换会合并成一个线性变换：

$$
W_2(W_1x)=W'x
$$

这样就失去“深层”的意义。ReLU、sigmoid、tanh 都是为了让模型能表示非线性语言规律。

### 4. Vectorization：从逐 neuron 计算到矩阵计算

课件强调 vectorization，因为神经网络的速度来自矩阵运算。逐个 neuron 写循环：

$$
z_j=w_j^Tx+b_j
$$

可以合并为：

$$
z=Wx+b
$$

这不仅代码更简洁，也能让 GPU 一次性并行处理大量乘加操作。后面的 Transformer、LoRA、FLOPs、quantization 都默认我们已经把模型看成矩阵计算。

### 5. RNN：用 hidden state 压缩历史

RNN 的核心递推是：

$$
h_t=\tanh(Wh_{t-1}+Ux_t+b)
$$

$$
o_t=Vh_t+c
$$

$$
\hat{y}_t=softmax(o_t)
$$

其中 $h_t$ 是到第 $t$ 步为止的历史摘要。语言模型中，训练目标通常是预测下一个 token：

$$
L=-\sum_t\log P_\theta(w_{t+1}|w_{\le t})
$$

POS tagging 中，输出可以是当前位置或下一位置的 POS tag。区别在输出含义，递归结构类似。

### 6. BPTT 与梯度消失 / 爆炸

RNN 训练时要把时间展开成一个很深的网络，再用 **Back-Propagation Through Time (BPTT)** 反传。长序列中，早期 hidden state 到后期 loss 的梯度会反复乘上 $W$ 的相关项。

如果矩阵乘法让梯度范数不断变小，就出现 **vanishing gradient (梯度消失)**；如果不断变大，就出现 **exploding gradient (梯度爆炸)**。tanh 把 hidden value 限制在 $[-1,1]$，能稳定数值，但不能彻底解决长程依赖。

### 7. Exam Focus

* RNN 的 hidden-to-hidden 矩阵 $W$ 形状由 hidden state 维度决定；若 hidden state 维度是 $n$，则 $W\in\mathbb{R}^{n\times n}$。
* 输入 embedding 维度影响 $U$，不是 $W$。
* RNN 不只用于 language modeling，也可用于 POS tagging、sequence classification、seq2seq encoder 等。
* n-gram 的窗口是固定的；RNN 理论上可保留任意长历史，但实际受优化和容量限制。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** 本讲从 n-gram 过渡到神经网络语言模型，重点是 RNN。
* **直觉：** 神经模型不再记忆每个 n-gram 频率，而是用参数化函数预测概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. n-gram 的动机问题

* **定义 / 内容：** n-gram 必须足够多次看到 `and asked the professor` 等序列，才能估计 $P(professor|and asked the)$；没见过则概率为 0，见得少则不准。
* **直觉：** count-based 模型依赖频率，数据稀疏时泛化差。长上下文会让组合数量爆炸。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. n-gram 三个问题

* **定义 / 内容：** data sparsity、model complexity、fixed-window architecture。增大 n 能看更长上下文，但会加剧稀疏和参数爆炸。
* **直觉：** n-gram 用固定长度历史，缺乏灵活地捕捉不同范围依赖的能力。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. 神经网络作为解决方案

* **定义 / 内容：** 神经网络用固定参数集做预测，通过 architecture 处理不同长度依赖；RNN 强加序列结构，Transformer 用 attention，SFT/RLHF 用任务和损失塑造模型。
* **直觉：** 参数共享让模型能泛化到未见序列；结构设计决定模型如何使用上下文。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Logistic regression 是神经网络

* **定义 / 内容：** $a=σ(Σ_i w_i x_i+b)$；binary cross-entropy loss：$ℓ(a,y)=-log(a^y(1-a)^{1-y})$。计算图支持 forward 与 back-propagation。
* **直觉：** forward 计算输出和损失；backprop 用链式法则求参数梯度。复杂神经网络本质上是更多层/更多节点的可微计算图。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. 从 logistic regression 到 MLP

* **定义 / 内容：** MLP 有输入层、隐藏层、输出层；隐藏层有多个 neuron；loss 可按任务设计，如 negative likelihood。
* **直觉：** 隐藏层引入中间表示，使模型能学习非线性特征，而不是直接从输入线性预测输出。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 7. 神经网络是堆叠的 logistic 模型

* **定义 / 内容：** 每个 neuron 计算 logit $z_j^{[1]}=W_j^{[1]T}x+b_j^{[1]}$，再经非线性 $a_j^{[1]}=σ(z_j^{[1]})$ 或 ReLU。
* **直觉：** 非线性是关键；没有非线性，多层线性映射会坍缩为单层线性映射。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Vectorization

* **定义 / 内容：** 逐个计算 hidden neuron 很慢；可向量化为 $z^{[1]}=W^{[1]}x+b^{[1]}$，$a^{[1]}=σ(z^{[1]})$。
* **直觉：** 向量化把多个 neuron 的计算合并为矩阵运算，是 GPU 加速神经网络的基础。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 9. RNN 基本思想

* **定义 / 内容：** RNN 用模型预测概率而非记忆概率；hidden state $h^(t)=f(h^(t-1),x^(t);θ)$ 总结过去信息；同一函数/参数在不同时间步复用。
* **直觉：** 参数共享让 RNN 能处理任意长度序列，并在理论上把过去历史压缩进 hidden state。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. RNN 结构、参数与训练数据

* **定义 / 内容：** 输入序列转为 embeddings；hidden states 传递历史；输出 units 给出 $y_t$ 概率；参数包括 $U,W,V,b,c$。语言模型中 $y_t=x_{t+1}$，POS tagging 中 $y_t=POS(x_{t+1})$。
* **直觉：** $U$ 映射输入到 hidden，$W$ 映射前一 hidden 到当前 hidden，$V$ 映射 hidden 到输出。RNN 可用于多种序列预测任务。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. RNN 计算公式

* **定义 / 内容：** 常见形式：$h_t=tanh(Wh_{t-1}+Ux_t+b)$；$o_t=Vh_t+c$；$ŷ_t=softmax(o_t)$。
* **直觉：** hidden state 把当前 token 与过去摘要融合；softmax 输出多类概率分布，如下一个词或 POS tag。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 12. RNN training 与 BPTT

* **定义 / 内容：** 使用 negative log-likelihood / perplexity 作为 loss；通过 back-propagation through time (BPTT) 更新参数；挑战是梯度消失/爆炸。
* **直觉：** RNN 展开后像一个很深的网络，长序列会让梯度连续乘很多矩阵，因此早期 token 的学习信号可能衰减或爆炸。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. 为什么需要 tanh 非线性

* **定义 / 内容：** 没有非线性，递归会坍缩为线性映射；语言不能只靠线性函数建模。$W^t$ 的矩阵幂可能导致信息消失或爆炸；tanh 把值压到 [-1,1]。
* **直觉：** tanh 有助于稳定 hidden state 和优化，但不能完全解决长程依赖问题，后续 LSTM/attention/Transformer 会继续改进。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 14. 扩展阅读

* **定义 / 内容：** Goodfellow《Deep Learning》第 6 和 10 章，RNN 的长程依赖、优化挑战、LSTM 变体。
* **直觉：** RNN 是理解后续 seq2seq 和 attention 的基础。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 15. Demo

* **定义 / 内容：** 定义 RNN；在 toy corpus 上优化 RNN。
* **直觉：** demo 关注输入 embedding、hidden state 更新、输出概率、NLL loss 和训练循环。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. Conclusion

* **定义 / 内容：** 神经网络用固定大小模型压缩文本数据；比 n-gram 更灵活；RNN 是序列神经模型，可建模词依赖；tanh 是让 RNN 可用的重要选择。
* **直觉：** 本讲承上启下：从 count-based LM 过渡到 neural sequence model。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 17. Quiz

* **定义 / 内容：** n-gram 不能建模超过窗口的长程依赖；RNN 不只用于 language modeling；若 hidden state 维度 n，则 $W$ 大小为 $n×n$。
* **直觉：** $W$ 是 hidden-to-hidden matrix，因此输入 embedding 维度 m 不影响 $W$，而影响 $U$ 的大小。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 9: Machine Translation、Rule-based MT、IBM Model 1

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；古代 Rosetta Stone 与翻译问题；翻译技巧与现代 Rosetta Stone；机器翻译与 LLM；IBM Model 1 与 LLM 翻译对比；语言差异：lexical 与 syntactic；语言差异：word ordering；Vauquois triangle
  * Rule-based direct translation；Direct method 的缺陷；Rule-based transfer method；翻译评价：fluency 与 faithfulness；翻译评价与“信达雅”；MT 目标的概率形式；Word alignment；Alignment matrix
  * Alignment 的复杂情况；IBM Model 1 generative story；IBM Model 1 概率；IBM Model 1 问题；扩展阅读；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Machine Translation Notes

### 1. MT 的基本问题：从 Rosetta Stone 到平行语料

机器翻译的核心是：给定源语言句子 $F$，生成目标语言句子 $E$。Rosetta Stone 的故事说明了 translation 的统计基础：如果同一内容以多种语言出现，就可以通过模式、位置和共现关系推断语言之间的映射。

现代统计 MT 使用 parallel corpus，例如 Canadian Hansards。平行语料提供大量句子对，让模型学习哪些词、短语或结构在两种语言之间对应。

### 2. 语言差异：翻译不是查字典

课件给了几类语言差异：

* **Lexical ambiguity (词汇歧义)**：`bass` 可能是乐器，也可能是鱼。
* **Lexical granularity (词义粒度差异)**：中文区分哥哥/弟弟，英语只说 brother。
* **Syntactic difference (句法差异)**：形容词位置、性别一致、PP 位置会变化。
* **Word order (词序差异)**：英语常见 SVO，日语常见 SOV。

所以翻译系统不仅要翻词，还要决定词序、语法一致性、上下文语义和目标语言自然度。

### 3. Rule-based MT 与 Vauquois Triangle

Vauquois triangle 把翻译方法按抽象程度分层：

* **Direct translation (直接翻译)**：词表替换 + 局部重排 + 形态生成。
* **Transfer method (转换法)**：先解析源语言结构，再转换成目标语言结构。
* **Semantic / interlingua level**：先抽象到语言无关的意义表示，再生成目标语言。

direct method 可处理 `green witch -> bruja verde` 这种局部重排，但难处理长程结构差异。transfer method 更强，但依赖 parser 和人工规则。

### 4. Fluency 和 Faithfulness 的概率分解

传统统计 MT 把好翻译理解成两个目标的乘积：

* **Fluency (流畅性)**：目标句 $E$ 是否像自然目标语言。
* **Faithfulness (忠实性)**：$E$ 是否保留源句 $F$ 的意义。

形式化目标：

$$
E^*=\arg\max_E P(E|F)
$$

用 Bayes rule：

$$
E^*=\arg\max_E P(F|E)P(E)
$$

其中：

* $P(E)$ 是 language model，负责 fluency。
* $P(F|E)$ 是 translation model，负责 faithfulness。
* $P(F)$ 与 $E$ 无关，所以在 argmax 中省略。

### 5. Word Alignment 是隐藏变量

word alignment 表示源语言词和目标语言词的对应关系。设外语句子：

$$
F=(f_1,\ldots,f_J)
$$

英语句子：

$$
E=(e_1,\ldots,e_I)
$$

alignment 用 $a_j$ 表示第 $j$ 个外语词 $f_j$ 对齐到英语第几个词：

$$
a_j\in\{0,1,\ldots,I\}
$$

$0$ 可表示 NULL，用来处理没有显式对应的词。真实翻译可能 one-to-one、one-to-many、many-to-one、many-to-many，因此 alignment 是统计 MT 的核心隐变量。

### 6. IBM Model 1

IBM Model 1 的 generative story：

1. 给定目标句 $E$。
2. 生成源句长度 $J$。
3. 生成 alignment $A$。
4. 对每个位置 $j$，根据 $e_{a_j}$ 生成 $f_j$。

联合概率：

$$
P(F,A|E)=P(J|I)P(A|I,J)\prod_{j=1}^J P(f_j|e_{a_j})
$$

因为 alignment 不可见，要求和：

$$
P(F|E)=\sum_A P(F,A|E)
$$

Model 1 的问题也很清楚：它把词当作 bag-of-words，不建模位置 locality；词翻译只依赖词对，不看上下文；也不能自然表示 many-to-many 短语翻译。

### 7. Exam Focus

* $P(E)$ 对应 fluency；$P(F|E)$ 对应 faithfulness。
* alignment matrix 可以表达 one-to-one、one-to-many、many-to-one。
* IBM Model 1 的 alignment 通常过于简单，所有位置可能等概率。
* direct translation 很难处理 long-range dependency 和大范围重排序。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** 本讲进入机器翻译（MT）。
* **直觉：** 机器翻译是 NLP 经典任务，也是 alignment、language model、sequence model 的重要应用场景。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. 古代 Rosetta Stone 与翻译问题

* **定义 / 内容：** Rosetta Stone 用三种文字记录同一内容：hieroglyphic、demotic、Greek。问题是如何用已知语言读未知语言。
* **直觉：** 机器翻译的核心思想与此类似：利用平行文本中的对应关系，学习语言之间的映射。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 3. 翻译技巧与现代 Rosetta Stone

* **定义 / 内容：** 通过未知语言模式统计、位置相似性、尝试多种 alignment、寻找一致映射来推断翻译。Canadian Hansards 是英法平行语料，可作为现代 Rosetta Stone。
* **直觉：** 统计机器翻译的基础是：大量平行文本中共同出现和位置对应的词/短语，暗示翻译关系。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. 机器翻译与 LLM

* **定义 / 内容：** 翻译曾是 NLP 活跃领域，需要大规模平行语料和计算基础设施；LLM 时代，模型可在任意上下文下生成语言。
* **直觉：** 传统 MT 是特定任务系统；LLM 把翻译吸收到通用语言生成能力中。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 5. IBM Model 1 与 LLM 翻译对比

* **定义 / 内容：** IBM Model 1 以单词为基本单位，基于统计频率；LLM 以上下文/token 为单位，更能处理语义推理、idiom、文化适配和用户约束。
* **直觉：** IBM Model 1 像“词袋翻译”，而 LLM 能根据上下文决定表达方式，例如 honorifics、单位换算、习语等。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 6. 语言差异：lexical 与 syntactic

* **定义 / 内容：** 词汇层差异：`bass` 在西语可为乐器或鱼，`wall` 在德语有室内/室外区分，`brother` 在中文区分哥哥/弟弟。句法层：法语/西语形容词有性别变化。
* **直觉：** 翻译不是简单查词典，还需要根据上下文、语法和目标语言习惯选择正确表达。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. 语言差异：word ordering

* **定义 / 内容：** 英语多为 SVO，日语多为 SOV；介词短语位置、形容词-名词顺序也不同。
* **直觉：** 逐词翻译会产生错误词序，例如日语直译可能像 “he music to listening adores”。翻译系统必须处理重排序。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 8. Vauquois triangle

* **定义 / 内容：** 翻译可分层：direct word-level transfer、syntactic transfer、semantic transfer、interlingua。
* **直觉：** 越往三角形上方，模型越抽象地理解源句再生成目标句；越下方越接近词表替换。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Rule-based direct translation

* **定义 / 内容：** direct translation 流程包括 morphology analysis、lexical transfer、local reordering、morphological generation。例子：英语 `green witch` 到西语 `bruja verde` 需要局部重排。
* **直觉：** direct 方法能处理短语内局部词序，但难处理大范围结构调整。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Direct method 的缺陷

* **定义 / 内容：** direct method 不能处理更大语言单位的顺序差异，如 PP 位置、句子级重排序。例子：英语到德语、中文到英语的短语位置不同。
* **直觉：** 要决定短语放哪里，需要全局句法信息，而不是只做词表查找和局部调整。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. Rule-based transfer method

* **定义 / 内容：** transfer method 包括 syntactic transfer（把源语言 parse tree 变成目标语言 parse tree）和 lexical transfer（词到词翻译）。
* **直觉：** 它比 direct 方法更结构化，能处理 SVO/SOV 等句法差异，但依赖高质量 parser 和大量规则。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. 翻译评价：fluency 与 faithfulness

* **定义 / 内容：** fluency 是目标语是否自然流畅；faithfulness 是是否忠实源文。例如 “the Lord will look after me” 流畅但不够忠实；冗长解释忠实但不流畅。
* **直觉：** 优秀翻译需要平衡自然表达和意义保真。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. 翻译评价与“信达雅”

* **定义 / 内容：** 形式化目标：$best T = argmax_T fluency(T) faithfulness(T,S)$。严复“信达雅”：信=意义准确，达=通顺明白，雅=表达得体优雅。
* **直觉：** 统计翻译中 fluency 通常由 language model 表示，faithfulness 由 translation model 表示。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. MT 目标的概率形式

* **定义 / 内容：** 设目标语言英文 $E=(e_1,...,e_I)$，源语言外语 $F=(f_1,...,f_J)$。通过 Bayes：$E*=argmax_E P(E|F)=argmax_E P(F|E)P(E)$。
* **直觉：** $P(E)$ 是 fluency language model，$P(F|E)$ 是 faithfulness translation model。$P(F)$ 与 E 无关，可省略。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Word alignment

* **定义 / 内容：** word alignment 是把目标词 E 映射到源词 F；多个源词可映射到同一个目标词。可用 $a_j$ 记录第 j 个源词对齐到哪个目标词。
* **直觉：** alignment 是统计翻译的隐变量：我们不知道翻译时哪个词对应哪个词，但模型要估计这种关系。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 16. Alignment matrix

* **定义 / 内容：** 矩阵行是目标词，列是源词，$X$ 表示对齐。例如 `implemented` 可对应 `mis/en/application` 多个法语词。
* **直觉：** 矩阵可表达一对一、多对一、一对多等关系，是可视化 alignment 的常用方式。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 17. Alignment 的复杂情况

* **定义 / 内容：** 一个源词可映射多个目标词；可存在 many-to-many mapping；可加入 NULL 处理无法对应的词。
* **直觉：** 真实翻译不是严格逐词对应，NULL 和多对多是处理虚词、省略、习语的重要机制。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 18. IBM Model 1 generative story

* **定义 / 内容：** IBM Model 1 输入 F、输出 E；生成过程：生成外语长度 J；生成 alignment A；根据对齐生成外语词 F。
* **直觉：** 它是 generative model：假设目标句 E 先存在，再生成源句 F 与 alignment。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 19. IBM Model 1 概率

* **定义 / 内容：** 生成长度概率 $P(J|I)$；alignment 概率通常均匀；词翻译概率 $P(f_j|e_{a_j})$；$P(F,A|E)=P(J|I)P(A|I,J)∏_j P(f_j|e_{a_j})$，$P(F|E)=Σ_A P(F,A|E)$。
* **直觉：** alignment 是隐藏变量，因此要求和边缘化。Model 1 简单、可训练，但假设很强。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. IBM Model 1 问题

* **定义 / 内容：** Bag-of-Words assumption：所有 alignment 距离等可能；Independent Word Translation：词翻译只依赖词对，不看上下文；One-to-Many 限制不能自然处理 many-to-many。
* **直觉：** 这导致它无法处理词序、上下文词义消歧和复杂短语/习语翻译。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. 扩展阅读

* **定义 / 内容：** FSNLP alignment/MT；2025 年关于 LLM 时代 MT 挑战的论文；古籍机器翻译相关论文。
* **直觉：** 传统 MT 的经典问题在 LLM 时代仍有研究价值，尤其在低资源语言、古籍、文化语境中。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 22. Demo

* **定义 / 内容：** 课件仅给出 Demo 页标题。
* **直觉：** 根据前后内容，demo 可能围绕 alignment 或 IBM Model 1 训练/可视化展开。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 23. Conclusion

* **定义 / 内容：** MT 是把一种语言文本转换为另一种语言；有 rule-based 和 statistics-based 方法；LLM 正在改变 MT。
* **直觉：** 本讲从规则系统过渡到统计系统，为下一讲 HMM alignment、decoding 和 seq2seq 做准备。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 24. Quiz

* **定义 / 内容：** alignment matrix 可表示 one-to-one、one-to-many、many-to-one；$P(E|F)$ 不是 fluency，而是整体翻译后验；direct translation 不能处理 long-range dependencies。
* **直觉：** fluency 单独由 $P(E)$ 表示；faithfulness 由 $P(F|E)$ 表示。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 10: HMM Alignment、Decoding Search、BLEU、Seq2Seq、Attention

## Part I: Lecture Map

* **本讲覆盖路径：**
  * 课程标题页；IBM Model 1 的弱点；用 HMM 改造 translation alignment；HMM translation 的 Markov 假设与模型；Alignment locality 与 jump model；Translation decoding；Search-based methods；Best-first search
  * A* search；Beam search；BLEU 直觉；BLEU 计算；BLEU pitfalls；Seq2seq encoder-decoder；Seq2seq 结构细节；Seq2seq training
  * Seq2seq issue：fixed-length bottleneck；Seq2seq issue：长程依赖、梯度与并行性；Attention mechanism 直觉；Attention mechanism 公式与优缺点；扩展阅读；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed MT Decoding / Seq2Seq / Attention Notes

### 1. HMM Alignment：给 alignment 加上 locality

IBM Model 1 的弱点是忽略词序。HMM alignment model 把 alignment 位置看成 hidden state，把源语言词看成 observation。

Markov 假设：

$$
P(a_j|a_1,\ldots,a_{j-1},E)\approx P(a_j|a_{j-1},I)
$$

发射假设：

$$
P(f_j|history,E,A)\approx P(f_j|e_{a_j})
$$

联合概率：

$$
P(F,A|E)=P(J|I)\prod_{j=1}^J P(a_j|a_{j-1},I)P(f_j|e_{a_j})
$$

这里 $P(a_j|a_{j-1},I)$ 是 jump model，鼓励相邻源词对齐到相邻目标词，因为真实翻译通常有 locality。

### 2. Translation Decoding 是搜索问题

训练 alignment/translation model 后，解码目标是：

$$
\hat{E}=\arg\max_E P(F|E)P(E)
$$

候选译文空间巨大，不能枚举。因此需要 search-based decoding：

* **Best-first search**：每次扩展当前分数最高的 partial translation，容易短视。
* **A\* search**：用 $f(p)=g(p)+h(p)$，其中 $g$ 是当前翻译质量，$h$ 是未来未翻译部分的估计。
* **Beam search**：每轮只保留 top-$k$ partial translations，是质量和速度的折中。

beam 太小会丢掉未来更优候选；beam 太大计算贵，也不一定保证更好。

### 3. BLEU：自动评价的优点和陷阱

BLEU 用候选翻译与参考翻译的 n-gram overlap 来近似质量。对 $n=1,2,3,4$ 分别算 precision，再做几何平均：

$$
BLEU\propto \exp\left(\frac{1}{N}\sum_{n=1}^N\log p_n\right)
$$

unigram precision 更像词义覆盖，高阶 n-gram 更像局部词序和流畅度。

主要陷阱：

* 很短的候选句可能 precision 很高，比如只输出 `the`。
* 重复词会虚增匹配，所以需要 modified precision，把命中次数限制在参考译文出现次数内。
* BLEU 是表面 n-gram 指标，不等同于真正语义正确；同义改写可能被低估。

### 4. Seq2Seq：从统计 MT 到神经 MT

seq2seq encoder-decoder 用一个 RNN 编码源句，用另一个 RNN 生成目标句。encoder 递推：

$$
h_i=f(h_{i-1},x_i)
$$

decoder 生成：

$$
P(y_t|y_{<t},x)=softmax(g(s_t))
$$

训练时常用 **teacher forcing**：decoder 的上一步输入用真实词 $y_{t-1}$，而不是模型预测词。loss 是目标序列的 NLL：

$$
L=-\sum_t\log P_\theta(y_t|y_{<t},x)
$$

### 5. Fixed-length Bottleneck 和 Attention

经典 seq2seq 把整个源句压进一个 fixed-length vector。长句时，这个向量必须同时保存所有词义和词序，负担过重。

attention 的想法是：每生成一个目标词，都动态查看源句 hidden states。设 decoder state 是 query，encoder states 是 keys/values：

$$
e_{t,i}=score(s_t,h_i)
$$

$$
\alpha_{t,i}=softmax(e_{t,i})
$$

$$
c_t=\sum_i\alpha_{t,i}h_i
$$

context vector $c_t$ 再和 decoder state 一起预测当前词。这样模型不需要把所有信息压缩进最后一个 hidden state。

### 6. Exam Focus

* HMM alignment 处理 locality，但不能自然处理 many-to-many。
* BLEU-1 precision 高不代表翻译好，尤其是短句。
* Seq2seq 的主要弱点：fixed-length bottleneck、长程依赖、错误累积、RNN 不能并行。
* Attention 缓解信息瓶颈，但如果 encoder 仍是 RNN，训练并行性问题还在。

## Part III: Concept Coverage from Lecture Materials

### 1. 课程标题页

* **定义 / 内容：** 本讲继续机器翻译，从 IBM Model 1 的弱点引出 HMM、搜索解码、BLEU、seq2seq 与 attention。
* **直觉：** 这是传统统计 MT 到神经 MT 的桥梁。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. IBM Model 1 的弱点

* **定义 / 内容：** Model 1 假设词独立生成、所有 alignment 等概率；但真实中多个词可能联合生成，且相邻源词通常对齐到相邻目标词。
* **直觉：** 例如法语最后三个词可共同对应英文一个词，需要联合分布；alignment locality 说明词序信息不可忽略。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 3. 用 HMM 改造 translation alignment

* **定义 / 内容：** HMM alignment model 生成 alignment 和 foreign sentence。POS tag 类比为目标句位置；observed word 类比为 foreign word；transition 负责下一个对齐位置；emission 负责根据对齐位置生成 foreign word。
* **直觉：** 把目标句中的位置当作 hidden state，源句词按顺序生成，就能建模 alignment 的连续性。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. HMM translation 的 Markov 假设与模型

* **定义 / 内容：** 假设 $P(a_j|history,E)=P(a_j|a_{j-1},I)$，$P(f_j|history,E,A)=P(f_j|e_{a_j})$。最终 $P(F,A|E)=P(J|I)∏_j P(a_j|a_{j-1},I)P(f_j|e_{a_j})$。
* **直觉：** 与 HMM POS tagging 完全类比：alignment 状态会转移，并发射源语言词。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Alignment locality 与 jump model

* **定义 / 内容：** $P(a_j|a_{j-1},I)$ 应鼓励 locality；若 $a_j$ 接近 $a_{j-1}$，概率更高。模型关注 jump $|a_j-a_{j-1}|$，而非绝对位置。
* **直觉：** 连续 foreign words 的英文来源通常也相近，所以大的跳跃应低概率。$P(7|6,I=15)=P(9|8,I=15)$ 体现相对位移思想。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Translation decoding

* **定义 / 内容：** 已知 alignment model 后，要找 $Ê=argmax_E P(F|E)P(E)$。这不同于已知 E/F 时做 alignment。一般带 bigram LM 的 decoding 是 NP-complete；HMM 在特定假设下可用 Viterbi。
* **直觉：** 翻译生成要在巨大候选句空间中搜索，难度高于对齐两个已知句子。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Search-based methods

* **定义 / 内容：** 一般翻译需要 greedy 和 heuristic search；状态节点是 partial translation；一个下划线覆盖 F 中一段词，其上方是该片段翻译。
* **直觉：** 解码可看成从空翻译逐步扩展候选短语，形成搜索树。评分函数指导扩展方向。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Best-first search

* **定义 / 内容：** Best-first 每次扩展当前最高分节点，并把扩展结果入队；缺点是昂贵且短视，容易落入局部最优。
* **直觉：** 高局部分数不保证后续能形成好句子；翻译需要考虑未来上下文。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 9. A* search

* **定义 / 内容：** A* 用 $f*(p)=g(p)+h*(p)$，其中 $g(p)$ 是当前 partial translation 质量，$h^*(p)$ 是未翻译部分的未来质量估计。
* **直觉：** 未来估计很贵，需要 heuristic，例如用 phrase table 给剩余词一个简化翻译概率。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Beam search

* **定义 / 内容：** Beam 是当前 top-k 状态集合；每轮扩展 beam 中所有状态，只保留 top-k 扩展。
* **直觉：** Beam search 是效率与质量折中。k 越大越接近全搜索但越慢；k 太小容易丢掉未来更优路径。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. BLEU 直觉

* **定义 / 内容：** BLEU 自动评估候选翻译与参考翻译的 n-gram 匹配频率。示例中计算 unigram precision。
* **直觉：** 候选译文如果很多 n-gram 与人工参考重合，通常更可能忠实且流畅；但 BLEU 只是表面匹配指标。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 12. BLEU 计算

* **定义 / 内容：** 对 n=1,2,3,4 计算 n-gram precision，再取几何平均作为 BLEU 主要部分。
* **直觉：** unigram 更偏词义覆盖，高阶 n-gram 更偏局部流畅度和词序。实际 BLEU 还会考虑短句惩罚和 modified precision。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 13. BLEU pitfalls

* **定义 / 内容：** 极短翻译可能 precision 很高；重复词会虚增匹配。因此需 modified precision，如 `the the...` 对参考 `the cat is on the mat` 的 `the` 命中最多按参考中出现次数计，为 2/7。
* **直觉：** BLEU 是自动指标，不等同于人类质量判断。它对语义等价但措辞不同的翻译可能不公平。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Seq2seq encoder-decoder

* **定义 / 内容：** 一个 RNN 编码源句，一个 RNN 生成目标句；encoder 从 x 到 h，decoder 从最终 h 到 y；输入输出长度可不同；编码器和解码器解耦。
* **直觉：** seq2seq 适合 MT、image captioning、music generation 等输入输出序列长度不同的任务。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 15. Seq2seq 结构细节

* **定义 / 内容：** Encoder 是 source RNN；Decoder 是 target RNN；decoder 生成时需要 previous predicted word，训练时常用 teacher forcing。
* **直觉：** teacher forcing 是训练时把真实前一个词喂给 decoder，使训练稳定；测试时只能用模型上一步预测。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. Seq2seq training

* **定义 / 内容：** 给定 input-output pair，用 MLE 最大化目标序列概率；decoder hidden states 依赖前面输出。
* **直觉：** 训练目标是让模型逐步生成正确目标词，loss 是每个目标位置 NLL 的累积。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 17. Seq2seq issue：fixed-length bottleneck

* **定义 / 内容：** 固定长度 hidden vector 必须总结任意长度源句及词序，这要求过高。
* **直觉：** 长句信息压缩到一个向量会丢失细节，尤其是早期词和长程依赖。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 18. Seq2seq issue：长程依赖、梯度与并行性

* **定义 / 内容：** RNN 有 sequential recency，近期 token 影响更强；如 `writer of the books is/are` 需要远距离主谓一致。BPTT 会遇到梯度消失/爆炸；RNN 不能并行，长序列慢。
* **直觉：** 这些问题直接推动 attention 和 Transformer 的出现。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Attention mechanism 直觉

* **定义 / 内容：** 生成目标词时应关注源句中有用位置；训练端到端学习 focus。类比搜索引擎：query Q 匹配网页关键词 K，取出 value V。
* **直觉：** attention 不再要求一个最终 hidden vector 包含所有信息，而是在每个生成步动态检索源句信息。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 20. Attention mechanism 公式与优缺点

* **定义 / 内容：** decoder hidden state 是 query；encoder hidden states 是 keys/values；attention weights 由 score 函数和 softmax 得到；context 是加权和 $c_t=Σ_i α_{t,i}h_i$。
* **直觉：** 优点是能结合输入全局位置的信息；缺点是 encoder 仍是 sequential RNN，计算瓶颈仍在。Transformer 会进一步移除 RNN。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. 扩展阅读

* **定义 / 内容：** FSNLP MT；Seq2Seq 2014；Bahdanau attention 2015；Attention is All You Need 2017。
* **直觉：** 这几篇是从统计 MT 到神经 MT，再到 Transformer 的关键路径。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 22. Demo

* **定义 / 内容：** 用 GRU 替代 RNN，定义带 attention decoder 的 seq2seq，在 toy data 上优化。
* **直觉：** GRU 是 RNN 变体，可缓解部分长程依赖问题；attention decoder 学习对源句不同位置加权。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 23. Conclusion

* **定义 / 内容：** HMM 可用于 MT 但仍是统计模型，难编码复杂语言模式；解码有多种搜索方法但难保证全局最优；RNN 构成 seq2seq；attention 缓解长程依赖。
* **直觉：** 本讲为 Transformer 讲解铺垫：attention 是核心突破点。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 24. Quiz

* **定义 / 内容：** IBM Model 1 中词独立翻译；HMM alignment 不能直接实现 many-to-many；短候选 `the` 的 BLEU-1 precision 可为 1/1；seq2seq 弱点包括难完整编码输入、早期错误影响未来。
* **直觉：** 注意 $precision=1$ 不代表翻译好，这正是 BLEU 短句陷阱。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 11: Transformer 架构、位置编码、注意力、LayerNorm 与残差连接

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 11；Transformer；Transformer；Transformer: positional embedding；Transformer: positional embedding；Transformer: positional embedding；Transformer: MLP layer；Transformer: attention
  * Transformer: attention C = nh * hs=n_embd；Transformer: attention q.shape=[B, nh, T, hs]；Transformer: attention；Transformer: layer normalization；Transformer: layer normalization；Transformer: residual connections；Transformer: residual connections；Transformers: mechanistic interpretability
  * Transformer: mechanistic interpretability；Transformer: training；Transformer: inference INEFFICIENCY: 1) need a for loop to generate token-by-token;；Research project 1；Extra readings；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Transformer Notes

### 1. Transformer 的基本动机

RNN 最大的问题是 sequential dependency：第 $t$ 步必须等第 $t-1$ 步算完，训练很难完全并行；长程依赖还会遇到梯度消失/爆炸。Transformer 用 self-attention 直接让每个 token 访问序列中其他 token，从而更适合大规模并行训练。

一个 Transformer block 可以粗略看成：

1. token embedding + positional embedding。
2. multi-head self-attention。
3. residual connection + layer normalization。
4. MLP / feed-forward layer。
5. residual connection + layer normalization。

### 2. Positional Embedding：给 attention 顺序感

self-attention 本身对顺序不敏感。如果只看 token 集合，`dog bites man` 和 `man bites dog` 的 token 一样，但意思不同。因此输入表示通常写成：

$$
x_i=e(w_i)+p_i
$$

其中 $e(w_i)$ 是 token embedding，$p_i$ 是 position embedding。没有 positional information，模型很难区分主语、宾语、相对位置和语序。

### 3. Scaled Dot-Product Attention

给定矩阵：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

attention 计算：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

解释每一项：

* $QK^T$：每个 query 和每个 key 的相似度。
* $\sqrt{d_k}$：缩放项，避免维度大时 dot product 太大导致 softmax 饱和。
* softmax：把相似度变成概率权重。
* 乘 $V$：按权重汇总 value 信息。

如果 batch size 是 $B$，head 数是 $nh$，序列长度是 $T$，每个 head 维度是 $hs$，课件里的典型形状是：

$$
q.shape=[B,nh,T,hs]
$$

### 4. Multi-head Attention

多个 attention head 不是简单重复。不同 head 可以学习不同关系：

* 局部搭配。
* 主谓一致。
* 指代关系。
* 句法边界。
* 长程语义依赖。

如果 embedding 维度是 $C$，通常有：

$$
C=nh\times hs
$$

每个 head 独立计算 attention，最后 concat 后投影回模型维度。

### 5. MLP、Layer Normalization 和 Residual

attention 做 token 间信息交换，MLP 做每个位置内部的特征变换：

$$
MLP(x)=W_2\sigma(W_1x+b_1)+b_2
$$

LayerNorm 对每个 token 的 hidden vector 做归一化：

$$
LN(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

它让训练更稳定。residual connection 则让模型学习增量：

$$
x_{l+1}=x_l+F(x_l)
$$

这能缓解深层网络训练困难，也让信息可以跨层保留。

### 6. Training、Inference 和 Mechanistic Interpretability

训练时，autoregressive Transformer 可用 causal mask 并行计算所有位置的 next-token loss。推理时却必须一个 token 一个 token 生成：

$$
w_t\sim P_\theta(w_t|w_{<t})
$$

因此推理效率低，后续 Lecture 20 的 KV Cache、PagedAttention、StreamingLLM 都是在处理这个系统瓶颈。

Mechanistic interpretability 尝试解释 Transformer 内部 circuit：哪些 head 负责复制、括号匹配、指代或特定 pattern。这不是 Transformer 架构组件，而是研究模型行为的方法。

### 7. Exam Focus

* attention 复杂度随序列长度通常是 $O(T^2)$，不是线性。
* Transformer 架构组件包括 attention、MLP、positional embedding、LayerNorm、residual；interpretability 不是架构组件。
* attention matrix 可写成 $softmax(QK^T/\sqrt{d_k})$，完整输出再乘 $V$。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 11

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 11，主题为 Transformer 架构、位置编码、注意力、LayerNorm 与残差连接。
* **直觉：** 确认 Lecture 11 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Transformer

* **定义 / 内容：** Transformer。• Started from the 2017 paper；“Attention is all you need”.；• Extended to computer vision and；robotics and data science.；• The current popular GPT-like；applications are based on Transformer.；A comprehensive survey on applications of transformers for deep learning tasks.；Saidul Islam, etc. 2024
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. Transformer

* **定义 / 内容：** Transformer。• A breakdown of the transformer As a foundation model, it needs to；• be a large model to model general and；diverse linguistic patterns (transformer-；based to unification).；• consume large data to see all patterns；（less inductive bias).；• Positional embedding；• Feed Forward (MLP) More expressiveness；• Masked Multi-head Attention；• Residual connections；Easier to train large models；• Layer normalization；Initially: encoder-decoder arch；Later: decoder-only arch
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Transformer: positional embedding

* **定义 / 内容：** Transformer: positional embedding。• Positional embedding；o To give a positional identification to a token.；o The position of a token is important in deriving languistic features.；Tokenization；Trust is what builds trust {trust, trust, is, what, builds}；embedding("trust")=embedding("trust")；o But they are different:；§ Syntactically: the first “trust” is subject, while the second is a object.；§ Semantically: the first “trust” the thing that creates something, while the second is；the thing being created.；o Another example；A man bites a dog Tokenization；vs. {a, man bites, dog}；A dog bites a man
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Transformer: positional embedding

* **定义 / 内容：** Transformer: positional embedding。• Positional embedding；Trust is what builds trust embedding("trust") + position_vector(0)；!=；0 1 2 3 4 embedding("trust") + position_vector(4)；• Positional encoding (fixed)；pos: position of a token；i: the i-th dimension of the embedding；d: the total number of dimensions of；the embedding.
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Transformer: positional embedding

* **定义 / 内容：** Transformer: positional embedding。• A running example；Example: sentence = [“The”, “cat”, “sat”]；“The” (E_The): [0.1, 0.2, 0.3, 0.4] “cat” (E_cat): [0.5, 0.6, 0.7, 0.8]
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Transformer: MLP layer

* **定义 / 内容：** Transformer: MLP layer。• MLP (multi-layered perception) Feed Forward Layer；o MLP first maps x to higher dimensional space, then takes the GeLU nonlinear mapping,；and finally projects back to dim(x).；• Why it is necessary?；o The self-attention layer can only find linear combination of value vectors.；o What if some non-linearity is needed?；dim(x) ”dessert”=[1,1]；It is important；that the input x2；Proj =；and output；have the same “pie” Convex hull of x1 and x2；dimensionality, GeLU dim(x) * 4 =[0,1]；since any number of；FC；transformer blocks x1=”apple”=[1,0]；can be appended.；dim(x)
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Transformer: attention

* **定义 / 内容：** Transformer: attention。• The central of Transformer: attention mechanism；o To address the gradient propagation issues of RNN/LSTM.；"I grew up in France, where I spent many years surrounded by my family and friends, so I speak fluent French."；Long-range dependencies；o Why it is hard to learn the dependency?；o If the model makes a wrong prediction “English” at the end, it is possible that the；training data have more frequent “fluent English” occurrences (short-range dep).；o The model forward passes are more influenced by the recent word such as “fluent”,；while the far-away word “France” has its semantic vector modified by all the other；words in the middle and the many multiplication of the weight matrix of RNN.；rollout
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Transformer: attention C = nh * hs=n_embd

* **定义 / 内容：** Transformer: attention C = nh * hs=n_embd。k.shape[B, T, nh * hs]；view；k.shape[B, T, nh, hs]；self.c_attn=[Wk, Wq, Wv], shape=(n_embd, 3*n_embd) transpose；This is a 5x5 matrix as；k.shape[B, nh, T, hs]；the input has T=5 tokens.；The size of the attention；matrix scales O(T2)；Note: hidden state has hs =5；and is just one of the heads’ size.；For a multiple head attention,；there are nh heads, so that；the total dimension is nh*hs.；x.shape=[Batch_size, Token_len, n_embd]
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Transformer: attention q.shape=[B, nh, T, hs]

* **定义 / 内容：** Transformer: attention q.shape=[B, nh, T, hs]。k.transpose(-2,-1).shape=[B, nh, hs, T]；(q @ k).shape=[B, nh, T, T]；To prevent the inner products from inflating the scales.；self.bias[:,:,:T,:T] has the upper triangle = 0；Causal Attention: don’t look into the future；>>> import torch；>>> a=torch.tensor([1,2,3,float('-inf'), float('-inf'), float('-inf')])；>>> F.softmax(a)；tensor([0.0900, 0.2447, 0.6652, 0.0000, 0.0000, 0.0000])
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. Transformer: attention

* **定义 / 内容：** Transformer: attention。att.shape=[B, nh, T, T]；v.shape=[B, nh, T, hs]；(att @ v).shape=[B, nh, T, hs]；[B, T, hs]；[B, T, nh * hs]=[B, T, C]；[B, T, hs]
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. Transformer: layer normalization

* **定义 / 内容：** Transformer: layer normalization。• Intuitions:；o For all neurons at the same layer, the “scale” of their activations should remain about the；same as the input.；o After going through many layers, the scale of activations should remain about the same.；• Why it is necessary?；o If several layers enlarge or shrink the activation scales dramatically, there is the so-called；“saturation of softmax” and consequently the “vanishing gradient” issue.；>>> a=torch.tensor([1.,2.,3.])；>>> b=torch.tensor([10.,20.,30.])；>>> softmax(a)；tensor([0.0900, 0.2447, 0.6652])；>>> softmax(b)；tensor([2.0611e-09, 4.5398e-05, 9.9995e-01])
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. Transformer: layer normalization

* **定义 / 内容：** Transformer: layer normalization。• Ignoring the learnable parameters, the；first sample’s features are normalized to；o (2-5.6)/2.73=-1.3186813187；o (6-5.6)/2.73=0.1465201465；The differences；o (8-5.6)/2.73=0.8791208791 beween two；activations；are much small.；o (9-5.6)/2.73=1.2454212454；o (3-5.6)/2.73=-0.9523809524；https://miro.medium.com/v2/resize:fit:1400/1*F4pVp_fr90NDdWhKwcTBLQ.png
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Transformer: residual connections

* **定义 / 内容：** Transformer: residual connections。• Residual connections；o x=MHA(x)+x and x=MLP(x)+x；• Intuitions:；o At any layer, if a mapping is unnecessary for a token, just let that mapping to be zero and；push the input to the next layer.；o Mapping from [9, 11, 19, 21] to [10, 12, 18, 20] is hard, but from [9, 11, 19, 21] to the；residual [-1,-1,1,1] is much easier.；• Why it is necessary?；o Allow information to directly forward to the next level without MHA or MLP (which are；just learned to be the zero mapping).；o Allow gradients (of loss function of supervision) to directly backprop to the；earlier layers, without too many multiplications and eventual gradient explosion；or vanishing.
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Transformer: residual connections

* **定义 / 内容：** Transformer: residual connections。• Another viewpoint of transformer: residual-centered；• Layers as "Add-Only" Modules；o the Attention and MLP layers aren't reforming the representation; they are；reading from the stream, performing a small calculation, and writing the result；back to it.；o Attention layers move information from one token‘s position in the stream to；another, thus connecting far-away information.；o MLP layers process the information at a single position, often acting like key-；value memories to refine the data (the "Processing").；• Linear Representation；o Because the updates are additive, the residual stream remains surprisingly；linear. You can think of the final output of transformer as the sum of the；original embedding plus a series of "delta" updates:
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. Transformers: mechanistic interpretability

* **定义 / 内容：** Transformers: mechanistic interpretability。• Since the stream is additive,；take the hidden state at any；intermediate layer and pass；it through the final In the figure, each position at each；Unembedding layer to see layer is the most likely word from；what the model "thinks" the unembedding.；answer is at that exact；moment. The color means the saturation；of the logits:；• The model's prediction Blue means less certainty, while；evolve from a simple bigram yellow means more certainty of；guess in layer 2 to the the predicted token.；correct complex fact by；layer 12.；https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/AcKRB8wDpdaN6v6ru/ccfmt4rt3aegjjfi7lo8
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. Transformer: mechanistic interpretability

* **定义 / 内容：** Transformer: mechanistic interpretability。• Information flow: how information flow from bottom to top to；predict the next token.；"When Mary and John went to the store, John gave a book to"；https://www.lesswrong.com/posts/xNgdJEep9DQQWhSbv/understanding-the-information-flow-inside-large-language
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Transformer: training

* **定义 / 内容：** Transformer: training。• Training；o Prepare training and validation data；o Load raw texts.；o Use an existing encoder to turn text token；into indices.；o Save to binary files.；o Sample a batch: randomly Load part (not all) of the file to memory.；batch_size=4 next token prediction；block_size；y1 y2 y3 y4；x1 x2 x3 x4；i=1 i=2 i=3 i=4；Total tokens in data
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Transformer: inference INEFFICIENCY: 1) need a for loop to generate token-by-token;

* **定义 / 内容：** Transformer: inference INEFFICIENCY: 1) need a for loop to generate token-by-token;。2) for each token, all previous tokens should be forwarded.；• Inference；idx_cond.shape；=[batch_size, block_size]；Take the final；Transformer positon of the；seq of vectors.；Umembedding is；logits.shape；performed inside；=[batch_size, voc_size]；the Transformer.；Find top-k；per row of；Sample；the batch probs；idx_next.shape=batch_size
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. Research project 1

* **定义 / 内容：** Research project 1。• We released the first project option on Canvas.；• Mechanistic interpretability of Transformer (GPT-2)；• Small enough to fine-tune and analyze on mid-end GPUs.；• Main tasks:；o Run the LRP algorithm to show contributions of input and parameters to；predicted tokens.；o Compare the contributions before and after fine-tuning.；o Research extension: how layer normalization impact the attribution? How to；discover input and parameter conflicts?；• Top teams will be featured on the course promotion materials, and；supported for paper publications.
* **直觉：** 本页展开 Transformer 的核心部件包括 positional embedding、masked multi-head attention、MLP、LayerNorm、residual connection、training/inference。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. Extra readings

* **定义 / 内容：** Extra readings。• For a very good visualization of transformer, see the Youtube video；• https://www.youtube.com/watch?v=9-Jl0dxWQs8；• For mechanistic interpretability of LLM, see the blog；• https://www.lesswrong.com；• For a very good implementation from scratches, see the following:；• https://github.com/karpathy/nanoGPT (by Andrej Karpathy)；• https://www.youtube.com/watch?v=kCc8FmEb1nY (explaining the above codes).
* **直觉：** 本页给出扩展阅读，用于把课堂概念连接到论文、系统或后续深入学习。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. Demo

* **定义 / 内容：** Demo。• Nothing is better than Karpathy’s nanoGPT, just go there and read；the codes and run them on your own computer for a pre-training.
* **直觉：** 本页是实践演示页，说明本讲概念如何落到代码、实验或可视化流程中。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. Conclusion

* **定义 / 内容：** Conclusion。• Transformers are now everywhere for many tasks in NLP, CV, and robotics.；• Diving deep into a Transformer.；• Mechanistic interpretability of Transformers.
* **直觉：** 本页总结本讲主线，适合作为复习时的高层索引。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (T/F): layer normalization is to make model training more stable.；• Q2 (T/F): attention mechanism scales linearly as the length of the sequence.；• Q3 (short answer): write down the equation to calculate attention matrix；using Q, K, and V matrices.；• Q4 (multiple choices): which of the following is not part of the architecture；of a transformer? A) attention; B) MLP layer; C) interpretability; D) positional；embedding.；• A: T, F, softmax(QKT/sqrt(d)), C
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 12: Pretraining、Mid-training、Post-training、BERT 与 GPT

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 12；Pretraining, mid-training, post-training；Pretraining, mid-training, post-training；Pretraining: downstream fine-tuning/prompting；Pretraining: why；Pretraining: why；Major factors affecting pre-training；Model architecture
  * What data to use；Data factors: quantity；Data factors: quality；Pretraining objective function；Masked language modeling；Masked language modeling: BERT；Masked language modeling: BERT；Masked language modeling: BERT
  * Masked language modeling: BERT；Autoregressive language modeling；Autoregressive language modeling；Autoregressive language modeling: GPT；Autoregressive language modeling: GPT；Autoregressive language modeling: GPT；Autoregressive language modeling: GPT；Autoregressive language modeling: GPT
  * Extra readings；Demo；Conclusion；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Pretraining Notes

### 1. 三阶段：Pretraining、Mid-training、Post-training

LLM 不再主要遵循传统小任务的 train/test paradigm，而是按阶段训练：

* **Pretraining (预训练)**：在大规模文本上学习语言、语法、世界知识、基本推理和代码模式，得到 base foundation model。
* **Mid-training (中训练 / 领域继续训练)**：在高质量特定领域数据上继续训练，让模型增强专业能力，同时尽量保留通用能力。
* **Post-training / Alignment (后训练 / 对齐)**：把 base model 变成能遵循指令的 assistant，常见方法包括 SFT、RLHF、DPO、GRPO。

课件用 chef 类比：pretraining 像学会基本厨艺和食材知识，mid-training 像专攻某菜系，post-training 像学会按顾客要求服务。

### 2. 为什么 pretraining 有迁移能力

pretraining 让模型参数落在一个更好的 loss landscape 区域。后续任务不需要从随机参数开始，而是在已经学到语言规律和概念组织的模型上微调或 prompting。

这解释了 downstream fine-tuning/prompting：

* sentiment analysis。
* named entity recognition。
* question answering。
* natural language inference。
* information extraction。

同一个 pretrained model 可以被初始化到不同任务上，因为它已经学到可复用表示。

### 3. Pretraining 成败因素

课件列出四类因素：

* **Model architecture**：通常是 Transformer，负责可扩展性和长程依赖建模。
* **Training objective**：MLM 或 autoregressive LM。
* **Data**：数量、质量、覆盖范围最关键。
* **Hyperparameters**：learning rate、batch size 等；大数据大模型下仍重要，但很多经验已较稳定。

数据方面，常见来源包括 books、Wikipedia、Common Crawl、social media、open-source code repositories。随着公开数据被有效用尽，private data 和 AI-generated data 变得重要。

### 4. Masked Language Modeling and BERT

MLM 把一部分 token 替换成 `[MASK]`，让模型根据双向上下文预测原词。目标只在 masked positions 上计算：

$$
L_{MLM}(\theta)=-\sum_{i\in M}\log P_\theta(x_i|x_{\setminus M})
$$

BERT 的 15% masking 规则：

* 80% 替换为 `[MASK]`。
* 10% 替换为随机 token。
* 10% 保持原 token 不变，但仍预测它。

原因是 fine-tuning 时不会出现 `[MASK]`，所以训练不能只让模型适应 mask token。

BERT 输入 embedding 是三者相加：

$$
x_i=token_i+segment_i+position_i
$$

BERT Base：12 layers、768 hidden、12 heads、110M parameters。BERT Large：24 layers、1024 hidden、16 heads、340M parameters。

### 5. Autoregressive Language Modeling and GPT

autoregressive objective 从左到右预测下一个 token：

$$
L_{AR}(\theta)=-\sum_{D}\sum_t\log P_\theta(w_t|w_{<t})
$$

优点：

* pretraining objective 和 inference process 一致。
* 目标简单统一。
* scaling behavior 强。

缺点：

* 只能使用 left-to-right context。
* generation 慢，因为必须逐 token 生成。
* 早期错误会在后续累积。

GPT 是 decoder-only Transformer。课件提到 GPT 2018 的 12 layers、117M parameters、768 hidden、3072 FFN hidden、BPE 40k merges，以及 Llama 系列用 trillion tokens 级别数据训练。

### 6. Exam Focus

* BERT 和 GPT 都用 Transformer，但 objective 不同：BERT 是 MLM，GPT 是 autoregressive LM。
* pretraining 让后续 fine-tuning/prompting 更容易。
* 数据质量和数据覆盖不只是“更多 token”，还决定模型学到什么。
* AR objective 要能写成 $-\log p_\theta(w_t|w_{<t})$ 的求和。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 12

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 12，主题为 Pretraining、Mid-training、Post-training、BERT 与 GPT。
* **直觉：** 确认 Lecture 12 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Pretraining, mid-training, post-training

* **定义 / 内容：** Pretraining, mid-training, post-training。• Pre-training: teach the model the fundamentals of language, grammar, reasoning,；and a broad range of world knowledge.；Large data Base foundation model Sentiment；analysis；Pre-train Adaptation；Information；extraction；Question；answering；• Mid-training: further train a base model on a specific, high-quality domain to；deepen its expertise without losing its general knowledge.；• Post-training: a.k.a, alignment, transform the base model into an “assistant” that；can follow instructions and align with human values (helpful, honest, harmless).；o SFT, RLHF (to be discussed later)
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. Pretraining, mid-training, post-training

* **定义 / 内容：** Pretraining, mid-training, post-training。LLM training；stages are similar to；training a chef.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Pretraining: downstream fine-tuning/prompting

* **定义 / 内容：** Pretraining: downstream fine-tuning/prompting。Base foundation model Example: Fine-tuned model；(1+1, 2)；Fine-tune；Problem；solving；Problem；solving；Example:；1+1= 2, so；2*(1+1)=?；https://www.veryicon.com/icons/education-technology/guangzhou-baiyun-district-ecological/icon_-task-tracking.html；https://www.alamy.com/problem-solving-concept-icon-planning-management-way-out-of-difficult-situations-decision-making-idea-thin-line-illustration-；vector-isolated-outl-image341371726.html；https://www.flaticon.com/free-icon/prompt_10817257
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Pretraining: why

* **定义 / 内容：** Pretraining: why。Transferability Across Tasks: Pretrained models can be adapted to a；wide range of tasks through fine-tuning；Downstream tasks；• Sentence classification；• Named entity recognition；• Question answering；(simple form)；• Language inference (if；sentence A entails；sentence B)；The pre-trained model parameters are used to initialize BERT for multiple down-stream tasks.；Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[C]//Proceedings of the 2019 conference of the North；American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 2019: 4171-4186.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 6. Pretraining: why

* **定义 / 内容：** Pretraining: why。Improved task performance: better results compared to training from scratchl；Validation error rates for supervised；and semi-supervised ULMFiT vs.；training from scratch with different；numbers of training examples on；IMDb, TREC-6, and AG.；Why it is easier with pre-training?；In deep learning, optimizers demonstrate a remarkable ability to；dynamically navigate complex loss landscapes, ultimately；converging to solutions that generalize well.；Pretraining moves the model parameters to a smoother landscape.；Howard J, Ruder S. Universal language model fine-tuning for text classification. ACL.(Volume 1: Long Papers). 2018: 328-339.；https://www.nature.com/articles/s41467-025-58532-9
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 7. Major factors affecting pre-training

* **定义 / 内容：** Major factors affecting pre-training。1. Model: the underlying；neural network；architecture. *important but fixed.；2. Training objective: the；objective used to train.；*important but fixed.；3. Data: the quantity and；quality of pre-training；datasets. *very important.；4. Hyperparameters: such as；learning rate, batch size.；*These are not too critical at the end with；large data.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Model architecture

* **定义 / 内容：** Model architecture。• Typically a Transformer, though the specifics can differ；o it learns quite fast from texts, absorbing significant linguisitc patterns；o handle long-range dependencies；o scaling law: gets stronger scalability as model and data sizes increase；• Size: Larger models tend to perform better on tasks within a specific model family；• Model details can differ.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. What data to use

* **定义 / 内容：** What data to use。• Commonly used: • Quantity: How much data do I have；o Books；• Quality: Is the data suitable for；o Wikipedia；training purposes?；o Common crawl: data from the whole internet；o Social media data • Coverage: Does the data adequately；o Open source code repositories cover the domains I care about, and；in the right proportions?；• Publicly available data, though still；increasing, are effectively used up.；o Consider private special data (train a robot；reasoning model using scene data collected in a；factory) and AI-generated data.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 10. Data factors: quantity

* **定义 / 内容：** Data factors: quantity。Data quantity refers to the total amount of data used to train a；model, often measured in tokens for large language models；Model Model size # of Tokens；Llama 1 7B, 13B, 33B, 65B 1.4 trillion；Llama 2 7B, 13B, 70B 1.8 trillion；Llama 3 8B, 70B, 405B 15 trillion；Deepseek3 671B 15 trillion
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 11. Data factors: quality

* **定义 / 内容：** Data factors: quality。Filtering Deduplication；Filter out unwanted text Remove duplicate content；Penedo G, Kydlíček H, Lozhkov A, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 2024, 37: 30811-30849.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 12. Pretraining objective function

* **定义 / 内容：** Pretraining objective function。• Masked language modeling: used more for fine-tuning.；Example: BERT；• Auto-regressive language modeling: used for fine-tuning, prompting.；Example: GPT 2/3/4, Lamma1/2
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 13. Masked language modeling

* **定义 / 内容：** Masked language modeling。• Idea: replace some fraction of words in the input with；special [MASK] token; predict these words.；ℎ! , … , ℎ " = Encoder 𝑤! , … , 𝑤"；•；𝑧# ∼ 𝐴ℎ# + 𝑏；• Only add loss terms from words that are “masked；out”. If 𝑥2 is the masked version of x, we are learning；2 called masked LM.；𝑝$ (𝑥|𝑥),
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Masked language modeling: BERT

* **定义 / 内容：** Masked language modeling: BERT。Some details about Masked LM for BERT；• Predict a random 15% of word tokens.；o Replace input word with [Mask] 80% of the time.；o Replace input word with a random token 10% of；the time.；o Leave input word unchanged 10% of the time (but；still predict it)；• No masks are seen at fine-tuning time；Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[C]//Proceedings of the 2019 conference of the North；American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 2019: 4171-4186.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Masked language modeling: BERT

* **定义 / 内容：** Masked language modeling: BERT。Special token added Special token；to the beginning of to separate；each input sequence two sentences；This embeddings；indicate whether it；belongs to sentence A；or sentence B；Position of the token in the entire sequence；The final embedding is the sum of all three；Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[C]//Proceedings of the 2019 conference of the North；American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 2019: 4171-4186.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. Masked language modeling: BERT

* **定义 / 内容：** Masked language modeling: BERT。• Achieved state-of-the-art performance across various tasks after fine-tuning；• QQP: Quora Question Pairs (detect paraphrase questions)；• QNLI: natural language inference over question answering data；• SST-2: sentiment analysis；• CoLA: corpus of linguistic acceptability (detect whether sentences are grammatical.)；• STS-B: semantic textual similarity；• MRPC: microsoft paraphrase corpus；• RTE: a small natural language inference corpus；Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[C]//Proceedings of the 2019 conference of the North；American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 2019: 4171-4186.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. Masked language modeling: BERT

* **定义 / 内容：** Masked language modeling: BERT。• Two model sizes；o Base: 12 layers, 768-dim hidden states, 12 attention heads, 110 million params.；o Large: 24 layers, 1024-dim hidden states, 16 attention heads, 340 million params.；o The larger the better!；Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[C]//Proceedings of the 2019 conference of the North；American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 2019: 4171-4186.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Autoregressive language modeling

* **定义 / 内容：** Autoregressive language modeling。Pre-trained model Pre-trained model for；downstream tasks；ℎ# , … , ℎ % = f 𝑤# , … , 𝑤%；𝑧" = 𝐴ℎ"$# + 𝑏；Add a linear layer on top of；𝑝! 𝑤" 𝑤# , … 𝑤"$# , 𝑥 : conditioned on a source the last hidden layer to make；context to generate from left-to-right it a classifier!；Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Autoregressive language modeling

* **定义 / 内容：** Autoregressive language modeling。min 0 0 − log 𝑝𝜽 𝑤" ∣ 𝑤*" • Pros:；𝜽；'∈)!"#$% " o The pre-training objective is unified；and simple, and the training procedure；is consistent with the inference process.；• 𝛉: model parameters o Strong scaling behavior；• 𝐷%&'() : training data；• 𝑝𝜽 𝑤+ ∣ 𝑤,+ : conditional language；• Cons:；modeling, conditioned on a；source context to generate from o Only uses left-to-right context；o Autoregressive generation is slow；left to right.；o Error accumulation during generation；Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. Autoregressive language modeling: GPT

* **定义 / 内容：** Autoregressive language modeling: GPT。• GPT=Generative Pretrained Transformer；• GPT was a big success in 2018.；o Transformer decoder with 12 layers, 117M parameters.；o 768 dimensional hidden states, 3072-dimensional feed-forward hidden layers；o Byte-pair encoding with 40,000 merges；o Trained on BooksCorpus: over 7000 unique books.；Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. Autoregressive language modeling: GPT

* **定义 / 内容：** Autoregressive language modeling: GPT。Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.
* **直觉：** 本页文字较少或主要依赖图示；知识点按页标题、可见文字和前后页面上下文保留。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. Autoregressive language modeling: GPT

* **定义 / 内容：** Autoregressive language modeling: GPT。• GPT results on various natural language inference datasets；• GPT results on question answering datasets；Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. Autoregressive language modeling: GPT

* **定义 / 内容：** Autoregressive language modeling: GPT。• Example : Llama；o Model: Transformer, {6.7B, 13B, 32B, 65B}；o Data: 1.4 trillion tokens, sources；Touvron H, Lavril T, Izacard G, et al. Llama: Open and efficient foundation language models[J]. arXiv preprint arXiv:2302.13971, 2023.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. Autoregressive language modeling: GPT

* **定义 / 内容：** Autoregressive language modeling: GPT。• Llama: training loss；Better；Loss；More data；Touvron H, Lavril T, Izacard G, et al. Llama: Open and efficient foundation language models[J]. arXiv preprint arXiv:2302.13971, 2023.
* **直觉：** 本页展开 foundation model 的 pretraining / mid-training / post-training，以及 BERT 的 MLM 和 GPT 的 autoregressive objective。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. Extra readings

* **定义 / 内容：** Extra readings。• For a very good visualization of transformer, see the Youtube video；• https://www.youtube.com/watch?v=9-Jl0dxWQs8；• For mechanistic interpretability of LLM, see the blog；• https://www.lesswrong.com；• For a very good implementation from scratches, see the following:；• https://github.com/karpathy/nanoGPT (by Andrej Karpathy)；• https://www.youtube.com/watch?v=kCc8FmEb1nY (explaining the above codes).
* **直觉：** 本页给出扩展阅读，用于把课堂概念连接到论文、系统或后续深入学习。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 26. Demo

* **定义 / 内容：** Demo。• Nothing is better than Karpathy’s nanoGPT, just go there and read；the codes and run them on your own computer for a pre-training.
* **直觉：** 本页是实践演示页，说明本讲概念如何落到代码、实验或可视化流程中。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 27. Conclusion

* **定义 / 内容：** Conclusion。• LLM does not go through the training-test paradigm, but uses the pre-；training, mid-training, post-training paradigm.；• Pre-training makes later stages easier.；• Pre-training is the key to model transfer.；• Several factors contribute to LLM’s success: data, architecture, objective,；and hyperparamters.；• Two sorts of training objectives based on Transformer.；• Data becomes very important once the others are (almost) fixed.
* **直觉：** 本页总结本讲主线，适合作为复习时的高层索引。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 28. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (short answer): write down the objective function of pre-training using；auto-regression.；• Q2 (T/F): Pretrained models can be further fine-tuned for other tasks.；• Q3 (T/F): both BERT and GPT use transformer architecture, but are pre-；trained using different training objective functions.；• A1: ∑!∈#!"#$% ∑$ − log 𝑝𝜽 𝑤$ ∣ 𝑤&$；• A2: T；• A3: T
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 13: Supervised Fine-Tuning (SFT)、Alignment 与数据构造

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 13；Supervised fine-tuning (SFT)；Pretraining vs. SFT；Why alignment?；Why alignment；Why alignment；Why alignment；Two kinds of SFT
  * Two kinds of SFT；Data for SFT；Data for SFT；Data for SFT；Data for SFT；Data for SFT: AI generated data；Data for SFT: available datasets；Data for SFT: available datasets
  * Data for SFT: available datasets；Data for SFT；SFT loss function；SFT loss function；SFT loss function；SFT performance；SFT performance；Industrial SFT
  * SWE-Lego: coding agent；SWE-Lego: pipeline；SWE-lego: data curation；SWE-Lego: synthetic data curation；SWE-Lego: curriculum learning for SFT；SWE-Lego: results；Multi-task SFT: catastropic fogetting；Multi-task SFT: Dual-stage mixed fine-tuning

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed SFT / Alignment Notes

### 1. SFT 和 Pretraining 数学相似，但数据分布不同

**Supervised Fine-Tuning (SFT)** 仍然是 next-token prediction。差别在于数据不是海量 raw text，而是 prompt/instruction 与高质量 response 的配对。

预训练：

$$
L_{pretrain}=-\sum_t\log p_\theta(w_t|w_{<t})
$$

SFT：

$$
L_{SFT}=-\sum_t m_t\log p_\theta(y_t|x,y_{<t})
$$

其中 $m_t$ 是 loss mask。通常 prompt 部分 $m_t=0$，response 部分 $m_t=1$，因为我们希望模型学习“如何回答”，而不是学习复述用户 prompt。

### 2. Alignment：Helpful、Harmless、Honest

base LM 擅长续写文本，但不一定擅长做 assistant。alignment 试图让模型满足：

* **Helpful**：可靠遵循用户指令。
* **Harmless**：遇到危险请求时拒绝或安全改写。
* **Honest**：不确定时表达不确定，减少 hallucination。

这解释了为什么 GPT-3 直接使用时可能不如 InstructGPT：base model 学的是“互联网文本续写”，aligned model 学的是“按人类偏好回答”。

### 3. Single-task SFT vs Multi-task SFT

**Single-task SFT** 用大量同一任务数据训练专门模型，例如只做 sentiment 或 bug fixing。优点是任务表现强，缺点是泛化窄。

**Multi-task SFT / instruction tuning** 混合多任务，训练 generalist assistant。它更适合开放式交互，但数据组成更难：不同任务可能语义冲突，混太多会导致 catastrophic forgetting。

### 4. SFT 数据来源和格式

人工标注优点是质量高，缺点是慢、贵、缺多样性。AI-generated data 出现后，Self-Instruct 等方法让强 teacher model 生成 instruction-response pair。

常见数据格式：

* Alpaca format：instruction / input / output，偏 single-turn。
* GPT-style format：多轮对话。
* ChatML / role format：system、user、assistant 分角色。

课件列出的数据集包括 Alpaca、Dolly-15k、COIG、OASST1；要记住它们说明了 SFT 数据可以是英文、中文、多语言、单轮、多轮、人写或模型生成。

### 5. Prompt Loss Weight

prompt loss weight 控制 prompt token 是否也参与 loss：

* $0$：只学 assistant response，强迫模型专注回答行为。
* $0.1$：保留少量 prompt loss，像 regularization，帮助减少 catastrophic forgetting。

直觉是：完全不看 prompt loss 可能更像 instruction follower，但也可能忘掉一些语言和世界知识；少量 prompt loss 可以让模型保持 base model 能力。

### 6. Industrial SFT: SWE-Lego

SWE-Lego 说明 coding-agent SFT 不只是普通 QA。它的数据包括：

* repositories。
* bug-fixing tasks。
* sandbox / execution environment。
* trajectory：一系列读文件、改文件、运行测试的步骤。
* verified result：修复是否通过测试。

合成 bug 可通过 LLM rewrite 或 AST reformulation，例如删除 conditional、修改 operator、改变依赖。curriculum learning 让模型先学简单读代码/修 bug，再学困难任务。

### 7. Catastrophic Forgetting and Dual-stage Mixed Fine-tuning

多任务 SFT 中，math、coding、general chat 的数据分布不同。直接大量混合可能让某些能力下降。dual-stage mixed fine-tuning 的思想是：用少量 general data 保持通用能力，同时加入目标任务数据学习专业技能。

### 8. Exam Focus

* SFT 和 pretraining 的公式相似，但数据和 mask 不同。
* SFT 主要教 assistant behavior，不是从零学习世界知识。
* SFT 可减少 hallucination，但也会受数据质量、覆盖和格式影响。
* 多任务 SFT 的常见风险是 catastrophic forgetting。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 13

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 13，主题为 Supervised Fine-Tuning (SFT)、Alignment 与数据构造。
* **直觉：** 确认 Lecture 13 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Supervised fine-tuning (SFT)

* **定义 / 内容：** Supervised fine-tuning (SFT)。• Previous lecture: Pre-train on large-scale data → Base foundation model；• This lecture: Base foundation model → Aligned model；https://cameronrwolfe.substack.com/p/understanding-and-using-supervised
* **直觉：** 本页展开 SFT、alignment、instruction data、AI-generated data、SFT loss、industrial SFT 与 multi-task forgetting。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. Pretraining vs. SFT

* **定义 / 内容：** Pretraining vs. SFT。• Similarity: SFT is mathematically similar to language model pretraining: both；optimize a next-token prediction (maximum likelihood) objective.；• Differences: The key difference is the training data distribution.；o Pretraining uses a massive corpus of raw text.；o SFT uses a supervised dataset of high-quality reference responses (human-；written or model-generated) paired with prompts/instructions.；o SFT masks the instruction part using a loss mask.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Why alignment?

* **定义 / 内容：** Why alignment?。A pretrained LM is great at continuing text, but it may fail at:；• Helpful: Following instructions reliably.；• Harmless: Respecting safety constraints.；• Honest: Being honest about uncertainty.；Go further:；• Mathematical Reasoning；• Code Generation；• ……
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 5. Why alignment

* **定义 / 内容：** Why alignment。Helpful: GPT-3 models aren’t trained to follow user instructions, directly use pretrained model.；InstructGPT uses model after alignment and generate much more helpful outputs.；https://openai.com/index/instruction-following/
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 6. Why alignment

* **定义 / 内容：** Why alignment。Harmless: Respecting safety constraints. The pretrained model might generate malicious or unsafe；content it asked. Aligned models are trained to refuse requests, prioritizing user and system safety.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 7. Why alignment

* **定义 / 内容：** Why alignment。Honest: Being honest about uncertainty/hallucination.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 8. Two kinds of SFT

* **定义 / 内容：** Two kinds of SFT。Single-task SFT: many examples from one task → a specialized model；Step 2；fine-tuning；Step 1；data curation
* **直觉：** 本页文字较少或主要依赖图示；知识点按页标题、可见文字和前后页面上下文保留。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Two kinds of SFT

* **定义 / 内容：** Two kinds of SFT。Multi-task SFT (a.k.a. instruction tuning): diverse tasks mixed in one dataset →；a generalist model.；Step 2；fine-tuning；Step 1；data curation
* **直觉：** 本页展开 SFT、alignment、instruction data、AI-generated data、SFT loss、industrial SFT 与 multi-task forgetting。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Data for SFT

* **定义 / 内容：** Data for SFT。• Collect examples pairs across many tasks；[4] Chung, Hyung Won, et al. "Scaling instruction-finetuned language models." Journal of Machine Learning Research 25.70 (2024): 1-53.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 11. Data for SFT

* **定义 / 内容：** Data for SFT。• How to get collect examples pairs across many tasks (instruction data)?；o InstructGPT (GPT3.5) collected 11,295 diverse；prompts written by annotators.；o Prompt types include QA, multi-turn chat, few-shot；imitation, etc.；o This means that early quality control instruction；data relied entirely on manual annotation.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 12. Data for SFT

* **定义 / 内容：** Data for SFT。• Advantages of manual annotations；o SFT does not require massive data: Unlike pretraining (learning world；knowledge), SFT mainly teaches assistant behaviors (instruction-following,；formatting, interaction style).；“LIMA: Less is More for Alignment” [5].；o No strong “teacher model” in early days: early base LMs (e.g., BERT, GPT1)；were not reliably instruction-following. Self-generated SFT data would amplify；mistakes.；[5] Zhou, Chunting, et al. "Lima: Less is more for alignment." Advances in Neural Information Processing Systems 36 (2023): 55006-55021.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 13. Data for SFT

* **定义 / 内容：** Data for SFT。• But manual annotations for SFT have some issues；o Slow and high-cost: hiring experts to write diverse prompts and responses is；expensive and time-consuming.；o Lack of diversity: human annotators suffer from cognitive fatigue; it's hard for；them to brainstorm thousands of truly distinct tasks.；o Strong teacher models appeared later: With the release of highly capable,；aligned models, we finally have reliable teachers.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 14. Data for SFT: AI generated data

* **定义 / 内容：** Data for SFT: AI generated data。• Self-Instruct: From Human Labels to Machine Labels；=GPT3；[6] Wang, Yizhong, et al. "Self-instruct: Aligning language models with self-generated instructions." Proceedings of the 61st annual meeting of the association for；computational linguistics (volume 1: long papers). 2023.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 15. Data for SFT: available datasets

* **定义 / 内容：** Data for SFT: available datasets。• Public datasets for SFT；Contents /；Dataset Language(s) Size Format；Generation Method；Single-turn QA. Generated via；Alpaca English 52K (Instruction, Input, Self-Instruct.；Output)；100% Human-；generated by；Dolly-15k English 15K Single-turn；Databricks employees.；Chinese NLP tasks.；Single-turn / Multi- Mixed human &；COIG Chinese 191K；turn model generation.；Human-annotated；Multi-turn conversational；OASST1 Multilingual (35 langs) 161K；(Conversation Trees) dialogue and assistant；responses.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 16. Data for SFT: available datasets

* **定义 / 内容：** Data for SFT: available datasets。Alpaca [7] (English Dataset)；[7] https://huggingface.co/datasets/tatsu-lab/alpaca
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 17. Data for SFT: available datasets

* **定义 / 内容：** Data for SFT: available datasets。COIG [8] (Chinese Dataset)；[8] https://huggingface.co/datasets/BAAI/COIG
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 18. Data for SFT

* **定义 / 内容：** Data for SFT。Mainstream SFT data format:；Alpaca Format GPT Format ChatML Format；(Single-turn) (Multi-turn) (Role)
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 19. SFT loss function

* **定义 / 内容：** SFT loss function。Consider the finetuning data pairs；The SFT loss has the same formulation as that of the pre-training loss:；The difference is that is very small and we do not count in the loss using a；loss mask.
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 20. SFT loss function

* **定义 / 内容：** SFT loss function。We do not count in the loss using a loss mask. (Change prompt loss weight)；https://towardsdatascience.com/to-mask-or-not-to-mask-the-effect-of-prompt-tokens-on-instruction-tuning-016f85fd67f4/
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 21. SFT loss function

* **定义 / 内容：** SFT loss function。• Why 0 vs. 0.1 for Prompt Loss Weight?；o Weight = 0: Forces the model to focus 100% on learning how to answer (assistant；behaviors), ignoring the prompt prediction.；o Weight = 0.1 (Regularization): Retains a small loss on the prompt to prevent；catastrophic forgetting. It helps the model maintain its pre-trained world knowledge；and general language understanding.；https://towardsdatascience.com/to-mask-or-not-to-mask-the-effect-of-prompt-tokens-on-instruction-tuning-016f85fd67f4/
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 22. SFT performance

* **定义 / 内容：** SFT performance。Before SFT After SFT；[6] Wang, Yizhong, et al. "Self-instruct: Aligning language models with self-generated instructions." Proceedings of the 61st annual meeting of the association for；computational linguistics (volume 1: long papers). 2023.
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. SFT performance

* **定义 / 内容：** SFT performance。• SFT is the most effective stage for minimizing hallucinations, making it even；more "honest" than the PPO stage (PPO is in our next lecture).；[1] Ouyang, Long, et al. "Training language models to follow instructions with human feedback." Advances in Neural Information Processing Systems 35 (2022): 27730-27744
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. Industrial SFT

* **定义 / 内容：** Industrial SFT。• SFT are invented in the industry to align large models for complex；tasks, there are many tricks to make SFT successful in the real-world.；• A few case studies to peek into the tricks:；o Huawei: SFT AI agent for coding, focusing on data curation for a specific tasks.；§ SWE-Lego: Pushing the Limits of Supervised Fine-tuning for Software Issue；Resolving (https://arxiv.org/html/2601.01426v2)；o Alibaba: multi-task SFT, dealing with catastropic forgetting.；§ How Abilities in Large Language Models are Affected by Supervised Fine-tuning；Data Composition (https://arxiv.org/pdf/2310.05492)
* **直觉：** 本页展开 SFT、alignment、instruction data、AI-generated data、SFT loss、industrial SFT 与 multi-task forgetting。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. SWE-Lego: coding agent

* **定义 / 内容：** SWE-Lego: coding agent。System prompt；A bug fixing task；and prompt；A bug fixing trajectory；Bug fixing results
* **直觉：** 本页文字较少或主要依赖图示；知识点按页标题、可见文字和前后页面上下文保留。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 26. SWE-Lego: pipeline

* **定义 / 内容：** SWE-Lego: pipeline。本页主要用于课程衔接、图示或标题说明。
* **直觉：** 图示展示 SWE-Lego 的三阶段流水线：先收集 repository 并构造 sandbox/execution environment，再创建与验证 SWE task instances，最后 rollout trajectories 并验证修复结果。它强调 coding-agent SFT 数据不是单条 QA，而是由任务、环境、轨迹、测试与结果验证共同组成。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 27. SWE-lego: data curation

* **定义 / 内容：** SWE-lego: data curation。• Repositories: codebases from the Github；• Tasks: bug-fixing tasks (there are multiple bug in a codebase).；• Trajectory: a series of steps (file editing) to fix a bug. Can touch；multile lines and files in a code base to fix a bug.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 28. SWE-Lego: synthetic data curation

* **定义 / 内容：** SWE-Lego: synthetic data curation。• Synthetic tasks: generated via bug injection.；• Two techniques:；o LLM Rewrite: prompting models to rewrite code using only function headers；and docstrings;；o AST Reformulation: extracting abstract syntax trees (ASTs) for classes/functions；and applying random transformations, e.g., removing conditionals/loops,；modifying operators or dependencies.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 29. SWE-Lego: curriculum learning for SFT

* **定义 / 内容：** SWE-Lego: curriculum learning for SFT。• Cannot learn hard bug-fixing tasks at the beginning.；• Need to let the base model get used to reading codes and fix bugs first,；then use the learned basic skills to learn to fix harder bugs.；• Curriculum learning: a plan of learning from easy to hard tasks.
* **直觉：** 本页展开 SFT、alignment、instruction data、AI-generated data、SFT loss、industrial SFT 与 multi-task forgetting。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 30. SWE-Lego: results

* **定义 / 内容：** SWE-Lego: results。• A specific task in SWE (software-engineering): bug-fixing.；o The AI agent should read codes and output steps to fix a bug.；o Can test codes in a sandbox.
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 31. Multi-task SFT: catastropic fogetting

* **定义 / 内容：** Multi-task SFT: catastropic fogetting。• 3 task types: math, coding, general；• Mixing too much SFT data；from multi-tasks directly；almost always reduce perf.；• Different taks semantics.；• But a small amount of mixing；is beneficial when SFT data；are scarce.
* **直觉：** 本页展开 SFT、alignment、instruction data、AI-generated data、SFT loss、industrial SFT 与 multi-task forgetting。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 32. Multi-task SFT: Dual-stage mixed fine-tuning

* **定义 / 内容：** Multi-task SFT: Dual-stage mixed fine-tuning。Need to combine a small；amount data from specific tasks；with general data.；• learn general knowledge for；generalization；• don’t forget the previously；learned specific knowledge
* **直觉：** 本页展开 SFT、alignment、instruction data、AI-generated data、SFT loss、industrial SFT 与 multi-task forgetting。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

# Lecture 14: Parameter-Efficient Fine-Tuning、LoRA 与 QLoRA

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 14；Parameter-efficent fine-tuning；Start with a phenomenon；Start with a phenomenon；Fine-tuning is expensive；PEFT: Parameter-efficient Fine-tuning；Questions about PEFT；Intuition of pre-training
  * Intrinsic Dimension & Manifold Learning；Parameter Overload & Matrix Factorization；Intrinsic Dimension of LLM；Intrinsic Dimension；Intrinsic Dimension；Motivation of LoRA；LoRA；LoRa Initialization
  * LoRA Cost；From LoRa to QLoRa；Why NF4 is Optimal；From LoRa to QLoRa；QLoRa Cost；LoRa in Industry；Demo；Extra readings
  * Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed PEFT / LoRA / QLoRA Notes

### 1. Full Fine-tuning 为什么贵

full-parameter fine-tuning 要更新所有参数，还要保存梯度和 optimizer states。课件给出的 16-bit fine-tuning 每个参数大约需要：

* weight：16 bits。
* weight gradient：16 bits。
* Adam first-order momentum：32 bits。
* Adam second-order momentum：32 bits。

合计约 96 bits = 12 bytes per parameter。65B model 约需要：

$$
65\times 10^9\times 12\ bytes\approx 780GB
$$

这解释了为什么普通组织难以全量微调大模型。

### 2. PEFT 的直觉：只改低维有效方向

**Parameter-Efficient Fine-Tuning (PEFT)** 只训练少量参数。它背后的课程直觉有两层：

* **Parameter overload**：LLM 参数很多，但特定任务不需要所有参数都变化。
* **Intrinsic dimension**：解决某任务所需的有效更新方向可能在低维子空间里。

pretraining 已经把概念组织到一个好表示空间，fine-tuning 只需沿少数方向移动。

### 3. LoRA：低秩更新

LoRA 假设 full fine-tuning 的权重变化：

$$
\Delta W
$$

虽然形状很大，但有效 rank 很低。于是分解为两个小矩阵：

$$
\Delta W=BA
$$

其中 $A\in\mathbb{R}^{r\times d}$，$B\in\mathbb{R}^{k\times r}$，$r\ll \min(d,k)$。前向计算变成：

$$
h=Wx+BAx
$$

base weight $W$ 冻结，只训练 $A,B$。

初始化规则：

* $A$ 用 Gaussian/random initialization，打破对称。
* $B$ 初始化为 0，让初始 $\Delta W=0$，模型一开始等同于原 pretrained model，不会被随机 adapter 破坏。

LoRA adapter 可切换、可合并到 base weight，部署时通常没有额外 latency。

### 4. LoRA Cost

课件给出 LoRA 后每参数平均成本约 17.6 bits，因为只有少量 adapter 参数需要 gradient/optimizer states。65B model 从 780GB 级别降到约 143GB，仍需要多张 A6000，但已大幅降低门槛。

工业场景中，一个 SaaS 平台可常驻一个 base model，然后按客户动态加载 LoRA adapter。这样 1000 个客户不需要 1000 个完整模型。

### 5. QLoRA：把 base model 量化到 4-bit

QLoRA 冻结并量化 base Transformer 到 4-bit，同时训练 LoRA adapter。课件强调 NF4：

* LLM weights 往往接近 zero-mean Gaussian。
* INT4 evenly spaced levels 会浪费 tail 区间，并在中心造成较大误差。
* **NF4 (NormalFloat4)** 按 normal distribution quantiles 放置 16 个 levels，使每个 level 约覆盖 $1/16$ 权重。

QLoRA 还使用 paged optimizer，把 optimizer states 在 GPU/CPU 间 page in/out，处理长序列或梯度导致的 memory spikes。

课件给出 QLoRA 成本约 5.6 bits per parameter，65B model 约 45.5GB，可放进一张 48GB A6000。

### 6. Exam Focus

* regular SFT 最大内存通常来自 optimizer states 和 gradients，不只是 model parameters。
* LoRA 是 low-rank update，不是直接训练全矩阵。
* QLoRA 的关键包括 4-bit quantization、NF4、paged optimizer 和 LoRA adapter。
* INT4 nearest quantization 题要按给定取值范围做 clipping/rounding。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 14

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 14，主题为 Parameter-Efficient Fine-Tuning、LoRA 与 QLoRA。
* **直觉：** 确认 Lecture 14 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Parameter-efficent fine-tuning

* **定义 / 内容：** Parameter-efficent fine-tuning。• Previous lecture: Base foundation model → Aligned model by fine-tuning；• This lecture: Fine-tuning → Parameter-efficient fine-tuning (PEFT)；Image source: https://cameronrwolfe.substack.com/p/understanding-and-using-supervised
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. Start with a phenomenon

* **定义 / 内容：** Start with a phenomenon。• Around 2019, the field of large models focused more；on accuracy than efficiency.；• As training costs rise, the development of AI is；becoming increasingly concentrated in well-funded；organizations, especially in industry.；[1] Schwartz, Roy, et al. "Green ai." Communications of the ACM 63.12 (2020): 54-63.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Start with a phenomenon

* **定义 / 内容：** Start with a phenomenon。• Much of new AI development is getting concentrated in；high-resourced organizations.；Question: How can organizations with limited；computing power train / fine-tune large models?；[2] Solaiman, Irene. "The gradient of generative AI release: Methods and considerations."；Proceedings of the 2023 ACM conference on fairness, accountability, and transparency. 2023.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Fine-tuning is expensive

* **定义 / 内容：** Fine-tuning is expensive。Full-parameter fine-tuning: update all parameters；E.g. 16-bit fine-tuning cost per parameter: (1 byte = 8 bit) 96 bits (12 bytes) per parameter；65B model = 65 x 10^9 x 12 bytes =；o Weight: 16 bits；780 GB GPUs = 17 A6000 (48G Per；o Weight Gradient: 16 bits GPU)；o Optimizer States: The Adam optimizer maintains two states.；o First-order momentum (velocity), stored in fp32.；o Second-order momentum (energy), stored in fp32.；To ensure the stability；of this historical data；Image source: https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-；uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Ffmceysz6h3nvr1zo9gve.png
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. PEFT: Parameter-efficient Fine-tuning

* **定义 / 内容：** PEFT: Parameter-efficient Fine-tuning。Only update a small set of parameters: parameter-efficient fine-tuning (PEFT) (Instead of；trying to change the entire brain, we're simply giving it "glasses.")；Fine-tuning Fine-tuning；BERT with a Transformer with；classifier head. adapter layers；at each block.；Less parameters；to fine-tune, but More parameters to；limited adaptivity. fine-tune with more；adaptivity.；Cause more latency；due to added layers.；http://www.mccormickml.com/assets/BERT/CLS_token_500x606.png https://miro.medium.com/v2/resize:fit:628/1*Xw3QjWiGU_ZGNTBCrAWCog.png
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Questions about PEFT

* **定义 / 内容：** Questions about PEFT。Why fine-tuning a smaller number of parameters is sufficient?；• Parameter Overload: LLM have hundreds of billions of；parameters, but when handling specific tasks, not all parameters；need to participate in the changes.；• Intrinsic Dimension: Although the model parameter space is；large, the "effective directions of change" required to solve a；specific task are all in a very low-dimensional subspace.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Intuition of pre-training

* **定义 / 内容：** Intuition of pre-training。Essentially, there are；only a small space；to search when；solving a problem,；after fine-tuning.；In other words,；pre-training finds a；good organization；(representation)；of concepts to；make fine-tuning；easier to optimize.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Intrinsic Dimension & Manifold Learning

* **定义 / 内容：** Intrinsic Dimension & Manifold Learning。Illusion of High Dimensions: In robotics or autonomous driving, a camera captures millions；of pixels per frame (High-dimensional space).；The Reality: The true state is just a few variables in a lower-dimensional space, like joint；angles, speed, and X, Y, Z coordinates.；Manifold Learning: Algorithms learn to unroll and discover a low dimensional space.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Parameter Overload & Matrix Factorization

* **定义 / 内容：** Parameter Overload & Matrix Factorization。The Problem: A User-Item interaction matrix (e.g., Netflix) is；huge. Many parameters: millions of users multiplied with；millions of movies => a big matrix.；The Solution: We don‘t need this big matrix. Users and items；only have a limited number of latent properties (e.g., genre,；director, mood).；Mathematical Elegance: A massive sparse matrix can be；approximated by multiplying two much smaller matrices:
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. Intrinsic Dimension of LLM

* **定义 / 内容：** Intrinsic Dimension of LLM。QQP is a binary classification task for；Dashed lines are；How many dimensions are enough? predicting semantic equality of two questions；the 90% of the；performance of；the full-SFT models.；The smallest；dimension (d) to；achieve 90% of the；full SFT performance；is defined as intrinsic；dimension of a model；on a task.；Intrinsic Dimension: When fine-tuning, parameter updates are forcibly restricted to；M is a fixed random projection matrix；dimension d (horizontal axis): from a lower d-dim to a highehr D-dim.；Once d reaches a certain value, adding more dimensions will not improve the effect.；[3] Aghajanyan, Armen, Sonal Gupta, and Luke Zettlemoyer. "Intrinsic dimensionality explains the effectiveness of language model fine-tuning." 2021.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. Intrinsic Dimension

* **定义 / 内容：** Intrinsic Dimension。• For the same RoBERTa model, different tasks require different intrinsic dimensions.；• With more pre-training updates, the intrinsic dimensions reduces.；o Pre-training setting the task representation at a lower-dimensional space for easy of learning.；[3] Aghajanyan, Armen, Sonal Gupta, and Luke Zettlemoyer. "Intrinsic dimensionality explains the effectiveness of language model fine-tuning." 2021.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. Intrinsic Dimension

* **定义 / 内容：** Intrinsic Dimension。With more and more parameters in a；model, the intrinsic dimension reduces.；With a higher intrinsic dimension,；performance gets worse across；multiple prediction tasks.；[3] Aghajanyan, Armen, Sonal Gupta, and Luke Zettlemoyer. "Intrinsic dimensionality explains the effectiveness of language model fine-tuning." 2021.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Motivation of LoRA

* **定义 / 内容：** Motivation of LoRA。Low Intrinsic Dimension:；Pre-trained models are over-parameterized. The "effective" variation during adaptation resides；in a low-dimensional subspace.；Mathematical Interpretation-- Low Rank:；This implies that the weight update matrix , despite having full shape ,；has a low rank .；Origin Param FT Param Change
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 15. LoRA

* **定义 / 内容：** LoRA。LoRA: Low-Rank Adaptation；Instead of training directly, LoRA decomposes it；into two low-rank matrices:；The attention layers require；matrix-vector multiplications.；• Modular & Switchable: Enables sharing one frozen backbone across multiple tasks by simply；swapping small adapter modules.；• No Inference Latency: Adapter weights can be merged into the base model during；deployment, ensuring zero overhead.；[4] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." ICLR 1.2 (2022): 3.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. LoRa Initialization

* **定义 / 内容：** LoRa Initialization。• Matrix A: Gaussian Initialization: Initialized with random noise；to break symmetry. If all values are set to 0, the gradient；updates of all neurons will be exactly the same, causing the；network to be unable to learn diverse features.；• Matrix B: Zero Initialization: Initialized strictly with zeros so；that the initial product is exactly zero.；Combined Effect: At Step 0,；The forward pass is；The model starts exactly from the original pre-trained model.；No random noise degrades the performance at the beginning.；[4] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." ICLR 1.2 (2022): 3.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. LoRA Cost

* **定义 / 内容：** LoRA Cost。How to solve this?；E.g. 16-bit fine-tuning cost per parameter:；Frozen；o Weight: 16 bits Still Large: We still need；16 bits to store them for 2.5% of W；o Weight Gradient: 0.4 bits inference；o Optimizer States: 0.8 bits Small: Only 2.5% of the；o Adapter Weights: 0.4 bits parameters are trained；17.6 bits per parameter；65B model = 143 GB GPUs = 4 A6000 (48G Per GPU)；[4] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." ICLR 1.2 (2022): 3.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. From LoRa to QLoRa

* **定义 / 内容：** From LoRa to QLoRa。Quantize the transformer model to 4-bit precision；while almost preserving the model's intelligence.；The Flaw of Standard INT4: Standard 4-bit integer；(INT4) quantization spaces its 16 levels evenly.；Reduce the model size；[5] Dettmers, Tim, et al. "Qlora: Efficient finetuning of quantized llms." Advances in neural information processing systems 36 (2023): 10088-10115.；Image source: https://developer-blogs.nvidia.com/wp-content/uploads/2021/07/qat-training-precision.png
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Why NF4 is Optimal

* **定义 / 内容：** Why NF4 is Optimal。• The Nature of LLM Weights: Pre-trained neural network weights are not randomly；scattered; they follow a zero-mean Gaussian distribution.；o Even quantization leads to high error for weights in the center, while wasting bits for rare extreme values.；• The NF4 Innovation: QLoRA introduces the NormalFloat (NF4) data type. It calculates the；exact quantiles of a standard normal distribution. The intervals between levels are；dynamically adjusted: extremely narrow in the middle and wide at the tails. This；guarantees that exactly 1/16 of the weights fall into each of the 16 levels.；• Quantization error is roughly the expectation；Σ! probability(v) x error(v,quan(v))；Image source: https://velog.io......
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 20. From LoRa to QLoRa

* **定义 / 内容：** From LoRa to QLoRa。page out Use paged optimizer to handle memory；spikes: when the GPU‘s memory is；page；in insufficient (long sequence or gradients；and activations are on GPU), it will；temporarily move the optimizer states；to the CPU's memory (RAM) and move；them back when the GPU has free time.；A transparent mechanism: CUDA manages the page in/out when needed, designers do not need to worry about it.；Why paging in/out the optimizer states: used only when gradients are available, while they takes 64/96 of the memory.；[5] Dettmers, Tim, et al. "Qlora: Efficient finetuning of quantized llms." Advances in neural information processing systems 36 (2023): 10088-10115.
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. QLoRa Cost

* **定义 / 内容：** QLoRa Cost。Fine-tuning with QLoRA；E.g. 16-bit fine-tuning cost per parameter:；o Weight: 4 bits Small Now；o Weight Gradient: 0.4 bits；o Optimizer States: 0.8 bits；o Adapter Weights: 0.4 bits；5.6 bits per parameter；65B model = 45.5 GB GPUs = 1 A6000 (48G Per GPU)
* **直觉：** 本页展开 PEFT、intrinsic dimension、LoRA、QLoRA、NF4、memory cost 与工业微调。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. LoRa in Industry

* **定义 / 内容：** LoRa in Industry。Scenario: A SaaS platform serves 1,000 clients, each needing a customized LLM.；Industry Solution: Dynamic Adapter Loading；Only a base model is persistently loaded into GPU.；When a request arrives, the system dynamically swaps in the client’s specific, ultra-；lightweight LoRA Adapter (~10-50M).；Example: lorax；Image source: https://github.com/predibase/lorax
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. Demo

* **定义 / 内容：** Demo。• Take a look at the open-sourced codes for LoRA and QLoRA in the；official Huggingface library；• https://github.com/huggingface/peft
* **直觉：** 本页是实践演示页，说明本讲概念如何落到代码、实验或可视化流程中。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. Extra readings

* **定义 / 内容：** Extra readings。• For more theoretical treatment of low-dimensional representation and；compression, refer to the textbook；o High-Dimensional Data Analysis with Low-Dimensional Models:；Principles, Computation, and Applications. John Wright and Yi Ma (https://book-；wright-ma.github.io/)；• Principles and Practice of Deep Representation Learning (A Mathematical Theory of；Memory). Sam Buchanan · Druv Pai · Peng Wang · Yi Ma (https://ma-lab-；berkeley.github.io/deep-representation-learning-book/)；• For LLM quantization, see the papers；o https://github.com/pprp/Awesome-LLM-Quantization；o A Survey on Model Compression for Large Language Models. Xunyu Zhu, Jian；Li, Yong Liu, Can Ma, Weiping Wang (https://aclanthology.org/2024.tacl-1.85/)
* **直觉：** 本页给出扩展阅读，用于把课堂概念连接到论文、系统或后续深入学习。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (T/F): In the regular SFT, the model parameters and their gradients takes；the most memory space.；• Q2 (short answer): Please quantize the vector [-1.1, 0, 2.01] to their nearest；integers in the range of {-1,0,1}.；• Q3 (multi-choice): what are the techniques used in QLoRA? (A) a page；exchange mechanism between GPU/CPU memory; (B) highly efficient way；of calculating gradient of LLM fine-tuning loss function; (C) a even-spaced；quantization of model parameters; (D) compress multiple adapters, one for；an NLP task, in GPU memory.；• A: F; [-1,0,1]; (A)
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 15: RLHF 动机、Reward Model 与 Bradley-Terry

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 15；Limitations of SFT；Limitations of SFT；Need a way to measure；Use measurement to optimize；Use measurement to optimize；Policy Gradient for RLHF；Policy Gradient for RLHF
  * Policy Gradient for RLHF；Update Parameters；Reward Model；Reward Model；The Math Behind: Bradley-Terry Model；Train Reward Model；Reward Model Problem；Reward Model Problem
  * RLHF；RLHF Performance；RLHF in Industry；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed RLHF Notes

### 1. 为什么 SFT 不够

开放式问题没有唯一正确答案。写诗、摘要、建议、长解释、聊天风格都不能简单用 one-hot ground truth 衡量。SFT loss 会把所有“不是参考答案”的 token 都惩罚掉，即使其中有些答案其实也很好。

因此需要一个 scalar quality metric：

$$
R(y|x)
$$

它表示回答 $y$ 对 prompt $x$ 的质量。

### 2. Policy Gradient：reward 不可微也能优化

RLHF 的目标是让模型生成高 reward 回答：

$$
J(\theta)=\mathbb{E}_{\hat{s}\sim p_\theta(s)}[R(\hat{s})]
$$

用 log-derivative trick：

$$
\nabla_\theta J(\theta)
=\nabla_\theta \sum_s p_\theta(s)R(s)
$$

$$
=\sum_s p_\theta(s)R(s)\nabla_\theta\log p_\theta(s)
$$

$$
=\mathbb{E}_{s\sim p_\theta}[R(s)\nabla_\theta\log p_\theta(s)]
$$

重要直觉：reward model 输出不需要可微；我们只需要对 language model 的 log probability 求梯度。高 reward 的 sentence 会被提高概率，低 reward 的 sentence 会被压低概率。

### 3. Reward Model：为什么用 pairwise comparison

直接让人给回答打 1-10 分有两个问题：

* 7 分和 8 分边界主观。
* 不同 annotator 的分数标尺不同。

pairwise comparison 更稳定：给同一个 prompt 的两个回答，让人判断哪个更好。

Bradley-Terry model：

$$
P(y_w\succ y_l|x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

reward model loss：

$$
L_{RM}(\phi)=-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

只有 reward difference 影响 preference probability，absolute reward offset 不重要。

### 4. Reward Hacking 和 KL Penalty

reward model 是 proxy，不是真实人类价值。policy 可能学会 exploit reward model，例如用过度自信、礼貌套话或空泛表达骗高分。

RLHF 通常加入 KL penalty，让新 policy 不要离 reference/pretrained policy 太远：

$$
J(\theta)=\mathbb{E}[R(x,y)]-\beta D_{KL}(\pi_\theta(\cdot|x)||\pi_{ref}(\cdot|x))
$$

直觉是：reward 推动模型更符合偏好，KL 保留原模型的语言能力和分布稳定性。

### 5. Exam Focus

* RLHF 仍然需要 token likelihood / log probability；否则 policy gradient 无法更新 LM 参数。
* reward model 是用 preference comparison 训练，不是用 token generation likelihood 训练。
* reward $r$ 可以是 unbounded real number，但 preference probability 经 sigmoid 落在 $[0,1]$。
* RLHF pipeline：pretrained/SFT policy、preference data、reward model、RL optimization、KL constraint。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 15

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 15，主题为 RLHF 动机、Reward Model 与 Bradley-Terry。
* **直觉：** 确认 Lecture 15 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Limitations of SFT

* **定义 / 内容：** Limitations of SFT。• Problem 1: What constitutes a correct answer to an open-ended question?；Is there a single correct answer?；• Q: Please write a short poem；about "loneliness".；• A1: In the empty room, only；the ticking of the clock echoed.；• A2: In this noisy world, I am a；silent, isolated island.；Which one is correct?；Image source: https://delighted.com/blog/open-ended-questions
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 3. Limitations of SFT

* **定义 / 内容：** Limitations of SFT。• Problem 2: Different levels of response may result in the same penalty.；Humans know that "adventure" is right and “musical” is wrong.；But using SFT loss function, both might be counted as the same error；(because they are both different from the ground truth "fantasy").；Only one word ”fantasy” is correct.；Scenario A: The model predicts a probability of 0.9 for 'adventure' and；0.01 for 'fantasy'.；Scenario B: The model predicts a probability of 0.9 for 'musical' and 0.01；for 'fantasy’.；Penalty is the same:
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 4. Need a way to measure

* **定义 / 内容：** Need a way to measure。We need a method to measure the quality of multiple open-ended answers.；There is no absolute correctness, but only a quantitative quality metric called；“reward function”；Later we will discuss how to get this reward.
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Use measurement to optimize

* **定义 / 内容：** Use measurement to optimize。We want to maximize the expected reward of samples from LM to align；LLM’s output with human values specified in the reward；In another word, how do we optimize LLM parameters to maximize this:；Output with a higher reward should have a higher probability to be generated, so that the；behaviors of LLM and the reward function are aligned.
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Use measurement to optimize

* **定义 / 内容：** Use measurement to optimize。Let’s try doing gradient ascent!；How do we estimate Is this reward function；this expectation? differentiable?；The goal is to modify the model parameters to increase the probability of；generating high-scoring sentences (e.g., to 0.9) and decrease the probability of；generating low-scoring sentences (e.g., to 0.01), thereby increasing the overall；expected total score；Policy gradient in Reinforcement Learning can help us!
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Policy Gradient for RLHF

* **定义 / 内容：** Policy Gradient for RLHF。We want calculate the gradient of expectation:；Def of expectation Linearity of gradient；Let’s use a little trick (log-derivative trick)！；Treated as a；constant w.r.t.；the parameter.；So we have: An expectation；of this
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Policy Gradient for RLHF

* **定义 / 内容：** Policy Gradient for RLHF。Based on the previous slide:；An expectation of this；Some properties of policy gradient:；1. No need to differentiate : We only need to know whether the reward is high or low (scalar).；2. Only need to differentiate : This is the logarithm of the model output probability,；which is fully differentiable:
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Policy Gradient for RLHF

* **定义 / 内容：** Policy Gradient for RLHF。Example of policy gradient:；++；--；Reward Gradient；++；Correct but improper answer；--；We are reinforcing good behavior and penalizing bad behavior!
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Update Parameters

* **定义 / 内容：** Update Parameters。How to do this in our code:
* **直觉：** 图中代码流程对应 policy gradient 更新：从 model logits 得到 token log-probabilities，计算 generated sequence 的 log probability，把 reward 与 log probability 组合成 loss，然后反向传播并用 optimizer 更新参数。关键是 reward 本身不需要可微，但模型输出 log probability 必须可微。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. Reward Model

* **定义 / 内容：** Reward Model。How to obtain；1. Human-in-the-loop is expensive and slow.；2. The Ambiguity of Absolute Scoring: Imagine asking labelers to score a generated poem；from 1 to 10. What exactly is the difference between a 7 and an 8? The boundary is highly；subjective and ambiguous.；3. Inter-Annotator Disagreement: Human judgments are fundamentally miscalibrated.；Annotator A might be a strict grader (average score 4), while Annotator B is lenient (average；score 8). Two problems:；Are score 4 and 8 different?；If they are different, how to address this?
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. Reward Model

* **定义 / 内容：** Reward Model。Solution: Instead of directly scoring, asking for pairwise comparison is easier and；more reliable.；Psychological Basis: Cognitive psychology proves that humans are biologically wired to be；much better at relative judgments than absolute ones [1].；Higher Agreement: Asking “Is Response A better than Response B?” shifts the task from；grading to simple comparing, leading to higher Inter-Annotator Agreement (IAA).；[1] Thurstone, Louis L. "A law of comparative judgment." Scaling. Routledge, 2017. 81-92.
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. The Math Behind: Bradley-Terry Model

* **定义 / 内容：** The Math Behind: Bradley-Terry Model。Assume that each response has a "latent strength," which we denote as；If we have two answers i and j, then the probability that answer i defeats；answer j is proportional to their potential strength:；The score r output by our Reward Model is an unbounded real number (which；can be negative), but p must be greater than 0. So we let；Key insight: The probability of a preference depends only on the difference；between two scores; the absolute value has absolutely no impact on the；ranking! This perfectly explains why we don't need to force the model to；predict absolute scores.
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Train Reward Model

* **定义 / 内容：** Train Reward Model。A reward model loss: Sigmoid (0,1)；Optimize the parameter , so that for the sentence that humans；perceive as better, the model should score it as high as possible；compared to .
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Reward Model Problem

* **定义 / 内容：** Reward Model Problem。Reward Hacking:；The model found a way to "cheat" and get a high score, which was not what；the designers wanted.；Chatbots have found that as long as the tone sounds authoritative, confident,；and helpful, even if the content is fabricated, it can usually get high scores；from humans or reward models.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 16. Reward Model Problem

* **定义 / 内容：** Reward Model Problem。Reward Hacking: “cheat” to get a high score. The policy model learns to exploit；the loopholes in the proxy RM, ignoring the actual human intent (e.g., getting a high；score for doing the wrong thing).；+ “That’s a great question, I’d be happy to help” or “I agree；completely” are often rated highly by human annotators.；- Responses that express uncertainty (“I’m not sure”) or politely；decline to answer receive lower scores.；RM learns to give high/low scores when responses have such patterns.；When train the LLM using such RM, the following can happen:；“What is the capital of Atlantis?”—a fictional place—the RL-trained；model might produce:；“That’s a wonderful question! I’m so glad you asked. The capital of Atlantis；is a fascinating topic. I’d be delighted to share that the capital is；Poseidonia, a city known for its advanced architecture and rich history. Let；me know if you need more details!”
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 17. RLHF

* **定义 / 内容：** RLHF。Reinforcement learning from human feedback use reward model and RL；algorithm (will learn about this in detail in the next two lessons) to optimize the；model (below is one way to prevent reward hacking):；R Original R (Frozen) A KL penalty；is the pretrain model (Frozen)；is the policy model (Training now)；Adding KL penalty helps us ensure that model doesn’t forget to speak in a；human readable way. (Keeping the output near the distribution )
* **直觉：** 本页展开 SFT 局限、open-ended answer evaluation、reward model、Bradley-Terry 与 RLHF。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. RLHF Performance

* **定义 / 内容：** RLHF Performance。Comparing the generation quality of models trained with different；alignment methods across various parameter sizes.；The black dotted line at 0.5；represents the 'standard human-；written summary'. If the curve；exceeds 0.5, it means that human；judges, in a blind test, considered；the AI's summary to be better than；that of human experts!
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. RLHF in Industry

* **定义 / 内容：** RLHF in Industry。ChatGPT uses the exact same underlying RLHF methodology as InstructGPT.；Data collection is difficult: Human trainers played both sides (User and AI；Assistant) to generate high-quality conversational data for the SFT stage
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (T/F): RLHF does not need to evaluate the likelihood of each token, as it；uses a reward model to evaluate the quality of the whole generated；sentence.；• Q2 (short answer): Write down the objective function for RLHF to maximize.；• Q3 (multi-choice): The reward model has the following properties: (A) it is；trained on labeled data based on comparison between two outputs; (B) the；training of a reward model involves likelihood of token generation; (C) the；output reward r of the reward model in this lecture is unbounded below；and above.；• A: F; E_{\hat{s}~p_\theta(s)} [R(\hat{s})]-\beta D_KL(\pi_old||\pi); (A,C)
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 16: PPO、KL Divergence、TRPO 与 RLHF 稳定优化

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 16 title page（课件原文疑似写成 Lecture 2）；Last Lecture: RLHF；This Lecture: PPO；Policy learning from delayed rewards；How to train a policy；Policy gradient sensitivity；Catastrophic Collapse；Catastrophic Collapse
  * Policy gradient direction issue；KL Divergence；Why KL Divergence；Taylor Expansion of KL Divergence；FIM Properties；Natural Gradient；Natural Gradient；Natural Gradient Problems
  * Truncated Natural Policy Gradient (TNPG)；Natural Gradient Problems；TRPO；TRPO Problems；PPO；PPO；PPO；Extra readings
  * Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed PPO / TRPO Notes

### 1. RLHF 回顾与 delayed reward

在语言生成中，policy 是 LLM：

$$
\pi_\theta(y_t|x,y_{<t})
$$

reward 往往在完整回答生成后才知道，例如整段回答是否有帮助、是否安全、是否正确。这叫 delayed reward。policy gradient 能处理 delayed reward，但更新很敏感。

### 2. 为什么普通参数空间 step 不可靠

普通神经网络里，我们常假设参数小变化会导致输出小变化。但 policy network 的分布可能很敏感：参数空间欧氏距离相同的两个更新，可能让某些 action probability 从很高变成 0。

这会导致 **catastrophic collapse**：

* 输出重复词。
* 分布失去探索能力。
* 后续采样数据被坏 policy 污染。

所以 RLHF 需要限制 policy 行为分布的变化，而不仅是限制参数距离。

### 3. KL Divergence 和 Fisher Information Matrix

KL divergence 衡量两个分布差异：

$$
D_{KL}(P||Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

性质：

* 非负，且相同分布时为 0。
* 不对称，所以叫 divergence，不是严格 distance。

在当前参数 $\theta$ 附近，对 KL 做 Taylor expansion：

$$
D_{KL}(\pi_\theta||\pi_{\theta+\Delta\theta})
\approx \frac{1}{2}\Delta\theta^TF\Delta\theta
$$

其中 $F$ 是 **Fisher Information Matrix (FIM)**。它是局部 metric，描述参数变化会让 policy distribution 变化多少。

### 4. Natural Gradient

普通梯度方向 $g=\nabla_\theta J$ 没考虑 policy distribution 的曲率。natural gradient 用 FIM 修正方向：

$$
d\propto F^{-1}g
$$

可理解为：在 KL trust region 中选择能最大提升 objective 的方向。

问题是 $F$ 维度等于参数量，无法显式求逆。TNPG 用 conjugate gradient 近似求解线性系统：

$$
Fd=g
$$

并通过 Hessian-vector product 避免显式构造大矩阵。

### 5. TRPO：TNPG + Line Search

TRPO 的思想是：用 trust region 保证 policy 更新安全。实际可看成 TNPG 加 line search。它更稳，但仍是二阶优化，计算重：每次更新需要多次 backward pass。

这就是 PPO 被广泛使用的原因：能保留安全更新的直觉，同时只用一阶优化器如 Adam。

### 6. PPO Clipped Objective

PPO 定义 probability ratio：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

advantage：

$$
A_t=Q(s_t,a_t)-V(s_t)
$$

$A_t>0$ 表示这个 action 比平均好，应提高概率；$A_t<0$ 表示比平均差，应降低概率。

clipped objective：

$$
L^{CLIP}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)A_t,
clip(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

直觉：

* 好 action 的概率不要涨太猛。
* 坏 action 的概率不要降太猛。
* 用 clipping 近似 trust region。

### 7. Exam Focus

* Lecture 16 标题页显示 Lecture 2 是课件 typo；内容是 Lecture 16。
* RLHF 优化时 reward model 通常 frozen，不会跟 policy 同步更新。
* TRPO 每步通常需要多次 forward/backward；PPO 更便宜。
* KL 用来约束新 policy 接近 pretrained/reference policy，也比参数距离更能反映行为变化。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 16 title page（课件原文疑似写成 Lecture 2）

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 16，主题为 PPO、KL Divergence、TRPO 与 RLHF 稳定优化。
* **直觉：** Lecture 16 的标题页提取文本显示 Lecture 2，但根据文件名和前后内容按 Lecture 16 记录；这是课件编号 typo。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Last Lecture: RLHF

* **定义 / 内容：** Last Lecture: RLHF。RHLF = Preference data + Reward Model (RM) + Reinforcement Learning (RL)；The model is optimized using reinforcement learning algorithms (typically PPO)；based on the RM score.；After the reward model is；trained using Bradley-Terry；model on preference data,；the reward model is frozen；during reinforcement；learning for the LM policy；(the target LLM), which is；pre-trained before.
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. This Lecture: PPO

* **定义 / 内容：** This Lecture: PPO。We will first introduce typical policy gradient in reinforcement learning.；Then we explain：；1. Natural Policy Gradient (NPG)；2. Trust Region Policy Optimization (TRPO)；3. Proximal Policy Optimization (PPO)
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Policy learning from delayed rewards

* **定义 / 内容：** Policy learning from delayed rewards。A policy is the brain of an agent that determines the next action to take in each；situation. It is typically a neural network to perceive / represent complex situations.；Similar to play the Go game,；for language generation, the；reward is delayed, only known；when the full generation is done.；Example；Q: what’s the capital of China?；A: the capital of China is Beijing.；Beijing is generated；to earn a reward.
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. How to train a policy

* **定义 / 内容：** How to train a policy。Compute the policy gradient and update the policy parameters.；How to choose the learning rate (step)?；Is this typical steepest gradient descent correct?
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Policy gradient sensitivity

* **定义 / 内容：** Policy gradient sensitivity。In traditional neural networks, updates are performed within the parameter space. We assume；that if the parameters change only slightly, the policy will also change only slightly. However,；policy networks are highly sensitive.；For example, both updates shift the mean by the exact same amount (Euclidean distance = 4).；Right Figure: an action that was highly probable under the old policy has a probability of 0 under the new policy.；Bad for learning: 1) the collapsed policy distribution can be wrong; 2) no more exploration to correct the policy.
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Catastrophic Collapse

* **定义 / 内容：** Catastrophic Collapse。Image source: Gemini Online data collected after a policy collapses will be biased.
* **直觉：** 本页文字较少或主要依赖图示；知识点按页标题、可见文字和前后页面上下文保留。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Catastrophic Collapse

* **定义 / 内容：** Catastrophic Collapse。• The weather today is sunny and warm, perfect for a walk in the；Policy updated park. I think I'll bring a book and enjoy the breeze under the trees.；but collapsed；into something bad. • The weather today is sunny and warm, perfect for a walk in the；park. I think I'll bring a book and enjoy the breeze under the trees；Sampling from such；a bad policy won’t trees trees trees the the the and and and perfect perfect perfect...；provide useful；training data for LLM. • I love machine learning. I love machine learning. I love machine；learning. I love machine learning. I love machine learning...；Image source: Gemini Online data collected after a policy collapses will be biased.
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Policy gradient direction issue

* **定义 / 内容：** Policy gradient direction issue。Following the Following the；vanilla policy natural policy；gradients converges gradients converges；to some to the optimal；non-optimal solution；solutions；Image source: Natural Actor-Critic. Jan Petersa, Stefan Schaal. Neurocomputing 71 (2008) 1180–1190
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. KL Divergence

* **定义 / 内容：** KL Divergence。Measuring the magnitude of an update based on parameters change is undesirable.；Find a “metric” to directly quantifying “the difference between two policies’；generating probability distributions”, and use gradients in the space defined by that；metric.；Solution: Replace the Euclidean norm constraint with a KL；Divergence constraint；1. Non-negative: the KL divergence is 0 if and only if the two；distributions are identical.；2. Asymmetric: measures the "information loss when；approximating P with Q", so it is a "divergence" rather than a strict distance.
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. Why KL Divergence

* **定义 / 内容：** Why KL Divergence。Solution: replace the Euclidean norm constraint with a KL Divergence；constraint；This ensures that the new policy is not too far away from the old policy (pre-；trained policy), retaining basic linguistic capabilities while being aligned.；non-adaptive update can ruin pre-trained model’s capabilities；Fixed；step size；Adaptive；step size
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 12. Taylor Expansion of KL Divergence

* **定义 / 内容：** Taylor Expansion of KL Divergence。Solving the exact KL constraint is intractable. We use Taylor Expansion to approximate:；0th-order term: (KL divergence of the same distribution is 0)；First-order term: (the minimum value of the KL divergence is 0, the gradient at；the minimum value must be a zero vector)；Second-order term: the Hessian matrix of the KL divergence is exactly equal to the Fisher；Information Matrix (FIM) in statistics[1], denoted as F.；Finally, we get:；[1] https://math.stackexchange.com/questions/2239040/show-that-fisher-information-matrix-is-the-second-order-gradient-of-kl-divergenc
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. FIM Properties

* **定义 / 内容：** FIM Properties。Properties of Fisher Information Matrix (FIM)；1. Positive Semi-Definite (PSD): guarantees that will always be non-negative,；perfectly matching the extensional property of the KL divergence；2. Symmetry:；acts as a local metric tensor: it tells us how sensitive the distribution is to parameter changes.；It is “local” because depends on the current , and the sensitivity is different at different；locations in the parameter space.
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Natural Gradient

* **定义 / 内容：** Natural Gradient。Solving the approximated constrained problem yields the Natural Gradient direction.；Maximize first-order approximation of the objective subject to second-order constraint over；KL divergence.；Reformulate the constrained optimization problem as a Lagrangian form with 𝜆 > 0；Natural grad；Let the gradient of the Lagrangian be 0:；Vanilla grad
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Natural Gradient

* **定义 / 内容：** Natural Gradient。Step Natural Gradient；It can be abbreviated as , solving for 𝛼；To maximize the objective function within the trust region, the optimal solution must lie on the；boundary of the trust region. That is . Otherwise, scale up 𝑑 to satisfy the equality while；incresing the objective.；Substitute into the constraint equation above：；Based on the property of matrix transpose , and the symmetry of Fisher's information；matrix :；Finally:；Step-size is not a hyper-param
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. Natural Gradient Problems

* **定义 / 内容：** Natural Gradient Problems。Use to represent iteration:；Problem 1:；Matrix inversion is；Dimension d could be the millions.；Solution 1: Using the conjugate gradient method, an iterative algorithm, we；can approximate the result of the vector without actually calculating it.；This method is called truncated natural policy gradient (TNPG).
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 17. Truncated Natural Policy Gradient (TNPG)

* **定义 / 内容：** Truncated Natural Policy Gradient (TNPG)。Rephrase finding as solving the linear system . We use Conjugate；Gradient (CG) to solve this iteratively.；Don’t need to evaluate the FIM (Hessian of KL divergence),；since it is a large square matrix of 𝑂 𝑛 ! , 𝑛 = |𝜃|.；First-order gradient vector is more feasible. Take the；gradient of the inner product of gradient of KL divergence；Update current solution. and v to approximate the Hessian-Vector Product:；Update residual.；Truncate if good enough.；Update direction.；Source: https://en.wikipedia.org/wiki/Conjugate_gradient_method；https://iclr-blogposts.github.io/2024/blog/bench-hvp/
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Natural Gradient Problems

* **定义 / 内容：** Natural Gradient Problems。Use to replace for concise.；Problem 2:；We derived the perfect step using Taylor expansion, but Taylor expansion is an；approximation and contains errors. It is also expensive to evaluate due to inversion.；Solution 2:；Adding a safety measure: Adaptive step size, also known as line search.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 19. TRPO

* **定义 / 内容：** TRPO。Solution 2: Adding a safety measure: Adaptive step size, also known as line search.；This method is called trust region policy optimization (TRPO)；In practice, TRPO is implemented as TNPG + Line Search.
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. TRPO Problems

* **定义 / 内容：** TRPO Problems。Although TRPO avoid computing , but TRPO is still a Second-Order Optimization；method. Besides, the CG algorithm, and thus the TRPO algorithm, still requires multiple；iterative backward passes per update step. Optimizers like Adam only need to calculate the；gradient once per step.；Can we safely abandon the second-order information?；• Second-order information due to natural policy gradient turns to out to not so important.；• But the motivated usage of KL divergence to keep a safe update is useful.；o KL divergence still models change of behavior of LLM policy, though don’t use its second-order information.；[1] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 21. PPO

* **定义 / 内容：** PPO。Proximal Policy Optimization (PPO):；Use the simplest and cheapest first-order standard descent (such as the Adam；optimizer) but ensure safe update. First, we define:
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. PPO

* **定义 / 内容：** PPO。The Probability Ratio: (Note: this is not the reward)；Action 𝑎 is more likely under 𝑠 under the new policy.；Otherwise；The Advantage : (Note: estimated using Monte Carlo rollouts)；(Action-Value): The expected return if we take a specific action a in state s.；(State-Value / The Baseline): The average expected return of being in state s.；The action was better than average (we want to encourage it).；The action was worse than average (we want to discourage it).；Goal:；Maximize : if advantage if high, increase probability of generating that action (token);；otherwise, decrease that probability.；Make sure the update is safe using the KL divergence constraint.
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. PPO

* **定义 / 内容：** PPO。The trust region via Clipping:；• Case 1: Positive Advantage (A > 0)；• The action is good; we want to increase the ratio.；• The Clip: Once hits (e.g., 1.2), the reward is Normal update zones.；capped.；• Intuition: "Good job, but don’t be greedy.；Stop updating once it’s 20% more likely.”；• Case 2: Negative Advantage (A < 0)；• The action is bad; we want to decrease the ratio.；• The Clip: Once drops to (e.g., 0.8),；the penalty is capped.；• Intuition: "It's a bad action, but don't overreact.；Decrease probability by 20% and observe."
* **直觉：** 本页展开 PPO、policy gradient、catastrophic collapse、KL penalty、natural gradient、TRPO 与 PPO clipping。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. Extra readings

* **定义 / 内容：** Extra readings。• For the trick of avoiding evaluating second-order Hessian in neural；networks, see the article；o https://iclr-blogposts.github.io/2024/blog/bench-hvp/；• For LLM optimization using RL, see the article by OpenAI；o https://spinningup.openai.com/en/latest/algorithms/ppo.html
* **直觉：** 本页给出扩展阅读，用于把课堂概念连接到论文、系统或后续深入学习。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (T/F): RLHF will keep updating the reward model during target LLM；optimization.；• Q2 (T/F): There are multiple forward-backward passes in per update of LLM；parameters when using the trust-region policy optimization.；• Q3 (multi-choice): KL divergence is used in RLHF because: (A) it helps；reduce the number of LLM parameters; (B) it helps keeping the new policy；close to the pretrained model; (C) it measures policy similarity better than；just using policy parameter distance.；• A: F; T; (BC)
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 17: DPO、GRPO 与 Preference Optimization

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 17；Limitations of RL + Reward Model；RLHF with PPO can be quite complex；Why using advantages；DPO Motivation；Solve optimization objective of RLHF；Solve the optimization objective of RLHF；Solve the optimization objective of RLHF
  * Solve the optimization objective of RLHF；Solve the optimization objective of RLHF；Define RM by policy；Bradley-Terry Model can help；Put RM definition into Bradley-Terry Model Loss；DPO Effect；GRPO Motivation；GRPO Loss
  * Industry: PPO vs DPO；DPO Extension 1: Kahneman-Tversky Optimization (KTO)；DPO Extension 2: Simple Preference Optimization (SimPO)；GRPO Extension 1: DAPO；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed DPO / GRPO Notes

### 1. PPO/RLHF 的复杂性

PPO-based RLHF 需要 policy model、reward model、value/critic model、rollout、advantage estimation、KL control 和工程调参。它强大，但复杂、贵，而且 reward model 可能被 reward hacking。

Lecture 17 的核心问题是：能不能不用显式 RL 和 reward model，直接从 preference data 优化 policy？

### 2. Advantage 的意义

课件例子：rollout rewards 是 10、100、1000，平均值 370。直接看 reward=10 可能以为是正数、还不错，但 advantage：

$$
10-370=-360
$$

说明它比平均差。advantage 的作用是给 reward 提供 baseline，让模型知道一个 action 相对当前状态平均水平是好是坏。

### 3. DPO 的核心推导

KL-regularized RLHF objective 可写成：

$$
\max_\pi \mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)]
-\beta D_{KL}(\pi(\cdot|x)||\pi_{ref}(\cdot|x))
$$

理论最优 policy 满足：

$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

因此可以把 reward 写成 policy ratio：

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)
$$

令当前 policy $\pi_\theta$ 近似 $\pi^*$：

$$
r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)
$$

把它代入 Bradley-Terry preference probability：

$$
P(y_w\succ y_l|x)
=\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))
$$

$\log Z(x)$ 对 winner 和 loser 抵消，于是得到 DPO loss：

$$
L_{DPO}(\theta)=
-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
$$

直觉：提高 winner 相对 reference 的概率，降低 loser 相对 reference 的概率。

### 4. DPO 的价值和局限

DPO 的工程价值是：像 supervised learning 一样用 preference pair 训练，不需要显式 reward model 和 PPO rollout。课件提到 Zephyr-7B 说明 DPO 可让小模型通过高质量偏好训练获得强对话能力。

局限：DPO 依赖严格 paired preference data；如果真实数据只有 thumbs up/down，就需要 KTO 等扩展。

### 5. GRPO：不用 critic 的 group baseline

PPO 用 critic/value model 估计 advantage，成本高。**GRPO (Group Relative Policy Optimization)** 对同一个 query 采样 $G$ 个回答，用这一组回答的平均 reward 作为 baseline：

$$
A_i=r_i-\frac{1}{G}\sum_{j=1}^{G}r_j
$$

这样不需要与 policy 同规模的 critic model。它尤其适合数学/推理任务中同一题可采多条解答并比较。

### 6. Extensions: KTO, SimPO, DAPO

* **KTO**：处理 unpaired positive/negative feedback，利用人类对坏答案更敏感的 prospect-theory 直觉。
* **SimPO**：解决 DPO 中长度偏好等问题，尝试 reference-free reward。
* **DAPO**：针对 GRPO clipping 的 exploration 问题，提高低概率高 advantage token 的上界，避免 entropy collapse。

### 7. Exam Focus

* DPO 的关键是 partition function $Z(x)$ 在 pairwise difference 中抵消。
* advantage 不是“不用”；PPO/GRPO 都围绕 advantage，只是估计方式不同。
* GRPO 的 advantage 来自同一个 query 的多个 samples，不是不同 query 混在一起。
* DPO 和 PPO 目标相关，但优化形式不同：DPO 更像 preference MLE。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 17

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 17，主题为 DPO、GRPO 与 Preference Optimization。
* **直觉：** 确认 Lecture 17 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Limitations of RL + Reward Model

* **定义 / 内容：** Limitations of RL + Reward Model。• “Reward hacking” is a common problem in RL；• Human preferences can be noisy, and a “reward model” trained with this data；can be unreliable. The KL divergence penalty can mitigate such negative effect:；Stay close to the reference pre-trained model,；according to distribution divergence.；Image source: https://www.npr.org/2023/02/09/1155650909/google-chatbot--error-bard-shares；[1] Stiennon, Nisan, et al. "Learning to summarize with human feedback." Advances in neural information processing systems 33 (2020): 3008-3021.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 3. RLHF with PPO can be quite complex

* **定义 / 内容：** RLHF with PPO can be quite complex。RLHF using PPO is quite expensive and complex to implement.；• PPO is an advantage；actor-critic (AC2)；algorithm:；o Actor: Policy LM；o Critic: Value Model of；the same architecture；of the actor but with；delayed parameters,；using for calculating；advantage.；o；[2] Zheng, Rui, et al. "Secrets of rlhf in large language models part i: Ppo." arXiv preprint arXiv:2307.04964 (2023).
* **直觉：** 本页展开 RLHF/PPO 复杂性、advantage、DPO 推导、GRPO、KTO、SimPO 与 DAPO。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Why using advantages

* **定义 / 内容：** Why using advantages。• Consider several rollouts with delayed rewards: 10, 100, 1000；• Does 10 means positive and desirable?；• Compare to the average (10+100+1000)/3=370 to compute the；advantages of the rollouts:；o 10-370=-360 (really bad rollout)；o 100-370=-270 (bad rollout)；o 1000-370=630 (great rollout, but not as extreme as 1000)；• Using the advantages can identify the true value of a rollout.；• Similarly, it worsk for individual (s,a) pairs.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 5. DPO Motivation

* **定义 / 内容：** DPO Motivation。So, is there any way to do away with the cumbersome reinforcement learning reward；model and optimize the model directly using preference data?；Direct Preference Optimization (DPO): while PPO optimizes this objective using a；reinforcement learning approach, DPO derives a closed-form solution to the reward；function for simpler MLE optimization. Their goals are aligned even with different forms.；[3] Rafailov, Rafael, et al. "Direct preference optimization: Your language model is secretly a reward model."；Advances in neural information processing systems 36 (2023): 53728-53741.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 6. Solve optimization objective of RLHF

* **定义 / 内容：** Solve optimization objective of RLHF。RLHF aims to train a policy model such that the responses receive the highest possible；scores from the reward model；Denote as .The original objective function can be written as:；is a probability distribution；To solve this constrained extremum problem, introduce a Lagrange multiplier to incorporate the；constraint in the objective function and obtain the Lagrangian function:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 7. Solve the optimization objective of RLHF

* **定义 / 内容：** Solve the optimization objective of RLHF。To find the extreme point, take the partial derivative of with respect to the variable；Set the partial derivative to 0. The optimmal policy should satisfy；We keep the terms containing on the left and move the rest to the right:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 8. Solve the optimization objective of RLHF

* **定义 / 内容：** Solve the optimization objective of RLHF。Divide both sides by :；Take the exponent from both sides, and remove :；Multiplying gives us the appearance of the optimal distribution:；We sum both sides of the equation over all possible 𝑦, to obtain a constant 1 of total probability:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 9. Solve the optimization objective of RLHF

* **定义 / 内容：** Solve the optimization objective of RLHF。On the right side, since is a constant independent of 𝑦, move it outside the summation:；Substituting back into obtained in previous steps, we get:；At he variational extremum of the objective of RLHF, we obtain a theoretical closed-form solution.；at its theoretical optima
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 10. Solve the optimization objective of RLHF

* **定义 / 内容：** Solve the optimization objective of RLHF。Note that the right-hand side depends on the known reference model and reward score, and the；difficult-to-compute partition function 𝑍(𝑥)；𝑍(𝑥) is a normalization constant that requires integrating all possible responses that the model is；capable of generating and it is intractible to compute.；It is precisely why reinforcement learning algorithms, such as PPO, is used to compute this；theoretical optimal policy.
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 11. Define RM by policy

* **定义 / 内容：** Define RM by policy。Since the optimal policy is determined by the reward model , we can express the reward；model in terms of the policy model:；Assume that for a language model policy , there is a reward function of language；evaluation, so that the policy generates samples to maximize the . Let a parameterized；RM be:；Problem: Still intractable；Definition, not derivation
* **直觉：** 本页展开 RLHF/PPO 复杂性、advantage、DPO 推导、GRPO、KTO、SimPO 与 DAPO。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. Bradley-Terry Model can help

* **定义 / 内容：** Bradley-Terry Model can help。We introduce the Bradley-Terry Model in RLHF lecture. For the same cue word x, the model；generates two responses and . Assume the latent reward function is . According to the；Bradley-Terry model, the probability that a human prefers to is:；With the Sigmoid function；Training of the reward model is to to minimize its negative log-likelihood:
* **直觉：** 本页展开 RLHF/PPO 复杂性、advantage、DPO 推导、GRPO、KTO、SimPO 与 DAPO。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. Put RM definition into Bradley-Terry Model Loss

* **定义 / 内容：** Put RM definition into Bradley-Terry Model Loss。But now we have a closed-form；of the reward model:；The term is perfectly canceled out:；We can directly optimizing the model:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 14. DPO Effect

* **定义 / 内容：** DPO Effect。Zephyr-7B was a landmark model launched by；HuggingFaceH4 at the end of 2023. Using only 7B；parameters, it surpassed Llama-2-Chat, which had 70B；parameters at the time, in dialogue capabilities (such as；MT-Bench scores) through highly skillful DPO training.；Image source: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
* **直觉：** 本页展开 RLHF/PPO 复杂性、advantage、DPO 推导、GRPO、KTO、SimPO 与 DAPO。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. GRPO Motivation

* **定义 / 内容：** GRPO Motivation。PPO: uses a critic (value model) for calculating the advantage. Taking GPU resources.；GRPO: Abandon Critic but just let the peers provide a reference；As large as the Policy Model, it is difficult to train.；For the same input question, the current Policy model generates G different answers in parallel. We obtain；G specific scores , and the average of these G scores is the natural baseline.；[4] Shao, et al., "Deepseekmath: Pushing the limits of mathematical reasoning in open language models." arXiv:2402.03300 (2024).
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 16. GRPO Loss

* **定义 / 内容：** GRPO Loss。The loss formula for GRPO is actually very similar to that of PPO, except that the source of the；advantage has changed.；[4] Shao, et al., "Deepseekmath: Pushing the limits of mathematical reasoning in open language models." arXiv:2402.03300 (2024).
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 17. Industry: PPO vs DPO

* **定义 / 内容：** Industry: PPO vs DPO。PPO (Proximal Policy DPO (Direct Preference；Feature / Dimension；Optimization) Optimization)；The Heavyweight's The Challenger's Open-；Industry Positioning；Computing Moat Source Hero；OpenAI (GPT-4 era), Hugging Face (Zephyr),；Key Representatives；Anthropic (Claude) Mistral, Llama 3；Global exploration of the Elegant, lightweight,；Core Advantage；"solution space" and plug-and-play；Extremely high (Best for Highly effective for；Performance Ceiling；complex logic & long text) general alignment；Massive compute & Accessible to open-；Resource Requirement；engineering bandwidth source developers；Image source: https://ai.gopubby.com/llm-alignment-with-sft-rlhf-dpo-and-grpo-026333dfaf1f
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. DPO Extension 1: Kahneman-Tversky Optimization (KTO)

* **定义 / 内容：** DPO Extension 1: Kahneman-Tversky Optimization (KTO)。DPO Problem: DPO relies heavily on strictly paired preference data. However, in real-world；scenarios, collecting perfectly aligned paired data is extremely costly, and we often only get a；large amount of unpaired data, such as "thumbs up" or "thumbs down".；Solution: Humans are not as sensitive to "receiving good answers" as they are to "avoiding bad；answers." Therefore, there is no need for pairwise comparisons; we can directly maximize the；implicit utility value of each individual answer based on whether it is a "praise" or a "dislike."；when 𝑦 is a good answer；when 𝑦 is a bad answer；[5] Ethayarajh, Kawin, et al. "Kto: Model alignment as prospect theoretic optimization." arXiv preprint arXiv:2402.01306 (2024).
* **直觉：** 本页展开 RLHF/PPO 复杂性、advantage、DPO 推导、GRPO、KTO、SimPO 与 DAPO。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. DPO Extension 2: Simple Preference Optimization (SimPO)

* **定义 / 内容：** DPO Extension 2: Simple Preference Optimization (SimPO)。DPO Problem: DPO discovered that by lengthening the generated sentences, even if the；advantage of each token is only slight, the cumulative effect can significantly widen the implicit；reward gap between good answers and bad answers.；Solution:；[6] Meng, Yu, Mengzhou Xia, and Danqi Chen. "Simpo: Simple preference optimization with a reference-free reward.”；Advances in Neural Information Processing Systems 37 (2024): 124198-124235.
* **直觉：** 本页展开 RLHF/PPO 复杂性、advantage、DPO 推导、GRPO、KTO、SimPO 与 DAPO。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. GRPO Extension 1: DAPO

* **定义 / 内容：** GRPO Extension 1: DAPO。GRPO Problem: Standard GRPO uses a symmetric clipping to limit update magnitude. This inhibit the；model from exploring high-reward young outputs, causing the model to repeat a single pattern.；Clipped Upper Bound；Scenario (Token Type) Initial Prob. (πold) Actual Max Prob. Allowed Fatal Impact；(πold×1.2)；Unrestricted: Probability；easily pushed to 0.999; the；A. Exploitation Token 0.90 0.90 x 1.2 = 1.08 nearly 1.0；model becomes；overconfident.；Strictly Bounded: Novel；tokens cannot emerge,；B. Exploration Token 0.01 0.01 x 1.2 = 0.012 0.012；killing diversity (Entropy；Collapse).；Solution: give higher upper bounds；to encourage low-probability tokens；with positive advantages.；[7] Yu, Qiying, et al. "Dapo: An open-source llm reinforcement learning system at scale." arXiv preprint arXiv:2503.14476 (2025).
* **直觉：** 本页展开 RLHF/PPO 复杂性、advantage、DPO 推导、GRPO、KTO、SimPO 与 DAPO。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (T/F): advantages is not used when training LLM with RLHF.；• Q2 (T/F): Due to the comparison between the winner and loser in the；optimization of reward model, we don’t need to calculate the partition；function 𝑍(𝑥).；• Q3 (T/F): In the GRPO algorithm, multiple samples from different queries；are used to calcualte the advantage.；• A: F; T; F
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 18: Synthetic Data：生成、评估、可靠性与局限

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 18；LLM’s training data；Running out of training data；Data synthesis: motivation；Data synthesis: motivation；Data synthesis: motivation；Reliability concerns；Alignment effectiveness
  * Data synthesis for evaluation；Type of synthetic data；Type of synthetic data；Data synthesis methods: prompting a teacher LLM；Data synthesis methods: retrieve and transform；Data synthesis methods: extract and re-write；Data synthesis methods: extract and re-write；Data synthesis methods: rephrasing
  * Data synthesis: extracting from knowledge graphs；Data synthesis: extracting from knowledge graphs；Data synthesis methods: AI rating；Data synthesis methods: prompting a teacher LLM；Data synthesis methods: self-instruct；Data synthesis methods: self-instruct；Data synthesis methods: self-guide；Data synthesis methods: self-guide
  * Data synthesis methods: Evol-instruct；Data synthesis methods: Evol-instruct；Data synthesis methods: mutli-agent methods；Data synthesis methods: mutli-agent methods；Synthetic data evaluation: correctness；Synthetic data evaluation: correctness；Synthetic data evaluation: complexity；Synthetic data evaluation: diversity
  * Synthetic data evaluation: diversity；Synthetic data evaluation: diversity；Synthetic data evaluation: fidelity；Synthetic data evaluation: fidelity；Limitation of synthetic data

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Synthetic Data Notes

### 1. 为什么需要 Synthetic Data

LLM 消耗的数据规模增长很快。课件提到 Llama 3、Qwen2.5 都是万亿 token 级别训练，而真实世界高质量公开文本增长更慢。简单重复已有数据在早期有用，但重复次数越多，收益越快下降。

synthetic data 的动机包括：

* 缓解高质量数据不足。
* 构造更安全、更可控的 alignment data。
* 降低人工标注成本。
* 增强 instruction following、math、code、preference data。
* 为 evaluation/red-teaming 覆盖自然数据中少见的危险场景。

### 2. Synthetic Data 的类型

课件列出几类：

* **Self-instruct data**：模型生成 instruction、input、output。
* **Code data**：编程题、解释、修复轨迹、测试结果。
* **Math data**：题目、推理步骤、答案。
* **Preference data**：同一 prompt 的多个 candidate response 和偏好关系。

这些数据分别服务 SFT、reward model、DPO/GRPO、evaluation 等阶段。

### 3. 主要合成方法

**Prompting a teacher LLM**：用强 teacher model 生成 labels 或 responses，训练 student model；本质上常和 distillation 相连。

**Retrieve and transform**：先检索真实文档或数据集，再改写为目标任务格式。优点是 grounded，缺点是依赖检索质量。

**Extract and rewrite**：从网页中抽取有用 QA 或内容，过滤低质量样本，再重写成统一 instruction-response 格式。

**Rephrasing**：重写表达但尽量保留知识，提升格式和多样性；风险是语气变得更像 teacher model。

**Knowledge graph extraction**：从结构化知识中生成自然语言样本，适合私有/领域数据；风险是 relation extraction 或 generation 出错。

**AI rating / hybrid feedback**：让系统判断哪些样本交给人、哪些交给 AI，以节省成本。

### 4. Self-Instruct、Self-Guide、Evol-Instruct、Multi-agent

Self-Instruct 从少量 seed tasks 扩展出大量新任务。优点是便宜、规模大；缺点是 seed 质量决定上限，生成任务可能表面多样但实际浅。

Self-Guide 增加结构化示例和过滤，让生成更可控。

Evol-Instruct 让任务逐步变难，例如 adding constraints、deepening、increasing reasoning、complicating input。它适合训练复杂指令、代码和数学能力，但任务可能演化得不现实或过难。

Multi-agent 方法让多个 agent 相互批评和修改，可以减少单一模型偏见，但也可能互相强化错误，计算成本高。课件中的“高中是否减少历史课”例子说明，多 agent 反馈可能把同一个价值立场越推越偏。

### 5. Synthetic Data Evaluation

**Correctness (正确性)**：回答是否遵循指令、代码是否能执行、自由文本是否可由 LLM-as-a-judge 或规则检查。

**Complexity (复杂度)**：是否覆盖难题，而不是只生成简单题。可看 CoT steps、约束数、领域知识深度。

**Diversity (多样性)**：self-BLEU 越高通常表示样本越相似；Vendi Score 用 similarity matrix 的谱/熵估计集合中“有效不同样本数”。

**Fidelity (忠实性)**：合成数据是否忠于源事实。课件例子中，知识库只说 `Penicillin treats Bacterial infections`，若生成回答解释 PBP 机制，就是引入了未给出的额外事实。

### 6. 三个核心风险

* **Mode collapse**：合成-训练循环会让细节越来越少，套话越来越多。
* **Lack of provenance**：数据被吸收到参数里后，很难追踪模型行为来源。
* **Data leakage**：复杂 rephrasing 可能绕过 n-gram/similarity 检查，让测试集信息进入训练。

### 7. Exam Focus

* synthetic data 不等于低质量数据；关键在生成、过滤、验证。
* LLM-as-a-judge 便宜但可能不可靠。
* fidelity 和 diversity 是两件不同的事：忠实不代表多样，多样不代表正确。
* red-teaming synthetic data 的目标是覆盖自然数据中罕见但重要的失败模式。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 18

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 18，主题为 Synthetic Data：生成、评估、可靠性与局限。
* **直觉：** 确认 Lecture 18 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 2. LLM’s training data

* **定义 / 内容：** LLM’s training data。• Llama 3 was trained on over 15,000B tokens, Qwen2.5 on 18,000B tokens, and；GPT-3 was trained on 300B tokens；• All three models use synthetic data to improve instruction following and alignment；Llama series GPT series Qwen series；Code and technical；Public web text Public web text </>；documents；Knowledge Licensed content Instruction；documents from partners tuning dialogue；Code and technical Human instructions High-quality；</>；documents and dialogue Chinese and；English text；A B Human-labeled；Image-text pairs A B；preference data Preference data
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 3. Running out of training data

* **定义 / 内容：** Running out of training data。• LLM are becoming larger and larger and consume more and more data, but；real-world data grow much slower.；Data are similar to oil: consumption grows faster than proven reserves；1000B；Villalobos et al., 2022, "Will we run out of data? An analysis of the limits of scaling datasets in Machine Learning", Epoch AI；Image source: https://epoch.ai/assets/images/posts/2024/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data/figure2-banner.png；https://imageio.forbes.com/blogs-images/judeclemente/files/2015/06/Screen-Shot-2015-06-23-at-3.48.52-PM.png?format=png&height=600&width=1200&fit=bounds
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 4. Data synthesis: motivation

* **定义 / 内容：** Data synthesis: motivation。• Repeating the same training data improves performance in the early stage, but；the return on compute declines rapidly as the number of repetitions increases.；Loss predicted by repeated data；Loss assumed repeated data is；worth the same as the new data；Muennighoff N, Rush A, Barak B, et al. Scaling data-constrained language models[J]. Advances in Neural Information Processing Systems, 2023, 36: 50358-50376.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 5. Data synthesis: motivation

* **定义 / 内容：** Data synthesis: motivation。• Reducing hallucination: construct data to fine-tune LLM so that it refuse to；answer when uncertainty is high.；Long-form factuality in large language models.；Simple synthetic data reduces sycophancy in large language models；FactKB: Generalizable factuality evaluation using language models enhanced with factual knowledge
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 6. Data synthesis: motivation

* **定义 / 内容：** Data synthesis: motivation。• Higher quality data: organic data can contain noises and error.；o Large language models (LLMs) strengthen instruction-following capability through；instruction-finetuning (IFT) on supervised instruction/response data. However, widely；used IFT datasets (e.g., ALPACA’s 52k data) surprisingly contain many lowquality instances；with incorrect or irrelevant responses – “AlpaGasus: raining a Better Alpaca with Fewer；Data”. ICLR 2024.；• Reduce training data and computation: organic data are too large and only a；small but high quality data is sufficient to train an LLM. E.g., AlpaGasus has；o 7B model: 80 minutes -> 14 min， $27.31 -> $4.78；o 13B model: 5.5 h -> 1 h， $225.28 -> $40.969；Yue X, Zheng T, Zhang G, et al. Mammoth2: Scaling instructions from the web[J]. Advances in Neural Information Processing Systems, 2024, 37: 90629-90660.；Muennighoff N, Rush A, Barak B, et al. Scaling data-constrained language models[J]. Advances in Neural Information Processing Systems, 2023, 36: 50358-50376.；Textbooks Are All You Need
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 7. Reliability concerns

* **定义 / 内容：** Reliability concerns。• Construct factual data to verify LLM output or fine-tune LLM so that it；refuses to answer when uncertainty is high.；Using verified sources, such as knowledge base and Wikipedia,；to construct dataset for fact-checking, retrieval, or fine-tuning.；Image source: https://www.reuters.com/legal/new-york-lawyers-sanctioned-using-；fake-chatgpt-cases-legal-brief-2023-06-22/；https://media.beehiiv.com/cdn-cgi/image/fit=scale- Long-form factuality in large language models.；down,format=auto,onerror=redirect,quality=80/uploads/asset/file/ac713c15-29a3- Simple synthetic data reduces sycophancy in large language models；49a0-9e23-acda3dc08599/Screenshot_2023-10-24_at_3.05.05_PM.png FactKB: Generalizable factuality evaluation using language models enhanced with factual knowledge
* **直觉：** 本页展开 LLM training data、数据耗尽、synthetic data generation、evaluation、fidelity 与 mode collapse。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Alignment effectiveness

* **定义 / 内容：** Alignment effectiveness。• Alignment can be expensive and also over-optimistic.；• RLHF uses 33k human preference data, taking several months to collect.；• Constitutional AI: RL from AI feedback (RLAIF):；o Use principles to self-critic and self-correct LLM’s own answers.；o User LLM’s preference data to train a reward model.；• Overly optimized reward model can lead to overfitting and reward hacking,；and synthetic data can increase diversity of reward model training data.；• Alignment tax: alignment can hurt other capabilities, e.g., training；effectiveness for safety, synthetic data can have both.；Scaling laws for reward model overoptimization；Constitutional AI: Harmlessness from ai feedback；Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization
* **直觉：** 本页展开 LLM training data、数据耗尽、synthetic data generation、evaluation、fidelity 与 mode collapse。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Data synthesis for evaluation

* **定义 / 内容：** Data synthesis for evaluation。• Red-teaming: test LLM using crafted malicious prompts under controlled.；o LLM should be tested thoroughly before open to public. However, organic data may not cover；all possible undesirable prompts (e.g., jailbreaking, anti-social personality, closed-mindedness,；etc.). Synthesize data to provide full coverage and test LLM more comprehensively.；Testing；multiple；political；personalities；sycophancy；and capabilities；of an LLM；Red Teaming Language Models with Language Models；Discovering Language Model Behaviors with Model-Written Evaluations SAGE-RT: Synthetic Alignment data Generation for Safety Evaluation and Red Teaming
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 10. Type of synthetic data

* **定义 / 内容：** Type of synthetic data。• Self instruct • Code
* **直觉：** 本页图示 synthetic data 的两类例子：self-instruct 数据通过模型生成 instruction、input、output 形成指令跟随样本；code 数据包含编程题、解释、输入输出或修复轨迹。它说明合成数据既可以训练通用指令能力，也可以强化代码能力。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. Type of synthetic data

* **定义 / 内容：** Type of synthetic data。• Math；• Preference data
* **直觉：** 本页图示 synthetic data 的 math 与 preference data：math 数据强调题目、逐步推理和答案；preference data 强调同一 prompt 下多个候选回答的比较或 ranking，用于 reward model、DPO/GRPO 等偏好优化。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 12. Data synthesis methods: prompting a teacher LLM

* **定义 / 内容：** Data synthesis methods: prompting a teacher LLM。• Why；o When human-annotated data is scarce, a；strong teacher model (e.g., GPT-4) can；guide a student model (e.g., Llama2-7B)；with more comprehensive knowledge；and diverse capabilities.；• How；o The model creates labeled sentence pairs；and uses them to train a smaller sentence；embedding model. This is also known as；model distillation.；Schick T, Schütze H. Generating datasets with pretrained language models[J]. arXiv preprint arXiv:2104.07540, 2021.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 13. Data synthesis methods: retrieve and transform

* **定义 / 内容：** Data synthesis methods: retrieve and transform。• Why；o Synthetic data may lack grounding in；real-world knowledge, which increases；the risk of unrealistic examples.；• How；o Retrieve relevant datasets or documents；o Transform them to align with target task；• Pros；o The produced data that is grounded in；real knowledge sources, which helps；reduce hallucination.；• Cons；o The final data quality depends heavily；on retrieval quality；Gandhi S, Gala R, Viswanathan V, et al. Better synthetic data by retrieving and transforming existing datasets[J]. arXiv preprint arXiv:2404.14361, 2024.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 14. Data synthesis methods: extract and re-write

* **定义 / 内容：** Data synthesis methods: extract and re-write。• Why；o The web contains informative question-answer. However, the useful contents are；embedded in irrelevant contexts and are unsuitable for instruction tuning.；• How；o Heuristic rules and small classifiers are used to extract web passages.；o LLM scores the extracted samples and keeps only high-quality examples.；o The selected content is rewritten into a consistent instruction-response format.；Yue X, Zheng T, Zhang G, et al. Mammoth2: Scaling instructions from the web[J]. Advances in Neural Information Processing Systems, 2024, 37: 90629-90660.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 15. Data synthesis methods: extract and re-write

* **定义 / 内容：** Data synthesis methods: extract and re-write。• Pros；o The rewriting step helps standardize formatting and may improve clarity or completeness；• Cons；o The extraction stage depends heavily on the quality of heuristic rules and classifiers；o The filtering stage can also be expensive when it relies on strong models such as GPT-4；Home > Math > Algebra；Limited-time offer: Get premium access for $1.99；Download our app for more exercises Instruction: Solve the equation 2x + 3 = 11. Good；Response: x = 4, because subtracting 3；Question: Solve 2x + 3 = 11 gives 2x = 8, and dividing by 2 gives x = 4 instruction；A. 2 B. 3 C. 4 D. 5；Answer: C；Raw web Explanation: 2x = 8, so x = 4.；page text Instruction: Solve 2x + 3 = 11 and；Recommended for you:；- 100 linear equation exercises；download our app for more exercises. Bad；Response: C. This problem is too easy.；- Best SAT prep courses Try 100 linear equation exercises instruction；User comments:；"This problem is too easy."；Privacy Policy | Contact Us | About；Yue X, Zheng T, Zhang G, et al. Mammoth2: Scaling instructions from the web[J]. Advances in Neural Information Processing Systems, 2024, 37: 90629-90660.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 16. Data synthesis methods: rephrasing

* **定义 / 内容：** Data synthesis methods: rephrasing。• Why；o Large web corpora contain repetitive and poorly；formatted text.；o Rephrasing can improve data quality and diversity；while preserving the original knowledge.；• How；o The authors use a language model to rewrite；documents into multiple styles.；• Pros；o This method changes expression rather than adding；new knowledge, so the hallucination risk is low.；• Cons；o Rewriting may introduce the preferences of the LLM,；which can make the data more uniform in tone.；Maini P, Seto S, Bai R, et al. Rephrasing the web: A recipe for compute and data-efficient language modeling
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 17. Data synthesis: extracting from knowledge graphs

* **定义 / 内容：** Data synthesis: extracting from knowledge graphs。• Why；o Domain-specific texts are often limited or even private – some are not on the web.；o Extract from structured knowledge bases and synthetic texts for fine-tuning.；• How；o Entity Extraction; Relation Graph Construction; Synthetic Text Generation；such as user-manuals Let LLM interacts with knowledge graphs；Yang Z, Band N, Li S, et al. Synthetic continued pretraining[J]. arXiv preprint arXiv:2409.07431, 2024.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 18. Data synthesis: extracting from knowledge graphs

* **定义 / 内容：** Data synthesis: extracting from knowledge graphs。• Pros；o This approach can expand a small source corpus into a much larger training set；• Cons；o The method depends on accurate entity and relation extraction, and these steps may；introduce errors；o The quality of the synthetic texts also depends on how well the language model；understands the knowledge graph；Sentence: In 2021, Company X acquired Startup Y, but Y remained；operationally independent；Even with a correct；Knowledge graph: knowledge graph, the；Company X → acquired → Startup Y synthetic corpus may still be；Startup Y → operationally independent from → Company X inaccurate if the model；Generated data:；misinterprets relations or；After the acquisition, Startup Y was fully merged into Company X constraints.；Company X absorbed all operations of Startup Y；Yang Z, Band N, Li S, et al. Synthetic continued pretraining[J]. arXiv preprint arXiv:2409.07431, 2024.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 19. Data synthesis methods: AI rating

* **定义 / 内容：** Data synthesis methods: AI rating。• Why；o Human feedback is useful, but it is expensive to collect. AI feedback is cheaper and；faster but can be biased or wrong. Which cases should go to humans or AI?；• How；o It learns to decide whether an example should be labeled by a human or by an LM；• Pros；o Reduces the cost of using only human annotations；• Cons；o It depends on the quality of the prediction model；Hybrid preferences: Learning to route instances for human vs. AI feedback
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 20. Data synthesis methods: prompting a teacher LLM

* **定义 / 内容：** Data synthesis methods: prompting a teacher LLM。• Pros；o No human-labeled data is required.；• Cons；o The dataset quality depends on the teacher model；o the generated labels may contain errors.；Ø The word bank has；Task: Write two sentences that are similar different meanings in the；two sentences.；Sentence1: The bank approved my loan application Ø The student model model；may learn from these；Sentence2: I sat by the river bank and watched the boats incorrect instruction-；response pairs, if；unchecked.；Schick T, Schütze H. Generating datasets with pretrained language models[J]. arXiv preprint arXiv:2104.07540, 2021.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 21. Data synthesis methods: self-instruct

* **定义 / 内容：** Data synthesis methods: self-instruct。• Why；o Prompting a teacher LLM is expensive and depends on commercial APIs.；o The student model expand training data using its generation ability to reduce cost.；• How；o begin with 175 manually written seed tasks (instructions), and use the LM to generate new；instructions and corresponding instances；From 175 seed examples；--> 100k new examples；Wang Y, Kordi Y, Mishra S, et al. Self-instruct: Aligning language models with self-generated instructions
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 22. Data synthesis methods: self-instruct

* **定义 / 内容：** Data synthesis methods: self-instruct。• Pros；o Scales instruction data without heavy human annotation；o Improves alignment through task-level diversity, not just more examples；• Cons；o The quality of the initial seed tasks strongly affects the final results.；o Generated instructions may be of low-quality or not practically meaningful.；Seed task: Translate this sentence in French As a result, the fine-tuned；model may perform well on；Generated tasks1: Translate this sentence into Spanish the surface but is not；Generated tasks2: Rewrite this sentence in formal English aligned with target；Generated tasks3: Translate this paragraph into German preference.；……；Wang Y, Kordi Y, Mishra S, et al. Self-instruct: Aligning language models with self-generated instructions
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 23. Data synthesis methods: self-guide

* **定义 / 内容：** Data synthesis methods: self-guide。• Why；o Self-Instruct generates instructions with uneven quality；o Self-Guide aims to add more structured guidance during data generation；• How；o The model first creates its own synthetic task examples.；o It filters these examples to keep better-quality data.；Zhao C, Jia X, Viswanathan V, et al. Self-guide: Better task-specific instruction following via self-synthetic finetuning[J]. arXiv preprint arXiv:2407.12874, 2024.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 24. Data synthesis methods: self-guide

* **定义 / 内容：** Data synthesis methods: self-guide。• Pros；o It makes the data generation process more controllable and transparent；o It can reduce low-quality outputs caused by random generation；• Cons；o It requires defining and maintaining task examples, which adds engineering cost；o The task examples may be incomplete or biased；Ø These examples do not；Suppose the demonstrations for a toxicity task only cover sarcasm, coded；include explicit insults like: harassment, or context-；dependent abuse.；You are stupid àtoxic Ø If the seed examples are；incomplete, the synthetic；Have a nice day ànon-toxic data will likely miss harder；cases as well.；Zhao C, Jia X, Viswanathan V, et al. Self-guide: Better task-specific instruction following via self-synthetic finetuning[J]. arXiv preprint arXiv:2407.12874, 2024.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 25. Data synthesis methods: Evol-instruct

* **定义 / 内容：** Data synthesis methods: Evol-instruct。• Why；o Models can learn complex tasks from；progressively harder tasks.；o Tasks evolve automatically from simple；ones to harder ones.；• How；o It starts with a set of seed instructions；and iteratively increases their complexity；through several evolution operations: In-；Breadth Evolving, Deepening, Add；Constraints, Concretizing, Increasing；Reasoning and Complicating Input；o After each step, a teacher LLM generates；responses to the evolved instructions.；Wizardlm: Empowering large language models to follow complex instructions；Wizardcoder: Empowering code large language models with evol-instruct；Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 26. Data synthesis methods: Evol-instruct

* **定义 / 内容：** Data synthesis methods: Evol-instruct。• Pros；o It reduces the need for manual creation of complex instructions. In fact, humans；may be less imaginative than LLM!；• Cons；o More complex instructions are not always more useful; some evolved instructions；may become unrealistic or meaningless.；o Repeated evolution may accumulate noise and lead to failure cases；It reflects complexity；Simple instruction: 1+1=? introduced by automatic；evolution, but can be too；Evolve hard for smaller models to；Complex instruction: How to prove 1 + 1 = 2 in the reason about Goldbach；Goldbach Conjecture（哥德巴赫猜想）? Conjecture (any even；number greater than 2 is a；sum of two primes—not；Wizardlm: Empowering large language models to follow complex instructions proven yet).；Wizardcoder: Empowering code large language models with evol-instruct；Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 27. Data synthesis methods: mutli-agent methods

* **定义 / 内容：** Data synthesis methods: mutli-agent methods。• Why；o When generating data, a single；model may repeat its own biases.；o Multi-agent methods can produce；more reliable training data than；one model alone.；• How；o Build a simulated society with；multiple AI agents. The agent；evaluates each other and revise；their responses.；o The filtered dialogues are then；used as training data.；Training socially aligned language models in simulated human society.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 28. Data synthesis methods: mutli-agent methods

* **定义 / 内容：** Data synthesis methods: mutli-agent methods。• Pros；o Multi-agent methods add external critique, helping reduce biases of a single model；• Cons；o The agents may reinforce each others errors The conversation can；be c ontroversial.；o They have much higher computational cost；Question: Should high schools reduce history classes and give more time to AI and programming courses?；Draft response: Yes. High schools should reduce history classes and shift more time to AI and programming courses,；because these subjects are more practical；Feedback 1: The answer is practical and forward-looking.；Feedback 2: The response is socially relevant because students need skills that match future economic demands.；Revised response1: High schools should significantly reduce history instruction and devote that time to AI and；programming, since technical skills are more relevant to modern careers.；Revised response2 : Schools should make AI and programming a clear priority and scale back history requirements；accordingly.；Training socially aligned language models in simulated human society.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 29. Synthetic data evaluation: correctness

* **定义 / 内容：** Synthetic data evaluation: correctness。Correctness: to make sure LLM responses generated can follow instructions.；Such instruction-response pairs are then used as synthetic data for fine-tuning.；• Code generating is relatively easy: execute the generated codes and check；the responses against the known ground truth.；• For free-text generation, one can use LLM-as-a-judge methods, but the；judging LLM may not be reliable. The other way is explicit checking.；Example prompt from Evol-Instruct: How to check the generated responses follow the instructions?
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 30. Synthetic data evaluation: correctness

* **定义 / 内容：** Synthetic data evaluation: correctness。Many explicit checking rules!；Instruction-Following Evaluation for Large Language Models. 2023
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 31. Synthetic data evaluation: complexity

* **定义 / 内容：** Synthetic data evaluation: complexity。• Complexity: synthetic data may be too easy and do not cover harder cases.；• It can be evaluated according to:；o Chain-of-thought steps; number of constraints; domain specific knowledge depth.；o For example, a coding question can be made more and more complex.；Example of SQL query with higher and higher complexity,；by adding more and more constraints.；WizardCoder-Empowering Code Large Language Models with Evol-Instruct. ICLR 2024.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 32. Synthetic data evaluation: diversity

* **定义 / 内容：** Synthetic data evaluation: diversity。• Diversity: self-BLEU evaluate average pair-wise data similarity.；• BLEU score: similarity between two sentences.；Average the；BLEU scores；to evaluate；diversity of a set；of texts.；https://cdn.prod.website-；https://stackoverflow.com/questions/44324681/variation-in-bleu-score files.com/67db22dd1e51de2e91403865/67db22dd1e51de2e914042c3_6719250ecdfb4；b34838cc07a_66b1ca340ca5dd11cb1be390_101-Card-sorting_08-image-01.png
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 33. Synthetic data evaluation: diversity

* **定义 / 内容：** Synthetic data evaluation: diversity。• Use similarity matrix to evaluate diversity of a group of data；o Visual inspection can tell the difference between low and high diversity.；similar background and shape different background and shape；VS (Vendi Score): the entropy of the diagonal of a；similarity matrix represents the diversity: it is；similar to the number of unique objects in the set.；Generalization: it does not have to be 0/1 similarity；(consider a soft version); diagonalize a similarity；matrix using SVD.；The Vendi Score: A Diversity Evaluation Metric for Machine Learning. TMLR 2023.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 34. Synthetic data evaluation: diversity

* **定义 / 内容：** Synthetic data evaluation: diversity。• Score first, diversity aware greedy selection；o Sort all generated data according to their scores (complexity, correctness, etc.)；o Greedly add generated data from top to bottom scores, discard samples that are too；close to the already selected ones.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 35. Synthetic data evaluation: fidelity

* **定义 / 内容：** Synthetic data evaluation: fidelity。• Fidelity: it is necessary to check that the synthetic data is faithful to world；facts or data before transformation (rephrasing).；Original:；“Only the director may authorize budget overrides.”；Rephrased:；“The director may only authorize budget overrides.”；Use RoBERTa as an embedding model to calculate a similarity score；and filter the transformed data if the similarity is too low.；RoBERTa-MNLI(“Only the director may authorize budget overrides.”,；“The director may only authorize budget overrides.” ) < 0.75
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 36. Synthetic data evaluation: fidelity

* **定义 / 内容：** Synthetic data evaluation: fidelity。You are a knowledgeable assistant that generates diverse, natural-；sounding question-answer pairs from structured knowledge graph facts.；Unfaithful synthetic data:；• Knowledge base 🎯 Task:；Given the knowledge graph triple below, generate 5 distinct question-；response pairs that a user might ask about this fact. Vary:；{；- Question types: factual, explanatory, comparative, yes/no, scenario- "question": "How does；{ based；penicillin work to kill bacteria?",；- Phrasing: formal, casual, clinical, patient-friendly；"subject": "Penicillin", - Perspective: patient, doctor, researcher, student "answer": "Penicillin inhibits；📥 Input Triple:；"predicate": "treats", - Subject: Penicillin bacterial cell wall synthesis by；"object": "Bacterial infections", - Predicate: treats；- Object: Bacterial infections；binding to penicillin-binding；"context": { - Context: proteins (PBPs), leading to cell；• Penicillin is an antibiotic drug；"subject_type": "Antibiotic drug", • Bacterial infections are a medical condition lysis.",；"object_type": "Medical；• Source: DrugBank (conﬁdence: 0.98) "question_type": "explanatory",；📤 Output Format (strict JSON array):；[ "audience": "researcher”；condition", { }；"source": "DrugBank", "question": "string",；"answer": "string",；"confidence": 0.98 "question_type": "factual|explanatory|comparative|yes_no|scenario",；"audience": "patient|clinician|researcher|student|general"；} }, Why the generated question-answer；...；} ] pair is not faithful?；⚠ Guidelines:；- Answers must be accurate, concise, and grounded ONLY in the provided；triple + context. PBP is not mentioned in the；- Do NOT hallucinate additional drugs, conditions, or mechanisms.；- If a question requires info not in the triple, rephrase or skip it. knowledge base and can be checked；- Prefer answers that could stand alone in a FAQ or chatbot response.；Generate exactly 5 pairs. by matching.
* **直觉：** 本页关注数据来源、构造、质量或评估；在 LLM 训练中数据常常与模型结构同样关键。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 37. Limitation of synthetic data

* **定义 / 内容：** Limitation of synthetic data。• Mode collapse: LLM is an approximation of training data, and details may be；ignored. As the synthesization-training cycle continues, more and more；details can be ignore, while the details are sometimes more useful.；o Boilerplate responses (“sure I am glad to help”) are not as useful as deep details.；• Lack of data provenance: as synthetic data are absorbed into LLM’s；parameters, it becomes hard to explain the (undesirable) behaviors of an；LLM. As the synthesization-training cycle continues, tracing is almost；impossible.；• Risk of data leakage: while explicit n-gram/similarity check can remove some；leakage of test data into training data, but as data synthesis techniques；become more complicated (high-order rephrasing), the leakage can be hard；to detect.；Miranda L J V, Wang Y, Elazar Y, et al. Hybrid preferences: Learning to route instances for human vs. AI feedback[C]//Proceedings of the 63rd Annual Meeting of the；Association for Computational Linguistics (Volume 1: Long Papers). 2025: 7162-7200.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

# Lecture 19: Scaling Laws、FLOPs、Kaplan 与 Chinchilla

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 19；Motivation of Scaling laws；Model Size vs Accuracy；Motivation of Scaling laws；What is Scaling；Constraints of Real world；FLOPs；FLOPs: Matrix-vector Multiplication
  * FLOPs: Matrix Multiplication；FLOPs: Matrix Multiplication；FLOPs: Matrix Multiplication；Transformer FLOPs；Calculating Transfomer’s N；Calculating Transfomer’s N；Calculating Transfomer’s C；Calculating Transfomer’s C
  * Calculating Transfomer’s C；Kaplan’s scaling laws；Kaplan’s scaling laws；Kaplan’s scaling laws；Chinchilla’s scaling law；Chinchilla’s scaling law；Chinchilla’s scaling law；Kaplan or Chinchilla?
  * Variants of scaling laws；Variants of scaling laws；Variants of scaling laws；Limitations of Scaling Laws；Why inverse scaling；Limitations of Scaling Laws；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Scaling Law Notes

### 1. Scaling 的三个变量：N、D、C

Scaling 不是单纯把模型变大，而是在三个资源之间平衡：

* $N$：model parameters。
* $D$：training tokens。
* $C$：training compute / FLOPs。

课件核心关系：

$$
C\approx 6ND
$$

如果 compute budget $C$ 固定，增大 $N$ 通常意味着减小 $D$，反之亦然。真正的问题是：固定预算下，钱该花在更大模型，还是更多数据？

### 2. FLOPs 基础

矩阵-向量乘法 $A\in\mathbb{R}^{m\times n}$ 与 $x\in\mathbb{R}^n$ 需要约：

$$
2mn
$$

其中乘法和加法各算一个 FLOP。

矩阵乘法 $A\in\mathbb{R}^{m\times n}$ 与 $B\in\mathbb{R}^{n\times p}$ 需要约：

$$
2mnp
$$

训练中 backward pass 通常约为 forward pass 的两倍，所以一次完整训练 step 可粗略看作 forward + backward：

$$
2ND+4ND=6ND
$$

这只是估算，实际还会受到 softmax、LayerNorm、communication、memory bandwidth、checkpointing、data loading、OOM recovery 等影响。

### 3. Transformer 参数量估计

课件给出 Transformer 参数估计形式：

$$
N\approx 12\times n_{layers}\times d_{model}^2
+(n_{vocab}+n_{pos})\times d_{model}
$$

前半部分是非 embedding 参数，后半部分是 token/position embeddings。实际模型还会因 FFN ratio、bias、normalization、tie embeddings 等细节略有差异。

### 4. Kaplan Scaling Law

Kaplan 发现 training/test loss 随 model size、data、compute 呈 predictable power-law。直觉是：更大模型 data efficiency 更高，能从同样数据中提取更多 pattern。

power-law 的意义是：在多个数量级上，loss 可以用较平滑的曲线预测。这让研究者可用小规模实验外推大规模训练趋势。

### 5. Chinchilla：Compute-optimal 不是越大越好

Chinchilla 的关键结论是：很多大模型参数太多、数据太少，即 data-undertrained。固定 compute 下，较小模型 + 更多 tokens 可能优于更大模型 + 较少 tokens。

课件例子：Gopher 280B 参数但 token 不足；Chinchilla 70B 参数配更多数据，在相似 compute 下表现更好。

IsoFLOP curves 的思想：固定不同 FLOPs $C$，扫描 model size，找每个 compute budget 下的 optimal $N$ 和 $D$。

### 6. Scaling Law 的变体和限制

变体：

* optimal batch size 随 compute 增长。
* optimal learning rate 随 compute 变化。
* 数据质量 scaling：FineWeb-Edu 用更少高质量 tokens 达到类似效果。
* multimodal scaling law：不只文本模型有 scaling 规律。

限制：

* pretraining cross-entropy 不总能预测 downstream task。
* 可能出现 inverse scaling：模型越大越会重复训练中的坏模式、memorized sequences 或 easy distractor。
* 数据过滤策略不是 compute-agnostic；小 compute 更适合干净数据，大 compute 可能能利用更多 noisy data。

### 7. Exam Focus

* 必背关系：$C\approx 6ND$。
* FLOPs 估算中 backward 约为 forward 两倍。
* Kaplan 更强调大模型和 power-law；Chinchilla 强调 fixed compute 下模型大小和 token 数的平衡。
* inverse scaling 可由数据中 undesirable patterns、memorization 或 misleading few-shot demonstrations 引起。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 19

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 19，主题为 Scaling Laws、FLOPs、Kaplan 与 Chinchilla。
* **直觉：** 确认 Lecture 19 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Motivation of Scaling laws

* **定义 / 内容：** Motivation of Scaling laws。• Model sizes have increased dramatically over time, moving from millions；of parameters to hundreds of billions and even trillions.；Image source: https://ourworldindata.org/grapher/exponential-growth-of-parameters-in-notable-ai-systems
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 3. Model Size vs Accuracy

* **定义 / 内容：** Model Size vs Accuracy。• Larger language models have better zero/few-shot performance.；• Each 10× increase requires much more compute and data.；Is the next 10× increase in scale(?) worth it, and how much performance；improvement will it deliver?；Cottier B, Rahman R, Fattorini L, et al. The rising costs of training frontier AI models[J]. arXiv preprint arXiv:2405.21015, 2024.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Motivation of Scaling laws

* **定义 / 内容：** Motivation of Scaling laws。• A million-dollar question: How should I invest a fixed amount of resources on；data or GPU?；o GPT-4 pretraining is estimated to have cost over $100 million.；o Decision-makers need to predict what model size, data scale, and compute budget are required to；reach a target level of performance.；Cottier B, Rahman R, Fattorini L, et al. The rising costs of training frontier AI models[J]. arXiv preprint arXiv:2405.21015, 2024.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 5. What is Scaling

* **定义 / 内容：** What is Scaling。• “Scaling”means larger model size (# of parameter 𝑁), more compute；(FLOS 𝐶) and more data (# of tokens 𝐷).；• Increasing only one dimension is often inefficient. Good performance comes；from balancing all three.；o Case 1 (overfitting): A 10× larger model with the same amount of data can；memorize the training set more easily, but it often gains little in generalization.；o Case 2 (underfitting): With 10× more data but the same small model, there is；more information to learn from, but the model may not have enough capacity to；absorb it effectively.；o Case 3 (right scaling): When model size, data, and compute all grow together,；the model can learn more patterns and achieve more consistent improvement.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Constraints of Real world

* **定义 / 内容：** Constraints of Real world。• Example of constraints: the total computing budget；o FLOPs provides a unified way to measure that budget.；Under the same compute budget, can a smaller model outperform Gopher with 280B parameters?；Use scaling law to make the right decision: Chinchilla (70B) with more data (1.4T tokens) and；achieved better performance at the same compute.；Hoffmann J, Borgeaud S, Mensch A, et al. Training compute-optimal large language models[J]. arXiv preprint arXiv:2203.15556, 2022, 10.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. FLOPs

* **定义 / 内容：** FLOPs。• Floating point operations per Example of matrix-vector；second (FLOPS, flops or flop/s). products in Transformer:；• Attention query: W! ℎ；• Each FLOP can represent an • Projection back: W" ℎ；addition, subtraction, • Feedforward (ff): W## ℎ；multiplication, or division of Ignore FLOPs for；floating-point numbers. • Bias vector addition；• Layer normalization；• The total FLOP of training a model；• Residual connections；(e.g., GPT-4) provides a basic • Non-linearities；Significant cost of training a；approximation of computational Transformer comes from matrix-；• Softmax；costs of training. vector products and we focus；on FLOPs of such products
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. FLOPs: Matrix-vector Multiplication

* **定义 / 内容：** FLOPs: Matrix-vector Multiplication。• Requires 2𝑚𝑛 (2× matrix size) operations to multiply 𝐴 ∈ ℝ !×# and B ∈ ℝ #；• The factor 2: 1 for multiplication, 1 for addition.；• For multiplying 𝐴 ∈ ℝ !×# and B ∈ ℝ #×$ , one needs 2𝑚𝑛𝑝 operations；• This is just for the forward propagation.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. FLOPs: Matrix Multiplication

* **定义 / 内容：** FLOPs: Matrix Multiplication。• Backward passes calculate the derivatives of loss with respect to both；hidden state (for further backprop) and parameters (for gradient)；Due to the chain-rule；and inner product We compute；$%；to propagate；between 𝑊 and 𝑋 , back $&；propagations are also the gradient to earlier layers；matrix-vector products:；$% $%；= 𝑊 (× $%；$& $) We compute to update；$'；$% $% model parameters；= 𝑋 (×；$' $)；• FLOPs for backward pass is roughly twice of forward pass.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. FLOPs: Matrix Multiplication

* **定义 / 内容：** FLOPs: Matrix Multiplication。• FLOPs for backward pass is roughly twice of forward pass.；This ratio depends on various parameters (layers, width-depth ratio, batch size)；Hobbhahn M, Sevilla J. What’s the backward-forward flop ratio for neural networks?[J]. Published online at epochai. org, 2021.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. FLOPs: Matrix Multiplication

* **定义 / 内容：** FLOPs: Matrix Multiplication。• If 𝑊 ∈ ℝ'!" ×'#$% , then the number of parameters for muplyling W with a；vector ℎ is 𝑊 = 𝑑() ×𝑑*+, .；FLOPs of a single layer with a single matrix-vector product is；6 x (# tokens) x (# of parameters)；• Example: If 𝑊 ∈ ℝ-./0×11..2 , with 8 tokens, the training FLOPs are；6×B× W = 6×8×4096×11008 ≈ 2.16×10-
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. Transformer FLOPs

* **定义 / 内容：** Transformer FLOPs。• Let N be number of parameters (the sum of sizes of all matrices)；• Let D be the number of tokens in pre-training dataset；• Forward pass:；o FLOPs for forward pass on a single token is roughly 2N；o FLOPs for forward pass for the entire dataset is roughly 2ND；• Backward pass:；o FLOPs for backward pass is roughly twice of forward pass；o FLOPs for backward pass for the entire dataset is roughly 4ND；• The total cost of pre-training on this dataset is: 𝐶 ≈ 6𝑁𝐷；If you have a fixed compute budget C, increasing D means decreasing N；For a detailed calculation of 6ND, see:；https://www.lesswrong.com/posts/fnjKpBoWJXcSDwhZk/what-s-the-backward-forward-flop-ratio-for-neural-networks
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. Calculating Transfomer’s N

* **定义 / 内容：** Calculating Transfomer’s N。For a detailed calculation, see；https://medium.com/data-science/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Calculating Transfomer’s N

* **定义 / 内容：** Calculating Transfomer’s N。:；𝑁 = 12×𝑛345678 ×𝑑9*'63 + (𝑛;<=>? + 𝑛@*8 )×𝑑9*'63；Non-embedding Embedding；For example:；Vocab size = 65536；Positional emb size = 1000；Askell A, Bai Y, Chen A, et al. A general language assistant as a laboratory for alignment
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Calculating Transfomer’s C

* **定义 / 内容：** Calculating Transfomer’s C。Given the pre-training data with；400B tokens；Training cost (FLOPs):；Askell A, Bai Y, Chen A, et al. A general language assistant as a laboratory for alignment
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. Calculating Transfomer’s C

* **定义 / 内容：** Calculating Transfomer’s C。• Consider HyperCLOVA, an 82B parameter model that was pre-trained on 150B tokens,；using a cluster of 1024 A100 GPUs.；• Training cost (FLOPs):；• The peak throughput of A100 GPUs if 312 teraFLOPS or 3.12×10%&；• How long would this take?；• In other words, 𝐶 = 2.7 𝑇𝐹 = 2.7×10'( 𝑃𝐹；• TF=TeraFLOPs, and PF=PeraFLOPs
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. Calculating Transfomer’s C

* **定义 / 内容：** Calculating Transfomer’s C。• According to the paper, training took 13.4 days. The estimate is 5 times off, why?；• These estimates can be slightly off in practice:；o ignore many operations like softmax, Relu/Gelu activations, layer Norm etc.；o In distributed training, cross-GPU All-Reduce communication can consume 15–；30% of training time.；o Memory bandwidth constraints often limit GPU utilization to 30–60% of the；theoretical peak.；o Cached activations can be swapped in and out between GPU caches and CPU memory；o More on KV-caching next lecture.；o Additional overheads include checkpointing, data loading, and recovery from；node failures or OOM (out-of-memory) exceptions.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Kaplan’s scaling laws

* **定义 / 内容：** Kaplan’s scaling laws。• Blue curves: Each blue curve shows the loss；of a fixed-size model over a fixed number of；training steps.；• Black points: The black points mark the；lowest achievable loss across models,；forming the compute-efficient frontier.；• Orange dashed line: a power-law fit to the；frontier, which is the scaling law.；• Scaling laws show that model performance；Note: power-law is used to describe；improves in a predictable way with scale quantities varying across magnitudes.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Kaplan’s scaling laws

* **定义 / 内容：** Kaplan’s scaling laws。Overfitting；• The x-axis shows the number of training tokens；• The y-axis shows the loss；• Each curve corresponds to a model of a；different size；• Training loss/test loss curves follow predictable；power laws Nice power law；• Larger models have steeper curves, meaning；that with the same amount of data, they；achieve greater loss reduction => # of data Slight；and parameters should grow simultaneously! overfitting；Kaplan J, McCandlish S, Henighan T, et al. Scaling laws for neural language models[J]. arXiv preprint arXiv:2001.08361, 2020.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. Kaplan’s scaling laws

* **定义 / 内容：** Kaplan’s scaling laws。• The x-axis shows the number of tokens processed during training；• The y-axis shows the test loss；• Each curve represents a model with a different number of；parameters；• Model size improves data efficiency: bigger models extract useful；patterns from data more effectively, so they need fewer examples；to learn the same thing.；• The Kaplan scaling law: Bigger models often achieve better；final performance.；Kaplan J, McCandlish S, Henighan T, et al. Scaling laws for neural language models[J]. arXiv preprint arXiv:2001.08361, 2020.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. Chinchilla’s scaling law

* **定义 / 内容：** Chinchilla’s scaling law。• Training runs for models from 70M to 10B parameters under different compute budget.；o Different from Kaplan’s experiments: larger model size, optimal learning rate schedules.；• Middle: Estimated optimal model size for each compute budget.；• Right: Estimated optimal number of training tokens for each compute budget.；• Example: With a compute budget of 6 × 10²³ FLOPs (about 1 million A100 GPU hours)；o Case 1: training a model of roughly 174B parameters with relatively limited data；o Case 2: a smaller model of about 67B parameters trained on around 1.3T tokens；o In experiments, case 2 performs better, challenging the earlier Kaplan’s scaling law “bigger is；always better”.；Hoffmann J, Borgeaud S, Mensch A, et al. Training compute-optimal large language models[J]. arXiv preprint arXiv:2203.15556, 2022.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. Chinchilla’s scaling law

* **定义 / 内容：** Chinchilla’s scaling law。The IsoFLOP curves；• Training runs given different FLOPs C. For each C, vary model size to find the optimal size.；• Middle: Estimated optimal model size for each compute budget.；• Right: Estimated optimal number of training tokens for each compute budget.；Hoffmann J, Borgeaud S, Mensch A, et al. Training compute-optimal large language models[J]. arXiv preprint arXiv:2203.15556, 2022.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. Chinchilla’s scaling law

* **定义 / 内容：** Chinchilla’s scaling law。Kaplan under-estimated；the utility of model；parameters due to:；1) smaller model sizes;；2) non–optimal；learning rates.；Gopher is a larger model with 280B parameters, but was；trained on insufficient number of tokens.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. Kaplan or Chinchilla?

* **定义 / 内容：** Kaplan or Chinchilla?。本页主要用于课程衔接、图示或标题说明。
* **直觉：** 本页作为对比页，提示复习 Kaplan 与 Chinchilla 的核心差异：Kaplan 更强调扩大模型规模带来的 loss 改善，而 Chinchilla 强调在固定 compute 下模型参数量与训练 token 数需要共同匹配，很多大模型其实 data-undertrained。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 25. Variants of scaling laws

* **定义 / 内容：** Variants of scaling laws。• The x-axis is training FLOPs, and the y-axis is the optimal batch size (left) or learning rate (right)；• As compute increases, the optimal batch size grows, while the optimal learning rate decreases；• Fitting these scaling curves on small-scale experiments, one can predict near-optimal；hyperparameters for large-scale training and avoid expensive hyperparameter search；Bi X, Chen D, Chen G, et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 26. Variants of scaling laws

* **定义 / 内容：** Variants of scaling laws。• The FineWeb-Edu reaches 33.6% accuracy at just；38B high-quality training tokens, while the Matrix；dataset requires about 300B tokens from a less；curated web corpus to reach a similar result.；• Traditional scaling laws treat token count as the；main data variable and assume similar token；quality. FineWeb shows that high-quality tokens；matters more than raw token count.；Penedo G, Kydlíček H, Lozhkov A, et al. The fineweb datasets: Decanting the web for the finest text data at scale[J]. Advances in Neural；Information Processing Systems, 2024, 37: 30811-30849.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 27. Variants of scaling laws

* **定义 / 内容：** Variants of scaling laws。• This figure shows that scaling laws also hold for multimodal language models,；not only for text-only LLMs.；Shukor M, Fini E, da Costa V G T, et al. Scaling laws for native multimodal models[C]//Proceedings of the IEEE/CVF International Conference on；Computer Vision. 2025: 12-23.
* **直觉：** 本页展开 scaling laws、FLOPs、Transformer 参数/计算量、Kaplan、Chinchilla、inverse scaling 与限制。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 28. Limitations of Scaling Laws

* **定义 / 内容：** Limitations of Scaling Laws。• Inverse scaling:；o The x-axis is cross-entropy loss, which is often；used as a proxy for pretraining quality, and the y-；Inverse axis is downstream task performance (varied；trend according to task types).；o Pretraining scaling laws do not automatically；transfer to downstream tasks that cannot be；measured by cross-entropy loss of next token；prediction.；o A smooth power law in loss does not guarantee；a smooth curve in other end-task metrics；Lourie N, Hu M Y, Cho K. Scaling laws are unreliable for downstream tasks: A reality check[J]. arXiv preprint arXiv:2507.00885, 2025.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 29. Why inverse scaling

* **定义 / 内容：** Why inverse scaling。• preference to repeat memorized sequences over following in-context instructions；• imitation of undesirable patterns in the training data；• tasks containing an easy distractor task which LMs could focus on, rather than the；harder real task；• correct but misleading few-shot demonstrations of the task；A demonstration；of memorization；when LLMs use；more training；computation.；Inverse Scaling: When Bigger Isn’t Better. TMLR 2023
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 30. Limitations of Scaling Laws

* **定义 / 内容：** Limitations of Scaling Laws。Cannot use；data with quality；lower than a；threshold.；• The optimal data filtering strategy depends on the compute budget:；o small compute: cleaner and smaller datasets work best;；o with large compute, including noisier data can be beneficial.；• Current scaling laws often assume that all web tokens are equally informative, ignoring；heterogeneity in data quality.；Goyal S, Maini P, Lipton Z C, et al. Scaling Laws for Data Filtering--Data Curation cannot be Compute Agnostic. Proceedings of the IEEE/CVF；Conference on Computer Vision and Pattern Recognition. 2024: 22702-22711.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 31. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (short answer): Write down the relationship between N, C, D mentioned；in this lecture.；• Q2 (T/F): Each matrix-vector calculation 𝐴𝒙 requires computation；complexity linear in the number of columns of the matrix 𝐴；• Q3 (multi-choice): which of the following is a property of scaling laws: (A)；Kaplan’s scaling law says pre-training cross-entropy loss decreases faster if；increase the model size than the number of tokens; (B) inverse scaling can；happen due to data containing undesirable patterns; (C) the IsoFLOPs；curves in Chinchilla scaling law is based on the optimal model sizes.；• Answers: C=6ND; N/A (due to ambiguity of the question); (AB)
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 20: LLM Inference、KV Cache、Memory Wall、PagedAttention、StreamingLLM

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 20；Motivation of KV Cache；Incremental Attention with Cache；What is Key-Value (KV) Cache?；The Two Stages of Token Generation；Phase 1: Prefill (processing the input)；Phase 2: Decode (Generation)；The "Memory Wall" in LLMs FLOPs
  * Linear Growth, Massive Impact；How big is the KV Cache A batch of 3 sequences. LLM generates a token；FlashAttention；Optimizing Attention Heads；Optimizing Attention Heads；The Memory Waste: Why Fragmentation Happens；PagedAttention (One request:)；PagedAttention (Multi requests:)
  * Bubbles and Preemption in Continuous Batching；Chunked prefills with decode-maximal batching；Memory Wall in Long Context；StreamingLLM: Attention Sink is All You Need；StreamingLLM: Attention Sink is All You Need；Attention Map；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed LLM Inference Systems Notes

### 1. Autoregressive Decoding 为什么慢

LLM 生成是逐 token 的：

$$
w_t\sim P_\theta(w_t|w_{<t})
$$

naive decoding 每生成一个新 token 都重新计算全部历史 token 的 Key/Value，造成大量重复。生成 $n$ 个 token 时，重复历史计算会近似形成 quadratic growth。

KV cache 用 memory 换 time：历史 token 的 $K,V$ 只算一次，后面直接读取。

### 2. KV Cache

对最新 token 的 query $q_t$，只需和缓存中的历史 keys 做 attention：

$$
Attention(q_t,K_{\le t},V_{\le t})
=softmax\left(\frac{q_tK_{\le t}^T}{\sqrt{d_k}}\right)V_{\le t}
$$

KV cache size 公式：

$$
Size_{KV}=Batch\times SeqLen\times 2\times Layers\times Heads\times Dim\times Bytes
$$

其中 $2$ 表示 K 和 V 两套矩阵；Bytes 对 FP16 通常是 2。注意模型 weights 固定，但 KV cache 随 sequence length 和 batch size 线性增长，长上下文 OOM 往往是 cache 造成的。

### 3. Prefill vs Decode

**Prefill**：处理完整 prompt，建立 initial cache。prompt tokens 可以并行，GPU 利用率高，偏 compute-bound。它决定 TTFT：

$$
TTFT=Time\ To\ First\ Token
$$

**Decode**：每次只生成一个 token。每步都要读取整个 KV cache，GPU core 常等 memory transfer，偏 memory-bandwidth bound。它决定 TPOT：

$$
TPOT=Time\ Per\ Output\ Token
$$

用户觉得慢，往往是 TPOT 高，而不是 prefill 慢。

### 4. Memory Wall

硬件 compute FLOPS 增长很快，但 memory bandwidth 跟不上。decode 时大量时间花在从 HBM 读取 KV cache，而不是做矩阵乘法。这就是 **memory wall**。

序列越长，KV cache 越大：

$$
KV\ memory\propto L
$$

因此 long-context serving 的瓶颈不只是算力，更是显存容量和带宽。

### 5. FlashAttention：online exact softmax

FlashAttention 的思想是 tiling：不把完整 attention matrix 写入显存，而是 block-by-block 计算 softmax 的分母和 weighted sum。

online softmax 维护 running max 和 denominator，因此不是近似 softmax，而是 exact online computation。它减少 HBM read/write，提升 attention 训练/推理效率。

### 6. MHA、MQA、GQA

**MHA (Multi-Head Attention)**：每个 query head 有自己的 K/V head，表达力强，但 KV cache 大。

**MQA (Multi-Query Attention)**：多个 query heads 共享一组 K/V，KV cache 压缩最强，但可能损失表达能力。

**GQA (Grouped-Query Attention)**：query heads 分组，每组共享 K/V，是 MHA 与 MQA 的折中，常用于降低 memory bandwidth，同时保留较好质量。

### 7. PagedAttention 和 Continuous Batching

serving 中每个 request 长度不同。如果为每个 request 按 max length 预留 cache，会有 internal fragmentation；物理显存不连续还会有 external fragmentation。

PagedAttention 把 KV cache 像 OS virtual memory 一样分 block：

* logical block：序列逻辑位置。
* physical block：GPU 中实际存储块。
* block table：负责映射。

这样不同 request 的 KV blocks 可以灵活放置，提高显存利用率。

continuous batching 中，decode 任务可能被别人的 prefill 阻塞，出现 bubbles。chunked prefill 把长 prefill 切成块，让 decode 可以 piggyback，减少 GPU 空转。

### 8. StreamingLLM and Attention Sink

长上下文超过显存后，简单 sliding window 会丢掉开头 token，模型 perplexity 可能突然崩。StreamingLLM 发现 **attention sink**：初始 tokens 经常吸收 attention，即使语义不强，也对 softmax 稳定很重要。

解决方法：保留少量初始 sink tokens，加上最近 window tokens：

$$
KV=\text{sink tokens}+\text{recent tokens}
$$

这样 cache size 固定，同时维持较稳定 attention。

### 9. Exam Focus

* KV cache 影响因素包括 layers、heads、head dim、sequence length、batch size、K/V 两套矩阵、precision。
* layer normalization parameters 不是 KV cache size 的直接因素。
* FlashAttention online softmax 是 exact，不是必须近似。
* long-context OOM 的 culprit 常是 KV cache，而不是固定 model weights。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 20

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 20，主题为 LLM Inference、KV Cache、Memory Wall、PagedAttention、StreamingLLM。
* **直觉：** 确认 Lecture 20 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Motivation of KV Cache

* **定义 / 内容：** Motivation of KV Cache。• In a naïve implementation, autoregressive generation would waste；massive compute power recalculating the entire historical context；(K and V matrices) just to predict one single new token.；• Input: "Tell me something about；llamas. Llamas are"；• Step 1: Predict "domesticated"；• Context of Step 2: "Tell me；something about llamas. Llamas are；domesticated"；Quadratic Growth；To generate the i-th token, you recompute the Key and Value vectors for tokens 1 through i-1.；These vectors have not changed since the last step! The time complexity is 𝑂(𝑛 ! ) when；generating 𝑛 tokens (ignoring the factor of the cost of computing the attention matrix).；Image source: https://blog.gopenai.com/anatomy-of-llms-transformer-overview-encoder-decoder-autoregressive-flow-6a59bc839710
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 3. Incremental Attention with Cache

* **定义 / 内容：** Incremental Attention with Cache。• Recall the core attention formula for a sequence length L:；• During naive decoding, are all recomputed every time；when L increases by 1.；• With KV cache, at step t, we only compute vectors for the latest generated；token and update the cache:；• The attention score for the query 𝑞% (from the latest generated token):
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. What is Key-Value (KV) Cache?

* **定义 / 内容：** What is Key-Value (KV) Cache?。• Trading Memory for Time；• Instead of recomputing the；Key (K) and Value (V)；matrices from previous；tokens from scratches, we；store them in GPU memory.；• Compute once:；• Cache:；• Used for generating:；Latest；• Result: 𝑂(𝑛) computation generated；instead of 𝑂(𝑛 ! ). token；A KV Cache saves results of intermediate steps while solving a problem. The next time you need to use those；same intermediate values; you can simply look them up from the cache.；Image source: https://ankitbko.github.io/blog/2025/08/prompt-engineering-kv-cache/
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. The Two Stages of Token Generation

* **定义 / 内容：** The Two Stages of Token Generation。• Most LLMs generate text token by token.；• The first output token depends on all prompt tokens, but the second output；token already depends on all prompt tokens plus the first output token, and；so on.；Image source: https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Phase 1: Prefill (processing the input)

* **定义 / 内容：** Phase 1: Prefill (processing the input)。for one head at one layer；• Encode the full prompt and build the initial cache.；• In the prefill phase, the calculations for；all input tokens can be executed in；parallel (across all prompt tokens and；all heads, but not over all layers).；• High GPU utilization.；• Compute-bound phase.；• Determines TTFT (Time to First Token).；What is the biggest animal?；Parameters for；E1 E2 E3 E4 E5 WK K KEY；generating keys.
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Phase 2: Decode (Generation)

* **定义 / 内容：** Phase 2: Decode (Generation)。• Computing later output token；• Low GPU compute utilization.；• Each decode step processes only 1 token.；• Memory-bandwidth bound.；• Every step must load the entire KV Cache；from High-bandwidth memory (HBM, e.g.,；A100 has 80GB HBM); the GPU spends；most of its time waiting for memory；transfer, not computing.；• Determines (Time Per Output Token).；• (TPOT > 200ms feels sluggish to users;；production systems like GPT-4o achieve；~30–50ms (~20 tokens/sec)) These are big chunks of memory；from GPU HBM (or worse from CPU；main memory) to GPU cache；(memory closest to GPU cores).
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. The "Memory Wall" in LLMs FLOPs

* **定义 / 内容：** The "Memory Wall" in LLMs FLOPs。needs；grows；faster；• Massive Data Load: Generating a new token forces than；the historical KV cache (of size L * d) to be re-read computing；power；from memory, a multi-gigabyte operation with a；moderate large context length.；o L: Context/sequence Length, d: dim Memory；needs；• 1 GB Loaded per Step: A real example for grows；faster；Llama-3 8B at seq=4096. than；GPU；• The gap between GPU core compute and memory；memory transfer speed growth.；Computing；• Compute Starvation: the compute power sits idle speed；grows；while waiting for the slow memory bandwidth to faster；fetch the cache. than；memory；transfer；speed；https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Linear Growth, Massive Impact

* **定义 / 内容：** Linear Growth, Massive Impact。• As sequence length L increases, the KV Cache grows linearly, acting as a hard；limit on the GPU's context window.；Grows too fast!；• Weights are fixed, but long document tasks；crash with CUDA out of memory. The culprit；is the KV Cache, not model size.；• Memory layout when serving an LLM；with13B parameters on NVIDIA A100.；vLLM stands for Virtual Large Language Model；and supports LLMs in inferencing and model；serving efficiently.；Kwon, Woosuk, et al. "Efficient memory management for large language model serving with pagedattention." symposium on operating systems principles. 2023.
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. How big is the KV Cache A batch of 3 sequences. LLM generates a token

* **定义 / 内容：** How big is the KV Cache A batch of 3 sequences. LLM generates a token。for each sequence in parallel in iterations.；• The size formula (FP16)；Size = Batch × SeqLen × 2 × Layers × Heads × Dim × 2 bytes；Comment:；Batch: The number of concurrent sequences being processed. • Increase batch size；for more parallelisim;；SeqLen: The max number of tokens of the sequences.；• SeqLen:；2: the two separate matrices need to cache: K (Key) and V (Value). unpredictable.；• Layers/Dim/Bytes:；Layers: The number of Transformer blocks.；should be increased；Heads: The number of attention heads in multi-head attention. due to scaling laws.；• Heads: may be；Dim: The hidden dimension size of each attention head.；decreased, but the；2 bytes (Data Precision): for storing a FP16 number. gain is limited (see；the MQA/GQA slides).
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. FlashAttention

* **定义 / 内容：** FlashAttention。• Tiling: compute attention softmax online block-by-block and don’t；cache the attention weights at all.；• A simple demonstration；• Suppose we have 3 keys/values, and a query generates 3 logits [𝑚! 𝑚" 𝑚# ].；• Softmax calculates the weights over 3 value vectors.；$%& '! (' ∗ $%& '# (' ∗ $%& '$ (' ∗；• 𝑤" 𝑤! 𝑤# = ( , , )；) ) )；• 𝑑 = ∑*+",!,# exp 𝑚* − 𝑚 ∗；• 𝑚 ∗ = max(𝑚" 𝑚! 𝑚# )；• Want to find the weighted sum s = 𝑤" 𝑣" + 𝑤! 𝑣! + 𝑤# 𝑣#；• The nominators can be calculated in parallel, but the denominator d is the sum of；all nominators: the weights can be found only after going through all logits.；• What if the logits and are in different places?；.；• Define the partial sum 𝑑. = ∑*+" exp 𝑚* − 𝑚 ∗；• 𝑠. = ((𝑠.(" ∗ 𝑑.(" ) + exp 𝑚. − 𝑚 ∗ 𝑣. )/𝑑.
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. Optimizing Attention Heads

* **定义 / 内容：** Optimizing Attention Heads。• Multi-Head Attention (MHA): Each Query has its own；dedicated Key and Value.；• The KV Cache expands drastically, serving as the primary culprit；behind the "memory wall" we encounter.；• To reduce the KV-cache bottleneck in MHA, Multi-；Query Attention (MQA) where the keys and values are；shared across all of the different attention heads.；• KV Cache Compressed to the Extreme.；• Can impact the model's expressive power and generation quality.
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. Optimizing Attention Heads

* **定义 / 内容：** Optimizing Attention Heads。• Grouped Query Attention (GQA)；• An interpolation between MHA and MQA by introducing subgroups；of query heads. Each group has a single Key and Value head.；• In contrast to MQA, GQA keeps the same proportional decrease in；memory bandwidth and capacity as model size increases.；Spend much less time but with；slight performance reduction.；Image source: https://medium.com/@atulit23/implementing-multi-head-latent-attention-from-scratch-in-python-1e14d03fbc91
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. The Memory Waste: Why Fragmentation Happens

* **定义 / 内容：** The Memory Waste: Why Fragmentation Happens。• An LLM may allow a max length of 2048 tokens.；• But in reality: User A sends 50 tokens. User B sends 300 tokens. User C sends 700 tokens.；• Internal fragment: LLM reserves 2048 tokens worth of memory for each user. However,；users can use much less, resulting in memory fragment due to the unused reserved full；chunk, and other threads cannot use the fragments. (Output length is unpredictable).；• External fragment: operating system (CUDA) may place the memory space of request；A and request B at non-consecutive physical space. The gap between them may be too；small to be used by another thread and becomes a fragment.；Kwon, Woosuk, et al. "Efficient memory management for large language model serving with pagedattention." symposium on operating systems principles. 2023.
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 15. PagedAttention (One request:)

* **定义 / 内容：** PagedAttention (One request:)。• Prompt: ”Four score and seven years ago our”；• A request as a process.；• Logical memory as virtual；memory within an operating；system.；• Block table translates virtual；memory to physical memory.；The blocks come from；• Physical memory blocks reside different layers/heads.；in the GPU's memory, and each When a memory block；block is analogous to a page in is full, future KV blocks；virtual memory. vLLM finds a new place.；Kwon, Woosuk, et al. "Efficient memory management for large language model serving with pagedattention." symposium on operating systems principles. 2023.
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. PagedAttention (Multi requests:)

* **定义 / 内容：** PagedAttention (Multi requests:)。• The logical blocks of the two sequences are mapped to different；physical blocks within the space reserved by the block engine.；Storing the KV cache of two requests at the same time in vLLM.；Kwon, Woosuk, et al. "Efficient memory management for large language model serving with pagedattention." symposium on operating systems principles. 2023.
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. Bubbles and Preemption in Continuous Batching

* **定义 / 内容：** Bubbles and Preemption in Continuous Batching。• The decoding iterations of a request (prompt) needs to wait for；other’s pre-fill to be done: this is a scheduling issue.；Iteration-level Scheduling leads to significant GPU resource underutilization and 'Bubbles'.；Agrawal, Amey, et al. "Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills." arXiv:2308.16369 (2023).
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Chunked prefills with decode-maximal batching

* **定义 / 内容：** Chunked prefills with decode-maximal batching。• Chunked Prefills: Instead of processing the entire Prefill task at once, it is split into；fixed-size chunks.；• Piggybacking: When processing a specific Prefill chunk (Cp1), utilize its remaining；computational capacity to "piggyback" an ongoing Decode task (Ad1).；significantly reduces pipeline bubbles and enables more efficient piggybacked decodes.；Agrawal, Amey, et al. "Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills." arXiv:2308.16369 (2023).
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Memory Wall in Long Context

* **定义 / 内容：** Memory Wall in Long Context。• When the text length exceeds the GPU memory capacity, the system will immediately；encounter an Out of Memory (OOM) error；• Limit the window size:；(a): Memory explodes; fails；beyond pre-train length.；(b): Collapses immediately；when first tokens are evicted.；(c): Prohibitively slow due to；repeated O(L2) recalculation.；PPL: Perplexity, the lower the；better.；the lower,；the better；Xiao, Guangxuan, et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. StreamingLLM: Attention Sink is All You Need

* **定义 / 内容：** StreamingLLM: Attention Sink is All You Need。With a window of limited size, the model performance drops due to missed information.；One interesting observation: attention sink. Hypothesis: due to the normalization；of softmax, though the initial tokens；do not carry much semantics；information, the position embedding；instructs the attention weights to be；placed at the beginning.；Why: as during auto-regression；training, the initial few positions are；Xiao, Guangxuan, et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024 seen the most often.
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. StreamingLLM: Attention Sink is All You Need

* **定义 / 内容：** StreamingLLM: Attention Sink is All You Need。• Key insight: The initial tokens are visible to all subsequent tokens; consequently, the；model tends to focus its attention on these initial tokens, forming an "attention sink."；The KV cache of Streaming LLM Some dummy characters；(now of a fixed size). (e.g., ‘\n’) also maintain；the performance (PPL)；Keeps the attention sink (several initial tokens) for stable；attention computation, combined with the recent tokens.；Xiao, Guangxuan, et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. Attention Map

* **定义 / 内容：** Attention Map。• Visualization of average attention logits over 256 sentences, each 16 tokens long,；comparing models pre-trained without (left) and with (right) a sink token.；• Without a sink token, models show local attention in lower layers and increased attention to；initial tokens in deeper layers.；• With a sink token, there is clear attention directed at it across all layers, effectively collecting；redundant attention.；• With the presence of the sink token, less attention is given to other initial tokens, supporting；the benefit of designating the sink token to enhance the streaming performance.；Xiao, Guangxuan, et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024
* **直觉：** 本页展开 KV cache、prefill/decode、memory wall、FlashAttention、MQA/GQA、PagedAttention 与 StreamingLLM。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (short answer): write down the attention equation involving Q, K, V；matrix Attention(Q, K, V).；• Q2 (T/F): exact online computation of softmax is impossible and；approximation of softmax is needed when calculating the normalization；factor online.；• Q3 (multi-choice): which of the following is a factor about the size of KV；cache: (A) number of layers; (B) layer normalization parameters; (C) the；differences in the length of user queries; (D) the length of the generated；token sequences; (E) the size of GPU memory.；• A: see page 3’s first equation; F; (AD)
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 21: LLM Compression：Quantization、Pruning、Distillation

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 21；The Efficiency Challenge；Numeric formats in computers；Mathematical mapping；Quantization accuracy cost；How to Quantize?；Quantization-Aware Training；Quantization-Aware Training (QAT)
  * Post-Training Quantization (PTQ)；Emergent outliers in LLMs；Why LLM has and needs outlier activations?；Compression efficiency；LLM.int8 (mixed precision quantization)；SmoothQuant Optimization；Concept of Pruning；Activation Pruning
  * Activation Pruning；Iterative pruning；Iterative pruning；Structured Sparsity (2:4)；Structured Sparsity (2:4)；Concept of Distillation；Concept of Distillation；Distillation
  * Distillation；Distillation: debates and lawsuits；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Compression Notes

### 1. Compression 要解决什么

scaling law 推动模型变大，但硬件显存、带宽和成本跟不上。70B model 用 FP16 权重约需：

$$
70B\times 2\ bytes=140GB
$$

compression 让模型更便宜地训练/部署，主要方法是：

* quantization。
* pruning。
* distillation。

### 2. Quantization 的数学映射

quantization 把连续值映射到离散整数。一般形式：

$$
q=clip\left(round\left(\frac{x}{S}\right)+Z,q_{min},q_{max}\right)
$$

dequantization：

$$
\hat{x}=S(q-Z)
$$

$S$ 是 scale，$Z$ 是 zero-point。symmetric quantization 常令 $Z=0$；asymmetric quantization 用 zero-point 对齐非对称范围。

bit width 越小，内存越省，但误差越大。课件强调 4-bit 常是大模型 memory/accuracy 的好折中；3-bit 往往会遇到 precision cliff。

### 3. QAT vs PTQ

**Post-Training Quantization (PTQ)**：训练后直接量化。优点是快、便宜；缺点是低 bit 下容易掉精度。

**Quantization-Aware Training (QAT)**：训练中模拟 quantization error，让模型适应低精度。优点是恢复精度好；缺点是贵，需要数据和训练。

round() 不可微：

$$
\frac{d}{dx}round(x)=0\quad \text{almost everywhere}
$$

QAT 用 **Straight-Through Estimator (STE)**：forward 用 round，backward 假装它是 identity：

$$
\frac{d}{dx}round(x)\approx 1
$$

### 4. LLM Outliers、LLM.int8 和 SmoothQuant

LLM 大模型中会出现 activation outliers：少数维度激活值巨大，而且对 perplexity 很重要。如果用全局 absmax scale，outlier 会把 $S$ 拉大，普通值量化后都变成 0。

LLM.int8 的思想是 mixed precision：大多数普通值用 INT8，少数关键 outlier 用 FP16 保留。

SmoothQuant 观察到 activations 有 outliers，而 weights 更容易量化，于是用 per-channel factor 把量化困难从 activation 平滑迁移到 weight：

$$
Y=XW=(XS^{-1})(SW)
$$

这样 activation 更平滑，weight 吸收 scale 后仍可较好量化。

### 5. Pruning

pruning 删除冗余参数或结构。

**Unstructured pruning**：删除单个权重，模型大小可降，但 GPU 很难跳过散乱 0，速度收益有限。

**Structured pruning**：删除 neuron、channel、attention head 等完整结构，剩余矩阵仍 dense，更容易获得真实 speedup。

activation pruning 不只看 weight magnitude，还看 calibration data 上的 activation。课件例子：

$$
w=[1,2],\quad x=[2,0.1]
$$

内积是：

$$
w^Tx=1\cdot2+2\cdot0.1=2.2
$$

虽然第二个 weight 更大，第一维贡献更大，因为 activation 更大。这就是 WANDA 这类方法要同时看 weight 和 activation 的原因。

### 6. Iterative Pruning、OBS 和 2:4 Sparsity

iterative pruning 反复执行：

1. 按重要性删除一部分 block/channel/weight。
2. fine-tune/retrain 恢复性能。
3. 重复直到达到 size/accuracy trade-off。

Optimal Brain Surgeon 用 Hessian 估计删除一个参数对 loss 的影响，并更新剩余参数补偿。

2:4 structured sparsity 是硬件友好折中：每连续 4 个值中固定 2 个为 0。NVIDIA Sparse Tensor Core 能利用这种模式加速。

### 7. Distillation

distillation 用大 teacher model 训练小 student model。hard label 只告诉正确类别；soft distribution 还包含 teacher 的相似性知识：

$$
p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

温度 $T>1$ 会让 distribution 更平滑，提高非目标类别之间的可学习差异，这些小概率就是 **dark knowledge**。

distillation 可让 student 不同架构、参数更少、速度更快；缺点是需要训练 student，且 teacher 输出来源可能涉及数据许可和法律争议。

### 8. Exam Focus

* structured pruning 删除结构，因此更可能带来真实速度提升。
* 只看 weight magnitude 会错过 activation outlier 的重要性。
* QAT 用 STE 处理 round 不可微。
* distillation 的 soft targets 比 one-hot hard targets 信息更多。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 21

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 21，主题为 LLM Compression：Quantization、Pruning、Distillation。
* **直觉：** 确认 Lecture 21 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. The Efficiency Challenge

* **定义 / 内容：** The Efficiency Challenge。Scaling Law vs. Hardware：；As LLMs scale up, their memory；and compute requirements grow；faster than hardware capabilities.；• Memory overhead: weights；for a 70B model take 140GB in；FP16.；• Memory wall: hardware；compute (FLOPS) outpaces；memory bandwidth by orders；of magnitude (right figure).；Real example: LLaMA-2 70B requires 8x A100 80GB GPUs to run in FP16 — costing ~$20K/month on AWS. After；INT4 quantization, it fits on a single RTX 3090 consumer GPU (~$1,500). This is the gap compression bridges.；[1] Gholami, Amir, et al. "Ai and memory wall." IEEE Micro 44.3 (2024): 33-39.
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. Numeric formats in computers

* **定义 / 内容：** Numeric formats in computers。Module 1: GPU memory；Slow Data Transfer；• FP16 (2 Bytes): Range [-65504, 65504]. (Example: Standard model weights) N: Number of parameters；C: Cache capacity in Bytes；Pbit: Bit Width；• INT8 (1 Byte): Range [-128, 127]. (Example: Standard post-training quantization)；• INT4 (0.5 Bytes): Range [-8, 7]. (Example: Aggressive compression formats) More parameters,；but less precisions!
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Mathematical mapping

* **定义 / 内容：** Mathematical mapping。Quantization maps continuous values to a discrete set of integers using a Scale；( ) and Zero-point ( ):；!"#("%&(')) !"# ' -!./(')；𝑆= (absmax quantization), or 𝑆 = (zeropoint quantization)；*+, +∗*+,；Example (INT8 symmetric quantization)： Setting:；• The range [-max, max] is mapped to [-127, 127]. X=2.14 S=0.0275 Z=0；qmin=-128 qmax=127；• Quantize (FP32 to INT8):；q=78；• Dequantization approximates the original:；x=2.14；Storing less information: 8 bits rather than 32 bits,；plus quantization parameters S and Z.
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Quantization accuracy cost

* **定义 / 内容：** Quantization accuracy cost。The Precision Cliff；Given a fixed memory budget, how Same model；should we trade off model size versus memory requirements；precision?；• A larger model quantized to 4-bit yields；higher accuracy than a smaller model at；8-bit or 16-bit.；• 4-bit precision strikes the optimal balance.；The gain from increasing the number of；parameters by 4x outweighs the；quantization noises.；• Scaling breaks down at 3-bit.；Quantization errors become so severe；that adding more parameters can no；longer recover the lost performance.；[2] Dettmers, Tim, and Luke Zettlemoyer. "The case for 4-bit precision: k-bit inference scaling laws." ICML, 2023.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. How to Quantize?

* **定义 / 内容：** How to Quantize?。Quantization introduces errors. To minimize this accuracy drop, the industry has；developed two primary pipelines.；Option 1: Quantization-Aware Training (QAT)；• What: Simulate quantization during training so the model adapts.；• Pros: Recovers accuracy perfectly, essential for 4-bit or lower.；• Cons: Expensive, requires training data and compute.；Option 2: Post-Training Quantization (PTQ)；• What: Quantize the model after it's fully trained.；• Pros: Fast, cheap, requires no training data.；• Cons: Vulnerable to the "Precision Cliff" at < 8-bit.；[3] Liu, Zhenhua, et al. "Post-training quantization for vision transformer. NIPS 2021；[4] Nagel, Markus, et al. "Overcoming oscillations in quantization-aware training.” ICML, 2022.
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Quantization-Aware Training

* **定义 / 内容：** Quantization-Aware Training。First, we need to solve non-differentiability with the Straight-Through Estimator (STE)；Why standard training fails；Forward Pass: Simulated Quantization uses the round() operator to simulate low-precision；formats in .；Try to simulate a quan/dequan process.；The Block: The derivative of round() is exactly 0 almost everywhere.；Backpropagation fails because gradients vanish, preventing weight updates.；Straight-Through Estimator (STE)；STE acts as a “gradient bypass.” During the；backward pass, it pretends the round() operator；is an identity function (derivative = 1).；[4] Nagel, Markus, et al. "Overcoming oscillations in quantization-aware training.” ICML, 2022.；Image source: https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-；estimators-with-pytorch-implementation-71d99d25d9d0
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Quantization-Aware Training (QAT)

* **定义 / 内容：** Quantization-Aware Training (QAT)。Unlike PTQ, QAT trains the model with quantized values in the forward path.；The gradients account for；the information loss as well.；1. PTQ Initialization；Runs a small calibration data；to compute the initial Scale 𝑆；and Zero-point 𝑍.；2. Forward Pass；Injects “Fake Quantization”；(rounding errors) into the；actual training flow.；The loss now can account；for information loss during；to quantization errors.；3. Backpropagation；Uses the Straight-Through Estimator (STE) to bypass the non-differentiable round() nodes.；4. The Update Loop；Accumulates the backpropagated gradients into hidden FP32 Shadow Weights.；Image source: https://developer.nvidia.cn/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Post-Training Quantization (PTQ)

* **定义 / 内容：** Post-Training Quantization (PTQ)。Quantizing a pre-trained model without requiring any further weight updates or；backpropagation.；Why 8-bit post-training quantization；• Memory & Speed: Converts both weights and activations from FP32 to INT8. This；shrinks the memory footprint to 1/4th of the original size and significantly accelerates；matrix multiplication throughput.；Before (Left):；Standard convolution operates using high-precision.；After (Right):；1. Weights converted to INT8.；2. Activations adaptively quantized.；3. The actual convolution is executed using INT8.；[3] Liu, Zhenhua, et al. "Post-training quantization for vision transformer. NIPS 2021
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Emergent outliers in LLMs

* **定义 / 内容：** Emergent outliers in LLMs。The Difficulty of LLMs；Large models (>6B) have activation channels；with magnitudes 100x larger than others.；Problem: standard quantization squashes all；non-outlier values to zero, destroying accuracy.；For example, in absmax quantization；The Squash Effect: When an outlier is 100x；larger than normal values, S becomes massive.；Normal values (e.g., x=1.2) divided by a massive；S become < 0.5, which the round() function；directly maps to 0.；[5] Dettmers, Tim, et al. "Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale."；Advances in neural information processing systems 35 (2022): 30318-30332.
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. Why LLM has and needs outlier activations?

* **定义 / 内容：** Why LLM has and needs outlier activations?。Definition (activation outlier): a；dimension of LLM activations (hidden；vectors), that has values at least 6.0,；and that dimension appears in at least；25% of the layers, and at least 6% of；the tokens.；Left: outlier dimensions correlats to；perplexity reduction (better model fitting).；Right: outlier only appear in a few；number of dimensions, leading to some；rare but significant events. Great；opportunity for compression!
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 12. Compression efficiency

* **定义 / 内容：** Compression efficiency。• In information theory, and coding theory in particular, the goal is to assign；some discrete (quantized) codes (numbers) to symbols to reduce overall；size of storing some information (e.g., pice of texts).；• Expected total length；3；𝐿=$ 𝑝1 × 𝑙1；12* symbol p codes；a 5 1100；• 𝑝! : probability of seeing the i-th symbol；• 𝑙! : code length assigned to the i-th symbol b 9 1101；• Want to minimize the expectation. c 12 100；• Huffman tree: generates the codes. d 13 101；• Average code length is close to the e 16 111；information entropy f 45 0；of the symbol distribution.；• Mixed precision model quantization! Rare events require longer code, while；frequent events use shorter codes.
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. LLM.int8 (mixed precision quantization)

* **定义 / 内容：** LLM.int8 (mixed precision quantization)。Matrix multiplication can be decomposed into independent row-column dot products,；allowing us to apply ”mixed precision“ quantization (assigning different scales for different；rows of X and columns of W ), instead of a single global scale (waste bits and zero outliers).；per-column；For frequent but small numbers element-wise；scaling；Outlier activation features product；(our VIP: can be 20x larger；than other dimensions)；remain in FP16 but they；represent only a tiny fraction per-row；scaling；of total weights.；𝐶! , 𝐶" : row/column max of X/W；For infrequent but significant numbers (VIP!)；[5] Dettmers, Tim, et al. "Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale."；Advances in neural information processing systems 35 (2022): 30318-30332.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. SmoothQuant Optimization

* **定义 / 内容：** SmoothQuant Optimization。The Observation:；Activation (X) contains massive outliers that destroy INT8 precision.；Conversely, Weight (W) distributions are extremely flat and "easy" to quantize.；Action:；Migrate the quantization bandwidth from weights；(need less bandwidth) to activations (need more；bandwidth). Apply a per-channel smoothing factor；s to "migrate" the quantization difficulty；smoother sharper；The Result:；Suppress the spikes in X by pushing them into W.；Since weights have "excess precision capacity,"；they can absorb these outliers with minimal；accuracy loss.；[6] Xiao, Guangxuan, et al. “Smoothquant: Accurate and efficient post-training quantization for large language models.”ICLR, 2023.
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Concept of Pruning

* **定义 / 内容：** Concept of Pruning。Removing Redundancy；Neural networks are often 30-50% redundant.；Pruning involves identifying and zeroing out；"unimportant" weights.；• Reduces Parameter count.；• Reduces FLOPs (if structured).；• Unstructured: Zeroes out individual weights based on；importance regardless of their spatial location.；• GPUs cannot efficiently skip these scattered zeros；due to irregular memory access (store zeros in the；sparsified vectors).；• Reduced model size, but minimal to no wall-clock；speedup on general-purpose hardware.；• Structured: Removes entire architectural components,；such as neurons, channels, or attention heads.；• The remaining weights form smaller, dense；matrices.；• Achieves instant, real-world speedup and；significant memory bandwidth savings.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. Activation Pruning

* **定义 / 内容：** Activation Pruning。Unlike weight pruning which looks at static parameters, Activation Pruning；identifies redundancy by observing the model in action.；1. We feed a representative dataset (calibration data)；through the pre-trained model.；2. We monitor the output (activations) of every neuron；across these samples. No backpropagation is required.；Green: activations with large absolute values (e.g., 0.5 and -；1.3), significant features, and should be retained.；Red: activations close to 0 (e.g., 0.007 and -0.002),；contributing almost nothing.；Image source: https://blog.dailydoseofds.com/p/activation-pruning-reduce-neural
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. Activation Pruning

* **定义 / 内容：** Activation Pruning。• Example: let w=[1,2], x=[2,0.1], then their inner product is 2.2.；o The first weight actually contributes more due to larger magnitude of the first；activation dimension, though the first weight is less than the second one.；• Need to check both weight and activation (WANDA).；o The significance of weight at (i,j) is determined by；Statistical significance of the j-th activation dimension；• Related to the significance score used by Optimal Brain Surgeon (OBS):
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Iterative pruning

* **定义 / 内容：** Iterative pruning。To minimize accuracy drop, pruning is performed incrementally. The model；undergoes multiple cycles of removing redundant sub-blocks (layer by layer,；channel by channel, etc.), followed by fine-tuning to recover performance.；Sub-blocks & Scores；Divide a weight matrix into non-overlapping N×N sub-blocks. Each block's importance is measured；by its L2-norm magnitude:；1. Pruning Phase: Blocks with scores below a dynamic；threshold (e.g., the weakest 10%) are masked to zero.；2. Retraining Phase: We "heal" the model by retraining；the remaining dense blocks. This allows the surviving；neurons to compensate for the lost information,；restoring the original accuracy.；3. Iteration: This loop continues until the desired balance；between model size and accuracy is reached.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 19. Iterative pruning

* **定义 / 内容：** Iterative pruning。Optimal Brain Surgeon:；For each layer, for each output channel,；find the optimal surgeon by solving the following:；o The big Hessian matrix per layer only needs to；o Pruning one weight will impact others, be calculated once during iterative pruning.；so that the 𝜹𝒑 vector updates the；remaining weights to minimize the；impact of removing one weight.；Image source: https://matthewmcateer.me/blog/optimal-brain-damage/；Optimal Brain Damage, NIPS 1989, Yann LeCun；Optimal Brain Surgeon and General Network Prunning. In IEEE International Conference on Neural Networks, 1993.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. Structured Sparsity (2:4)

* **定义 / 内容：** Structured Sparsity (2:4)。The Problem: random sparsity is slow for GPUs. However, cutting entire channels；(Structured) hurts accuracy too much.；The Solution: 2:4 is a ”semi-structured" compromise. For every block of 4；contiguous values in a row, we force exactly 2 to be zeros.；NVIDIA designed the Sparse Tensor Core specifically to skip the zeros in this exact；2:4 pattern, yielding 2x higher throughput while maintaining enough "granularity"；to keep accuracy high.；Pros:；1. efficient memory accesses；2. a low-overhead compressed format；3. 2x math throughput increase on the NVIDIA Ampere GPU architecture；Image source: Accelerating Sparse Deep Neural Networks
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. Structured Sparsity (2:4)

* **定义 / 内容：** Structured Sparsity (2:4)。A fine-grained 2:4 structured pruning approach to make sparsity adoption practical.；For every block of 4 horizontal elements, at least 2 must be zeroed out.；• Data Matrix (R x C/2): Stores only the 2 non-zero values per block, cutting weight memory in half.；• Indices Matrix (Metadata): Uses 2-bit indices to store the spatial location of each non-zero element；for each block.；• During inference, the Tensor Core uses the Indices to pull data from the compressed matrix and；"match" it with the corresponding activation values in real-time.；Keeping hightest weights ensures that the；Image source: Accelerating Sparse Deep Neural Networks most "significant" features are preserved.
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. Concept of Distillation

* **定义 / 内容：** Concept of Distillation。LLMs achieve SOTA performance by scaling parameters to hundreds of billions.；We need the wisdom of a 70B model but the efficiency of a 7B model.；Why Distillation?；• Pruning often struggles to maintain the；complex reasoning capabilities of LLMs.；• Distillation serves as a "knowledge；transfer" mechanism.；Teacher-Student Framework:；o Teacher Model: A large, cumbersome, but high-performing pre-trained model.；o Student Model: A smaller, compact model that aims to mimic the Teacher’s behavior.；Goal:；• Enable a small “Student” model to mimic the Teacher model’s behaviors.；What knowledge can be transferred: input-output relationship.；Image source: https://www.britannica.com/technology/knowledge-distillation
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. Concept of Distillation

* **定义 / 内容：** Concept of Distillation。Hard vs. soft output；• Hard output: one-hot vectors. Only tells the Student which class is correct.；• Soft output: The probability distribution from the Teacher's output.；• Dark Knowledge: These tiny probabilities (e.g., 0.1% for Deer) are not random noise. They；represent the Teacher’s internal logic: "A horse shares structural similarities with a deer；(quadruped, similar stature) but looks nothing like a car."；https://dailyinterlake.com/news/2014/mar/05/horses-and-；deer-6/?=/&subcategory=285%7CWine
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. Distillation

* **定义 / 内容：** Distillation。The Problem:；Softmax outputs are often too “sharp” (target ≈1.0, others ≈ 0.0). This extreme compression；crushes the subtle differences between non-target classes to near zero (thus little；significance for the student to learn from).；• Without Temperature:；• With Temperature:；Zi means probability of each category.；At T=1, the distribution has low entropy. Applying a Temperature factor (T>1) scales down the logits；before the exponential function, increasing distribution entropy and preventing the dominant class；from saturating the gradients.
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. Distillation

* **定义 / 内容：** Distillation。Pros:；1. Student can be a completely different architecture[7]；Transformer CNN；2. DistilBERT[8]: 40% smaller, 60% faster, retains 97% of BERT accuracy.；Cons:；1. Requires training the student from scratch — expensive compared to PTQ.；2. Teacher must be available during student training — not always feasible for；proprietary models.；[7] Hu, Chengming, et al. "Teacher-student architecture for knowledge distillation: A survey." arXiv preprint arXiv:2308.04268 (2023).；[8] Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).
* **直觉：** 本页展开 numeric format、quantization、QAT/PTQ、outlier activations、LLM.int8、SmoothQuant、pruning 与 distillation。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 26. Distillation: debates and lawsuits

* **定义 / 内容：** Distillation: debates and lawsuits。本页主要用于课程衔接、图示或标题说明。
* **直觉：** 图示引用围绕 distillation 的争议：如果用闭源模型输出或可疑来源数据训练 student model，可能引发数据许可、模型条款、版权和竞争伦理问题。知识蒸馏不仅是压缩技术，也涉及 data provenance 与合规风险。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 27. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (T/F): structured model pruning removes some neurons of a layer so；that it gains more computational efficiency.；• Q2 (T/F): pruning using weight magnitudes without considering activation；magnitudes can miss the emergent outlier activations in an LLM.；• Q3 (calculation): quantize the vector x=[0.1, 100] to INT8 (ranging from 0 to；255) data precision, using the equation (S=255/100， Z=0)；• A: T; T; q=[0,39]
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 22: Mixture of Experts (MoE)、Routing 与 Sparse Upcycling

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 22；Motivation of Mixture of Experts；Road map of Mixture of Experts；Overview of Mixture of Experts；Mixture of Experts in LSTM；Mixture of Experts in Transformer；Routing algorithms — Top 1 routing；Routing algorithms — Routing collapse
  * Routing algorithms — Top 2 routing；Case of Mixture of Experts；Case of Mixture of Experts--DeepSeek；Case of Mixture of Experts—DeepSeek；Case of Mixture of Experts—DeepSeek；MoE training: backprop through expert selection；MoE training；MoE training
  * Routing algorithms — BASE routing；Routing algorithms — BASE routing；Routing algorithms — Reinforcement learning；Building MoE LLMs；Building MoE LLMs—Sparse Upcycling；Sparse Upcycling-Mixtral MoE；Sparse Upcycling-Qwen MoE；Building MoE LLMs—Sparse Splitting
  * models of similar scale；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed MoE Notes

### 1. MoE 的目标：参数多，但每次只激活一小部分

dense model 每个 token 都经过所有参数。MoE 的想法是：模型可以拥有很多 experts，但每个 token 只路由到少数 experts，从而提高总参数容量，同时控制 activated FLOPs。

基本形式：

$$
G(x)=softmax(W_gx)
$$

选择 top-$k$ experts：

$$
I=TopK(G(x))
$$

输出是 selected experts 的加权和：

$$
y=\sum_{i\in I}s_iE_i(x)
$$

### 2. Transformer 中的 MoE

Switch Transformer 把标准 FFN 替换成 MoE layer。对每个 token hidden state $h$：

$$
p=softmax(W_rh)
$$

Top-1 routing 时：

$$
MoE(h)=p_iE_i(h)
$$

只有被选中的 expert 处理该 token。attention 层通常仍 dense，MoE 主要替换 FFN，因为 FFN 占 Transformer 计算量很大。

### 3. Routing Collapse

Top-1 routing 省 FLOPs，但容易 **routing collapse**：router 总把 token 分给少数 experts，其他 experts 不训练，整体容量浪费。

Top-2 routing 让每个 token 走两个 experts：

* 表达力更强。
* 可缓解 collapse。
* 计算和跨设备通信更贵。

auxiliary load-balancing loss 鼓励专家使用更均匀。课件中：

* $f_i$：实际路由到 expert $i$ 的 token fraction，来自 hard routing，通常不可微。
* $P_i$：router 给 expert $i$ 的平均概率，可微。

训练时梯度主要通过 $P_i$ 流动，$f_i$ 当作常量。

### 4. 离散路由为什么难训练

expert selection 是 discrete choice，普通 backprop 无法穿过 argmax/top-k。可用近似：

* softmax gate 的概率梯度。
* straight-through / Gumbel-Softmax 思路。
* auxiliary loss。
* RL-based routing，把 router 当 agent，把 expert selection 当 action，把任务表现当 reward。

RL routing 的问题是 reward sparse、variance 高。

### 5. BASE Routing

BASE routing 把 token-to-expert assignment 视为 linear assignment problem，强制每个 expert 接收 $T/E$ 个 tokens，从而严格 load balance。

优点：专家利用均衡。  
缺点：assignment 求解昂贵，不适合 autoregressive decoding 中逐 token 生成的场景。

### 6. MoE 案例：GLaM、DeepSeek、Mixtral、Qwen

课件的共同结论：在相似 activated FLOPs 下，MoE 往往比 dense model 更强。

DeepSeek-MoE 例子：

* shared experts 学通用语法和常见语义。
* routed experts 学更专门的领域知识。
* 总参数很多，但每 token 只激活一部分。

Mixtral、Qwen MoE 说明 sparse upcycling 可从 dense checkpoint 出发构建 MoE，降低从零训练成本。

### 7. Sparse Upcycling vs Sparse Splitting

**Sparse Upcycling**：复制原 dense FFN 形成多个 experts。优点是复用 pretrained knowledge；缺点是总参数按 expert 数增加，存储/通信成本变高。

**Sparse Splitting**：把原 FFN 切分成多个 experts，总参数量不增加，activated parameters 减少；缺点是每个 expert 容量变小。

### 8. Exam Focus

* MoE 可以 Top-1、Top-2 或 Top-k，不是只能一个固定专家。
* 课件 quiz 中第二题答案给 F，理解为“MoE 并不必然只能选 1 个 expert”；Top-1 只是一个 routing choice。
* 当前 token 未选某 expert，不代表未来 token 不能选它。
* routing collapse 是 MoE 的核心训练风险。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 22

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 22，主题为 Mixture of Experts (MoE)、Routing 与 Sparse Upcycling。
* **直觉：** 确认 Lecture 22 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Motivation of Mixture of Experts

* **定义 / 内容：** Motivation of Mixture of Experts。• As model size increases, activating • Not all model parameters are useful；all parameters for each inference in the forward pass. We activate only；step becomes computationally a small subset of parameters,；expensive. thereby reducing computational cost.；𝑋 Full model 𝑦"；Subset of the；𝑋 full model 𝑦"；Image source: https://ourworldindata.org/grapher/exponential-growth-of-parameters-in-notable-ai-systems
* **直觉：** 本页解释问题动机或现有方法的局限，说明为什么需要引入本讲的新模型、算法或系统设计。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。

### 3. Road map of Mixture of Experts

* **定义 / 内容：** Road map of Mixture of Experts。• This figure summarizes the evolution；of MoE models from 2017 to 2024.；• MoE has expanded from an early；research idea to a widely used；architecture in NLP, vision,；multimodal, and recommender；systems.；Cai W, Jiang J, Wang F, et al. A survey on mixture of experts in large language models[J]. IEEE Transactions on Knowledge and Data Engineering, 2025.
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Overview of Mixture of Experts

* **定义 / 内容：** Overview of Mixture of Experts。• Input x is sent to a gating network G；𝐺 𝑥 = [0,1,0]；• The gating network decides which expert is the most；suitable for the current input (using a one-hot vector).；• Instead of using all experts 𝐸! , 𝐸" , …, 𝐸# , the model；activates only a small subset.；• In the example on the left, only 𝐸" is selected, while the；other experts remain inactive.；• The final output is generated from the selected expert:
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Mixture of Experts in LSTM

* **定义 / 内容：** Mixture of Experts in LSTM。• For input X, the gating network computes routing；scores: 𝐺 = Softmax 𝑊! 𝑋；• The model selects the top-K experts, e.g. K=2, 𝐼 =；𝑖" , 𝑖# = Top𝐾 𝐺 𝑘 = 2；• The selected experts are assigned weights:；𝐺$! 𝐺$"；𝑠" = , 𝑠# =；𝐺$! + 𝐺$" 𝐺$! + 𝐺$"；• Each selected expert is a feed-forward network.；The final output is a weighted combination of the；selected experts:；𝑌 = s$ FFN%! 𝑋 + 𝑠! FFN%" 𝑋；Shazeer N, Mirhoseini A, Maziarz K, et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Mixture of Experts in Transformer

* **定义 / 内容：** Mixture of Experts in Transformer。• Switch Transformer block, where the standard FFN；is replaced by a Mixture-of-Experts layer.；• For each input token, the router computes a；probability: P = Softmax 𝑊% ℎ；• Each expert is a feed-forward network 𝐸$ , The；model selects the expert with the highest routing；probability；• Only the selected expert processes the token:；𝑀𝑜𝐸(ℎ) = 𝑝& 𝐸& (ℎ)；token 1 token 2；Fedus W, Zoph B, Shazeer N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity[J]. Journal of Machine Learning Research, 2022,；23(120): 1-39.
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Routing algorithms — Top 1 routing

* **定义 / 内容：** Routing algorithms — Top 1 routing。• Advantages: each token activates only one；expert, minimizing forward-pass FLOPs.；• Limitations: routing collapse. During；training, the gating network may over-select；a small subset of experts, leaving others；rarely used and therefore under-trained.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 8. Routing algorithms — Routing collapse

* **定义 / 内容：** Routing algorithms — Routing collapse。• Maximum Routing Imbalance (%) measures how；unevenly tokens are distributed across experts；within a layer.；• A higher value means that routing is more；concentrated on a small number of experts,；indicating poorer load balance；• MoE routers may overuse a few experts while；underutilizing others in early layers；Thérien B, Joseph C É, Sarwar Z, et al. Continual Pre-training of MoEs: How robust is your router?. arXiv preprint arXiv:2503.05029, 2025.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 9. Routing algorithms — Top 2 routing

* **定义 / 内容：** Routing algorithms — Top 2 routing。• Advantages: this improves representation；power and can alleviate routing collapse；• Limitations: Each token activates two experts,；so the computational cost is higher.；Communication overhead is also larger, since；more tokens must be transferred across；devices.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 10. Case of Mixture of Experts

* **定义 / 内容：** Case of Mixture of Experts。• Using similar FLOPs per token prediction,；MoE models have better performance；than the dense variants.；• GLaM has better performance while；using 1/3 of the energy and 1/2 of；serving cost of GPT-3.；Glam: Efficient scaling of language models with mixture-of-experts. International conference on machine learning, 2022
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. Case of Mixture of Experts--DeepSeek

* **定义 / 内容：** Case of Mixture of Experts--DeepSeek。• Each MoE layer consists of 2 shared experts；and 64 routed experts (select 6 experts)；• Shared experts: each token passes through；(no routing). Shared experts learn general；knowledge, such as syntax and common；semantics.；• Routed experts: selectively activated for；each token by a router. Routed experts focus；more on domain-specific knowledge；Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.ACL. 2024
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. Case of Mixture of Experts—DeepSeek

* **定义 / 内容：** Case of Mixture of Experts—DeepSeek。DeepSeek-MoE-；abc 16B, with only；2.8B activated；parameters；A few cases；where MoE is；worse than the；dense model.；DeepSeek-MoE-16B, matches or even surpasses；LLaMA2-7B, which activates all 7B parameters；Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.ACL. 2024
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. Case of Mixture of Experts—DeepSeek

* **定义 / 内容：** Case of Mixture of Experts—DeepSeek。Deepseek-v2 Deepseek-Coder-V2；236B total parameters, 21B are Continue pretraining from an；activated. intermediate checkpoint of Deepseek-；It achieves strong V2 on 6T, adding to a total of 10.2T；performance with much pre-training tokens. Comparable to Deepseek-Coder-V2；lower activated computation GPT4-Turbo in code-specific tasks. Reduced price per 1M tokens；Liu A, Feng B, Wang B, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. MoE training: backprop through expert selection

* **定义 / 内容：** MoE training: backprop through expert selection。• Panel (1): ordinary neural network training. In a standard；dense model, every operation is differentiable；• Panel (2): MoE routing creates a stochastic or discrete node.；The selection of experts step is like the discrete node z in the；figure. The gradients cannot pass through that choice in the；normal way.；Jang E, Gu S, Poole B. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. MoE training

* **定义 / 内容：** MoE training。Given a token sequence of length T, (𝑥! , 𝑥" , ... 𝑥# ), the training loss can be written as:；Non-differentiable；• ℒ &'() is the cross-entropy loss for next-token prediction；• ℒ '*+ encourages balanced expert utilization (details in the next page).；Jang E, Gu S, Poole B. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. MoE training

* **定义 / 内容：** MoE training。For the auxiliary loss:；• 𝑓$ is the fraction of tokens actually routed to expert i, so it depends；on a hard routing decision and is not differentiable；• 𝑃$ is the average router probability for expert i, so it is；differentiable；Panel (4): In the forward pass, the；model still uses a hard or sampled • As a result, the gradient mainly flows through 𝑃$ , while 𝑓$ is treated；value z. In the backward pass, it as a constant；approximates the gradient.；Jang E, Gu S, Poole B. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. Routing algorithms — BASE routing

* **定义 / 内容：** Routing algorithms — BASE routing。• Advantages: treat token-to-expert routing as a；linear assignment problem. It provides strict；load balancing across experts；• Limitations: BASE routing unsuitable for；autoregressive decoding, where tokens are；generated one by one. Solving the assignment；problem is computationally expensive.；Lewis M, Bhosale S, Dettmers T, et al. Base layers: Simplifying training of large, sparse models. International Conference on Machine Learning. PMLR, 2021: 6265-6274.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 18. Routing algorithms — BASE routing

* **定义 / 内容：** Routing algorithms — BASE routing。Linear assignment problem；• ℎ& : the representation of token t；• 𝑤'# : the parameter of expert assigned to token t；• 𝑎& : the expert selected for token t；• T: the total number of tokens in the batch；• E: the number of experts；• This problem maximizes the sum of token–；• 𝕝'# ,- : an indicator function, equal to 1 if token t；expert matching scores；is assigned to expert e, and 0 otherwise.；• The constraint requires every expert to receive；exactly T/E tokens. Therefore, all experts handle；the same number of tokens；Lewis M, Bhosale S, Dettmers T, et al. Base layers: Simplifying training of large, sparse models. International Conference on Machine Learning. PMLR, 2021: 6265-6274.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 19. Routing algorithms — Reinforcement learning

* **定义 / 内容：** Routing algorithms — Reinforcement learning。• In Top-k routing, expert selection is a discrete；decision. In practice, training usually updates only；the softmax gate values, not the discrete expert；assignment itself.；• RL-based routing addresses this by treating routing；as a policy learning problem:；o the router is the agent；o expert selection is the action；o task performance is the reward；• Limitations: the reward signal is often sparse and；training usually has high variance；Zuo S, Liu X, Jiao J, et al. Taming sparsely activated transformer with stochastic experts. arXiv preprint arXiv:2110.04260, 2021.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 20. Building MoE LLMs

* **定义 / 内容：** Building MoE LLMs。Sparse Upcycling；Copy FFNs to Parameters；form experts increased；Sparse；transformation；Split FFNs to Parameters；form experts unchanged；Sparse Splitting
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. Building MoE LLMs—Sparse Upcycling

* **定义 / 内容：** Building MoE LLMs—Sparse Upcycling。• Advantages: it can reuse the pretrained；knowledge of an dense model. • To build a Mixture-of-Experts (MoE) model,；"recycling" a pretrained dense model is；• Limitations: the total number of parameters；more economical and effective than；increases by a factor of N, leading to higher；starting from zero.；storage and communication costs；Komatsuzaki A, Puigcerver J, Lee-Thorp J, et al. Sparse upcycling: Training mixture-of-experts from dense checkpoints. arXiv preprint arXiv:2212.05055, 2022.
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. Sparse Upcycling-Mixtral MoE

* **定义 / 内容：** Sparse Upcycling-Mixtral MoE。• Mixtral 8x22B (only 39B active；parameters) outperforms much；larger dense models like Command；R+ (104B) and LLaMA 2 70B across；nearly all benchmarks.；• Mixtral 8x7B achieves stronger results；than LLaMA 2 70B on several tasks；Jiang A Q, Sablayrolles A, Roux A, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. Sparse Upcycling-Qwen MoE

* **定义 / 内容：** Sparse Upcycling-Qwen MoE。*A2.7B=2.7B active parameters；• Upcycled from Qwen-1.8B, 14.3B parameters in total and 2.7B A remarkable reduction of 75% in training；activated parameters；Inference speed up by 174%；• It use shared 4 experts and routing experts 60 experts, choose 4；https://qwen.ai/blog?id=qwen-moe
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 24. Building MoE LLMs—Sparse Splitting

* **定义 / 内容：** Building MoE LLMs—Sparse Splitting。• Advantages: the total number of；parameters remains unchanged, while the；number of activated parameters during；inference is reduced, leading to higher；computational efficiency.；• Limitations: the expressive capacity of each；expert is limited, as each expert has only；1/N of the original FFN capacity；Zhu T, Qu X, Dong D, et al. Llama-moe: Building mixture-of-experts from llama with continual pre-training. Proceedings of the 2024 conference on；empirical methods in natural language processing. 2024: 15913-15923.
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 25. models of similar scale

* **定义 / 内容：** models of similar scale。Building MoE LLMs—Sparse Splitting；Start from LLaMA-2-7B；Split into multiple experts；Baselines with similar number of active params；Zhu T, Qu X, Dong D, et al. Llama-moe: Building mixture-of-experts from llama with continual pre-training. EMNLP. 2024: 15913-15923.
* **直觉：** 本页展开 MoE、expert routing、routing collapse、Top-1/Top-2/BASE/RL routing、DeepSeek、sparse upcycling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 26. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (T/F): creating multiple experts from an existing LLM can be done by；copy the original feed-forward network for several times.；• Q2 (T/F): an MoE can select only 1 out of multiple experts per token.；• Q3 (T/F): if an expert is not selected for generating a token 𝑤E during；training, that expert won’t be used in generating future tokens starting；from 𝑤E；• A: T; F; F
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 23: Diffusion Models：Forward Process、Reverse Process 与 DDPM

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 23；GPT: generative pretrained transformer(only?)；Generative Models Overview；Iterative Refinement；The Forward Stochastic Process；Mathematical Formulation of the Forward Process；Analyzing Stochastic Degradation；Variance Schedules
  * Deriving a key property of forward noising；Bayesian Inference for the Reverse Process；Objectives and Training；Objectives and Training；Bayesian Challenges；Tractable Posterior via Clean data Conditioning；Deriving the Tractable Posterior Mean；Substituting to derive
  * Deriving the Reverse Mean；The Simplified Loss；U-Net and Time Embeddings；Training Algorithm；Sampling Algorithm - Generation；Case Study - CIFAR-10 Progressive Generation；Quiz

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed DDPM Diffusion Notes

### 1. Diffusion 的生成观

Diffusion model 把生成分成两个过程：

* **Forward process**：逐步加噪，把真实数据变成接近 Gaussian noise。
* **Reverse process**：学习逐步去噪，从 noise 生成数据。

它不是一次性生成，而是 iterative refinement。

### 2. Forward Stochastic Process

常见 DDPM forward step：

$$
q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

定义：

$$
\alpha_t=1-\beta_t
$$

$$
\bar{\alpha}_t=\prod_{s=1}^t\alpha_s
$$

Gaussian 闭包给出直接采样公式：

$$
q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
$$

等价写法：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

variance schedule $\beta_t$ 控制每一步加多少噪声。

### 3. Reverse Process 为什么难

真实 reverse distribution：

$$
q(x_{t-1}|x_t)
$$

需要数据边缘分布，难以直接计算。但如果额外知道 clean data $x_0$，posterior：

$$
q(x_{t-1}|x_t,x_0)
$$

是 tractable Gaussian。训练时我们知道 $x_0$，所以能推导出可学习目标；推理时不知道 $x_0$，于是用神经网络预测噪声或 reverse mean。

### 4. 从 ELBO 到 Simplified Loss

diffusion 最大化 log-likelihood 的 variational lower bound (ELBO)。DDPM 推导中，固定 variance 后，高斯 KL 可化为 mean 的 L2 距离。进一步发现预测噪声效果好，于是常用 simplified loss：

$$
L_{simple}=
\mathbb{E}_{t,x_0,\epsilon}
\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2
$$

这就是“训练 diffusion model 可归约为预测噪声”的考试重点。

### 5. Reverse Mean

如果模型能预测噪声 $\epsilon_\theta(x_t,t)$，就能估计 $x_0$：

$$
\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}}
$$

再用 posterior mean 公式得到反向采样均值。直觉是：模型不是直接画图，而是在每个噪声级别估计“当前噪声是多少”，然后把它减掉。

### 6. U-Net、Time Embedding、Training/Sampling

U-Net 通过 downsample/upsample 和 skip connections 同时保留细节与全局结构。time embedding 告诉模型当前噪声级别 $t$，因为不同 $t$ 的 denoising 难度和策略不同。

训练算法：

1. 采样 clean data $x_0$。
2. 采样 timestep $t$。
3. 采样 noise $\epsilon$。
4. 构造 $x_t$。
5. 训练 $\epsilon_\theta(x_t,t)$ 预测 $\epsilon$。

采样算法：

1. 从 $x_T\sim\mathcal{N}(0,I)$ 开始。
2. 对 $t=T,\ldots,1$ 逐步用模型预测噪声并采样 $x_{t-1}$。
3. 得到 $x_0$。

### 7. Exam Focus

* Gaussian 之和仍是 Gaussian。
* inference 从随机噪声开始，不需要知道真实 $x_0$。
* simplified loss 是预测 $\epsilon$ 的 MSE。
* time embedding 是必要条件，因为模型必须知道当前噪声强度。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 23

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 23，主题为 Diffusion Models：Forward Process、Reverse Process 与 DDPM。
* **直觉：** 确认 Lecture 23 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. GPT: generative pretrained transformer(only?)

* **定义 / 内容：** GPT: generative pretrained transformer(only?)。• Generation can be done not just by transformer, but other models.
* **直觉：** 本页文字较少或主要依赖图示；知识点按页标题、可见文字和前后页面上下文保留。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. Generative Models Overview

* **定义 / 内容：** Generative Models Overview。When evaluating generative frameworks, we encounter the "Generative Trilemma".；sample fidelity, fast sampling, and mode coverage (diversity).；• GANs: optimize；D represents the Discriminator, and G is the Generator；(mapping noise z to data space x).；• VAEs: maximize ，；is the Encoder (mapping data to a latent distribution)， is the；Decoder (reconstructing data from the latent space), is the；Prior Distribution of the latent variables.；• Diffusion Models: utilize a learnable reverse Markov chain；, represents the；learned Gaussian transition kernel at each denoising step.；Diffusion model has multiple steps of iterative sampling, while GAN/VAEs generate sample in 1 step.；Image source: https://javiersolisgarcia.com/posts/ddpm/
* **直觉：** 本页展开 diffusion 的 forward stochastic process、variance schedule、reverse process、posterior、simplified loss、U-Net 与 sampling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Iterative Refinement

* **定义 / 内容：** Iterative Refinement。Diffusion models introduce iterative refinement. Instead of one giant leap from a prior to the；data distribution, diffusion models break the generation process into many small, reversible；steps. By decomposing the transformation into a Markov chain of T steps (e.g., 1000) each；step becomes tractable and easier for neural networks (such as U-Net) to learn.；• VAE: one large step of transformation that can be hard for a neural network to fit for diverse samples.；• Diffusion: many small steps of transformation that can be easiers for a neural network to fit.；each step done；by a neural network Image source: https://javiersolisgarcia.com/posts/ddpm/
* **直觉：** 本页展开 diffusion 的 forward stochastic process、variance schedule、reverse process、posterior、simplified loss、U-Net 与 sampling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. The Forward Stochastic Process

* **定义 / 内容：** The Forward Stochastic Process。Physical Intuition: Thermodynamics and Entropy；• Diffusion Models stems from non-equilibrium thermodynamics.；• Consider a drop of ink (low entropy) diffusing in a container of water; the system naturally evolves；toward a state when the ink is uniformly dispersed (high entropy).；• Simulate this forward process by systematically adding Gaussian noise to the current state xt-1 to get；the next state xt , eventually transforming data into isotropic noise (standard Gaussian, which is the；distribution with the highest entropy given mean and standard deviation).；xt-1 xt Markovian Degradation: The forward process；acts as a Markovian information decay,；incrementally replacing structured signal；(something informative, such as；location/shape/scale) with isotropic uncertainty；(something non-informative).；Image source: https://www.sciencefacts.net/diffusion.html
* **直觉：** 本页展开 diffusion 的 forward stochastic process、variance schedule、reverse process、posterior、simplified loss、U-Net 与 sampling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Mathematical Formulation of the Forward Process

* **定义 / 内容：** Mathematical Formulation of the Forward Process。The forward process q is a predefined Markov chain that adds Gaussian noise；according to a variance schedule . The transition kernel is defined as:；(If 𝛽! → 0, it collapse to just 𝑥!"# )；• As t increases, the original structure of x0 is dispersed. The term acts as a scaling；factor to prevent the variance from exploding during the noise injection steps (sudden；changes are harder for a neural network to learn).；• Visualizing the distribution q(xt | xt-1) in 2D:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Analyzing Stochastic Degradation

* **定义 / 内容：** Analyzing Stochastic Degradation。In the forward process, the original signal x0 is systematically attenuated. As ,；the influence of x0 vanishes, and the distribution converges to a standard normal；distribution , regardless of the initial input x0 .；This process contains no learnable parameters; it serves as a hypothetical “destruction schedule” that；provides the intuition for a learnable reverse process (learn to generate x0 from standard normal).
* **直觉：** 本页展开 diffusion 的 forward stochastic process、variance schedule、reverse process、posterior、simplified loss、U-Net 与 sampling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Variance Schedules

* **定义 / 内容：** Variance Schedules。Define and its cumulative product .；The choice of and thus is crucial.；• Linear schedule: the signal drops to zero too early.；• Cosine schedule: preserves the signal for longer and；avoids the "abrupt collapse" near the end.
* **直觉：** 本页展开 diffusion 的 forward stochastic process、variance schedule、reverse process、posterior、simplified loss、U-Net 与 sampling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Deriving a key property of forward noising

* **定义 / 内容：** Deriving a key property of forward noising。expand by substituting the expression；Substituting this into the equation for；The sum of two independent Gaussians is also a Gaussian:；So the two noise terms are:；that’s why the squared roots；in the noise schedule；The combined variance is:；Thus, we can rewrite the two noise terms as a single Gaussian；We obtain:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 10. Bayesian Inference for the Reverse Process

* **定义 / 内容：** Bayesian Inference for the Reverse Process。Data generation process is to reverse the forward process .；Learn a model 𝜃 to take a noisy sample 𝑥! and estimate what it looked like 𝑥!"# one；step earlier. Since each noising step is infinitesimal (a tiny step of change), a reverse；transition can also be modeled as a Gaussian distribution, where the network predicts；the noise’s mean and variance .
* **直觉：** 本页展开 diffusion 的 forward stochastic process、variance schedule、reverse process、posterior、simplified loss、U-Net 与 sampling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. Objectives and Training

* **定义 / 内容：** Objectives and Training。1. We want to maximize the likelihood of the data x0 :；2. The marginal likelihood is intractable. Introduce a variational distribution q(x1:T | x0)；and apply Jensen’s Inequality to get a upper bound:；3. Define the Variational Upper Bound:；4. Expand the terms:；5. Group terms into KL Divergences:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 12. Objectives and Training

* **定义 / 内容：** Objectives and Training。Objective Function: Variational Lower Bound；We optimize the model by maximizing the Variational Lower Bound (ELBO) of the log-；likelihood.
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 13. Bayesian Challenges

* **定义 / 内容：** Bayesian Challenges。The true reverse distribution is intractable because it requires knowledge of；the marginal distribution , which involves an integral over the entire data space.；We cannot compute this directly. Instead, we use a neural network to approximate；this unknown distribution. This challenge is the central motivation for the training；objectives we will derive next.
* **直觉：** 本页展开 diffusion 的 forward stochastic process、variance schedule、reverse process、posterior、simplified loss、U-Net 与 sampling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Tractable Posterior via Clean data Conditioning

* **定义 / 内容：** Tractable Posterior via Clean data Conditioning。is intractable but the posterior conditioned on x0 , is tractable!；By using x0 as a reference, the reverse step can be solved using Bayes' Rule as a；product of known Gaussian forward kernels:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 数据决定模型学到什么，复习时要同时关注数据来源、构造方式、质量控制和潜在偏差。

### 15. Deriving the Tractable Posterior Mean

* **定义 / 内容：** Deriving the Tractable Posterior Mean。We want to find . Using Bayes' Rule and the Markov property:；Substituting the Gaussian forms (ignoring constants that do not depend on ):；Expanding the squares and collecting terms for :；For a Gaussian ,the exponent is .Thus, the mean is the ratio of the；coefficient of to the coefficient of :；Time-dependent interpolation between 𝑥! and 𝑥"
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 16. Substituting to derive

* **定义 / 内容：** Substituting to derive。In the reverse process (inference), is unknown. We use the forward marginal formula to；express in terms of and the noise :；Substitute this into the formula for :；Combining the terms for :；For the noise term:；Factoring out :
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 17. Deriving the Reverse Mean

* **定义 / 内容：** Deriving the Reverse Mean。By expanding the Gaussian exponents and completing the square, we find that the；mean c. of the tractable posterior is a linear combination of xt and x0:；During inference, x0 is unknown. However, we can express x0 in terms of xt and the；noise 𝜀 added to it. Substituting this into the mean formula reveals a striking result:；predicting the reverse mean is mathematically equivalent to predicting the noise 𝜀 :
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 18. The Simplified Loss

* **定义 / 内容：** The Simplified Loss。Ho et al. (2020) discovered that the KL Divergence between two Gaussians with fixed；variances simplifies to an L2 distance between their means. Crucially, they found that；dropping the weighting coefficients yields even better sample quality:；Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion；probabilistic models." NIPS 2020: 6840-6851.
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 19. U-Net and Time Embeddings

* **定义 / 内容：** U-Net and Time Embeddings。The standard architecture for diffusion model is a U-Net with skip connections.；• Skip connections allow the network to preserve fine-grained spatial details while；understanding global structure through the bottleneck.；• Since the denoising logic depends on the current noise level, we must inject a Time；Embedding (often sinusoidal) into the network to condition its predictions on t.
* **直觉：** 本页展开 diffusion 的 forward stochastic process、variance schedule、reverse process、posterior、simplified loss、U-Net 与 sampling。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. Training Algorithm

* **定义 / 内容：** Training Algorithm。The training procedure is efficient: each iteration only requires one forward noising step.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 21. Sampling Algorithm - Generation

* **定义 / 内容：** Sampling Algorithm - Generation。The sampling (reverse) procedure: each iteration requires one backward denoising step.
* **直觉：** 本页描述具体算法流程或工程技术，需要掌握输入、输出、优化目标和主要 trade-off。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 22. Case Study - CIFAR-10 Progressive Generation

* **定义 / 内容：** Case Study - CIFAR-10 Progressive Generation。Observing the progressive generation on CIFAR-10 reveals how the model prioritize features.
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 23. Quiz

* **定义 / 内容：** Quiz。1. Pen and paper; no compute or cellphone allowed.；2. Turn in your answer sheet when you leave the classroom.；• Q1 (T/F): training a diffusion model can be reduced to letting a neural；network predict the noise 𝜖 added to 𝑥% to obtain 𝑥& .；• Q2 (T/F): the sum of two Gaussian variables is another Gaussian variable.；• Q3 (T/F): during inference time of a diffusion model, the starting point 𝑥%；should be known as the the distribution requires it.；• A: T; T; F
* **直觉：** 本页是课堂测验页，保留题目和答案线索，用于复习考试重点。
* **为什么重要：** 这部分常直接变成判断题、选择题或短答题；复习时要把题目背后的概念关系说清楚。
* **易错点：** quiz 页往往考最小概念差异，例如是否是同一个概率、是否需要归一化、复杂度是否来自 DP 而不是 greedy。

# Lecture 24: Score-Based Diffusion、Text Diffusion 与 Block Diffusion

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 24；Score-Based Perspective: The Score Function；Score-Based Perspective: The Score Function；Score-Based Perspective: The Score Function；Deriving；Langevin Dynamics for inference (reverse process)；Unified Framework: Denoising vs. Score Matching；Why Diffusion for NLP?
  * Advanced Diffusion: From Images to Text；Diffusion of text samples；Need to be careful about text perturbations；From Gaussian Kernels to Transition Matrices；How to Forward Step Works；The Forward Posterior in Discrete Space；Diffusion in Continuous Latent Spaces；Solution
  * How to control our text generation?；Deriving；How to train a classifier；Experiment；Case Study: Block Diffusion；Autoregressive vs. Diffusion

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Score / Text Diffusion Notes

### 1. Score Function

score-based modeling 学的是数据分布 log-density 的梯度：

$$
score(x)=\nabla_x\log p(x)
$$

这个向量指向概率密度上升最快的方向，也就是把 noisy sample 推回高概率数据流形的方向。denoising 可以理解成沿 score field 往真实数据分布移动。

### 2. Langevin Dynamics

如果已知 score，可从随机点开始迭代采样：

$$
x_{k+1}=x_k+\eta\nabla_x\log p(x_k)+\sqrt{2\eta}z_k
$$

其中 $z_k\sim\mathcal{N}(0,I)$。第一项把样本推向高密度区域，第二项保留随机探索。

实际不知道 $p(x)$，所以训练神经网络 $s_\theta(x)$ 近似 score。

### 3. Denoising Score Matching

直接学习 $\nabla_x\log p(x)$ 很难。score matching 通过给 clean data 加噪得到 $x_t$，条件分布的 score 可计算：

$$
\nabla_{x_t}\log q(x_t|x_0)
$$

对 Gaussian perturbation：

$$
q(x_t|x_0)=\mathcal{N}(x_0,\sigma^2I)
$$

有：

$$
\nabla_{x_t}\log q(x_t|x_0)
=-\frac{x_t-x_0}{\sigma^2}
$$

于是训练目标可让 $s_\theta(x_t,t)$ 拟合这个方向。它和 DDPM 预测噪声本质相通。

### 4. 为什么 NLP 上 diffusion 更难

图像像素是连续变量，有自然距离和局部结构；文本 token 是离散 vocabulary：

$$
w_i\in V
$$

vocabulary 中 token 没有天然几何距离。随便替换 token 可能破坏语法和语义。因此 text diffusion 要么在 discrete token space 设计 transition matrix，要么把 token 映射到 continuous embedding/latent space。

### 5. Discrete Text Diffusion

离散扩散用 transition matrix 替代 Gaussian kernel：

$$
x_t=Q_tx_{t-1}
$$

其中 $x_t$ 可是 token distribution / one-hot vector。常见 forward step：

* **Keep**：保留原 token。
* **Mask**：替换为 `[MASK]`，absorbing state。
* **Swap**：替换成 vocabulary 中另一个 token。

训练 reverse model 时，用 Bayes rule 写 forward posterior，再用 categorical cross-entropy 学习反向去噪。

### 6. Continuous Latent Diffusion for Text

Diffusion-LM 把 token sequence 映射到 continuous embedding sequence：

$$
w_i\mapsto e_i\in\mathbb{R}^d
$$

然后在 embedding space 做 Gaussian diffusion。这样有距离和方向，适合 gradient-based denoising。最后需要把 continuous vectors rounding / decoding 回 discrete tokens。

### 7. Controllable Generation

如果想控制文本属性 $c$，目标是从 posterior 生成：

$$
p(x|c)\propto p(c|x)p(x)
$$

在 diffusion 每一步中，可以用 classifier 或 discriminator 提供梯度，引导 sample 满足控制条件。课件中的 FUDGE 和 controlled text generation 体现了这种思路。

### 8. Block Diffusion

纯 discrete diffusion 可能固定长度，且不容易像 autoregressive model 一样复用 KV cache。Block Diffusion 在 AR 和 diffusion 之间折中：按 block 生成或修正，试图结合 diffusion 的 self-correction 和 AR 的可扩展推理。

### 9. Exam Focus

* score function 是 $\nabla_x\log p(x)$。
* text diffusion 的难点是 token 离散、没有自然距离。
* discrete diffusion 用 transition matrix；continuous latent diffusion 用 embedding space。
* diffusion 相比 AR 的潜在优势是 whole-sequence refinement 和 self-correction，但系统实现更复杂。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 24

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 24，主题为 Score-Based Diffusion、Text Diffusion 与 Block Diffusion。
* **直觉：** 确认 Lecture 24 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Score-Based Perspective: The Score Function

* **定义 / 内容：** Score-Based Perspective: The Score Function。An alternative view of Diffusion comes from Score-based modeling. Instead of probability；densities, we learn the Score Function:；This vector field points toward the high-density regions of the data distribution (the "peaks").；Denoising is essentially the act of moving a noisy sample back toward the high-probability；manifold.；Song, Yang, and Stefano Ermon. "Generative modeling by estimating gradients of the data distribution.” NIPS 2019
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. Score-Based Perspective: The Score Function

* **定义 / 内容：** Score-Based Perspective: The Score Function。Langevin Dynamics describes the trajectory of a particle x moving in a potential field U(x) = -；log p(x) towards the point with the lowest potential, with stochastic thermal fluctuations.；Equivalent to maximize the p(x),；which can be the real data distribution,；thus generating realistic data.；In practice, we do not know p(x).；Train a neural network to approximate the score function:
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Score-Based Perspective: The Score Function

* **定义 / 内容：** Score-Based Perspective: The Score Function。To solve the “unknown score” issue, we perturb the data x0 with noise to create xt . The log-gradient of；the conditional distribution is easy to calculate:；The gradient of the conditional density provides an unbiased estimate of the gradient of the marginal；density.；Denoising Score Matching:
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Deriving

* **定义 / 内容：** Deriving。In score-based modeling and diffusion processes,；This implies that the conditional distribution:；So:；The first term is a constant with respect to xt；Obtaining the score function:；Unbiasedness, refer to [Vincent, P.]；Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. Neural Computation.
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 6. Langevin Dynamics for inference (reverse process)

* **定义 / 内容：** Langevin Dynamics for inference (reverse process)。Starting from a random location 𝑥 ! , use the trained score function 𝑠" (𝑥# ) to iteratively move 𝑥 ! to 𝑥%$
* **直觉：** 本页文字较少或主要依赖图示；知识点按页标题、可见文字和前后页面上下文保留。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Unified Framework: Denoising vs. Score Matching

* **定义 / 内容：** Unified Framework: Denoising vs. Score Matching。本页主要用于课程衔接、图示或标题说明。
* **直觉：** 图表把 diffusion model 和 score matching 对齐：两者都包含 sampling/generation、denoising step、key connection 与 training objective。diffusion 可看作学习 reverse diffusion step，score matching 可看作学习 score field；二者在目标上都让模型沿着从噪声回到数据分布的方向移动。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 8. Why Diffusion for NLP?

* **定义 / 内容：** Why Diffusion for NLP?。本页主要用于课程衔接、图示或标题说明。
* **直觉：** 图中对比 autoregressive generation 与 diffusion generation。自回归模型从左到右逐 token 决策，早期错误可能级联影响后续；diffusion 通过 whole-sequence refinement 反复修正整段文本，理论上能先建立全局语义再修局部语法，具有 self-correction 潜力。
* **为什么重要：** 这是引入新方法的原因，复习时要能说明旧方法哪里不够，以及新方法解决了哪个痛点。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 9. Advanced Diffusion: From Images to Text

* **定义 / 内容：** Advanced Diffusion: From Images to Text。Treating with image pixels as continuous sentence=[w1, w2 ,w3], wi∈ 𝑉, 𝑖 = 1,2,3；variables are easy: the distance metric and；ordering are well-defined. There is no order or distance metric defined on 𝑉；But languages, on the other hand, are not；too much different from images.；These are discrete values too.
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Diffusion of text samples

* **定义 / 内容：** Diffusion of text samples。本页主要用于课程衔接、图示或标题说明。
* **直觉：** 图示把文本 diffusion 表示为 token 序列的逐步加噪与反向去噪：forward diffusion 从真实句子逐渐替换/扰动 token，reverse diffusion 从 noisy/generated text 逐步恢复目标文本。它强调文本 diffusion 的状态是离散 token 或其表示，而不是连续像素。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 11. Need to be careful about text perturbations

* **定义 / 内容：** Need to be careful about text perturbations。本页主要用于课程衔接、图示或标题说明。
* **直觉：** 图中说明 text perturbation 不能随意做：token 可替换成 valid word、unknown/mask、random word 等，但不同扰动会改变语法和语义。embedding-space 可视化也提示语义相近词在向量空间接近，因此 continuous latent diffusion 需要利用 embedding 的距离结构，而 discrete token diffusion 需要设计合理 transition。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。
* **补充说明：** 原页主要依赖图示或标题，本笔记已在本讲前面的主题解释中补足上下文。

### 12. From Gaussian Kernels to Transition Matrices

* **定义 / 内容：** From Gaussian Kernels to Transition Matrices。To define diffusion over a discrete vocabulary V, we replace the Gaussian kernel with a；Transition Matrix .；Instead of shifting a mean, we define the probability of a token xt-1 "jumping" to another token；xt. The forward process is now a categorical Markov chain:；𝑥# = 𝑄# 𝑥#%& : transiting from a probability distribution 𝑥#%& over tokens to another distribution 𝑥# .；Examples:
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. How to Forward Step Works

* **定义 / 内容：** How to Forward Step Works。Given current token xt-1 with one-hot vector xt-1；Three kinds of transitions:；Keep:；The token remains unchanged at its current position.；In the early stages of the forward process (close to t=0), most；tokens are "kept" to maintain the original data structure.；Represents the probability that a token survives the；transition step without being corrupted.；Mask:；The token is replaced by a special [MASK] token, which is；unknown to the model.；This is the most popular form of discrete diffusion, often referred；to as "absorbing" because the [MASK] state acts as a sink.；Swap:；The original token is replaced by another token chosen from the；vocabulary, either uniformly or according to a pre-defined；probability matrix.；This introduces noise by replacing valid data with random；incorrect data, sometimes referred to as "uniform diffusion".
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. The Forward Posterior in Discrete Space

* **定义 / 内容：** The Forward Posterior in Discrete Space。1. Forward Process in Discrete Space: 5. Training the Reverse Model:；We train a model to approximate the true；reverse posterior；2. We define: .；3. Forward Posterior via Bayes’ Rule:；based on variational lower bound from the last lecture.；4. In terms of transition matrices:；For discrete distribution, this KL is minimized；by the Category Cross-Entropy loss:；6. We can train a categorical predictor to；reverse the corruption process.
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Diffusion in Continuous Latent Spaces

* **定义 / 内容：** Diffusion in Continuous Latent Spaces。When the vocabulary is large, discrete diffusion becomes computationally expensive；(matrices are V x V). Diffusion-LM takes a different path: map tokens to a continuous；embedding space (with order and metric!) and apply Gaussian Diffusion.；Diffusion-LM iteratively denoises a sequence of Gaussian vectors into word vectors, yielding a；intermediate latent variables of decreasing noise level {X T, ,,, Xo}；Li, Xiang, et al. "Diffusion-lm improves controllable text generation." NIPS (2022): 4328-4343.
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. Solution

* **定义 / 内容：** Solution。Suppose we have a sequence of words: . An embedding function maps；each word into a vector: .；Thus, the entire sequence is encoded into:；Loss function:；diffusion Loss:；] rounding to discrete tokens using softmax :；words with the same part-of-speech tags；(syntactic role) tend to be clustered
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. How to control our text generation?

* **定义 / 内容：** How to control our text generation?。We can now generate the correct text, but how do we generate the text we want?；We iteratively perform gradient updates on these continuous embeddings to optimize for；fluency and satisfy control requirements c (parametrized by a classifier).；Controlling is equivalent to decoding from the posterior；and we decompose this joint inference problem to a sequence of control problems at each；diffusion step:；(Bayes Theorem)
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Deriving

* **定义 / 内容：** Deriving。Controlling is equivalent to decoding from the posterior；and we decompose this joint inference problem to a sequence of control problems at each；diffusion step:；To simplify the second term, we adopt conditional independence assumptions from prior；work on controlling diffusions:；To perform gradient-based optimization, we take the logarithm of both sides:；So:
* **直觉：** 本页给出目标函数、推导或数学定义，是理解训练/推理算法的核心。
* **为什么重要：** 这是算法或训练目标的核心，复习时要能写出公式、解释每一项含义，并说明它为什么可以降低复杂度或优化模型。

### 19. How to train a classifier

* **定义 / 内容：** How to train a classifier。A labeled dataset consisting of text sequences w and their corresponding；control labels c.；Classifier
* **直觉：** 本页文字较少或主要依赖图示；知识点按页标题、可见文字和前后页面上下文保留。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 20. Experiment

* **定义 / 内容：** Experiment。1. This represents two sentences (clauses) joined by a single；connector word (*). The first clause requires a single-word；subject (NP *) and a deeply nested verb phrase. The second；clause requires a subject that is precisely three words long (NP；* * *) followed by a verb and an adjective phrase.；2. The notation includes an SBAR, which requires the model to；generate a relative clause.；FUDGE: Controlled text generation with future discriminators. ACL 2021；FT Fine-tuning Oracle
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 21. Case Study: Block Diffusion

* **定义 / 内容：** Case Study: Block Diffusion。Discrete diffusion models are limited by；fixed-length generation and the inability；to reuse computation via KV-caching.；Block Diffusion introduces a hybrid；paradigm that interpolates between；Autoregressive (AR) and Diffusion models.；Arriola, Marianne, et al. "Block diffusion: Interpolating between autoregressive and diffusion language models.”ICLR 2025.
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 22. Autoregressive vs. Diffusion

* **定义 / 内容：** Autoregressive vs. Diffusion。Unlike GPT, where a single "bad" word can derail the entire sentence (exposure bias), Diffusion；can self-correct. In early steps, it sets the global semantics; in later steps, it fixes local syntax.
* **直觉：** 本页展开 score function、Langevin dynamics、score matching、text diffusion、discrete transition matrix、latent diffusion、classifier guidance 与 block diffusion。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

# Lecture 25: Agentic Systems：Memory、RAG、Reflection、Tools 与 MCP

## Part I: Lecture Map

* **本讲覆盖路径：**
  * AIAA 4051 Lecture 25；Everyone is talking about agent；What is an agentic system；Comparing LLM and agentic systems；Agentic system examples；Agentic system for robotics；Agentic system for electricity trading；Comparing agentic system and computers
  * Context engineering；Memory；Memory examples in Codex；RAG (Retrieval-Augmented Generation)；Reflection；Reflection: LLM-as-a-judge；Tools；Tools
  * Tools；Collaboration

* **阅读方式：** 先读 Part II 的详细主题解释，再用 Part III 对照课件覆盖点查漏补缺。

## Part II: Detailed Agentic System Notes

### 1. Agentic System 是 LLM 加上可行动的循环

plain LLM 通常是一次输入、一次输出。agentic system 把 LLM 放进一个反复执行的 loop：

1. 读取任务和环境状态。
2. 规划下一步。
3. 调用工具或输出动作。
4. 观察结果。
5. 反思并修正。

所以 agent 不只是 model，而是 LLM + memory + tools + environment + feedback + control flow。

### 2. LLM vs Agentic System

LLM 更像语言模型本体，擅长根据上下文生成文本。agentic system 面向任务完成，常需要：

* 多步 planning。
* 文件/网页/API/代码工具。
* 长期 memory。
* error recovery。
* self-check 或 external judge。
* 与人或其他 agent 协作。

课件中的 robotics、电力交易等例子说明：agent 的 action 会改变外部世界，不只是写回答。

### 3. Context Engineering and Memory

context engineering 是决定把什么信息放入 prompt/context：

* 当前任务目标。
* 相关文件或检索片段。
* 用户偏好。
* 工具输出。
* 历史决策和约束。

memory 可分为短期上下文和长期记忆。长期 memory 保存偏好、项目背景、常用路径、错误经验等；但 memory 也可能过期或错误，所以需要检索、更新和冲突处理。

### 4. RAG

**Retrieval-Augmented Generation (RAG)** 用外部文档补充模型知识：

1. **Indexing**：切块、embedding、建索引。
2. **Retrieval**：根据 query 找 top-$k$ chunks。
3. **Generation**：把 chunks 放进 context，让模型基于证据回答。

RAG 的价值：

* 更新知识。
* 降低 hallucination。
* 提供可追踪依据。
* 让模型访问私有文档。

RAG 的风险：

* 检索不到关键文档。
* 检索到相似但错误的文档。
* context 太长导致关键信息被淹没。

### 5. Reflection and LLM-as-a-judge

reflection 让 agent 检查自己的计划和输出。它可以发现格式错误、遗漏步骤、代码失败、事实不一致。

LLM-as-a-judge 是常见反思机制：让另一个或同一个模型评价回答质量。但 judge 也可能偏见、过度自信或被 prompt 误导，所以对高风险任务要结合规则、测试和外部验证。

### 6. Tools

tools 让模型把语言决策变成外部动作，例如：

* 执行代码。
* 搜索资料。
* 读写文件。
* 调用数据库或 API。
* 操作浏览器/GUI。

工具调用的核心不是“模型会说要做什么”，而是系统真的执行动作并返回 observation。agent 的可靠性取决于工具权限、错误处理和验证闭环。

### 7. MCP and Collaboration

MCP 试图给模型和外部服务提供统一连接协议。若有 $N$ 个模型和 $M$ 个工具/服务，逐一适配需要：

$$
N\times M
$$

通过统一协议，可降为：

$$
N+M
$$

collaboration 则强调 agent 与人类或其他 agent 协作：人类给目标、约束和反馈，agent 做检索、执行、检查和迭代。

### 8. Exam Focus

* agentic system 不等于单次 LLM completion。
* RAG 是 retrieval + generation，不是 fine-tuning。
* reflection 可改善质量，但 judge 本身也会错。
* MCP 的核心价值是统一连接，降低模型-工具集成复杂度。

## Part III: Concept Coverage from Lecture Materials

### 1. AIAA 4051 Lecture 25

* **定义 / 内容：** AIAA 4051 Introduction to NLP，Lecture 25，主题为 Agentic Systems：Memory、RAG、Reflection、Tools 与 MCP。
* **直觉：** 确认 Lecture 25 的课程主题与讲义入口。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 2. Everyone is talking about agent

* **定义 / 内容：** Everyone is talking about agent。• ”Agent” is not a new term, but it’s re-gaining popularity.；Crowd；Universities；Companies Government
* **直觉：** 本页文字较少或主要依赖图示；知识点按页标题、可见文字和前后页面上下文保留。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 3. What is an agentic system

* **定义 / 内容：** What is an agentic system。• Agentic system • LLM as the brain:；o planning；o rethinking；• Memory: persistent perception and；experience (trajectory and feedback).；• Tools: execute actions from LLM over；the environments.；• Environment:；o interact with the agents；o provide state transition and；description；o reward as feedback；"If you're not the model, you're the harness.”；-- from LangChain's Vivek Trivedy
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 4. Comparing LLM and agentic systems

* **定义 / 内容：** Comparing LLM and agentic systems。• Solving complex tasks requires；o Planning: generating solution consisting of multiple steps.；o Remember history: cannot start over but needs to continue from break.；o Reflection: learn from errors and reasoning about mistakes；o Using tools: hands & eyes beyond brains；o Collaboration: complex worlds consist of multiple players.；Agent 1 Agent 2 critic；generating；a plan. goal；• LLM can only: tool Agent 1 reflection；• Passive generation upon user queries: one output per user input.；• Limited context window: cannot remember too much, with memory degradation.；• Forward generate texts.；• Generate language upon input.；• A stand-alone model and no interaction.
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 5. Agentic system examples

* **定义 / 内容：** Agentic system examples。• Complex task requiring；more than one-short；generation.；• Multiple steps of；execution；• Remember execution；history；• Calling tools for；execution；• Learn from outcomes to；change the plan.；Image source: https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 6. Agentic system for robotics

* **定义 / 内容：** Agentic system for robotics。3D scene graphs as a memory of the environment. Navigation requires multiple steps of actions.；Use tools such as cameras, segmentation models.；ConceptGraph: https://concept-graphs.github.io/
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 7. Agentic system for electricity trading

* **定义 / 内容：** Agentic system for electricity trading。• Setting trading price；requires multiple steps.；• Tools include weather；forecasting, grid status；checking, market；demand prediction, etc.；• Remember execution；history and trading rules；and constraints.；• The plan is quite fixed；but parameters can be；adjusted from past；experience.；Image source: Gemini (may contain errors)
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 8. Comparing agentic system and computers

* **定义 / 内容：** Comparing agentic system and computers。• LLM-based agentic system vs. computer system；computing-intensive；faster but limit size；larger size but slower；do what the LLM cannot；a big controlled while loop；agent-defined softwares；https://www.beren.io/2023-04-11-Scaffolded-LLMs-natural-language-computers/；Image source: https://x.com/akshay_pachaar/status/2041146899319971922
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 9. Context engineering

* **定义 / 内容：** Context engineering。• How to optimize the contexts sent to an LLM；o Limited size, lost in the middle (left figure)；o Overloaded context with irrelevant and conflicting information (middle figure)；o Context conflicts with model parametric information (middle figure)；o Rich structrure: images/graphs cannot be fully expressed as just sequences of tokens；(right figure) – better solved using an agent.；Lost in the Middle: How Language Models Use Long Contexts；Luo, 2024, GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 10. Memory

* **定义 / 内容：** Memory。• Reduce training data and computation: organic data are too large and only a；small but high quality data is sufficient to train an LLM.；Define global behaviors Define local behaviors；https://cdn-uploads.huggingface.co/production/uploads/64838b28c235ef76b63e4999/0kvM1n8gmc96N0dGQ9mZH.png；https://cdn-uploads.huggingface.co/production/uploads/64838b28c235ef76b63e4999/QFvOpCHwS_kDgXYlK3NJA.png
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 11. Memory examples in Codex

* **定义 / 内容：** Memory examples in Codex。Memory for resolving reference ambiguity Memory for persistent skills
* **直觉：** 本页通过案例、实验或工业系统说明方法在真实模型和任务中的效果与取舍。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 12. RAG (Retrieval-Augmented Generation)

* **定义 / 内容：** RAG (Retrieval-Augmented Generation)。RAG allows LLM to interact with memory to address the following issues:；• Lack of transparency of information sources；o Mechanical interpretability can find MLP for knowledge storage and recall,；but that’s not deterministic and only based on empirical post-analysis.；• Outdated information embedded in LLM;；o LLM was pre-pretrained on archived large Internet data.；o Me: What’s the gold price today? LLM: ehhh, it is XXX today.；• Hallucination in the generation.；o Due to wrong attention, decoding strategy, or lack of knowledge, LLM tends to hallucinate；RAG has 3 steps:；• Indexing: build a indexed database for documents；• Retrieval: find top-k relevant text chunks from the；database for a user query.；• Generation: insert the text chunks into the context of；an LLM and prompt it to generate the final answers.
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 13. Reflection

* **定义 / 内容：** Reflection。• CoT: a simple prompt addition – let’s think step by step.；• The intermediate steps provide more tokens and context to constrain future；token generation, potentially invoking relevant knowledge from the LLM.；https://storage.ghost.io/c/bc/c6/bcc6be81-dfc2-48d1-b46c-a8c08936114e/content/images/2024/08/image-2.png；https://learnprompting.org/docs/assets/basics/chain_of_thought_example.webp
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 14. Reflection: LLM-as-a-judge

* **定义 / 内容：** Reflection: LLM-as-a-judge。• Use another LLM to evaluate the output of the main LLM.；Image source: https://images.ctfassets.net/otwaplf7zuwf/GAaa71n2hPZ3iX5zLqN78/15cca50b3791774cb0265d628ccb5417/single-output-llm-judge.png
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 15. Tools

* **定义 / 内容：** Tools。• Web search；Recognize when need a web search:；• limited context to understand user query；• beyond the learned konwledge of the LLM；How to do a web search:；• MCP (search engine companies；provide the service).；Process the search results.；Image source: https://substackcdn.com/image/fetch/$s_!CRpe!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-；media.s3.amazonaws.com%2Fpublic%2Fimages%2F905d3033-c2c2-49dd-aa10-1ca1c8033022_2993x1313.png
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 16. Tools

* **定义 / 内容：** Tools。• Coders and interpreter；executing python codes, calling external pdf parsers；file manipulation
* **直觉：** 图示强调 agent 的工具不仅是搜索，还包括 code interpreter 与文件操作：模型可生成并执行 Python、调用 PDF parser、读取/移动/处理文件。这让 agent 能把语言计划转化为可验证的外部动作。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 17. Tools

* **定义 / 内容：** Tools。• GUI tools；Sense the location of a click；Recognize the cell；Action: type in contents；Image source: https://preview.redd.it/cuabot-v1-0-released-an-mit-licensed-tool-to-run-any-gui-v0- LLM-Powered GUI Agents in Phone Automation: Surveying；qaapo5x98ihg1.png?width=640&crop=smart&auto=webp&s=8edbad56a802fb03e703d4508a1d280b54968754 Progress and Prospects. TMLR 2025
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

### 18. Collaboration

* **定义 / 内容：** Collaboration。• MCP: model context protocol, proposed by OpenAI；• Allow a unified interface across N models and M services.；• Reduce interaction patterns from NxM to N+M；https://pbs.twimg.com/media/GorWYgDXgAAWzDI.jpg；https://levelup.gitconnected.com/getting-started-with-model-context-protocol-mcp-a-；beginners-guide-ec61058505b1
* **直觉：** 本页展开 agentic system、planning、memory、RAG、reflection、LLM-as-a-judge、tools、collaboration 与 MCP。 这些核心概念中的一个局部，需与本讲前后页面一起理解。
* **为什么重要：** 复习时不要只记关键词，要能把它放回本讲主线中：它解决什么问题、依赖什么假设、会带来什么代价。

# Formula Cheat Sheet / 公式速查表

## Probability and MLE

$$
P(A|B)=\frac{P(A\cap B)}{P(B)}
$$

$$
P(B|A)=\frac{P(A|B)P(B)}{P(A)}
$$

$$
\hat{\theta}_i=\frac{c_i}{n}
$$

## n-gram

$$
P(w_1,\ldots,w_n)=\prod_i P(w_i)
$$

$$
P(w_i|w_{i-1})=\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
$$

$$
P_L(w_2|w_1)=\frac{c(w_1,w_2)+1}{c(w_1)+|V|}
$$

## HMM

$$
P(Q)=\pi_{q_1}\prod_{t=2}^T a_{q_{t-1},q_t}
$$

$$
P(O|Q)=\prod_{t=1}^T b_{q_t}(o_t)
$$

$$
\alpha_t(j)=\sum_i \alpha_{t-1}(i)a_{ij}b_j(o_t)
$$

$$
\beta_t(i)=\sum_j a_{ij}b_j(o_{t+1})\beta_{t+1}(j)
$$

$$
v_t(j)=\max_i v_{t-1}(i)a_{ij}b_j(o_t)
$$

$$
\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{P(O)}
$$

$$
\xi_t(i,j)=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{P(O)}
$$

## CFG / PCFG

$$
G=(N,\Sigma,R,S)
$$

$$
A\rightarrow BC,\quad A\rightarrow a
$$

$$
CYK=O(n^3|R|)
$$

$$
\beta_A(i,j)=
\sum_{A\rightarrow BC}\sum_{k=i}^{j-1}
P(A\rightarrow BC)\beta_B(i,k)\beta_C(k+1,j)
$$

## Transformer / Training / Systems

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
L_{AR}(\theta)=\sum_D\sum_t-\log p_\theta(w_t|w_{<t})
$$

$$
L_{SFT}(\theta)=-\sum_t\log p_\theta(y_t|x,y_{<t})
$$

$$
P(y_w\succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))
$$

$$
Size_{KV}=Batch\times SeqLen\times2\times Layers\times Heads\times Dim\times2\ bytes
$$

$$
L_{diffusion}=\mathbb{E}_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|^2
$$

$$
score(x)=\nabla_x\log p(x)
$$
