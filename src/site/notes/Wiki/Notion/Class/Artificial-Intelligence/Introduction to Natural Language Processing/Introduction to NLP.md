---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/introduction-to-nlp/"}
---


# AIAA 4051 Introduction to Natural Language Processing

> 覆盖范围：Lecture 1-25。整理方式按主题重组，不保留逐页课件页标题；标题页、demo、conclusion、quiz 中的复习重点已合并到对应主题或考试提示中。

# Lecture 1: 课程介绍、数学基础、概率基础与 MLE

## Part I: Course Big Picture

* **Natural Language Processing (自然语言处理)** 研究如何让计算机处理、建模和生成自然语言。自然语言不只是字符串，它也是人类交流、保存知识文化、组织思维与社会协作的核心媒介。
* **Generative AI (生成式 AI)** 放大了 NLP 的影响：语言本身具有生成性，现代模型可以根据上下文生成回答、翻译、摘要、代码和多模态指令。
* **NLP model as probability distribution (概率分布视角)**：课程中的 n-gram、word vector、text classifier、POS tagger、parser、entity/relation extractor、translator、chatbot 等，都可以看成带参数 $\theta$ 的统计模型，用来定义文本或标签上的概率分布。
* **Typical pipeline (一般训练流程)**：
  1. 收集文本数据。
  2. 构造概率模型。
  3. 写出 likelihood / log-likelihood。
  4. 最大化目标函数得到参数。
  5. 对复杂模型用 SGD、AdaGrad、自动微分和矩阵运算优化。
* **Course logistics / exam hints (课程与考试提示)**：quiz 占比高，课堂核心概念、公式、算法步骤需要即时掌握；课程项目可继续发展为研究、应用、poster 或更大规模实验。

## Part II: Math Foundations

* **Gradient (梯度)** 是多元函数各变量偏导组成的向量：
$$
\nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)
$$
  训练模型就是沿着损失下降方向调整参数。
* **Vector / Matrix / Tensor (向量、矩阵、张量)** 是 NLP 的基本表示结构。词向量、hidden states、attention matrix、神经网络权重都以矩阵形式计算，GPU/TPU 也最擅长这类运算。
* **Norm and Dot Product (范数与点积)**：
  * $L_1, L_2, L_\infty$ 范数衡量向量大小。
  * 点积与夹角相关，可衡量两个向量方向是否接近。
  * 在 embedding 和 Transformer 中，相似 token 往往有相近方向，点积越大，模型越倾向认为二者相关。

## Part III: Probability Foundations

* **Sample Space (样本空间)** $\Omega$ 包含所有可能对象；**Event (事件)** 是 $\Omega$ 的子集；**Random Variable (随机变量)** 把对象映射为数字。
* **Probability Axioms (概率公理)**：
$$
P(\Omega)=1,\quad 0\le P(A)\le 1,\quad P(A\cup B)=P(A)+P(B)-P(A\cap B)
$$
  例如“正面情感词”和“经济相关词”可能有交集，求并集时必须减掉重复计算的交集。
* **Joint Distribution (联合分布)** 列出多个随机变量所有取值组合的概率。完整联合分布包含边缘概率和条件概率所需信息，但在 NLP 中变量极多，完整存储和估计通常不可行。
* **Conditional Probability (条件概率)**：
$$
P(A|B)=\frac{P(A\cap B)}{P(B)}
$$
  表示在已知 $B$ 发生后 $A$ 的概率。
* **Bayes Rule (贝叶斯公式)**：
$$
P(B|A)=\frac{P(A|B)P(B)}{P(A)}
$$
  其中 posterior 是看到证据后的信念，prior 是先验信念，likelihood 是假设成立时看到证据的概率，marginal 是归一化项。
* **Law of Total Probability (全概率展开)**：
$$
P(A)=P(A|B)P(B)+P(A|B^c)P(B^c)
$$
  机器学习中常省略与优化变量无关的分母，但分母保证 posterior 是合法概率。
* **Expectation and Variance (期望与方差)**：
$$
E[X]=\sum_x p(X=x)x
$$
$$
Var[X]=E[(X-E[X])^2]
$$
  期望衡量中心趋势，方差衡量不确定性或分散度。

## Part IV: Common Distributions and Independence

* **Bernoulli Distribution (伯努利分布)** 用于二分类或两种结果，如 coin flip、正/负情感。
* **Binomial Distribution (二项分布)** 是 $n$ 个 i.i.d. Bernoulli 变量之和。
* **Categorical Distribution (类别分布)** 用于一次多类取样，如下一个 token 在词表上的分布。
* **Multinomial Distribution (多项分布)** 是多次 categorical 取样后的类别计数分布。注意 categorical 是“一次抽样”，multinomial 是“多次抽样后的计数”。
* **Independence Assumption (独立性假设)**：
$$
P(X,Y)=P(X)P(Y)
$$
  或
$$
P(X|Y)=P(X)
$$
  NLP 中独立性多是建模假设，不是真实事实。它可以显著降低参数数量，但会牺牲表达能力。

## Part V: Maximum Likelihood Estimation

* **MLE (最大似然估计)** 的目标是找参数 $\theta$，使观察到的数据最可能出现。直觉上，大量掷骰子后，数字 1 出现次数除以总次数就是 $P(1)$ 的 MLE。
* **Categorical MLE (类别分布 MLE)**：令 $\theta_i=P(X=i)$，观测计数为 $c_i$，总数为 $n$：
$$
L(\theta)=\prod_i \theta_i^{c_i},\quad \sum_i \theta_i=1
$$
$$
\hat{\theta}_i=\frac{c_i}{n}
$$
* **NLP relevance (在 NLP 中的地位)**：预训练预测 next/masked token，SFT 用 query 预测人工答案，n-gram 估计词概率，本质上都与最大化数据 likelihood 有关。
* **Demo / review focus (实践与复习重点)**：能从语料读取文本、tokenize、统计 count、估计概率、构造矩阵并可视化。MLE 可解析求解，也可在复杂模型中通过数值优化求解。

# Lecture 2: Tokenization、n-gram、语义表示、Word2Vec / GloVe / FastText

## Part I: Tokenization and Language Units

* **Language hierarchy (语言层级)** 包括 phonology、morphology、syntax、semantics、pragmatics。本讲主要关注 token、word semantics 和 word vector。
* **Word Tokenization (分词)** 是把文本切成更小单位。方法包括简单 `split()`、正则表达式、NLTK 等。
  * `split()` 会把 `world,` 和逗号粘在一起。
  * 正则可去掉标点。
  * NLTK 可把标点作为独立 token。
  * 标点是否保留取决于任务；在情感、句法、生成中标点可能有用。
* **Complex token decomposition (复杂 token 拆解)**：
  * `state-of-the-art` 可拆成多个词。
  * `camelCaseWord` 可按大小写边界拆分。
  * `COVID19` 可拆为字母部分和数字部分。
  正确拆解可减少 OOV，并提高对 morphology 和组合结构的泛化能力。

## Part II: n-gram Language Models

* **Language Model (语言模型)** 定义句子概率：
$$
P(w_1,\ldots,w_n)
$$
  可用于 spell-checking、next-word prediction 等。完整联合分布参数太多，因此需要 unigram、bigram、trigram 等近似。
* **Unigram Model (一元模型)** 假设词彼此独立：
$$
P(w_1,\ldots,w_n)=\prod_i P(w_i)
$$
  MLE 为：
$$
P(w)=\frac{c(w)}{N}
$$
  缺点是完全忽略词序和上下文。
* **Bigram / Trigram (二元 / 三元模型)**：
$$
P(w_i|w_{i-1})=\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
$$
$$
P(w_i|w_{i-2},w_{i-1})=\frac{Count(w_{i-2},w_{i-1},w_i)}{Count(w_{i-2},w_{i-1})}
$$
  核心是用短历史近似长历史。
* **Zero Probability Problem (零概率问题)**：未出现组合会被估为 0，但没见过不代表真实语言中不可能。
* **Laplace Smoothing (加一平滑)**：
$$
P_L(w)=\frac{c(w)+1}{N+|V|}
$$
$$
P_L(w_2|w_1)=\frac{c(w_1,w_2)+1}{c(w_1)+|V|}
$$
  平滑避免未见事件概率为 0，但会把概率质量从高频事件挪给低频或未见事件。

## Part III: Symbolic Meaning and Word Semantics

* **Symbolic vs. Semantics (符号与意义)**：二进制序列本身没有固定意义；按 ASCII 可解读成文本，按 RGB 可解读成颜色，按波形可解读成声音。Chinese Room 思想实验说明“符号操作”不等于“理解语义”。
* **Grounding problem (语义落地问题)**：计算机操作的是符号和数字，而人类语义来自感官、世界经验和社会互动。`cat` 对计算机只是编码，对人类则关联动物概念。
* **Dictionary and WordNet (字典与 WordNet)**：
  * 字典以 lemma 为条目，一个 lemma 可有多个 sense。
  * WordNet 以 word sense 为基本单位，包含 synset、hypernym、hyponym 等关系。
  * 优点是编码人类知识；缺点是静态、更新慢、难覆盖新词，且偏 symbolic。
* **One-hot Vector (独热向量)** 为每个 token 分配唯一 ID。优点是唯一可索引；缺点是语义相关词相似度为 0，无法表达多个语义方面。

## Part IV: Distributional Word Vectors

* **Distributional Hypothesis (分布假说)**：“You shall know a word by the company it keeps.” 一个词的上下文定义了它的语义。
* **Word Embedding (词嵌入)** 是低维稠密向量，通常 50-300 维。向量支持线性代数操作，例如：
$$
king - man + woman \approx queen
$$
* **PCA Visualization (PCA 可视化)** 可把高维词向量降到 2 维观察聚类，但二维图只是高维结构的投影，不能完全代表真实语义空间。
* **Context window (上下文窗口)** 决定语义粒度：小窗口偏句法/局部关系，大窗口偏主题。

## Part V: Word2Vec, GloVe, and FastText

* **Word2Vec Skip-gram (Skip-gram 模型)**：在滑动窗口中构造 `(center, context)` 正样本，用中心词预测附近上下文词。
* **Two-vector parameterization (中心词向量与上下文词向量)**：每个词有中心词向量 $v_w$ 和上下文词向量 $u_w$：
$$
P(w_o|w_c)=\frac{\exp(u_o^T v_c)}{\sum_w \exp(u_w^T v_c)}
$$
* **Negative Log-Likelihood (负对数似然)**：
$$
L=-u_o^T v_c+\log\sum_w\exp(u_w^T v_c)
$$
* **Gradient intuition (梯度直觉)**：
$$
\frac{\partial L}{\partial v_c}=-u_o+\sum_w P(w|w_c)u_w
$$
  第一项把中心向量拉向真实上下文词，第二项把它推离模型当前认为可能的平均上下文。
* **Negative Sampling (负采样)**：softmax 分母需要遍历整个词表，代价高；negative sampling 只采 $K$ 个负样本，是 Word2Vec 实用化关键。
* **GloVe (Global Vectors)** 显式利用全局共现统计，核心建模为：
$$
w_i^T\tilde{w}_j+b_i+\tilde{b}_j\approx \log X_{ij}
$$
  并最小化带权平方误差：
$$
\sum_{i,j} f(X_{ij})(w_i^T\tilde{w}_j+b_i+\tilde{b}_j-\log X_{ij})^2
$$
* **FastText (子词表示)** 用字符 n-gram 表示词，解决 morphology 和 OOV 问题。词向量由子词向量组合，因此 `run` / `running` 可共享参数，新词也能由已有 n-gram 组合出表示。
* **Review focus (复习重点)**：negative sampling 解决 softmax 分母过贵；GloVe 先构造全局共现矩阵；FastText 用字符 n-gram 与词本身组合表示词。现代 LLM 常用 BPE 等 subword 方法。

# Lecture 3: POS Tagging 与 Hidden Markov Model 建模

## Part I: POS Tagging Task

* **POS Tagging (词性标注)** 是典型序列标注任务，把每个 token 标成 NN、NNP、VB、JJ、RB、DT、IN、CC 等词类。
* **POS as syntax signal (词性作为句法信号)**：同一 POS 的词有相似语法属性。名词、动词、形容词承载主要语义；副词、介词、particle、determiner 等组织句法。
* **Why POS matters (为什么重要)**：
  * 提供 noun-verb、determiner-noun、adjective-noun、verb-adverb、preposition-noun 等模式。
  * 辅助 spelling correction，例如 there/their、passed/past、effect/affect、loose/lose。
  * 支持机器翻译、关系抽取、事件抽取、实体抽取。
* **Lexical Ambiguity (词汇歧义)** 是 POS tagging 的难点。`back` 可为 RB/NN/JJ/VB，`like` 可为 IN/VB/JJ，`fast` 可为 JJ/RB/NN。
* **Buffalo example (Buffalo 例子)**：`Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo` 语法正确，因为 Buffalo 可表示地名、动物、动词“欺负”。
* **Social media and OOV (社交媒体与未登录词)**：现代文本出现 hashtag、@mention、RT、lol、emoji 等，需要新标签或 subword 处理。LLM 时代 POS 不一定是必需中间任务，但仍有助于建立 NLP 直觉。

## Part II: HMM Motivation and Variables

* **Hidden Markov Model (隐马尔可夫模型)** 同时建模：
  * **Transition probability (转移概率)**：tag 如何转移，如 $P(NN|PRP\$)$。
  * **Emission probability (发射概率)**：tag 如何生成词，如 $P(\text{back}|NN)$。
* **Variables (变量定义)**：
  * 词表 $V$。
  * POS tag 集合 $S=\{s_1,\ldots,s_N\}$。
  * 观测句子 $O=[o_1,\ldots,o_T]$。
  * 隐状态序列 $Q=[q_1,\ldots,q_T]$。
* **Prediction objective (预测目标)**：
$$
Q^*=\arg\max_Q P(Q|O)
$$
  直接枚举有 $N^T$ 种 tag 序列，复杂度指数级。

## Part III: Bayes Rule and Markov Assumptions

* **Bayes decomposition (贝叶斯分解)**：
$$
P(Q|O)=\frac{P(O|Q)P(Q)}{P(O)}
$$
  因为 $O$ 固定，预测可转为最大化：
$$
P(O|Q)P(Q)
$$
* 只看 $P(O|Q)$ 会忽略 tag 序列是否合理；只看 $P(Q)$ 会忽略实际词。
* **Markov Assumption (马尔可夫假设)**：下一状态只依赖当前状态，不依赖更早历史。它不是说历史真实无关，而是用当前状态作为历史摘要。
* **Tag sequence probability (tag 序列概率)**：
$$
P(Q)=P(q_1)\prod_{t=2}^T P(q_t|q_{t-1})
$$
  参数量从指数级降到 $|S|^2$ 级。
* **Emission independence (发射独立假设)**：当前词只依赖当前 tag：
$$
P(O|Q)=\prod_{t=1}^T P(o_t|q_t)
$$
  这是真实语言的强简化，但换来了可估计性和可推理性。

## Part IV: HMM Parameters and Tasks

* **Initial probability (起始概率)** $\pi_i=P(q_1=i)$。
* **Transition matrix (转移矩阵)**：
$$
a_{ij}=P(q_t=j|q_{t-1}=i)
$$
  每一行和为 1。高概率转移对应常见语法模式，如 determiner 后接 adjective 或 noun。
* **Emission matrix (发射矩阵)**：
$$
b_{j,o}=P(o_t=o|q_t=j)
$$
  例如 D 高概率发射 `the/a`，A 高概率发射 `big/red`，N 高概率发射 `dog/cat/car`。
* **Three HMM tasks (三个任务)**：
  * **Estimation (估计参数)**：从数据学习 $\pi,A,B$。
  * **Inference (推理)**：给定模型计算句子概率 $P(O)$。
  * **Prediction / Decoding (预测 / 解码)**：预测最可能 POS tag 序列。
* **Review focus (复习重点)**：transition matrix 不一定对称；emission matrix 空间复杂度为 $|S|\times |V|$；VB 是 verb，RB 是 adverb。

# Lecture 4: HMM Forward / Backward Algorithm

## Part I: Inference Problem

* **Inference (推理)**：给定观测 $O=[o_1,\ldots,o_T]$ 和 HMM 参数 $(A,B,\pi)$，计算：
$$
P(O|A,B,\pi)
$$
* **Marginalization over hidden states (对隐状态边缘化)**：
$$
P(O)=\sum_Q P(O,Q)
$$
  暴力枚举所有 $Q$ 的复杂度为 $O(|S|^T)$。
* **Dynamic Programming (动态规划)** 利用：
  * **Overlapping subproblems (重复子问题)**。
  * **Optimal / compositional substructure (可组合子结构)**。
  HMM 链式结构允许把过去所有路径压缩成每个状态上的概率。

## Part II: Forward Algorithm

* **Forward probability (前向概率)**：
$$
\alpha_t(j)=P(o_1,\ldots,o_t,q_t=j)
$$
  表示到时间 $t$ 为止、当前状态为 $j$ 的联合概率，是过去所有可能路径的汇总。
* **Base case (初始条件)**：
$$
\alpha_1(i)=\pi_i b_i(o_1)
$$
* **Recursion (递推)**：
$$
\alpha_t(j)=\sum_{k=1}^N \alpha_{t-1}(k)a_{kj}b_j(o_t)
$$
  对所有前一状态求和，表示 marginalization。
* **Sentence probability (句子概率)**：
$$
P(O)=\sum_j \alpha_T(j)
$$
* **Complexity (复杂度)**：每个时间步对每个当前状态枚举 $N$ 个前一状态，总复杂度为 $O(TN^2)$，远低于 $O(N^T)$。
* **Vectorization (矩阵化)**：前向概率向量可乘 transition matrix，再按 emission 概率逐元素缩放，提高实现效率。

## Part III: Backward Algorithm

* **Backward probability (后向概率)**：
$$
\beta_t(i)=P(o_{t+1},\ldots,o_T|q_t=i)
$$
  表示从当前状态 $i$ 出发，未来观测出现的概率。
* **Base case (初始条件)**：
$$
\beta_T(i)=1
$$
  句尾之后的空未来序列概率为 1。
* **Recursion (递推)**：
$$
\beta_t(i)=\sum_j a_{ij}b_j(o_{t+1})\beta_{t+1}(j)
$$
  对所有下一状态求和。
* **Forward vs. Backward (前向与后向)**：Forward 从过去到未来，Backward 从未来到过去；二者都可计算同一个 $P(O)$，也会在 EM / Baum-Welch 中配合使用。

## Part IV: Message Passing and Review

* **Message passing view (消息传递视角)**：Forward / Backward 是链式概率图模型上的 message passing，也是更一般概率图模型推理的特例。
* **Demo focus (实践重点)**：用 POS-tag 文本估计 HMM 参数后，运行 forward 和 backward，验证两个方向得到一致的句子概率。
* **Review focus (复习重点)**：
  * 每个时间步 forward 复杂度是 $O(N^2)$。
  * Forward / Backward 都是 DP。
  * Backward 在句尾初始化为 1。

# Lecture 5: HMM Viterbi Decoding 与 EM / Baum-Welch

## Part I: Structured Prediction and Viterbi Goal

* **Prediction / Decoding (预测 / 解码)**：给定句子、tag 集合和 HMM 参数，求：
$$
Q^*=\arg\max_Q P(Q|O;\theta)
$$
* **Structured prediction (结构化预测)**：输出不是独立标签集合，而是相互依赖的完整 tag 序列。
* **Why independent tagging fails (为什么不能逐词独立标注)**：`Time flies like an arrow` 中逐词选 most frequent tag 可能得到不自然序列；tag 之间的 transition 必须一起考虑。
* 暴力搜索仍是 $O(|S|^T)$，需要 DP。

## Part II: Viterbi Algorithm

* **Viterbi state (Viterbi 状态)**：
$$
v_t(j)=\max_{q_1,\ldots,q_{t-1}}P(q_1,\ldots,q_{t-1},q_t=j,o_1,\ldots,o_t)
$$
  表示到时间 $t$、以状态 $j$ 结尾的最佳路径概率。
* **Base case (初始条件)**：
$$
v_1(j)=\pi_j b_j(o_1)
$$
* **Recursion (递推)**：
$$
v_t(j)=\max_k v_{t-1}(k)a_{kj}b_j(o_t)
$$
  Viterbi 与 Forward 很像，但把求和换成取最大。
* **Backpointer (回溯指针)**：
$$
p_t(j)=\arg\max_k v_{t-1}(k)a_{kj}b_j(o_t)
$$
  只保存最大概率不够，还要保存它来自哪个前一状态，最终才能回溯完整 tag 序列。
* **Termination (终止)**：
$$
q_T^*=\arg\max_k v_T(k)
$$
  然后沿 backpointer 从后往前恢复 $Q^*$。
* **Complexity (复杂度)**：时间复杂度 $O(TN^2)$，保存 backpointers 的空间通常为 $O(TN)$。
* **Historical note (背景)**：Viterbi 算法最初用于通信问题，后来广泛用于语音识别、搜索、DNA sequencing 和 NLP 解码。

## Part III: Supervised HMM Estimation

* 如果 hidden states 已知，可直接用 MLE 估计：
$$
a_{ij}=\frac{C(i\rightarrow j)}{C(i)}
$$
$$
b_{i,o}=\frac{C(i\rightarrow o)}{C(i)}
$$
$$
\pi_i=\frac{C(q_1=i)}{m}
$$
* 有 POS 标注语料时，估计 HMM 参数就是数起始 tag、tag transition、tag emits word 的相对频率。

## Part IV: EM and Baum-Welch

* **EM Algorithm (期望最大化算法)** 用于带 unlabeled data 的参数估计。
  * **E-step**：用当前模型估计 hidden labels 的概率，得到 pseudo labels / soft labels。
  * **M-step**：用 soft counts 做 MLE 更新参数。
  * 循环迭代直到收敛。
* **EM properties (性质)**：每轮 EM 保证数据 likelihood 不下降，但通常只保证局部最优，初始化很重要。
* **Soft transition count (转移软计数)**：
$$
\xi_t(i,j)=P(q_t=i,q_{t+1}=j|O,\lambda)
=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{P(O)}
$$
* **Soft state count (状态软计数)**：
$$
\gamma_t(i)=P(q_t=i|O,\lambda)=\frac{\alpha_t(i)\beta_t(i)}{P(O)}
$$
  若位置 $t$ 的观测词为 $o$，就把 $\gamma_t(i)$ 计入 tag $i$ 发射 $o$ 的 soft count。
* **Baum-Welch Algorithm (Baum-Welch 算法)** 是 HMM 的 EM：
  1. 初始化 $(A,B,\pi)$。
  2. E-step 运行 Forward / Backward 得到 soft counts。
  3. M-step 更新 $A,B,\pi$。
  4. 迭代至收敛。
* **Review focus (复习重点)**：Viterbi 复杂度是 $O(TN^2)$；EM 的 E-step 用 DP，M-step 用 soft counts 做 MLE。

# Lecture 6: Syntax、CFG、CYK Parsing

## Part I: Linguistic Hierarchy and Syntax Basics

* **Levels of Language (语言层级)**：
  * **Phonology (语音学)**：研究声音。
  * **Morphology (形态学)**：研究单词内部结构，如 `un-happi-ness`。
  * **Syntax (句法学)**：研究词如何组织成短语和句子。
  * **Semantics (语义学)**：研究字面意义。
  * **Pragmatics (语用学)**：研究隐含意义，如隐喻、讽刺、幽默和沉默。
* **Syntax (句法)** 是定义词如何组织成更大单位的规则，可区分合法/非法句子。
* **Fluency (流利度)** 是使用句法规则的熟练程度。母语者通常通过 acquisition 隐式掌握，二语学习者常显式学习规则。
* **Shallow Syntax (浅层句法)** 关注局部结构，如 `the + noun`、主谓一致。n-gram 和 HMM 是 Markovian 模型，只利用近距离历史。
* **Deep Syntax (深层句法)** 关注全局和长程依赖。例如：
  * `The books that I bought yesterday are expensive.`
  * `are` 依赖远处复数名词 `books`，局部模型很难捕捉。
* **Constituent (语法成分)** 是作为单一语法单位行动的一组词。
  * 可以在句中移动而语义基本不变。
  * 不能随意拆开仍保持原语义。
  * 例如时间短语 `On September seventeenth` 可整体移动，但拆开后语义和语法会坏掉。

## Part II: Context-Free Grammar

* **Context-Free Grammar (CFG, 上下文无关文法)** 用数学规则建模 constituent structure。context 指 constituent 外部元素；context-free 表示短语内部结构可独立于外部环境递归处理。
* **Formal definition (形式化定义)**：
$$
G=(N,\Sigma,R,S)
$$
  * $N$：非终结符集合，如 NP、VP。
  * $\Sigma$：终结符集合，即具体单词。
  * $R$：产生式集合，形式为 $A\rightarrow \beta$。
  * $S$：起始符号，且 $S\in N$。
* **Derivation (推导)**：若 $A\rightarrow\beta$，则：
$$
\alpha A\gamma \Rightarrow \alpha\beta\gamma
$$
* **Language generated by grammar (文法生成的语言)**：
$$
L_G=\{w\in\Sigma^* \mid S\Rightarrow^* w\}
$$
* **Parse tree (解析树)** 表示从 $S$ 到终结符序列的多步推导。底层终结符规则类似 POS tagging，上层规则组合短语结构。
* **Phrasal categories (短语范畴)**：
  * **NP (Noun Phrase, 名词短语)**：指代人、事、物，如 `a flight`。
  * **VP (Verb Phrase, 动词短语)**：描述动作或状态，如 `prefer a morning flight`。
  * **PP (Prepositional Phrase, 介词短语)**：表示地点、时间、方式等，如 `from Atlanta`。
  * **AP (Adjective Phrase, 形容词短语)**：以形容词为核心，如 `least expensive`。
* **Word / clausal categories (词类与句子范畴)**：
  * **S (Sentence)**：文法最高层级。
  * **N / NN (Noun)**：名词。
  * **V / VB (Verb)**：动词。
  * **Adj / JJ (Adjective)**：形容词。
  * **Adv / RB (Adverb)**：副词。
  * **Aux (Auxiliary)**：助动词，如 `do/can/will/have`。
  * **Det (Determiner)**：限定词，如 `a/the/this`。
  * **Nominal (名词性成分)**：介于 noun 和 NP 之间，如 `morning flight`。
  * **Wh-NP / Wh-Adv**：疑问名词短语 / 疑问副词。
  * **P / IN (Preposition)**：介词。
  * **Conj / CC (Conjunction)**：连词，是 coordination ambiguity 的来源之一。
  * **Pro / PRP (Pronoun)**：代词，常可直接作为 NP。
  * **Card / CD、Ord、Quant**：基数词、序数词、数量词。
  * **SBAR / CP (Complementizer Phrase)**：补语从句或关系从句。

## Part III: English CFG Rules

* **Sentence structures (句子结构)**：
$$
S\rightarrow NP\ VP
$$
$$
S\rightarrow VP
$$
$$
S\rightarrow Aux\ NP\ VP
$$
$$
S\rightarrow Wh\text{-}NP\ VP
$$
$$
S\rightarrow Wh\text{-}NP\ Aux\ NP\ VP\ PP
$$
  分别覆盖陈述句、祈使句、是非问句和 wh 结构。
* **Noun Phrase (名词短语)**：
$$
NP\rightarrow (Det)(Card)(Ord)(Quant)(AP)Nominal
$$
  括号表示可选成分。
* **Adjective Phrase (形容词短语)**：
$$
AP\rightarrow (RB)JJ
$$
* **Possessive recursion (所有格递归)**：
$$
Det\rightarrow NP's
$$
  可生成 `Denver's mayor's mother's canceled flight` 这类深层嵌套。
* **Nominal rules (名词性成分规则)**：
$$
Nominal\rightarrow Noun
$$
$$
Nominal\rightarrow Nominal\ Noun
$$
$$
Nominal\rightarrow Nominal\ PP
$$
$$
Nominal\rightarrow Nominal\ Gerundive\text{-}VP
$$
$$
Nominal\rightarrow Nominal\ ed\text{-}VP
$$
$$
Nominal\rightarrow Nominal\ infinitive
$$
$$
Nominal\rightarrow Nominal\ relative\text{-}clause
$$
  例如 `flight to Boston`、`flight leaving before 10`、`flight that serves breakfast`。
* **Verb Phrase (动词短语)**：
$$
VP\rightarrow Verb
$$
$$
VP\rightarrow Verb\ NP
$$
$$
VP\rightarrow Verb\ NP\ PP
$$
$$
VP\rightarrow Verb\ PP
$$
$$
VP\rightarrow Verb\ VP
$$
$$
VP\rightarrow Verb\ S
$$
  其中 $S$ 可作 **sentential complement (句法补足语)**。
* **Penn Treebank (宾州树库)** 是带 parse tree 标注的语料库，可从中提取 CFG，也可作为 parser 训练数据。

## Part IV: Parsing and Ambiguity

* **Syntactic Parsing (句法解析)**：给定 CFG 和句子，为句子分配合法 parse tree。目标树的 root 必须是 $S$，leaves 必须是句子中的单词。Parsing 是 CFG 生成句子的反过程。
* **Top-down Search (自顶向下搜索)**：从 $S$ 开始扩展非终结符；若叶子无法匹配输入则剪枝；匹配完整输入后停止。优点是目标导向，缺点是会构造很多不匹配输入的树。
* **Bottom-up Search (自底向上搜索)**：从输入词作为 leaves 开始，用 CFG 右侧匹配生成非终结符，最终尝试到达 $S$。优点是直接基于输入，缺点是可能构造无法连接到根的局部结构。
* **Repeated Subproblems (重复子问题)**：搜索中会反复构造相同局部树，如 `Det -> that`、`Noun -> flight`、`NP -> Det Nominal`。这正是动态规划能优化 parsing 的原因。
* **Attachment Ambiguity (附着歧义)**：修饰成分依附对象不明确。
  * `I saw the Grand Canyon flying to New York.`
  * `flying to New York` 可修饰 `saw` 的动作，也可句法上修饰 `Grand Canyon`。
* **Coordination Ambiguity (并列歧义)**：连词连接范围不明确。
  * `old men and women`
  * 可理解为 `[old men] and [women]`，也可理解为 `old [men and women]`。

![Wiki/Image/Class/Introdution to NLP/1.png](/img/user/Wiki/Image/Class/Introdution%20to%20NLP/1.png)

## Part V: CYK Algorithm and CNF

* **CYK (Cocke-Younger-Kasami) Algorithm** 是 CFG parsing 的动态规划算法。它缓存每个 span 可生成的非终结符，避免重复解析相同子串。
* **Core DP idea (动态规划思想)**：
  * **Overlapping subproblems**：同一子串的解析结果复用。
  * **Optimal / compositional substructure**：大 constituent 由左右子 constituent 组合而成。
* **Chomsky Normal Form (CNF, 乔姆斯基范式)** 要求规则只有两种：
$$
A\rightarrow BC
$$
$$
A\rightarrow a
$$
  第一种右侧是两个非终结符，第二种右侧是一个终结符。
* **CNF conversion (CNF 转换)**：
  * Mixed RHS:
$$
A\rightarrow Bc \Rightarrow A\rightarrow BC,\quad C\rightarrow c
$$
  * Unit production:
$$
A\rightarrow B
$$
    需要消除并替换为 $B$ 的展开。
  * Long RHS:
$$
A\rightarrow BCD \Rightarrow A\rightarrow XD,\quad X\rightarrow BC
$$
  任意 CFG 可转为 CNF 而不损失表达能力。
* **CYK table (CYK 表格)**：句长为 $n$ 时，构建 $(n+1)\times(n+1)$ 矩阵。$table[i,j]$ 存储能生成位置 $i$ 到 $j$ 子串的所有非终结符。
* **Initialization (初始化)**：对每个词 $w_j$，若存在规则 $A\rightarrow w_j$，则把 $A$ 加入 $table[j-1,j]$。
* **Recursion (递推)**：对 span 从 2 到 $n$，枚举起点 $i$、终点 $j$、切分点 $k$。若：
$$
B\in table[i,k],\quad C\in table[k,j],\quad A\rightarrow BC
$$
  则把 $A$ 加入 $table[i,j]$。
* **Termination (终止)**：检查 $table[0,n]$ 是否包含起始符号 $S$。若包含，则解析成功；若多个结构可到 $S$，则有多棵 parse tree。
* **Efficiency (效率)**：通过缓存 $table[i,j]$，CYK 将指数级搜索降为多项式级，典型复杂度为 $O(n^3)$ 乘以语法规则匹配代价。
* **Review focus (复习重点)**：能判断 CFG 是否为 CNF；能写出简单句子的 derivation；能解释 CYK cell 的 span 含义，例如 `(0,2)` 表示前两个词组成的子串。

## Part VI: The Bitter Lesson

* **The Bitter Lesson (苦涩教训)** 讨论“手工构建人类知识规则”与“构建可扩展模型让数据自发现知识”的长期竞争。
* 在传统 NLP 中，syntax、CFG、parser 是显式中间结构；在 LLM 时代，模型通过 next-token prediction 可从大规模数据中学习大量语言规律。
* 这不代表 syntax 不重要。学习 CFG、parsing、CYK 的价值在于理解语言结构、动态规划、树形预测和传统 NLP 思路。

# Lecture 7: Probabilistic CFG、Inside / Outside、最优解析树

## Part I: From CFG to PCFG

* **CFG limitation (CFG 局限)**：CFG 只能判断哪些 parse tree 合法，不能比较哪棵树更可能。
* **Probabilistic CFG (PCFG, 概率上下文无关文法)** 给 CFG 每条 production 分配概率；同一 left-hand side 的规则概率和为 1。
* **Parse tree probability (解析树概率)** 是树中所有 production 概率的乘积：
$$
P(t,w_1,\ldots,w_m|G)=\prod_{r\in t}P(r)
$$
* **Sentence probability (句子概率)** 对所有可能 parse trees 求和：
$$
P(w_1,\ldots,w_m|G)=\sum_t P(t,w_1,\ldots,w_m|G)
$$
  合法树数量可能指数多，因此仍需要 DP。

## Part II: PCFG Assumptions and Tasks

* **Place invariance (位置不变性)**：规则概率不依赖子树出现在句子中的位置。
* **Context-free (上下文无关)**：规则概率不依赖外部上下文。
* **Ancestor-free (祖先无关)**：规则概率不依赖祖先节点。
* 这些假设让局部 rule 概率可以独立相乘，并使概率解析可分解。
* **Three PCFG tasks (三个任务)**：
  * 求句子概率。
  * 求最可能 parse tree。
  * 从 treebank 用 MLE 学 PCFG 参数。
  它们对应 HMM 中的 inference、decoding、estimation。

## Part III: Inside Probability

* **Inside probability (内部概率)** 类似 HMM Forward，表示某个非终结符生成某个 span 内部词序列的概率。
* 记 $N^j_{pq}$ 表示非终结符 $j$ 推导从 $p$ 到 $q$ 的词。
* **Base case (初始条件)**：
$$
\beta_j(k,k)=P(N^j\rightarrow w_k)
$$
* **Recursion in CNF (CNF 下递推)**：
$$
\beta_j(p,q)=\sum_{r,s,d}P(N^j\rightarrow N^rN^s)\beta_r(p,d)\beta_s(d+1,q)
$$
  长 span 的概率由所有可能切分点和左右孩子规则贡献相加。
* **Sentence probability (句子概率)** 是起始符号生成全句的 inside probability。

## Part IV: Outside Probability

* **Outside probability (外部概率)** 类似 HMM Backward，表示某个非终结符覆盖 span 时，生成该 span 外部词和上层结构的概率。
* **Base case (初始条件)**：起始符号覆盖整句的 outside probability 为 1，其他符号覆盖整句为 0。
* **Recursion intuition (递推直觉)**：
  * 如果目标节点是父节点的左孩子，需要父节点 outside 与右 sibling inside。
  * 如果目标节点是父节点的右孩子，需要父节点 outside 与左 sibling inside。
* **Inside + Outside (内外概率结合)**：若非终结符 $j$ 覆盖 span $[p,q]$，整句概率中与该节点相关的贡献包含：
$$
\alpha_j(p,q)\beta_j(p,q)
$$
  这也是估计 PCFG soft counts 的基础。

## Part V: Best Parse Tree

* **Viterbi-style PCFG parsing (Viterbi 风格 PCFG 解析)**：寻找最优树时，把 Inside algorithm 的求和换成取最大，并记录 backpointers。
* 在 cell $(p,q)$ 中保存：
  * 最大概率 $\delta_j(p,q)$。
  * 来源 $\psi_j(p,q)$，即左右孩子和切分点。
* 最后从右上角起始符号回溯重构最优 parse tree。
* 例子 `saw stars with ears` 中，VP 可由 `V NP` 或 `VP PP` 等方式生成；decoding 保留最大概率结构，而 inside 会累加所有结构。
* **Review focus (复习重点)**：同一 LHS 的 PCFG 规则概率和为 1；outside 依赖 sibling 的 inside；用 CYK-like 方法找最优树通常要求 PCFG 为 CNF。

# Lecture 8: Neural Network 与 RNN Language Modeling

## Part I: Why Neural Language Models

* **n-gram problems (n-gram 三个问题)**：
  * **Data sparsity (数据稀疏)**：没见过的序列概率为 0 或估计不准。
  * **Model complexity (模型复杂度)**：增大 $n$ 会导致组合数量爆炸。
  * **Fixed-window architecture (固定窗口)**：只能看固定长度历史，难捕捉不同范围依赖。
* **Neural networks as solution (神经网络作为解决方案)**：用固定参数集做预测，通过 architecture 处理上下文。RNN 强加序列结构，Transformer 用 attention，SFT/RLHF 用任务和损失塑造模型。
* 神经模型不再记忆每个 n-gram 的频率，而是学习可泛化的参数化函数。

## Part II: Logistic Regression and MLP

* **Logistic regression (逻辑回归)** 可看作最简单神经网络：
$$
a=\sigma(w^Tx+b)
$$
* **Binary cross-entropy loss (二元交叉熵)**：
$$
\ell(a,y)=-\log(a^y(1-a)^{1-y})
$$
* **Computation graph (计算图)** 支持 forward 和 back-propagation。复杂神经网络本质是更多层、更多节点的可微计算图。
* **MLP (多层感知机)** 包含输入层、隐藏层、输出层；隐藏层引入中间表示，使模型能学习非线性特征。
* **Neuron computation (神经元计算)**：
$$
z_j^{[1]}=W_j^{[1]T}x+b_j^{[1]}
$$
$$
a_j^{[1]}=\sigma(z_j^{[1]})
$$
  或使用 ReLU 等非线性。
* **Vectorization (向量化)**：
$$
z^{[1]}=W^{[1]}x+b^{[1]},\quad a^{[1]}=\sigma(z^{[1]})
$$
  向量化把多个 neuron 的计算合并为矩阵运算，是 GPU 加速的基础。

## Part III: RNN Architecture

* **Recurrent Neural Network (RNN, 循环神经网络)** 用 hidden state 总结过去信息：
$$
h^{(t)}=f(h^{(t-1)},x^{(t)};\theta)
$$
  同一函数和参数在不同时间步复用，因此可处理任意长度序列。
* **RNN language model (RNN 语言模型)**：
  * 输入 token 转成 embedding。
  * hidden states 传递历史。
  * 输出 units 给出下一个 token 或 tag 的概率。
* **Parameters (参数)** 通常包括 $U,W,V,b,c$：
  * $U$：input-to-hidden。
  * $W$：hidden-to-hidden。
  * $V$：hidden-to-output。
* **Common RNN equations (常见公式)**：
$$
h_t=\tanh(Wh_{t-1}+Ux_t+b)
$$
$$
o_t=Vh_t+c
$$
$$
\hat{y}_t=softmax(o_t)
$$
* 语言模型中常令 $y_t=x_{t+1}$；POS tagging 中可令 $y_t=POS(x_t)$ 或下一个位置的 tag。

## Part IV: Training and Limitations

* **Training objective (训练目标)**：常用 negative log-likelihood / perplexity。
* **BPTT (Back-propagation Through Time)**：把 RNN 沿时间展开后反向传播更新参数。
* **Vanishing / exploding gradients (梯度消失 / 爆炸)**：长序列中梯度连续乘许多矩阵，早期 token 的学习信号可能衰减或爆炸。
* **Why tanh matters (为什么需要 tanh)**：
  * 没有非线性，递归会坍缩为线性映射。
  * $W^t$ 的矩阵幂可能导致信息消失或爆炸。
  * tanh 把值压到 $[-1,1]$，有助于稳定 hidden state。
* tanh 不能完全解决长程依赖问题，后续 LSTM、GRU、attention、Transformer 继续改进。
* **Review focus (复习重点)**：n-gram 不能建模超过窗口的长程依赖；RNN 不只用于 language modeling；若 hidden state 维度为 $n$，hidden-to-hidden 矩阵 $W$ 大小为 $n\times n$。

# Lecture 9: Machine Translation、Rule-based MT、IBM Model 1

## Part I: Machine Translation Motivation

* **Machine Translation (机器翻译)** 是把一种语言文本转换成另一种语言，是 NLP 经典任务，也是 alignment、language model、sequence model 的重要应用。
* **Rosetta Stone analogy (罗塞塔石碑类比)**：同一内容用多种文字记录，可通过已知语言推断未知语言。现代统计 MT 利用大规模 parallel corpus，如 Canadian Hansards。
* **Translation in the LLM era (LLM 时代翻译)**：传统 MT 是特定任务系统；LLM 把翻译吸收到通用语言生成能力中，能处理上下文、语义推理、idiom、文化适配和用户约束。

## Part II: Language Differences

* **Lexical differences (词汇差异)**：
  * `bass` 在西语中可能对应乐器或鱼。
  * `wall` 在德语中可区分室内/室外墙。
  * `brother` 在中文中需区分哥哥/弟弟。
* **Syntactic differences (句法差异)**：法语/西语形容词有性别变化；英语多为 SVO，日语多为 SOV；介词短语位置、形容词-名词顺序也可能不同。
* 翻译不是查词典，还要根据上下文、语法和目标语言习惯选择表达。

## Part III: Rule-Based MT and Vauquois Triangle

* **Vauquois Triangle (Vauquois 三角)** 把翻译方法分为 direct word-level transfer、syntactic transfer、semantic transfer、interlingua。越上层越抽象地理解源句再生成目标句，越下层越接近词表替换。
* **Direct translation (直接翻译)** 流程：
  * morphology analysis。
  * lexical transfer。
  * local reordering。
  * morphological generation。
  例如英语 `green witch` 到西语 `bruja verde` 需要局部重排。
* **Direct method limitations (直接法缺陷)**：难处理大范围结构调整，如 PP 位置、SVO/SOV 词序差异、句子级重排序。
* **Transfer method (转换法)**：
  * **Syntactic transfer**：把源语言 parse tree 转成目标语言 parse tree。
  * **Lexical transfer**：再做词汇翻译。
  它能处理更多句法差异，但依赖高质量 parser 和大量规则。

## Part IV: Evaluation and Noisy Channel Objective

* **Fluency (流畅性)**：目标语是否自然。
* **Faithfulness (忠实性)**：是否保留源文含义。
* “信达雅”中，信对应意义准确，达对应通顺明白，雅对应表达得体优雅。
* 形式化目标可写为：
$$
best\ T=\arg\max_T fluency(T)\ faithfulness(T,S)
$$
* **Noisy-channel MT (噪声信道机器翻译)**：设英文目标句 $E=(e_1,\ldots,e_I)$，外语源句 $F=(f_1,\ldots,f_J)$：
$$
E^*=\arg\max_E P(E|F)
$$
$$
E^*=\arg\max_E P(F|E)P(E)
$$
  其中 $P(E)$ 是 fluency language model，$P(F|E)$ 是 faithfulness translation model，$P(F)$ 与 $E$ 无关可省略。

## Part V: Word Alignment and IBM Model 1

* **Word Alignment (词对齐)** 是把目标词和源词建立对应关系。可用 $a_j$ 表示第 $j$ 个源词对齐到哪个目标词。
* **Alignment matrix (对齐矩阵)**：行是目标词，列是源词，标记表示对应关系。可表达 one-to-one、one-to-many、many-to-one，也可加入 NULL 处理无法对应的词。
* **IBM Model 1** 是单词级统计翻译模型：
  * 生成外语长度 $J$。
  * 生成 alignment $A$。
  * 根据对齐的英文词生成外语词 $F$。
* **IBM Model 1 probability (概率形式)**：
$$
P(F,A|E)=P(J|I)P(A|I,J)\prod_j P(f_j|e_{a_j})
$$
$$
P(F|E)=\sum_A P(F,A|E)
$$
  alignment 是隐藏变量，因此需要对所有 alignment 求和。
* **IBM Model 1 limitations (局限)**：
  * **Bag-of-Words assumption**：所有 alignment 距离等可能，忽略词序。
  * **Independent Word Translation**：词翻译只依赖词对，不看上下文。
  * **One-to-Many limitation**：不能自然处理 many-to-many、习语和复杂短语翻译。
* **Review focus (复习重点)**：alignment matrix 可表示多种对应；$P(E|F)$ 是整体翻译后验，不是单独 fluency；direct translation 难处理 long-range dependencies。

# Lecture 10: HMM Alignment、Decoding Search、BLEU、Seq2Seq、Attention

## Part I: HMM Alignment Model

* **IBM Model 1 weakness (IBM Model 1 弱点)**：词独立生成、所有 alignment 等概率；真实翻译中相邻源词通常对齐到相邻目标词，多个词也可能联合生成。
* **HMM alignment model (HMM 对齐模型)** 类比 POS tagging：
  * hidden state 是目标句位置 $a_j$。
  * observed word 是源语言词 $f_j$。
  * transition 负责下一个对齐位置。
  * emission 负责根据对齐位置生成源词。
* **Markov assumption for alignment (对齐马尔可夫假设)**：
$$
P(a_j|history,E)=P(a_j|a_{j-1},I)
$$
$$
P(f_j|history,E,A)=P(f_j|e_{a_j})
$$
* **HMM translation probability (HMM 翻译概率)**：
$$
P(F,A|E)=P(J|I)\prod_j P(a_j|a_{j-1},I)P(f_j|e_{a_j})
$$
* **Alignment locality and jump model (局部性与跳跃模型)**：若 $a_j$ 接近 $a_{j-1}$，概率应更高。模型关注 jump $|a_j-a_{j-1}|$，而非绝对位置。

## Part II: Translation Decoding and Search

* **Translation decoding (翻译解码)**：
$$
\hat{E}=\arg\max_E P(F|E)P(E)
$$
  这比已知 $E,F$ 时做 alignment 更难，因为要在巨大候选句空间中搜索。
* 带 bigram LM 的一般 decoding 是 NP-complete；HMM 在特定假设下可用 Viterbi。
* **Search-based methods (搜索方法)**：状态节点是 partial translation，逐步扩展候选短语，评分函数指导搜索。
* **Best-first search (最佳优先搜索)**：每次扩展当前最高分节点，缺点是昂贵且短视，可能落入局部最优。
* **A* search (A 星搜索)**：
$$
f^*(p)=g(p)+h^*(p)
$$
  其中 $g(p)$ 是当前 partial translation 质量，$h^*(p)$ 是未翻译部分未来质量估计。
* **Beam search (束搜索)**：每轮扩展 beam 中所有状态，只保留 top-$k$。$k$ 越大质量通常越好但越慢；$k$ 太小容易丢掉未来更优路径。

## Part III: BLEU Evaluation

* **BLEU** 自动评估候选翻译与参考翻译的 n-gram 匹配。unigram 偏词义覆盖，高阶 n-gram 偏局部流畅度和词序。
* BLEU 主要计算 $n=1,2,3,4$ 的 modified n-gram precisions，并取几何平均，同时配合短句惩罚。
* **BLEU pitfalls (BLEU 陷阱)**：
  * 极短翻译可能 precision 很高。
  * 重复词会虚增匹配。
  * modified precision 会截断重复命中次数，例如候选 `the the ...` 对参考 `the cat is on the mat` 的 `the` 最多按参考出现次数计数。
* BLEU 是自动表面指标，不等同于人类质量判断；语义等价但措辞不同的翻译可能得分不公平。

## Part IV: Seq2Seq Encoder-Decoder

* **Seq2Seq (序列到序列)**：一个 RNN 编码源句，一个 RNN 生成目标句。输入输出长度可不同，适合 MT、image captioning、music generation 等。
* **Encoder (编码器)** 从 source tokens 得到 hidden states；**Decoder (解码器)** 根据编码信息和之前输出生成目标 tokens。
* **Teacher forcing (教师强制)**：训练时把真实前一个词喂给 decoder；测试时只能用模型上一步预测。
* **Training objective (训练目标)**：用 MLE 最大化目标序列条件概率：
$$
P(y_1,\ldots,y_T|x_1,\ldots,x_S)
$$
  loss 是每个目标位置 NLL 的累积。
* **Fixed-length bottleneck (固定长度瓶颈)**：固定长度 hidden vector 必须总结任意长度源句及词序，长句容易丢失信息。
* **RNN limitations (RNN 局限)**：
  * sequential recency 使近期 token 影响更强。
  * 长程主谓一致如 `writer of the books is/are` 难处理。
  * BPTT 有梯度消失/爆炸。
  * RNN 不能很好并行，长序列慢。

## Part V: Attention

* **Attention mechanism (注意力机制)**：生成每个目标词时，动态关注源句中有用位置，而不是把所有信息压进一个最终 hidden vector。
* 类比搜索引擎：query $Q$ 匹配 keys $K$，取出 values $V$。
* 在 seq2seq attention 中：
  * decoder hidden state 是 query。
  * encoder hidden states 是 keys / values。
  * attention weights 由 score 函数和 softmax 得到。
* **Context vector (上下文向量)**：
$$
c_t=\sum_i \alpha_{t,i}h_i
$$
$$
\alpha_{t,i}=softmax(score(s_{t-1},h_i))
$$
* **Advantage (优点)**：每个生成步都能结合输入全局位置的信息，缓解长程依赖。
* **Remaining issue (剩余问题)**：若 encoder 仍是 RNN，计算仍是 sequential bottleneck。Transformer 后续进一步移除 RNN。
* **Review focus (复习重点)**：IBM Model 1 词独立翻译；HMM alignment 不能直接实现 many-to-many；短候选可能 BLEU-1 很高但翻译差；seq2seq 弱点包括输入压缩困难和早期错误影响未来。

# Lecture 11: Transformer 架构、位置编码、注意力与训练推理

## Part I: Transformer Big Picture

* **Transformer** 起源于 2017 年的 *Attention is All You Need*，后来扩展到 CV、robotics、data science，也成为 GPT-like 应用的基础架构。
* 作为 **foundation model (基础模型)**，Transformer 需要足够大的模型容量去覆盖多样语言模式，也需要大规模数据去看到广泛 pattern；它的 inductive bias 比 RNN/CNN 更弱，因此更依赖数据和规模。
* 早期 Transformer 是 **encoder-decoder architecture**，后来 GPT 类模型主要采用 **decoder-only architecture**。
* 核心组件包括 **positional embedding**、**masked multi-head attention**、**feed-forward / MLP layer**、**layer normalization** 和 **residual connections**。

## Part II: Positional Embedding and Attention

* **Positional Embedding (位置编码)** 给 token 加入位置信息；否则同一个 token 在不同位置 embedding 相同，模型无法区分其句法角色。比如 `Trust is what builds trust` 中两个 `trust` 的语法和语义角色不同。
* **Attention (注意力)** 用 Query、Key、Value 计算 token 间依赖：
$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
* **Masked Multi-head Attention (掩码多头注意力)** 让每个位置只能看见历史 token，符合 autoregressive generation；multi-head 把 embedding 维拆成多个 head，让模型学习不同类型的关系。
* 典型实现中，batch size、head 数、sequence length、head size 会决定 `q/k/v` tensor 的 shape；所有 head 拼接后回到 embedding dimension。

## Part III: MLP, LayerNorm, and Residuals

* **MLP layer (前馈层)** 对每个 token 独立做非线性变换，增加表达能力；attention 负责 token 间交互，MLP 负责每个位置内部的 feature transformation。
* **Layer Normalization (层归一化)** 对一个 token 的 hidden vector 做 feature 维 normalization，使深层训练更稳定。
* **Residual Connection (残差连接)** 让层输出保留输入：
$$
x_{l+1}=x_l+F(x_l)
$$
  它缓解梯度传播困难，使训练大模型更容易。

## Part IV: Interpretability, Training, and Inference

* **Mechanistic Interpretability (机制可解释性)** 试图理解 Transformer 内部哪些 head、neuron、layer 承担具体功能。
* **Training (训练)** 可并行处理序列位置，适合 GPU 大矩阵计算。
* **Inference inefficiency (推理低效)** 来自 autoregressive generation：必须 token-by-token for loop 生成；每一步还可能重复读历史上下文。
* **Review focus**：LayerNorm 稳定训练；attention 随序列长度通常是 quadratic 而不是 linear；Transformer 架构包括 attention、MLP、LayerNorm、residual，不包括 interpretability 本身。

# Lecture 12: Pretraining、Mid-Training、Post-Training、BERT 与 GPT

## Part I: Training Stages

* **Pretraining (预训练)** 用大规模数据教模型语言、语法、推理和世界知识，得到 base foundation model。
* **Mid-training (中训练)** 在高质量领域数据上继续训练，让模型深化专业能力，同时尽量不丢通用知识。
* **Post-training / Alignment (后训练 / 对齐)** 让模型适配人类指令、偏好和具体任务，例如 sentiment analysis、information extraction、question answering。
* 预训练后的模型可通过 **fine-tuning** 或 **prompting** 适配下游任务。

## Part II: Why Pretraining Works

* 预训练通过海量自监督任务学习通用表示；相比从零训练任务模型，pretrained model 可迁移到更多任务。
* 影响 pretraining 的主要因素包括 **model architecture**、**data quantity**、**data quality**、**objective function**。
* 数据数量提供覆盖面，数据质量决定模型学到的模式是否可靠；低质量数据会让模型吸收噪声和偏见。

## Part III: Masked Language Modeling and BERT

* **Masked Language Modeling (MLM)** 随机 mask 输入 token，并训练模型预测被 mask 的词。
* **BERT** 使用 Transformer encoder 和 MLM objective，适合理解类任务，因为它能双向看上下文。
* MLM 的优势是利用左右上下文学习表示；限制是预训练目标与自回归生成不完全一致。

## Part IV: Autoregressive Language Modeling and GPT

* **Autoregressive Language Modeling (自回归语言建模)** 用历史 token 预测下一个 token：
$$
L(\theta)=\sum_D\sum_t -\log p_\theta(w_t|w_{<t})
$$
* **GPT** 使用 decoder-only Transformer 和 autoregressive objective，天然适合生成任务。
* GPT 生成时从左到右逐 token 采样；训练时可并行计算每个位置的 next-token loss。
* **Review focus**：BERT 和 GPT 都用 Transformer，但预训练目标不同；pretrained model 可以进一步 fine-tune。

# Lecture 13: Supervised Fine-Tuning、Alignment 与 SFT 数据

## Part I: SFT and Alignment Motivation

* **Supervised Fine-Tuning (SFT)** 把 base foundation model 调成 aligned model，让模型更好遵循指令和任务格式。
* **Alignment (对齐)** 关注 helpful、honest、harmless 等行为目标。模型不仅要给答案，还要知道不确定性、避免幻觉、遵守安全和用户意图。
* Pretraining 学语言和知识，SFT 学“怎样回答”。

## Part II: Two Kinds of SFT and Data

* SFT 可分为 **task-specific SFT** 和 **instruction / general SFT**。前者强化某类任务，后者提升通用指令跟随。
* SFT 数据来自人工标注、公开 instruction datasets、AI-generated data、对话数据、代码修复轨迹等。
* 课程提到 Alpaca、COIG 等可用数据集，也讨论了 AI generated data 在 SFT 中的作用。

## Part III: SFT Loss and Performance

* SFT 通常最大化参考答案 token likelihood，即对 response 部分做 teacher forcing 的 NLL。
* 如果 prompt 为 $x$、目标回答为 $y$：
$$
L_{SFT}(\theta)=-\sum_t \log p_\theta(y_t|x,y_{<t})
$$
* SFT 性能取决于数据质量、任务覆盖、格式一致性和 curriculum。低质量 SFT 会让模型学到模板化或错误行为。

## Part IV: Industrial SFT and SWE-Lego

* 工业 SFT 常关注 coding agent、bug fixing trajectory、system prompt、任务 prompt、结果验证等完整流程。
* **SWE-Lego** 示例强调 data curation、synthetic data curation、curriculum learning for SFT 和最终效果评估。
* **Multi-task SFT** 需要处理 **catastrophic forgetting (灾难性遗忘)**：模型学新任务时可能忘掉旧能力。
* **Dual-stage mixed fine-tuning** 用少量 specific task data 加 general data，兼顾特定能力和泛化。

# Lecture 14: PEFT、LoRA 与 QLoRA

## Part I: Why PEFT

* 常规 full fine-tuning 需要保存模型参数、梯度、优化器状态和激活，显存成本高。
* **Parameter-Efficient Fine-Tuning (PEFT)** 只训练少量新增参数或低维参数，从而降低显存和存储成本。
* PEFT 的直觉来自 **intrinsic dimension (内在维度)**：大模型适配特定任务时，真正需要改变的方向可能远低于总参数维度。

## Part II: LoRA

* **LoRA (Low-Rank Adaptation)** 假设权重更新 $\Delta W$ 可用低秩矩阵分解：
$$
\Delta W=BA
$$
  其中 rank $r$ 远小于原矩阵维度。
* 训练时冻结原始权重，只训练 $A,B$；推理时可把 LoRA 更新合并回原权重。
* LoRA 初始化通常让初始 $\Delta W$ 接近 0，避免一开始破坏 base model 行为。
* LoRA cost 远低于 full fine-tuning，但效果常能接近全量微调。

## Part III: QLoRA and NF4

* **QLoRA** 把 base model 量化存储，同时在低秩 adapter 上训练，进一步降低显存。
* **NF4 (NormalFloat4)** 针对近似正态分布的权重设计，比普通 4-bit 表示更适合 LLM 权重。
* QLoRA 常结合 quantization、LoRA adapter、paged optimizer 等技术，解决大模型微调显存问题。

## Part IV: Industry Use and Review

* LoRA / QLoRA 在工业中适合多租户、多任务、低成本适配：一个 base model 可挂多个 adapter。
* **Review focus**：full SFT 中参数、梯度和优化器状态占显存；quantization 是把连续/高精度数映射到低精度离散表示；QLoRA 结合量化和 LoRA。

# Lecture 15: RLHF、Reward Model 与 Bradley-Terry

## Part I: Limitations of SFT

* SFT 难处理 open-ended question：开放问题可能没有唯一正确答案，例如诗歌、创意写作、长文本建议。
* 只用 reference answer 的 likelihood 不一定等于人类偏好；模型可能生成 token-level 高概率但整体质量差的答案。
* 因此需要一种能评价整体输出质量的 measurement。

## Part II: Reward Model

* **Reward Model (奖励模型)** 学习给完整回答打分，用于度量 helpfulness、honesty、harmlessness 或人类偏好。
* 奖励模型通常用成对比较数据训练：给同一 prompt 的两个回答，标注 winner 和 loser。
* **Bradley-Terry Model** 把偏好概率建模为：
$$
P(y_w \succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))
$$
* Reward model 的问题包括偏好噪声、reward hacking、无法完美代表人类价值。

## Part III: RLHF

* **RLHF (Reinforcement Learning from Human Feedback)** = preference data + reward model + reinforcement learning。
* 先训练 reward model，再冻结 reward model，用 RL 优化 target LLM 让生成回答获得更高 reward。
* Policy gradient 用样本回报更新参数，但高方差、对 reward scale 敏感，后续 PPO/KL 约束用于稳定训练。
* **Review focus**：RLHF 仍需要 token likelihood / policy probability 来计算 policy gradient；reward model 评价完整句子质量；reward model 由比较标注训练。

# Lecture 16: PPO、KL Divergence、TRPO 与 RLHF 稳定优化

## Part I: From RLHF to PPO

* 本讲标题页提取文本显示 `Lecture 2`，但文件名和上下文为 Lecture 16，应视为课件编号 typo。
* RLHF 中 reward model 训练好后会冻结，target LLM 作为 policy 继续优化。
* **Delayed reward (延迟奖励)**：完整回答生成后才得到 reward，但需要把反馈分配给 token-level policy decisions。

## Part II: Policy Gradient and Collapse

* Policy gradient 直接用 reward 推动参数更新，但对采样、reward scale 和更新步长敏感。
* **Catastrophic collapse (灾难性崩塌)**：policy 若为了高 reward 偏离太远，后续 online data 会被坏 policy 污染，模型行为可能快速退化。
* 单纯沿普通梯度方向更新可能导致 policy distribution 变化过大，因此需要约束。

## Part III: KL Divergence, Natural Gradient, TRPO

* **KL Divergence** 衡量新旧 policy 分布差异，用于约束模型不要远离 reference / old policy。
* KL 可通过 Taylor expansion 与 **Fisher Information Matrix (FIM)** 关联。
* **Natural Gradient** 用参数空间中的分布几何修正梯度方向，使更新更符合 policy distribution 的实际变化。
* **TRPO** 通过 trust region 约束 KL，追求稳定提升，但计算复杂，需要多次 forward/backward 或近似。

## Part IV: PPO

* **PPO (Proximal Policy Optimization)** 用 clipped objective 近似 trust region，使训练更简单稳定。
* PPO 限制新旧 policy probability ratio，不让单步更新过大。
* 在 RLHF 中，PPO 常配合 reward、KL penalty、advantage estimation 使用。
* **Review focus**：RLHF 不会在 target LLM 优化时继续更新 reward model；KL 用于保持 policy 不偏离太远，而不是减少参数量。

# Lecture 17: DPO、GRPO 与 Preference Optimization

## Part I: Why Move Beyond PPO

* RLHF with PPO 较复杂：需要 reward model、policy sampling、advantage、KL penalty、PPO update 等组件。
* Reward model 可能不可靠，且 RL optimization 容易 reward hacking。
* 新方法试图直接从 preference data 优化 policy，减少 RL pipeline 复杂度。

## Part II: DPO

* **DPO (Direct Preference Optimization)** 从 RLHF objective 出发，把最优 reward 写成 policy 和 reference policy 的函数。
* 把 reward definition 代入 Bradley-Terry loss，可直接用 preference pairs 训练 policy，无需显式训练 reward model。
* DPO 保留 reference model 作为 KL 风格约束，使 policy 不至于偏离太远。
* 工业中 DPO 常比 PPO 简洁，但效果依赖 preference data 质量和超参数。

## Part III: GRPO and Extensions

* **GRPO (Group Relative Policy Optimization)** 用同一 query 的多个 samples 计算组内相对 reward / advantage，减少对 critic 的依赖。
* GRPO 的 advantage 是 group-relative，不是来自不同 query 的混合比较。
* 扩展方法包括 **KTO (Kahneman-Tversky Optimization)**、**SimPO (Simple Preference Optimization)** 和 **DAPO**。
* **Review focus**：advantage 在 RLHF 中会使用；Bradley-Terry 比较 winner/loser 时不需要显式 partition function $Z(x)$；GRPO 不是用不同 queries 的 samples 计算同一个 advantage。

# Lecture 18: Synthetic Data：生成、评估与局限

## Part I: Why Synthetic Data

* Llama、GPT、Qwen 等大模型训练都使用 public web text、knowledge documents、code、instruction dialogue、preference data 等多源数据。
* 模型和数据需求增长很快，真实高质量数据增长较慢，可能出现 data bottleneck。
* 重复使用同一训练数据早期有帮助，但重复次数增加后收益快速下降。
* **Synthetic Data (合成数据)** 用 teacher LLM 或其他结构化来源生成新训练样本，用来补充 instruction following、alignment、evaluation、code、math、preference data。

## Part II: Synthesis Methods

* **Prompting a teacher LLM**：直接让强模型生成任务、答案、解释或偏好比较。
* **Retrieve and transform**：检索真实材料后改写成训练样本。
* **Extract and rewrite / rephrasing**：从文本中抽取知识点，再转换为 QA、instruction 或多样表达。
* **Knowledge graph extraction**：从知识图谱抽取事实关系，转成自然语言任务。
* **AI rating**：让模型给样本质量打分或筛选。
* **Self-Instruct / Self-Guide / Evol-Instruct**：让模型自举生成指令并逐步增加复杂度。
* **Multi-agent methods**：多个模型或角色协作生成、批评、修正数据。

## Part III: Evaluation Dimensions

* **Correctness (正确性)**：答案是否事实正确、推理是否成立。
* **Complexity (复杂度)**：任务是否足够有挑战，不只是模板化简单问题。
* **Diversity (多样性)**：覆盖不同领域、格式、推理类型、语言风格。
* **Fidelity (保真度)**：合成数据是否忠实来源材料，不引入幻觉。
* 合成数据也可用于 evaluation，但要防止模型和评测数据同源导致虚高分数。

## Part IV: Limitations

* **Mode collapse (模式坍塌)**：模型生成数据可能丢失训练数据中的细节；反复“生成-训练”会让细节越来越少。
* **Boilerplate responses** 如 “sure I am glad to help” 价值低于深度细节。
* **Lack of data provenance (缺少数据来源追踪)**：合成数据吸收到模型参数后，很难解释模型不良行为来源。

# Lecture 19: Scaling Laws、FLOPs、Kaplan 与 Chinchilla

## Part I: Scaling Motivation

* 模型参数从百万级增长到千亿、万亿级，准确率和能力随规模提升出现规律性趋势。
* **Scaling laws (缩放定律)** 研究 loss / accuracy 如何随 model size、data size、compute 改变。
* 真实世界受 compute budget、data availability、hardware memory 和 inference cost 限制。

## Part II: FLOPs and Transformer Compute

* **FLOPs** 衡量浮点运算次数，是训练计算量的重要单位。
* Matrix-vector multiplication 和 matrix multiplication 的复杂度与矩阵维度相关。
* Transformer 参数量 $N$ 可由层数、hidden dimension、attention heads、MLP dimension 等估计。
* Training compute $C$ 常近似与参数量 $N$ 和训练 token 数 $D$ 成正比。

## Part III: Kaplan vs. Chinchilla

* **Kaplan scaling laws** 强调增大模型尺寸、数据和 compute 会按幂律降低 pretraining cross-entropy loss。
* Kaplan 结论倾向于大模型在固定 compute 下更有效，但可能低估数据量的重要性。
* **Chinchilla scaling law** 强调 compute-optimal training 需要平衡 model size 和 data size；很多模型其实 under-trained，应使用更多 token 训练较小或适中模型。
* 课程问题中的核心关系是 $N$、$C$、$D$ 之间的约束和平衡。

## Part IV: Variants and Limitations

* Scaling law 有很多变体，可针对 downstream performance、data quality、inference compute、architecture 等建模。
* **Inverse scaling (反向缩放)** 指某些任务上模型变大反而表现变差，说明能力提升不是所有维度单调改善。
* 限制包括数据质量、评估污染、任务选择、训练分布变化、硬件瓶颈和社会/经济成本。

# Lecture 20: LLM Inference、KV Cache、Memory Wall、PagedAttention、StreamingLLM

## Part I: KV Cache and Generation Stages

* Autoregressive generation naive 实现会重复计算历史 token 的 K/V；**KV Cache** 保存历史 Key/Value，用内存换时间。
* Decode step 只计算最新 token 的 $q_t,k_t,v_t$，再 attend 到 cached K/V：
$$
K_{cached}=[K_{prev};k_{new}],\quad V_{cached}=[V_{prev};v_{new}]
$$
* **Prefill** 处理完整 prompt，建立初始 cache，通常 compute-bound，决定 TTFT。
* **Decode** 每步生成一个 token，反复读 KV cache，通常 memory-bandwidth bound，决定 TPOT。

## Part II: Memory Wall and Attention Optimizations

* **Memory Wall**：GPU FLOPs 增长快于 memory bandwidth，decode 阶段算力常等待 HBM 读写。
* KV cache 大小随 batch、sequence length、layers、heads、head dim、precision 线性增长：
$$
Size=Batch\times SeqLen\times 2\times Layers\times Heads\times Dim\times 2\ bytes
$$
* **FlashAttention** 用 tiling 和 exact online softmax 减少 attention weights 的 HBM 读写。
* **MQA / GQA** 通过共享 K/V heads 减少 KV cache 和带宽；GQA 是 MHA 与 MQA 的折中。

## Part III: PagedAttention and Scheduling

* 静态分配 KV cache 会产生 internal fragmentation 和 external fragmentation。
* **PagedAttention** 像操作系统虚拟内存一样，用 block table 把 logical KV blocks 映射到 physical GPU blocks。
* 多请求服务中，vLLM 的块式管理提高显存利用率和吞吐。
* Continuous batching 中 prefill 和 decode 混合会产生 bubbles；chunked prefills 和 piggybacked decodes 用调度减少等待。

## Part IV: StreamingLLM

* 长上下文保存完整 KV 会 OOM，简单丢弃早期 token 会导致性能崩溃。
* **Attention Sink** 指初始 tokens 即使语义弱也吸引大量注意力；删除它们会破坏注意力分布。
* **StreamingLLM** 保留若干初始 sink tokens 与最近 window，使 KV cache 固定大小。
* **Review focus**：online softmax 可以精确计算；layer normalization 参数不是 KV cache size 因素。

# Lecture 21: Quantization、Pruning 与 Distillation

## Part I: Efficiency Challenge and Numeric Formats

* Scaling law 推动模型变大，但硬件 memory 和 compute 增长跟不上。70B FP16 权重约 140GB，真实推理还需要 KV cache 和激活。
* **Numeric formats** 决定数值范围、精度和存储成本。低精度可减少内存和带宽，但会带来 accuracy cost。
* **Quantization (量化)** 把高精度数映射到低精度表示，典型形式：
$$
q=round(Sx+Z)
$$

## Part II: QAT, PTQ, and Outliers

* **QAT (Quantization-Aware Training)** 在训练中模拟量化误差，让模型适应低精度。
* **PTQ (Post-Training Quantization)** 在训练后直接量化，成本低但可能更伤精度。
* LLM 中存在 **emergent outlier activations**，只看权重大小可能漏掉重要激活。
* **LLM.int8** 使用 mixed precision 处理 outliers；**SmoothQuant** 在 weight 和 activation 间迁移 scale，使量化更平滑。

## Part III: Pruning

* **Pruning (剪枝)** 删除不重要参数、神经元、attention heads 或结构，降低计算和存储。
* **Activation pruning** 根据激活重要性剪枝；iterative pruning 逐步剪掉部分结构并恢复训练。
* **Structured sparsity (2:4)** 每 4 个权重中保留 2 个，可更容易获得硬件加速。
* 只按 weight magnitude 剪枝可能忽视 activation outliers。

## Part IV: Distillation

* **Knowledge Distillation (知识蒸馏)** 用 teacher model 的输出训练 student model，让小模型模仿大模型。
* 蒸馏可用 soft labels、logits、rationales、step-by-step outputs 或 preference-style signals。
* 蒸馏有产业争议和法律问题，因为 teacher 输出可能来自闭源模型或受版权限制的数据。
* **Review focus**：structured pruning 删除层/神经元等结构，更可能带来真实计算效率；INT8 映射需根据 scale 和 zero point 计算。

# Lecture 22: Mixture of Experts、Routing 与 Sparse Upcycling

## Part I: MoE Motivation

* **Mixture of Experts (MoE)** 的核心是每个 token 只激活模型的一小部分参数，降低每步计算成本。
* Dense model 每次 forward 激活所有 FFN 参数；MoE 用多个 experts 替换或扩展 FFN，并由 router 选择 expert。
* MoE 可在参数总量很大时保持较低 activated parameters。

## Part II: MoE Architecture and Routing

* 在 Transformer 中，MoE 通常替换 MLP / FFN 层。
* **Top-1 routing** 每个 token 选择一个 expert；计算便宜但容易 routing collapse。
* **Top-2 routing** 选择两个 experts，提升稳定性和表达能力但增加计算。
* **Routing collapse** 指 router 总把 token 分到少数 experts，导致负载不均和专家退化。
* 需要 load balancing loss、capacity factor、BASE routing 或其他约束。

## Part III: Training and Building MoE LLMs

* Expert selection 是离散/稀疏操作，但 router score 和被选 expert 可通过 backprop 更新。
* DeepSeek、Mixtral、Qwen MoE 等案例展示了大规模 MoE 的工业实践。
* **Sparse Upcycling** 从已有 dense model 的 FFN 复制或拆分出多个 experts，再继续训练。
* **Sparse Splitting** 则更显式地拆分已有结构形成 experts。
* **Review focus**：从现有 LLM 复制 FFN 创建 experts 是可行路径；MoE 可每 token 选择多个 experts；某 expert 当前 token 未被选中不代表未来 token 不能选中它。

# Lecture 23: Diffusion Models：Forward / Reverse Process 与 DDPM

## Part I: Generative Models and Iterative Refinement

* 生成模型不只 Transformer；diffusion 是另一类重要生成方法。
* **Diffusion model** 通过 iterative refinement 从噪声逐步生成数据。
* Forward stochastic process 把 clean data 逐步加噪；reverse process 学习从噪声恢复数据。

## Part II: Forward Process

* Forward process 通常设计为逐步添加 Gaussian noise，使数据分布逐渐接近简单先验。
* Variance schedule 控制每一步加噪强度。
* 一个关键性质是可直接从 clean data $x_0$ 采样任意时间步 $x_t$，无需逐步模拟所有中间步。

## Part III: Reverse Process and Training Objective

* Reverse process 要学习 $p_\theta(x_{t-1}|x_t)$，但真实 posterior 通常难直接求。
* 若条件在 clean data $x_0$ 上，posterior 可变得 tractable。
* DDPM 的 simplified loss 常可化为训练神经网络预测噪声 $\epsilon$：
$$
L=\mathbb{E}_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|^2
$$
* **U-Net and time embeddings** 是图像 diffusion 的典型实现：网络既看 noisy input，也看时间步。

## Part IV: Sampling and Review

* Sampling 从随机噪声开始，按 reverse steps 逐步去噪生成样本。
* CIFAR-10 progressive generation 展示图像从噪声到清晰对象的过程。
* **Review focus**：训练 diffusion 可归约为预测加入的噪声；两个 Gaussian 变量之和仍是 Gaussian；inference 起点通常从噪声分布采样，不需要已知 clean $x_0$。

# Lecture 24: Score-Based Diffusion、Text Diffusion 与 Block Diffusion

## Part I: Score-Based Perspective

* **Score function** 是：
$$
\nabla_x \log p(x)
$$
  它指向数据分布高密度区域。
* Denoising 可理解为把 noisy sample 沿 score vector field 推回高概率数据流形。
* **Langevin dynamics** 描述粒子在势能场中带随机扰动地移动；若势能 $U(x)=-\log p(x)$，则向低势能移动等价于提高 $p(x)$。

## Part II: Score Matching and Diffusion

* 真实 $p(x)$ 不知道，因此训练神经网络近似 score function。
* 通过给 clean data 加噪得到 $x_t$，条件分布的 log-gradient 容易计算。
* **Denoising Score Matching** 用带噪样本训练模型预测 score，把 score-based modeling 与 diffusion 联系起来。
* Denoising 与 score matching 可放进统一框架。

## Part III: Diffusion for NLP

* 文本是离散 token，不像图像像素那样天然连续；对文本加 Gaussian noise 不一定有语义。
* 离散文本 diffusion 可用 transition matrix 表示 token corruption / replacement。
* 也可在 **continuous latent space** 中做 diffusion，先把文本映射到连续表示，再生成或解码。
* 需要谨慎设计 text perturbation，否则噪声过程可能破坏语义或语法结构。

## Part IV: Control and Block Diffusion

* 可用 classifier 或 classifier-free guidance 控制生成方向。
* 训练 classifier 可帮助模型朝目标类别或条件移动。
* **Block Diffusion** 尝试在文本块级别生成，结合 autoregressive 和 diffusion 的优点。
* 与 autoregressive 相比，diffusion 早期可设定全局语义，后期修局部语法，具备一定 self-correction 能力。

# Lecture 25: Agentic Systems、Memory、RAG、Reflection、Tools 与 MCP

## Part I: What Is an Agentic System

* **Agent** 不是新词，但在 LLM 时代重新流行。
* **Agentic system** 通常包括：LLM as brain、memory、tools、environment、feedback。
* LLM 负责 planning 和 rethinking；memory 保存 persistent perception 和 experience；tools 执行动作；environment 提供状态转移和反馈。
* “If you're not the model, you're the harness” 强调系统编排和上下文工程的重要性。

## Part II: Why Agents Differ from Plain LLMs

* 复杂任务需要 planning、history、reflection、tool use、collaboration。
* 普通 LLM 常是被动地对用户输入生成一次输出，受上下文窗口限制，不能天然持久记忆，也不直接与外部环境交互。
* Agentic systems 可用于 robotics、electricity trading、coding、search、workflow automation 等。

## Part III: Context Engineering, Memory, and RAG

* **Context engineering (上下文工程)** 是给模型组织正确上下文、工具结果、记忆和任务状态。
* **Memory (记忆)** 可保存长期偏好、任务轨迹、反馈、文件状态和中间结论。
* **RAG (Retrieval-Augmented Generation)** 从外部知识库检索相关文档，把证据注入上下文，提升 factuality 和可追溯性。

## Part IV: Reflection, Tools, and Collaboration

* **Reflection (反思)** 让模型根据错误、失败轨迹、critic 或 feedback 修正计划。
* **LLM-as-a-judge** 可用模型评估输出质量，但也要注意 judge bias 和可靠性。
* **Tools (工具)** 包括代码解释器、文件操作、PDF parser、搜索、API、数据库等，让模型拥有“手和眼睛”。
* **Collaboration (协作)** 涉及多 agent、多模型、多服务之间的接口。
* **MCP (Model Context Protocol)** 提供统一接口，允许 $N$ 个模型和 $M$ 个服务从 $N\times M$ 的交互模式降到 $N+M$。

# Formula Cheat Sheet / 公式速查表

## 概率与 MLE

$$
P(A\cup B)=P(A)+P(B)-P(A\cap B)
$$

$$
P(A|B)=\frac{P(A\cap B)}{P(B)}
$$

$$
P(B|A)=\frac{P(A|B)P(B)}{P(A)}
$$

$$
E[X]=\sum_x p(X=x)x
$$

$$
Var[X]=E[(X-E[X])^2]
$$

$$
\hat{\theta}_i=\frac{c_i}{n}
$$

## n-gram 与 Smoothing

$$
P(w_1,\ldots,w_n)=\prod_i P(w_i)
$$

$$
P(w_i|w_{i-1})=\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
$$

$$
P_L(w)=\frac{c(w)+1}{N+|V|}
$$

$$
P_L(w_2|w_1)=\frac{c(w_1,w_2)+1}{c(w_1)+|V|}
$$

## Word2Vec / GloVe

$$
P(w_o|w_c)=\frac{\exp(u_o^Tv_c)}{\sum_w \exp(u_w^Tv_c)}
$$

$$
L=-u_o^Tv_c+\log\sum_w \exp(u_w^Tv_c)
$$

$$
w_i^T\tilde{w}_j+b_i+\tilde{b}_j\approx \log X_{ij}
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
\xi_t(i,j)=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{P(O)}
$$

## CFG / PCFG

$$
G=(N,\Sigma,R,S)
$$

$$
L_G=\{w\in\Sigma^* \mid S\Rightarrow^* w\}
$$

$$
\beta_j(p,q)=\sum_{r,s,d}P(N^j\rightarrow N^rN^s)\beta_r(p,d)\beta_s(d+1,q)
$$

## Transformer / Attention

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
x_{l+1}=x_l+F(x_l)
$$

## Pretraining / SFT

$$
L_{AR}(\theta)=\sum_D\sum_t -\log p_\theta(w_t|w_{<t})
$$

$$
L_{SFT}(\theta)=-\sum_t \log p_\theta(y_t|x,y_{<t})
$$

## RLHF / Preference Optimization

$$
P(y_w \succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))
$$

DPO 把 reward 写成 policy/reference policy 的函数，直接用 preference pairs 优化 policy；GRPO 用同一 query 的多样本组内 reward 估计 advantage。

## Scaling / Inference / Compression

$$
Size_{KV}=Batch\times SeqLen\times 2\times Layers\times Heads\times Dim\times 2\ bytes
$$

$$
q=round(Sx+Z)
$$

Scaling laws 关注 loss 与 model size $N$、data size $D$、compute $C$ 的幂律关系；Chinchilla 强调 compute-optimal 需要同时扩大模型和数据。

## Diffusion / Score

$$
L=\mathbb{E}_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|^2
$$

$$
score(x)=\nabla_x\log p(x)
$$

## Agent / MCP

Agentic system = LLM brain + memory + tools + environment + feedback；MCP 把 $N$ 个模型和 $M$ 个服务的连接从 $N\times M$ 降到 $N+M$。
