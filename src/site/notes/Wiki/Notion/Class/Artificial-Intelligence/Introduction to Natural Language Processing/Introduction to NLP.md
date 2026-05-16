---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/introduction-to-nlp/"}
---


# AIAA 4051 Introduction to Natural Language Processing

> 覆盖范围：Lecture 1-10 与 Lecture 20。整理方式按主题重组，不保留逐页课件页标题；标题页、demo、conclusion、quiz 中的复习重点已合并到对应主题或考试提示中。

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

# Lecture 20: LLM Inference、KV Cache、Memory Wall、PagedAttention、StreamingLLM

## Part I: Autoregressive Inference and KV Cache

* **Autoregressive generation (自回归生成)** 逐 token 生成。naive 实现每预测一个新 token 都重新计算历史 token 的 Key / Value，造成重复计算。
* **Attention formula (注意力公式)**：
$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
* **KV Cache (KV 缓存)** 保存过去 token 的 Key / Value，用 GPU memory 换时间。
$$
K_{cached}=[K_{prev};k_{new}],\quad V_{cached}=[V_{prev};v_{new}]
$$
  decode step $t$ 只需计算最新 token 的 $q_t,k_t,v_t$，然后用 $q_t$ attend 到所有 cached K/V。
* KV cache 避免重复计算历史 K/V，使 decode 阶段 K/V 计算按 token 增量进行；代价是显存占用随序列长度线性增长。

## Part II: Prefill, Decode, and Memory Wall

* **Prefill phase (预填充阶段)**：编码完整 prompt 并构建初始 cache。输入 token 可并行处理，GPU utilization 高，通常 compute-bound，决定 **TTFT (Time to First Token)**。
* **Decode phase (解码阶段)**：每步只处理 1 个 token，GPU compute utilization 低，每步要从 HBM 读取整个 KV cache，通常 memory-bandwidth bound，决定 **TPOT (Time Per Output Token)**。
* **Memory Wall (内存墙)**：GPU 计算速度增长快于内存传输速度，导致算力空闲等待内存。长上下文推理尤其容易被 KV cache 读取拖慢。
* **KV cache growth (KV 缓存增长)**：模型权重大小固定，但每个请求的 cache 随 prompt 和生成长度增长。长文档 OOM 常由 KV cache 导致，而不只是模型权重。
* **KV cache size formula (KV cache 大小公式)**，FP16 下：
$$
Size=Batch\times SeqLen\times 2\times Layers\times Heads\times Dim\times 2\ bytes
$$
  第一个 2 表示 K 和 V，最后的 2 bytes 表示 FP16。

## Part III: Attention and Head-Level Optimizations

* **FlashAttention** 用 tiling block-by-block 在线计算 softmax，不缓存巨大 attention weights。
* **Online softmax (在线 softmax)** 维护最大值、分母部分和加权和，可以精确计算而不是近似。
* **MHA (Multi-Head Attention)** 每个 query head 有独立 K/V，KV cache 较大。
* **MQA (Multi-Query Attention)** 让所有 query heads 共享一组 K/V，显著压缩 KV cache 和内存带宽，但可能损失表达能力。
* **GQA (Grouped Query Attention)** 介于 MHA 与 MQA 之间：query heads 分组，每组共享一个 K/V head。它是实际大模型常用折中，比 MHA 省内存，比 MQA 保留更多表达能力。

## Part IV: Memory Fragmentation and PagedAttention

* **Internal fragmentation (内部碎片)**：系统为每个请求预留最大长度，如 2048 tokens，但用户实际只用 50/300/700 tokens，剩余空间不可用。
* **External fragmentation (外部碎片)**：CUDA 物理空间不连续，小空洞无法分配给新请求。
* **PagedAttention** 借鉴操作系统虚拟内存：
  * 请求像进程。
  * logical memory 像虚拟内存。
  * block table 把逻辑块映射到 GPU physical blocks。
  * block 类似 page。
* KV cache 不必连续存放，block 满了再找新物理块，从而减少大块连续分配需求和碎片。
* **vLLM multi-request cache (多请求缓存)**：多个请求的 KV cache 映射到 block engine 管理的不同 physical blocks，提高显存利用率和吞吐。

## Part V: Continuous Batching and Scheduling

* **Continuous batching (连续批处理)** 中，iteration-level scheduling 可能造成 GPU resource underutilization 和 bubbles。
* 一个请求的 decode 可能等待其他请求 prefill 完成；prefill 和 decode 的计算特性不同，调度不当会让 GPU 空等。
* **Chunked Prefills (分块预填充)** 把完整 prefill 拆成固定大小 chunks。
* **Piggybacked Decodes (搭载式解码)** 在处理 prefill chunk 时利用剩余计算容量搭载 decode task。
* 这种调度减少 pipeline bubbles，让 decode 不必长时间等待大 prefill，提高连续 batching 效率。

## Part VI: Long Context and StreamingLLM

* **Long Context Memory Wall (长上下文内存墙)**：
  * 保存完整 memory 会爆炸并 OOM。
  * 简单驱逐早期 tokens 会让性能崩溃。
  * 反复 $O(L^2)$ 重算极慢。
* **Attention Sink (注意力汇)**：观察到初始 tokens 即使语义不强，也会吸引注意力。可能原因包括 softmax 归一化、位置编码，以及自回归训练中早期位置被看到最多。
* 删除初始 tokens 会破坏注意力分布，导致 perplexity 恶化；PPL 越低越好。
* **StreamingLLM** 保留 attention sink（若干初始 tokens）和最近 tokens，使 KV cache 固定大小。
* 一些 dummy characters（如换行）也可作为 sink 维持性能。
* **Attention map observation (注意力图观察)**：无 sink token 时，低层偏局部注意力，深层对初始 tokens 注意力增加；有 sink token 时，各层明显关注 sink token，让冗余注意力集中到稳定位置。
* **Review focus (复习重点)**：online softmax 可以精确计算；KV cache 大小因素包括 layers 和生成序列长度，layer normalization 参数不是 KV cache 因素。

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
\alpha_1(i)=\pi_i b_i(o_1)
$$

$$
\alpha_t(j)=\sum_i \alpha_{t-1}(i)a_{ij}b_j(o_t)
$$

$$
\beta_T(i)=1
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
P(w_1,\ldots,w_m|G)=\sum_t P(t,w_1,\ldots,w_m|G)
$$

$$
\beta_j(p,q)=\sum_{r,s,d}P(N^j\rightarrow N^rN^s)\beta_r(p,d)\beta_s(d+1,q)
$$

Viterbi-style PCFG parsing：把 Inside 的 $\sum$ 换成 $\max$，并记录 backpointer。

## RNN / Seq2Seq / Attention

$$
a=\sigma(w^Tx+b)
$$

$$
h_t=\tanh(Wh_{t-1}+Ux_t+b)
$$

$$
\hat{y}_t=softmax(Vh_t+c)
$$

$$
P(y_1,\ldots,y_T|x_1,\ldots,x_S)
$$

$$
c_t=\sum_i \alpha_{t,i}h_i
$$

$$
\alpha_{t,i}=softmax(score(s_{t-1},h_i))
$$

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## MT 与 BLEU

$$
E^*=\arg\max_E P(F|E)P(E)
$$

$$
P(F,A|E)=P(J|I)P(A|I,J)\prod_j P(f_j|e_{a_j})
$$

$$
P(F,A|E)=P(J|I)\prod_j P(a_j|a_{j-1},I)P(f_j|e_{a_j})
$$

BLEU 主要由 $n=1,\ldots,4$ 的 modified n-gram precisions 的几何平均构成，并配合短句惩罚。

## LLM 推理与 KV Cache

$$
Size=Batch\times SeqLen\times 2\times Layers\times Heads\times Dim\times 2\ bytes
$$

$$
K_{cached}=[K_{prev};k_{new}],\quad V_{cached}=[V_{prev};v_{new}]
$$

$$
softmax\left(\frac{q_tK_{cached}^T}{\sqrt{d_k}}\right)V_{cached}
$$
