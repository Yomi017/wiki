---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/proof/inside-algorithm-derivation/"}
---


### 故事的起点：Inside 是概率版 CYK

CYK cell 存 boolean：某个 non-terminal 能不能生成 span。Inside cell 存概率：

$$
\beta_A(i,j)=P(A\Rightarrow^* w_i\ldots w_j)
$$

### 推理过程

若 grammar 为 CNF，长 span 的最后一步是：

$$
A\to BC
$$

并选择切分点 $k$，左边由 $B$ 生成，右边由 $C$ 生成：

$$
\beta_A(i,j)=\sum_{A\to BC}\sum_k P(A\to BC)\beta_B(i,k)\beta_C(k+1,j)
$$

这里求和是因为可能有多条 rule 和多个 split 都能生成同一个 span。

### Exam Focus

Inside 是 sum over derivations；Viterbi PCFG 则把 sum 换成 max。

### 和 CYK 的对应关系

CYK cell 是 boolean/set：

$$
A\in table[i,j]
$$

Inside cell 是 probability：

$$
\beta_A(i,j)=P(A\Rightarrow^*w_i\ldots w_j)
$$

如果某个 split/rule 可行，CYK 只记录“可行”；Inside 要把这条 derivation 的概率加进去：

$$
P(A\to BC)\beta_B(i,k)\beta_C(k+1,j)
$$

如果有多个 split 或多条 rule 都能生成同一 span，Inside 对它们求和，因为它在 marginalize all parse subtrees。
