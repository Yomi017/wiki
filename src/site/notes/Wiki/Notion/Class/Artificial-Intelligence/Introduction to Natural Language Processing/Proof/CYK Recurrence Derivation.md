---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/proof/cyk-recurrence-derivation/"}
---


### 故事的起点：CYK 为什么能用 span 动态规划

CNF grammar 只有两类规则：

$$
A\to BC,\qquad A\to a
$$

如果 $A$ 能生成 span $(i,j)$，且 span 长度大于 1，那么最后一步一定是某条 binary rule：

$$
A\to BC
$$

### 推理过程

因为 $A\to BC$，所以 $B$ 生成左半段，$C$ 生成右半段。存在某个切分点 $k$：

$$
i<k<j
$$

使得：

$$
B\Rightarrow^* w_i\ldots w_{k-1}
$$

$$
C\Rightarrow^* w_k\ldots w_{j-1}
$$

因此：

$$
A\in table[i,j]
$$

当且仅当存在 $k,B,C$：

$$
B\in table[i,k],\quad C\in table[k,j],\quad A\to BC
$$

### Exam Focus

CYK 是 bottom-up；先填 lexical cells，再按 span length 递增填表。复杂度来自 span、split、rule 的枚举。

### 常见易错点

如果一个 cell 里有多个 non-terminals，不代表最终 parse 一定使用它们；它只表示“这个 span 可以被这些 non-terminals 生成”。最终是否能 parse 成句子，还要看起始符号：

$$
S\in table[1,n]
$$

CNF 中每个内部节点都是二叉合并，所以长度为 $n$ 的句子在严格二叉 parse tree 中有：

$$
n\ \text{lexical rule applications}+(n-1)\ \text{binary rule applications}=2n-1
$$

这类题常用来检查你是否理解 CNF parse tree 的结构。
