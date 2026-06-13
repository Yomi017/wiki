---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-06-formula/"}
---



# Lecture 6 Formula: CFG, CNF, and CYK

## Core Formula Map

- CFG 描述句法结构，CYK 在 CNF 上用动态规划解析。
- 核心公式是 grammar definition、CNF rule form、span recurrence 和复杂度。

## Formula Details

### CFG Definition

**Formula**

$$
G=(N,\Sigma,R,S),\qquad L_G=\{w\in\Sigma^*\mid S\Rightarrow^*w\}
$$

**Meaning**：$N$ 非终结符，$\Sigma$ 终结符，$R$ 产生式，$S$ 起始符号。

**Intuition**：CFG 定义哪些 terminal strings 可由起始符号推导。

**When to Use**：formal definition、derivation、language。


### CNF Rule Forms

**Formula**

$$
A\to BC,\qquad A\to a
$$

**Meaning**：$A,B,C$ 是 non-terminals；$a$ 是 terminal。

**Intuition**：二叉结构让 span 可以由两个子 span 合成。

**When to Use**：CYK 前置条件、CNF conversion。


### CYK Recurrence

**Formula**

$$
A\in table[i,j]\ \text{if}\ \exists k,B,C:\ B\in table[i,k],\ C\in table[k,j],\ A\to BC
$$

**Meaning**：$table[i,j]$ 存能生成 span $i..j$ 的 non-terminals。

**Intuition**：从短 span 合成长 span，避免重复解析相同子串。

**When to Use**：CYK trace、ambiguity detection。

**Derivation / Proof**：完整证明见 [[Wiki/Notion/Class/Artificial-Intelligence/Introduction to Natural Language Processing/Proof/CYK Recurrence Derivation\|CYK Recurrence Derivation]]。


### CYK Complexity

**Formula**

$$
O(n^3|R|)
$$

**Meaning**：$n$ 是句长，$|R|$ 是规则数量。

**Intuition**：枚举 span 起点/终点/切分点，再检查规则。

**When to Use**：复杂度选择题。

**Exam / Homework Trap**：CYK 是 bottom-up DP；不是 greedy parser。


### Strict CNF Rule Count

**Formula**

$$
\text{rule applications}=n+(n-1)=2n-1
$$

**Meaning**：$n$ 个词需要 $n$ 个 lexical rules 和 $n-1$ 个 binary rules。

**Intuition**：一棵二叉 parse tree 有 $n$ 个叶子和 $n-1$ 个内部合并。

**When to Use**：Homework CNF rule application 题。


## Derivation / Proof Notes

CYK recurrence 的关键是二叉划分 span；CNF 把任意长 RHS 变成二叉组合。

CYK table 的标准索引可以按闭区间或半开区间写，做题时先统一。若用闭区间 $[i,j]$：

1. 初始化 lexical cells：

$$
A\in table[i,i]\quad \text{if}\quad A\to w_i
$$

2. 对长度 $\ell=2,\ldots,n$ 的 span，枚举切分点 $k$：

$$
[i,j]=[i,k]\cup[k+1,j]
$$

3. 若有 rule $A\to BC$ 且：

$$
B\in table[i,k],\qquad C\in table[k+1,j]
$$

则：

$$
A\in table[i,j]
$$

4. 终止条件：

$$
S\in table[1,n]
$$

则句子可由 grammar 生成。

复杂度直觉：span 有 $O(n^2)$ 个，每个 span 枚举 $O(n)$ 个 split，再检查 rules，所以常写 $O(n^3|R|)$。如果按每个 split 遍历所有二元规则，也得到同阶。

CNF conversion 的目的不是改变语言，而是让每次合并只有两个子 span；这正是 CYK 能用二维 table 的原因。

## Exam / Homework Traps

- 一个 non-terminal 出现在局部 cell 不代表它一定在最终 parse 中使用。
- attachment ambiguity 会导致同一 span/全句有多种 derivation。
