---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/formula/lecture-05-formula/"}
---



# Lecture 5 Formula: Viterbi and Baum-Welch

## Core Formula Map

- Viterbi 找单条最可能 hidden path。
- Baum-Welch 是 HMM 的 EM，用 Forward/Backward 得到 soft counts。

## Formula Details

### Viterbi State

**Formula**

$$
v_t(j)=\max_{q_{1:t-1}}P(q_{1:t-1},q_t=j,o_{1:t})
$$

**Meaning**：$v_t(j)$ 是到 $t$ 并以 $j$ 结尾的最佳路径概率。

**Intuition**：和 Forward 类似，但把所有路径求和换成只保留最佳路径。

**When to Use**：Algorithm trace、best tag sequence。


### Viterbi Recurrence

**Formula**

$$
v_1(j)=\pi_jb_j(o_1),\qquad v_t(j)=\max_i v_{t-1}(i)a_{ij}b_j(o_t)
$$

**Meaning**：$i$ 枚举前一状态；同时保存 backpointer。

**Intuition**：当前最佳路径一定由某个前一状态最佳路径延伸而来。

**When to Use**：They base、POS decoding。

**Derivation / Proof**：principle of optimality：最佳完整路径的前缀也必须最佳。


### Posterior State Probability

**Formula**

$$
\gamma_t(i)=P(q_t=i|O,\lambda)=\frac{\alpha_t(i)\beta_t(i)}{P(O)}
$$

**Meaning**：$\gamma_t(i)$ 是时间 $t$ 属于状态 $i$ 的 posterior soft count。

**Intuition**：过去概率 × 未来概率，再除以整句概率归一化。

**When to Use**：Baum-Welch E-step、soft tagging。


### Posterior Transition Probability

**Formula**

$$
\xi_t(i,j)=P(q_t=i,q_{t+1}=j|O,\lambda)=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{P(O)}
$$

**Meaning**：$\xi_t(i,j)$ 是从 $i$ 到 $j$ 的 transition soft count。

**Intuition**：把一条跨步 transition 放在中间，左右分别接 forward/backward。

**When to Use**：EM 更新 transition matrix。


### Baum-Welch Updates

**Formula**

$$
a_{ij}^{new}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)},\qquad b_i(o)^{new}=\frac{\sum_{t:o_t=o}\gamma_t(i)}{\sum_{t=1}^{T}\gamma_t(i)}
$$

**Meaning**：分子是期望次数，分母是归一化总 soft count。

**Intuition**：把 supervised MLE 的 hard counts 换成 posterior soft counts。

**When to Use**：EM/Baum-Welch 概念题。

**Exam / Homework Trap**：EM 不保证全局最优，只保证局部改进/收敛到局部最优。


## Derivation / Proof Notes

Viterbi 是 max-product；Forward 是 sum-product；Baum-Welch 用 posterior soft counts 做 M-step。

Viterbi 和 Forward 的表面形式很像，差别只有运算符：

$$
\alpha_t(j)=\sum_i\alpha_{t-1}(i)a_{ij}b_j(o_t)
$$

$$
v_t(j)=\max_i v_{t-1}(i)a_{ij}b_j(o_t)
$$

Forward 要整句概率，所以所有路径都要加起来；Viterbi 要最优路径，所以只保留最大来源，并记录 backpointer：

$$
bp_t(j)=\arg\max_i v_{t-1}(i)a_{ij}b_j(o_t)
$$

Baum-Welch 的 E-step 把隐藏路径变成 soft counts：

$$
ExpectedCount(i\to j)=\sum_{t=1}^{T-1}\xi_t(i,j)
$$

$$
ExpectedCount(i)=\sum_{t=1}^{T-1}\gamma_t(i)
$$

M-step 再像 supervised MLE 一样归一化：

$$
a_{ij}^{new}=\frac{ExpectedCount(i\to j)}{ExpectedCount(i)}
$$

emission 同理：

$$
b_i(o)^{new}=\frac{ExpectedCount(i\ emits\ o)}{ExpectedCount(i)}
$$

考试中若给你 $\alpha,\beta$ table，先算 $P(O)$，再算 $\gamma,\xi$；不要直接把 $\alpha_t(i)$ 当成状态 posterior。

## Exam / Homework Traps

- $\gamma$ 不能只由 $\alpha$ 决定，还要乘 $\beta$。
- Viterbi backpointer 用来恢复路径，不只是概率。
