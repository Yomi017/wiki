---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/introduction-to-natural-language-processing/proof/forward-backward-identity/"}
---


### 故事的起点：为什么 Forward 和 Backward 能拼出整句概率

Forward variable:

$$
\alpha_t(j)=P(o_{1:t},q_t=j)
$$

Backward variable:

$$
\beta_t(j)=P(o_{t+1:T}|q_t=j)
$$

我们要证明：

$$
P(O)=\sum_j\alpha_t(j)\beta_t(j)
$$

### 推理过程

对任意状态 $j$：

$$
\alpha_t(j)\beta_t(j)
=P(o_{1:t},q_t=j)P(o_{t+1:T}|q_t=j)
$$

根据 HMM 条件独立性，未来观测在给定 $q_t$ 后与过去独立：

$$
P(o_{1:T},q_t=j)=P(o_{1:t},q_t=j)P(o_{t+1:T}|q_t=j)
$$

所以：

$$
\alpha_t(j)\beta_t(j)=P(O,q_t=j)
$$

对所有可能当前状态求和：

$$
\sum_j\alpha_t(j)\beta_t(j)=\sum_jP(O,q_t=j)=P(O)
$$

### Exam Focus

不要写成 $\sum_j\beta_1(j)=P(O)$。Backward alone 缺少初始状态和第一个 emission。

### 做题模板

如果题目给出某个时间点 $t$ 的 forward/backward table：

1. 先用任意一种方式求整句概率：

$$
P(O)=\sum_j\alpha_T(j)
$$

或：

$$
P(O)=\sum_j\alpha_t(j)\beta_t(j)
$$

2. 再算 posterior：

$$
P(q_t=j|O)=\frac{\alpha_t(j)\beta_t(j)}{P(O)}
$$

3. 如果要算 transition posterior：

$$
P(q_t=i,q_{t+1}=j|O)=
\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{P(O)}
$$

最重要的检查：posterior 对所有状态求和必须等于 1。
