---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/machine-learning/machine-learning/"}
---

# Lecture 1: Logistics and introduction

## Definition:
A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in **T**, as measured by **P**, improves with **experience E**.
*   **Experience E (data)**: games played by the program or human
*   **Performance measure P**: winning rate
*   **Task T**: to win

## Taxonomy of Machine Learning (A Simplistic View)

### 1. Supervised Learning
*   **Core idea**: Learns from **labeled data**.
*   **Example Tasks**:
    *   **Regression**: The prediction result is a **continuous** variable.
        *   e.g., price prediction
        *   (x = Area) → (y = Price)?
    *   **Classification**: The prediction result is a **discrete** variable.
        *   e.g., type prediction
        *   (x = Area, y = Price) → (z = Type)?

### 2. Unsupervised Learning
*   **Core idea**: Learns from **unlabeled data**.
*   **Example Tasks**:
    *   **Clustering**:
        *   Given a dataset containing *n* samples:
            (x⁽¹⁾, y⁽¹⁾), (x⁽²⁾, y⁽²⁾), (x⁽³⁾, y⁽³⁾), ..., (x⁽ⁿ⁾, y⁽ⁿ⁾)
        *   **Task (vague)**: find interesting structures in the data.

### 3. Semi-supervised Learning
*   **Core idea**: Learns from a mix of labeled and unlabeled data.

### 4. Reinforcement Learning
*   **Core idea**: Learns from **environment feedback** (rewards/penalties).
*   **Example Task**: Multi-armed bandit problem (MAB)
    *   it involves a feedback loop between an **Agent** and an **Environment**:
        1.  The **Agent** takes an **Action** (e.g., pull an arm).
        2.  The **Environment** returns **Feedback** (e.g., a corresponding reward).
        3.  The agent learns from this feedback to make better decisions in the future.

### Learning Modes (When do we collect data?)
*   **Offline learning**: The model is trained on a static dataset before deployment.
*   **Online learning**: The model is trained incrementally as new data becomes available.

## Mathematical Tools

[[Wiki/Notion/Class/Artificial-Intelligence/Mathematics for AI/Mathematics for AI\|Mathematics for Ai]]

[[Wiki/Notion/Class/Artificial-Intelligence/Mathematics for AI/Mathematics for AI (Chinese)\|Mathematics for Ai (Chinese)]]

### Parameter Estimation: Maximum Likelihood Estimation (MLE)
A foundational method for estimating the parameters of a statistical model from data.

*   **Core Principle**: Find the parameter values ($θ$) that maximize the **likelihood function**. in other words, we find the parameters that make the observed data most probable.
    $\hat{θ_{ML}} = \displaystyle\arg \max_θ P(Data | θ)$

*   **Example: MLE for Bernoulli Distribution**
    *   **Scenario**: We have a dataset $D = {X₁, ..., Xₘ}$ from a Bernoulli distribution (e.g., $m$ coin flips), where $X_i$ is 1 (heads) or 0 (tails).
    *   **Goal**: Estimate the probability of heads, $θ = P(X=1)$.
    *   **Derivation Steps**:
        1.  **Likelihood Function**: Assuming data points are independent and identically distributed (i.i.d.), the likelihood of observing the entire dataset is the product of individual probabilities:
            $L(θ) = \prod_i p(X_i; θ)$
        2.  **Log-Likelihood Function**: To simplify calculation (turning products into sums) and for numerical stability, we maximize the log-likelihood, which is equivalent.
            $\log L(θ) = \sum_i \log p(X_i; θ)$
        3.  **Substitute Bernoulli PMF**: The probability mass function for a single $Xi$ can be written as $p(X_i; θ) = θ{(X_i)} * (1-θ){(1-X_i)}$. its logarithm is $\log p(Xi; θ) = Xi \log(θ) + (1-X_i) \log(1-θ)$. The log-likelihood becomes:
            $\log L(θ) = \sum_i [X_i log(θ) + (1-X_i) log(1-θ)]$
        4.  **Simplify**: Let $M₁ = \sum_i X_i$ be the total count of 1s (heads). The total count of 0s is $m - M₁$. The expression simplifies to:
            $\log L(θ) = M₁ \log(θ) + (m - M₁) \log(1-θ)$
    *   **Result**: To find the maximum, we take the derivative of this final expression with respect to $θ$, set it to 0, and solve. The resulting estimate is highly intuitive:
        **$\hat{θ_{ML}} = \dfrac {M_1} m$** (The proportion of 1s in the data)