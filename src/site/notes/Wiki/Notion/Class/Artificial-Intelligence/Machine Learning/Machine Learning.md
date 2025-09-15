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

# Lecture 3: Supervised Learning: Regression and Classification II

### **Ridge Regression: Study Notes**

#### **1. Core Idea & Philosophy**

Ridge Regression is an enhancement of Ordinary Least Squares (OLS). Its core idea is to address the problem of **overfitting** by **sacrificing a small amount of bias to achieve a significant reduction in model variance**.

**Analogy:**
*   **OLS** is like a "novice detective" who tries to create a complex theory that perfectly explains 100% of the current evidence. This theory is often fragile and performs poorly on new cases (high variance).
*   **Ridge Regression** is like a "veteran detective" who knows that evidence contains noise and coincidences. He seeks a **simpler, more general theory** that might not explain every tiny detail perfectly but is more robust and predictive for new cases (low variance).

---

#### **2. Motivation: Why is Ridge Regression Necessary?**

Ridge Regression is primarily designed to solve critical failures of OLS that occur in a specific, common scenario.

**The Problem Scenario: High-Dimension, Low-Sample Data (d >> m)**
*   The number of **features $d$** is much larger than the number of **samples $m$**.
*   This is common in modern datasets like genetics, finance, and text analysis.

**Problems for OLS in this Scenario:**

**A. Conceptual Level: Severe Overfitting**
*   With more features than samples, the model has enough flexibility to "memorize" the training data, including its noise and random fluctuations.
*   This results in learned model weights ($w$) that are **absurdly large**, assigning huge importance to potentially irrelevant features.
*   The model performs perfectly on training data but fails to generalize to unseen test data.

**B. Mathematical Level: The Matrix $XᵀX$ is Non-Invertible (Singular)**
*   The analytical solution for OLS is: $w* = (XᵀX)⁻¹Xᵀy$.
*   This solution requires the matrix $XᵀX$ to be invertible.
*   **$XᵀX$ is often non-invertible or ill-conditioned (nearly non-invertible) in two cases:**
    1.  **When $m < d+1$ (Fewer Samples than Features):** This is the main reason. By linear algebra, $rank(XᵀX) = rank(X) ≤ min(m, d+1)$. If $m < d+1$, the rank of $XᵀX$ is less than its dimension ($d+1$), meaning it is not full rank and is therefore **guaranteed to be singular (non-invertible)**.
    2.  **Multicollinearity:** When features are highly correlated (e.g., including both "house size in sq. meters" and "house size in sq. feet"). This makes the columns of $X$ linearly dependent, which in turn makes $XᵀX$ singular.

---

#### **3. The Solution: How Ridge Regression Works**

Ridge Regression introduces a **penalty term** into the objective function to constrain the size of the model's weights.

**Step 1: Modify the Objective Function**

*   **Ordinary Least Squares (OLS) Objective:**
    *   Minimize the **Sum of Squared Errors (SSE)**
    *   $J_{OLS}(w) = ||y - Xw||₂²$

*   **Ridge Regression Objective:**
    *   Minimize **SSE + λ * Penalty on Model Complexity**
    *   $J_{Ridge}(w) = ||y - Xw||₂² + λ||w||₂²$

*   **Dissecting the Penalty Term:**
    *   $||w||₂² = Σ(wᵢ)²$: This is the squared **L2-Norm** of the weight vector. It is the sum of the squares of all weights. A large value implies a complex model with large weights.
    *   **$λ$ (Lambda)**: The **regularization parameter**. This is a hyperparameter that we set to control the strength of the penalty.
        *   **Large $λ$**: Strong penalty. The model is forced to shrink the weights towards zero to avoid a large penalty.
        *   **Small $λ$**: Weak penalty. The model behaves more like OLS.
        *   **$λ = 0$**: No penalty. Ridge Regression becomes identical to OLS.

**Step 2: Derive the New Analytical Solution**

By taking the derivative of the new objective function with respect to $w$ and setting it to zero, we get the analytical solution for Ridge Regression:

> $w^*_{ridge} = (XᵀX + λI)⁻¹ Xᵀy$

*   $I$ is the **Identity Matrix**.
*   Compared to the OLS solution, the only difference is the addition of the $+ λI$ term. This single term is what solves the non-invertibility problem.

---

#### **4. The Core Insight: Why $+ λI$ is the "Special Medicine"**

This term works by altering the **eigenvalues** of the matrix to guarantee its invertibility.

1.  **Invertibility and Eigenvalues**: A matrix is **singular (non-invertible)** if and only if it has **at least one eigenvalue that is 0**.

2.  **Eigenvalues of $XᵀX$**: $XᵀX$ is a positive semi-definite matrix, which means all its eigenvalues $μᵢ$ are **non-negative ($μᵢ ≥ 0$)**. When it's singular, it has at least one eigenvalue $μᵢ = 0$.

3.  **The Effect of $+ λI$**: When we add $λI$ to $XᵀX$, the eigenvalues of the new matrix $(XᵀX + λI)$ become $(μᵢ + λ)$.

4.  **The Result**:
    *   We choose $λ$ to be a **small positive number ($λ > 0$)**.
    *   The original eigenvalues were $μᵢ ≥ 0$.
    *   The new eigenvalues are $μᵢ + λ$.
    *   Therefore, **all new eigenvalues are strictly positive ($> 0$)**.
    *   A matrix whose eigenvalues are all greater than zero is **guaranteed to be invertible**.

**Conclusion:** The $λI$ term acts as a "stabilizer" by shifting all eigenvalues of $XᵀX$ up by a positive amount $λ$, ensuring that none are zero and thus making the matrix $(XᵀX + λI)$ invertible.

---

#### **5. Summary & Comparison**

| Aspect | Ordinary Least Squares (OLS) | Ridge Regression |
| :--- | :--- | :--- |
| **Objective Function** | Minimize SSE | Minimize [SSE + λ * L2-Norm of weights] |
| **Model Complexity** | Unconstrained, weights can be very large | Constrained by L2 penalty, forcing weights to be smaller |
| **Handling $d > m$** | $XᵀX$ is singular; no stable solution exists | $(XᵀX + λI)$ is always invertible; provides a stable solution |
| **Analytical Solution** | $(XᵀX)⁻¹Xᵀy$ | $(XᵀX + λI)⁻¹Xᵀy$ |
| **Key Property** | Unbiased estimate, but can have very high variance | Biased estimate, but with significantly lower variance |
| **Best Use Case** | Low-dimensional data, no multicollinearity | High-dimensional data (esp. $d>m$), multicollinearity is present |

---

#### **6. How to Choose $λ$?**

$λ$ is a critical hyperparameter that controls the trade-off between bias and variance. It is not learned from the training data. The optimal value of $λ$ is typically found using **Cross-Validation**, a technique that evaluates the model's performance on unseen data for different $λ$ values and selects the one that generalizes best.