---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/machine-learning/machine-learning/"}
---

# Lecture 1: Logistics and Introduction

## Definition:
A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in **T**, as measured by **P**, improves with **experience E**.
*   **Experience E (data)**: games played by the program or human
*   **Performance measure P**: winning rate
*   **Task T**: to win

## Taxonomy of Machine Learning (A Simplistic View)

### 1. Supervised Learning
*   **Core Idea**: Learns from **labeled data**.
*   **Example Tasks**:
    *   **Regression**: The prediction result is a **continuous** variable.
        *   e.g., price prediction
        *   (x = Area) → (y = Price)?
    *   **Classification**: The prediction result is a **discrete** variable.
        *   e.g., type prediction
        *   (x = Area, y = Price) → (z = Type)?

### 2. Unsupervised Learning
*   **Core Idea**: Learns from **unlabeled data**.
*   **Example Tasks**:
    *   **Clustering**:
        *   Given a dataset containing *n* samples:
            (x⁽¹⁾, y⁽¹⁾), (x⁽²⁾, y⁽²⁾), (x⁽³⁾, y⁽³⁾), ..., (x⁽ⁿ⁾, y⁽ⁿ⁾)
        *   **Task (vague)**: find interesting structures in the data.

### 3. Semi-supervised Learning
*   **Core Idea**: Learns from a mix of labeled and unlabeled data.

### 4. Reinforcement Learning
*   **Core Idea**: Learns from **environment feedback** (rewards/penalties).
*   **Example Task**:
    *   Multi-armed bandit

### Learning Modes (When do we collect data?)
*   **Offline learning**: The model is trained on a static dataset before deployment.
*   **Online learning**: The model is trained incrementally as new data becomes available.
