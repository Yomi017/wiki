---
{"dg-publish":true,"permalink":"/notion/class/artificial-intelligence/mathematics-for-ai/"}
---

# Lecture 1: Linear Algebra: Systems of Linear Equations, Matrices, Vector Spaces, Linear Independence

## Part I

*   **Vector:**
    Objects that can be added together and scaled (multiplied by scalars). These operations must satisfy certain axioms (e.g., commutativity of addition, distributivity of scalar multiplication over vector addition).
    *   **Examples:** Geometric vectors (arrows in 2D/3D space), Polynomials ($ax^2+bx+c$), Audio signals, Tuples in $\mathbb{R}^n$ (e.g., $(x_1, x_2, \ldots, x_n)$).

*   **Closure (封闭性):**
    A fundamental property of a set with respect to specific operations (here, vector addition and scalar multiplication). It means that if you take any two elements from the set and perform the operation, the result will *always* also be an element of that same set. If a non-empty set of vectors satisfies closure under vector addition and scalar multiplication (along with other axioms), it forms a **Vector Space**.
    *   **Significance:** Closure ensures that the algebraic structure (the set and its operations) is self-contained and consistent. It's a cornerstone for defining what a vector space is.
*   **Solution of the linear equation system:**
    An *n*-tuple $(x_1,\cdots,x_n)\in\mathbb R^n$ that simultaneously satisfies *all* equations in a given system of linear equations. Each component $x_i$ represents the value for the corresponding variable.
    *   **Connection to Vectors:** Each such *n*-tuple is itself a **vector** in $\mathbb{R}^n$. Therefore, finding the solutions to a linear system is equivalent to finding specific vectors that fulfill the given conditions.
*   **A system have:**
	*   **No solution (inconsistent)**
	*   **Exactly one solution (unique)**
	*   **Infinity solutions (undetermined)**
*   **Matrix Notation**![Pasted image 20250616183101.png](/img/user/Pasted%20image%2020250616183101.png)

	* **could be written as:** $Wx=b$

## Part II
*   **Matrix Addition:**
	* For $A,B\in \mathbb R^{n\times m}$:$$(A+B)_{ij}=a_{ij}+b_{ij}$$
*   **Matrix Multiplication**
	* For $A\in\mathbb R^{m\times n},B\in\mathbb R^{n\times k},C=AB\in\mathbb R^{m\times k}$:$$c_{ij}=\sum_{l=1}^na_{il}b_{lj}$$
	* **Multiplication is only defined if the inner dimensions match:  $$A_{m\times n}B_{n\times k}$$
	* **Elementwise multiplication** is called the **Hadamard product: $$(A\circ B)_{ij}=a_{ij}b_{ij}$$
	* **Identity matrix: $$l_n=\begin{pmatrix}1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1\end{pmatrix}\in \mathbb R^{n\times n}$$
	* **Multiplicative identity:$$I_nA=AI_n=A\in \mathbb R^{m\times n}$$
	* **Algebraic properties:**
		* Associativity: $(AB)C=A(BC)$
		* Distributivity: $(A+B)C=AC+BC,A(C+D)=AC+AD$
	* 