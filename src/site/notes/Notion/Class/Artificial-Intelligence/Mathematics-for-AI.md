---
{"dg-publish":true,"permalink":"/notion/class/artificial-intelligence/mathematics-for-ai/"}
---

# Lecture 1: Linear Algebra: Systems of Linear Equations, Matrices, Vector Spaces, Linear Independence

## Part I: Concept

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
	*   **Infinity solutions (underdetermined)**
*   **Matrix Notation**![Pasted image 20250616183101.png](/img/user/Pasted%20image%2020250616183101.png)

	* **could be written as:** $Wx=b$

## Part II: Matrix Operation

*   **Matrix Addition:**
	* For $A,B\in \mathbb R^{n\times m}$:$$(A+B)_{ij}=a_{ij}+b_{ij}$$
*   **Matrix Multiplication**
	* For $A\in\mathbb R^{m\times n},B\in\mathbb R^{n\times k},C=AB\in\mathbb R^{m\times k}$:
	$$c_{ij}=\sum_{l=1}^na_{il}b_{lj}$$
	* **Multiplication is only defined if the inner dimensions match:**  
	$$A_{m\times n}B_{n\times k}$$
	* **Elementwise multiplication** is called the **Hadamard product:**
	$$(A\circ B)_{ij}=a_{ij}b_{ij}$$
	* **Identity matrix:** 
	$$l_n=\begin{pmatrix}1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1\end{pmatrix}\in \mathbb R^{n\times n}$$
	* **Multiplicative identity:**
	$$I_nA=AI_n=A\in \mathbb R^{m\times n}$$
	* **Algebraic properties:**
		* Associativity: $(AB)C=A(BC)$
		* Distributivity: $(A+B)C=AC+BC,A(C+D)=AC+AD$
*   **Matrix Inverse:** For $A,B\in R^{n\times n}$ is called the inverse of $A$ if $B$ if $AB=I_n=BA$. Denoted as $A^{-1}$
	* **Invertibility:** $A$ is called regular/invertible/nonsingular if $A^{-1}$ exists. Otherwise, it is singular/noninvertible.
	* **Uniqueness:** If $A^{-1}$ exists, it is unique.
	* $A^{-1}$ exists $\iff$ $|A|\neq 0:$ 
	$$A^{-1}=\dfrac{1}{|A|}\text{adj}(A)$$
	* **Properties:**
		* $AA^{-1}=I=A^{-1}A$
		* $(AB)^{-1}=B^{-1}A^{-1}$
*   **Matrix Transpose**
	* **Transpose:** For $A\in R^{m\times n}$ , $A^T\in R^{n\times m}$ is defined by $(A^T)_{ij}=a_{ji}$
	* **Properties:**
		* $(A^T)^T=A$
		* $(AB)^T=B^TA^T$
		* $(A+B)^T=A^T+B^T$
		* If $A$ is invertible, $(A^{-1})^T=(A^T)^{-1}$
*   **Symmetric Matrix:** $A\in \mathbb R^{n\times n}$ is symmetric if $A=A^T$
	* **Sum:** The sum of symmetric matrix is symmetric
	* **Properties:**
		* **Symmetry under Congruence Transformation: **
		$$PAP^T=(PAP^T)^T$$
		* **Diagonalizability of Real Symmetric Matrices:**
			* Every real symmetric matrix is **orthogonally diagonalizable**. This means there exists an orthogonal matrix $Q$ (where $Q^TQ=I$) and a diagonal matrix $D$ such that $A=QDQ^T$. The diagonal entries of $D$ are the eigenvalues of $A$ , and the columns of $Q$ are the corresponding orthonormal eigenvectors.
*   **Scalar Multiplication:** 
$$(\lambda A)_{ij}=\lambda(A_{ij})$$
*  **Solution:**
	*   **Consider the system:**
	$$ \begin{bmatrix}1 & 0 & 8 & -4 \\ 0 & 1 & 2 & 12 \end{bmatrix}
	    \begin{bmatrix}
	    x_1 \\
	    x_2 \\
	    x_3 \\
	    x_4
	    \end{bmatrix}
	    =
	    \begin{bmatrix}
	    42 \\
	    8
	    \end{bmatrix}
	    $$
	*   **Two equations, four unknowns:** The system is **underdetermined**, so we expect infinitely many solutions.
	*   The first two columns form an identity matrix. This means $x_1$ and $x_2$ are **pivot variables** (or basic variables), and $x_3$ and $x_4$ are **free variables**.
	    *   To find a particular solution, we can set the free variables to zero.
	    *   Setting $x_3 = 0$ and $x_4 = 0$ gives:
	        *   From the first row: $1 \cdot x_1 + 0 \cdot x_2 + 8 \cdot 0 - 4 \cdot 0 = 42 \implies x_1 = 42$
	        *   From the second row: $0 \cdot x_1 + 1 \cdot x_2 + 2 \cdot 0 + 12 \cdot 0 = 8 \implies x_2 = 8$
	*   **Thus, $[42, 8, 0, 0]^T$ is a particular solution (also called a special solution).**
	*   To find the **general solution** for the non-homogeneous system `Ax = b` (which describes *all* infinitely many solutions), we need to understand the solutions to the associated **homogeneous system**: `Ax = 0`.
	*   **Consider the homogeneous system:**
	$$\begin{bmatrix}
	    1 & 0 & 8 & -4 \\
	    0 & 1 & 2 & 12
	    \end{bmatrix}
	    \begin{bmatrix}
	    x_1 \\
	    x_2 \\
	    x_3 \\
	    x_4
	    \end{bmatrix}
	    =
	    \begin{bmatrix}
	    0 \\
	    0
	    \end{bmatrix}
	    $$
	*   Again, $x_1$ and $x_2$ are pivot variables, and $x_3$ and $x_4$ are free variables. We express the pivot variables in terms of the free variables:
	    *   From the first row: $x_1 + 8x_3 - 4x_4 = 0 \implies x_1 = -8x_3 + 4x_4$
	    *   From the second row: $x_2 + 2x_3 + 12x_4 = 0 \implies x_2 = -2x_3 - 12x_4$
	*   Let the free variables be parameters: $x_3 = s$ and $x_4 = t$, where $s, t \in \mathbb{R}$.
	*   The **homogeneous solution ($x_h$)** can be written in vector form:
	$$x_h = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} = \begin{bmatrix} -8s + 4t \\ -2s - 12t \\ s \\ t \end{bmatrix} = s \begin{bmatrix} -8 \\ -2 \\ 1 \\ 0 \end{bmatrix} + t \begin{bmatrix} 4 \\ -12 \\ 0 \\ 1 \end{bmatrix}
	 $$
	    The vectors $\begin{bmatrix} -8 \\ -2 \\ 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 4 \\ -12 \\ 0 \\ 1 \end{bmatrix}$ form a basis for the **null space** of matrix $A$, denoted as $N(A)$. These are also sometimes called **special solutions** to $Ax=0$.
	
	*   **The General Solution for `Ax = b`:**
	    The complete set of solutions for a consistent linear system `Ax = b` is the sum of any particular solution $x_p$ and the entire null space $N(A)$.
	$$\mathbf{x} = x_p + x_h = x_p + N(A)
	    $$
	    Using our specific example:
	$$\mathbf{x} = \begin{bmatrix} 42 \\ 8 \\ 0 \\ 0 \end{bmatrix} + s \begin{bmatrix} -8 \\ -2 \\ 1 \\ 0 \end{bmatrix} + t \begin{bmatrix} 4 \\ -12 \\ 0 \\ 1 \end{bmatrix} \quad \text{for any } s, t \in \mathbb{R}
	    $$
	    This formula describes all the infinitely many solutions to the original system `Ax = b`.
*  **Rank-Nullity Theorem**

	*   **Theorem Statement:**
	    For an $m \times n$ matrix $A$, the **rank** of $A$ plus the **nullity** of $A$ equals the number of columns $n$.
	    That is:
	    `rank(A) + nullity(A) = n`
	    This also implies:
	    `nullity(A) = n - rank(A)`
	
	*   **Explanation of Terms:**
	    *   **Rank of A (rank(A)):**
	        *   Definition: The dimension of the **Column Space (Col(A))** of matrix $A$. It is equal to the number of **pivot variables** in the Reduced Row Echelon Form (RREF) of $A$.
	    *   **Nullity of A (nullity(A)):**
	        *   Definition: The dimension of the **Null Space (Nul(A))** of matrix $A$. It is equal to the number of **free variables** in the Reduced Row Echelon Form (RREF) of $A$.
	    *   **$n$ (Number of Columns / Variables):**
	        *   Definition: The number of columns of matrix $A$, which represents the total number of unknowns in the system.
	
	*   **Intuitive Meaning:**
	    This theorem fundamentally shows that the total number of variables in a system ($n$) is divided into two parts: one part is constrained by the equations, whose count is the **rank**; the other part consists of variables that can be freely chosen in the solution, whose count is the **nullity**. That is, **(Number of Pivot Variables) + (Number of Free Variables) = (Total Number of Variables)**.
	
	*   **Example (Using the 2x4 Matrix):**
	    *   Consider the matrix $A = \begin{bmatrix} 1 & 0 & 8 & -4 \\ 0 & 1 & 2 & 12 \end{bmatrix}$ from our previous discussion.
	    *   Here, the number of columns $n = 4$ (as there are four unknowns $x_1, x_2, x_3, x_4$).
	    *   This matrix is already in Reduced Row Echelon Form.
	        *   **Pivot Variables:** $x_1, x_2$ (corresponding to the leading 1s in each row). Thus, `rank(A) = 2`.
	        *   **Free Variables:** $x_3, x_4$ (variables not corresponding to pivot positions). Thus, `nullity(A) = 2`.
	    *   **Verifying the Theorem:**
	        *   `rank(A) + nullity(A) = 2 + 2 = 4`. This matches the number of columns $n=4$.
	        *   `nullity(A) = n - rank(A) \implies 2 = 4 - 2`. This also holds perfectly true.
	    *   This example perfectly illustrates the Rank-Nullity Theorem.

* **Elementary Row Transformations (Elementary Row Operations)**
	*   **Definition:**
	    Elementary row transformations are a set of operations that can be performed on the rows of a matrix. These operations are crucial because they transform a matrix into an equivalent matrix (meaning they preserve the solution set of the corresponding linear system, and the row space, column space dimension, and null space of the matrix).
	
	*   **Types of Elementary Row Transformations:**
	    There are three fundamental types of elementary row transformations:
	
	    1.  **Row Swap (Interchange Two Rows):**
	        *   **Description:** Exchange the positions of two rows.
	        *   **Notation:** $R_i \leftrightarrow R_j$ (swap Row $i$ with Row $j$)
	        *   **Example:**
	        $$ \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \xrightarrow{R_1 \leftrightarrow R_2} \begin{bmatrix} 4 & 5 & 6 \\ 1 & 2 & 3 \\ 7 & 8 & 9 \end{bmatrix}
	            $$
	
	    2.  **Row Scaling (Multiply a Row by a Non-zero Scalar):**
	        *   **Description:** Multiply all entries in a row by a non-zero constant scalar.
	        *   **Notation:** $k R_i \to R_i$ (multiply Row $i$ by scalar $k$, where $k \neq 0$)
	        *   **Example:**
	        $$ \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \xrightarrow{2R_1 \to R_1} \begin{bmatrix} 2 & 4 & 6 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
	            $$
	
	    3.  **Row Addition (Add a Multiple of One Row to Another Row):**
	        *   **Description:** Add a scalar multiple of one row to another row. The row being added to is replaced by the result.
	        *   **Notation:** $R_i + k R_j \to R_i$ (add $k$ times Row $j$ to Row $i$, and replace Row $i$)
	        *   **Example:**
	            $$
	            \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \xrightarrow{R_2 - 4R_1 \to R_2} \begin{bmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 7 & 8 & 9 \end{bmatrix}
	            $$
	
	*   **Purpose and Importance:**
	    *   **Solving Linear Systems:** Elementary row transformations are the foundation of **Gaussian elimination** and **Gauss-Jordan elimination**, which are algorithms used to solve systems of linear equations by transforming the augmented matrix into row echelon form or reduced row echelon form.
	    *   **Finding Matrix Inverse:** They can be used to find the inverse of a square matrix.
	    *   **Determining Rank:** They help in finding the rank of a matrix (number of pivots/non-zero rows in REF/RREF).
	    *   **Finding Null Space Basis:** They are essential for transforming the matrix to RREF to identify free variables and determine the basis for the null space.
	    *   **Equivalence:** Two matrices are **row equivalent** if one can be transformed into the other using a sequence of elementary row transformations. Row equivalent matrices have the same row space, null space, and therefore the same rank.