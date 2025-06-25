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
*   **Matrix Notation:**![Image/Class/Mathematics-for-AI/1.png](/img/user/Image/Class/Mathematics-for-AI/1.png)
    A system of linear equations can be compactly represented using matrix multiplication. For a system of $m$ linear equations in $n$ unknowns:
    $$\begin{aligned}
    a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
    a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
    \vdots \\
    a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
    \end{aligned}
    $$
    *   **could be written as:** $Ax=b$
        Where:
        *   $A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$ is the **coefficient matrix**.
        *   $x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$ is the **variable vector** (or unknowns vector).
        *   $b = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}$ is the **constant vector** (or right-hand side vector).

*   **Augmented Matrix (增广矩阵):**
    When solving a system of linear equations $Ax=b$ using row operations (like Gaussian elimination), it's convenient to combine the coefficient matrix $A$ and the constant vector $b$ into a single matrix called the **augmented matrix**.
    *   **Notation:** It is typically written as `[A | b]`, where the vertical line separates the coefficient matrix from the constant vector.
    *   **Structure:**
        If $A$ is an $m \times n$ matrix and $b$ is an $m \times 1$ column vector, then the augmented matrix `[A | b]` is an $m \times (n+1)$ matrix.
    $$[A | b] = \begin{bmatrix}
        a_{11} & a_{12} & \cdots & a_{1n} & \bigm| & b_1 \\
        a_{21} & a_{22} & \cdots & a_{2n} & \bigm| & b_2 \\
        \vdots & \vdots & \ddots & \vdots & \bigm| & \vdots \\
        a_{m1} & a_{m2} & \cdots & a_{mn} & \bigm| & b_m
        \end{bmatrix}
        $$
    *   **Purpose:**
        This notation allows us to perform elementary row operations on the entire system (both coefficients and constants) simultaneously, simplifying the process of finding the solutions. Each row of the augmented matrix directly corresponds to an equation in the linear system.**

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
	        $$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \xrightarrow{R_2 - 4R_1 \to R_2} \begin{bmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 7 & 8 & 9 \end{bmatrix}
	            $$
	
	*   **Purpose and Importance:**
	    *   **Solving Linear Systems:** Elementary row transformations are the foundation of **Gaussian elimination** and **Gauss-Jordan elimination**, which are algorithms used to solve systems of linear equations by transforming the augmented matrix into row echelon form or reduced row echelon form.
	    *   **Finding Matrix Inverse:** They can be used to find the inverse of a square matrix.
	    *   **Determining Rank:** They help in finding the rank of a matrix (number of pivots/non-zero rows in [[Notion/Class/Concept/REF, RREF\|REF, RREF]]).
	    *   **Finding Null Space Basis:** They are essential for transforming the matrix to RREF to identify free variables and determine the basis for the null space.
	    *   **Equivalence:** Two matrices are **row equivalent** if one can be transformed into the other using a sequence of elementary row transformations. Row equivalent matrices have the same row space, null space, and therefore the same rank.
	*   **Importance:**
		* If the matrix is:
	$$\begin{bmatrix}
		\mathbf{1} & 0 & 0 & 5 & \bigm| & 10 \\
		0 & \mathbf{1} & 0 & -2 & \bigm| & 7 \\
		0 & 0 & 0 & 0 & \bigm| & a+1
		\end{bmatrix}$$
		    
		* If and only if $a=-1$ , it is sovlable.
		*  **What does the row `[0 0 0 0 | 0]` mean?**
		    *   A row of all zeros, including the constant term, means that the original equation corresponding to this row was a **linear combination of other equations** in the system. In other words, this equation was redundant and provides no new information about the variables.
		    *   Crucially, `0 = 0` is always a true statement. This indicates that the system is **consistent** (it has solutions). It does **not** imply that there are no solutions (an inconsistent system would have a row like `[0 0 0 0 | c]` where `c ≠ 0`).

* **Row Equivalent Matrices**
	
	*   **Definition:**
	    Two matrices are said to be **row equivalent** if one can be obtained from the other by a finite sequence of elementary row transformations.
	
	*   **Mechanism:**
	    The concept is built upon the three **Elementary Row Transformations** (Row Swap, Row Scaling, Row Addition), which were previously discussed. Applying these operations one or more times will transform a matrix into a row equivalent one.
	
	*   **Notation:**
	    If matrix $A$ is row equivalent to matrix $B$, we write $A \sim B$, $B$ is written as $\overset{\sim}A$.
	*   **Key Properties of Row Equivalent Matrices (What is Preserved):**
	    Elementary row transformations are powerful because they preserve several fundamental properties of a matrix, which are critical for solving linear systems and understanding matrix spaces:
	
	    1.  **Same Solution Set for Linear Systems:** If an augmented matrix $[A | b]$ is row equivalent to another augmented matrix $[A' | b']$, then the linear system $Ax=b$ has exactly the same set of solutions as $A'x=b'$. This is the underlying principle that allows us to solve systems by row reducing their augmented matrices.
	    2.  **Same Row Space:** The row space (the vector space spanned by the row vectors of the matrix) remains unchanged under elementary row transformations.
	    3.  **Same Null Space:** The null space (the set of all solutions to the homogeneous equation $Ax=0$) remains unchanged.
	    4.  **Same Rank:** Since the dimension of the row space and the dimension of the null space are preserved, the rank of the matrix (which is the dimension of the column space, and equals the dimension of the row space) is also preserved.
	    5.  **Same Reduced Row Echelon Form (RREF):** Every matrix is row equivalent to a unique Reduced Row Echelon Form (RREF). This unique RREF is often used as a canonical (standard) form for a matrix.
	
	*   **Importance and Applications:**
	    *   **Solving Linear Systems:** By transforming an augmented matrix into its RREF, we can directly read off the solutions, because the RREF is row equivalent to the original matrix and thus has the same solution set.
	    *   **Finding Matrix Inverse:** A square matrix $A$ is invertible if and only if it is row equivalent to the identity matrix $I$.
	    *   **Basis for Subspaces:** Row operations are used to find bases for the row space, column space, and null space of a matrix.
	
	*   **Example:**
	    Consider matrix $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$.
	    We can perform the elementary row operation $R_2 - 2R_1 \to R_2$:
	$$ \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \xrightarrow{R_2 - 2R_1 \to R_2} \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}
	    $$
	    Thus, $A \sim \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}$. These two matrices are row equivalent.
	
*   **Calculating the Inverse via Augmented Matrix:**
    The Reduced Row Echelon Form (RREF) is extremely useful for inverting matrices. This strategy is also known as the **Gauss-Jordan elimination method for inverses**.

    *   **Requirement:** For this strategy, we need the matrix **A to be square** ($A \in \mathbb{R}^{n \times n}$). An inverse only exists for square matrices.
    *   **Core Idea:** To compute the inverse $A^{-1}$ of an $n \times n$ matrix $A$, we essentially solve the matrix equation $AX = I_n$ for the unknown matrix $X$. The solution $X$ will be $A^{-1}$. Each column of $X$ represents the solution to $A \mathbf{x}_j = \mathbf{e}_j$, where $\mathbf{e}_j$ is the $j$-th standard basis vector (a column of $I_n$).
    *   **Procedure:**
        1.  **Write the augmented matrix `[A | I_n]`:**
            *   **Definition:** This is an augmented matrix formed by concatenating the square matrix $A$ on the left side with the $n \times n$ identity matrix $I_n$ on the right side.
            *   **Purpose:** This unified matrix allows us to perform elementary row operations on $A$ and, simultaneously, apply the *same* operations to $I_n$. Each row operation on $[A | I_n]$ is equivalent to multiplying the original matrix $A$ (and $I_n$) by an elementary matrix from the left. By transforming $A$ into $I_n$, we are effectively finding the product of elementary matrices that "undo" $A$, which is precisely $A^{-1}$.
            *   **Example for a 2x2 matrix:** If $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, then $[A | I_2] = \begin{bmatrix} a & b & \bigm| & 1 & 0 \\ c & d & \bigm| & 0 & 1 \end{bmatrix}$.
        2.  **Perform Gaussian Elimination (Row Reduction):** Use elementary row transformations to bring the augmented matrix to its reduced row-echelon form. The goal is to transform the left block (where $A$ was) into the identity matrix $I_n$.
            $$
            [A | I_n] \xrightarrow{\text{Gauss Elimination}} [I_n | A^{-1}]
            $$
        3.  **Read the Inverse:** If the left block successfully transforms into $I_n$, then the right block of the final matrix will be $A^{-1}$.
        4.  **Case of Non-Invertibility:** If, during the row reduction process, you cannot transform the left block into $I_n$ (e.g., if you end up with a row of zeros in the left block), then the matrix $A$ is singular (non-invertible), and $A^{-1}$ does not exist.
        5.  **Proof:** [[Notion/Class/Proof/Compatibility of Matrix Multiplication with Partitioned Matrices\|Compatibility of Matrix Multiplication with Partitioned Matrices]]
	        $$C[A|I_n]=[CA|CI_n]=[I_n|C]\Rightarrow C=A^{-1}$$

    *   **Limitation:** For non-square matrices, the augmented matrix method (to find a traditional inverse) is not defined because non-square matrices do not have inverses.
* **Algorithms for Solving Linear Systems (`Ax = b`): Direct Methods**
	
	This section outlines various direct algorithms used to find solutions for the linear system `Ax = b`.
	
	*   **1. Direct Inversion:**
	    *   **Applicability:** This method is used if the coefficient matrix $A$ is **square and invertible** (i.e., non-singular).
	    *   **Formula:** The solution $x$ is directly computed as:
	        $$x = A^{-1}b$$
	    *   **Mechanism:** If $A$ is invertible, its inverse $A^{-1}$ exists, and multiplying both sides of $Ax = b$ by $A^{-1}$ from the left yields $A^{-1}Ax = A^{-1}b \implies I_nx = A^{-1}b \implies x = A^{-1}b$.
	
	*   **2. Pseudo-inverse ([[Notion/Class/Proof/Moore Penrose Pseudo inverse\|Moore Penrose Pseudo inverse]]):**
	    *   **Applicability:** This method is used if $A$ is **not square but has linearly independent columns** (i.e., full column rank). This is common in overdetermined systems (more equations than unknowns, $m > n$) where an exact solution might not exist, but we seek a "best fit" solution.
	    *   **Formula:** The solution $x$ is given by:
	        $$x = (A^T A)^{-1} A^T b$$
	    *   **Result:** This formula provides the **minimum-norm least-squares solution**. It finds the vector $x$ that minimizes the Euclidean norm of the residual, $\|Ax - b\|_2^2$. If an exact solution exists, this method finds it. If not, it finds the solution that is "closest" to satisfying the equations in a least-squares sense.
	
	*   **Limitations (Common to Inversion and Pseudo-inversion Methods):**
	    *   **Computationally Expensive:** Calculating matrix inverses or pseudo-inverses is generally **computationally expensive**, especially for large systems. The computational cost typically scales with $O(n^3)$ for an $n \times n$ matrix.
	    *   **Numerically Unstable:** These methods can be **numerically unstable** for large or ill-conditioned systems, meaning small errors in input data or floating-point arithmetic can lead to large errors in the computed inverse and solution.
	
	*   **3. Gaussian Elimination:**
	    *   **Mechanism:** This is a systematic method that **reduces the augmented matrix `[A | b]` to row-echelon form or reduced row-echelon form** to solve `Ax = b`. It involves a series of elementary row operations.
	    *   **Scalability:** Gaussian elimination is generally **efficient for thousands of variables**. However, it is **not practical for very large systems** (e.g., millions of variables) because its computational cost scales cubically with the number of variables ($O(n^3)$), making it too slow and memory-intensive for extremely large problems.

## Part III: Vector Spaces and Groups

*   **Group:**
	A **group** is a set $G$ together with a binary operation $*$ (that combines any two elements of $G$ to form a third element also in $G$) that satisfies the following four axioms:
	
	1.  **Closure:** For all $a, b \in G$, the result of the operation $a * b$ is also in $G$.
	    *   (Formally: $\forall a, b \in G, \quad a * b \in G$)
	
	2.  **Associativity:** For all $a, b, c \in G$, the order in which multiple operations are performed does not affect the result.
	    *   (Formally: $\forall a, b, c \in G, \quad (a * b) * c = a * (b * c)$)
	
	3.  **Identity Element:** There exists an element $e \in G$ (called the identity element) such that for every element $a \in G$, operating $e$ with $a$ (in any order) leaves $a$ unchanged.
	    *   (Formally: $\exists e \in G \text{ s.t. } \forall a \in G, \quad a * e = e * a = a$)
	
	4.  **Inverse Element:** For every element $a \in G$, there exists an element $a^{-1} \in G$ (called the inverse of $a$) such that operating $a$ with $a^{-1}$ (in any order) yields the identity element $e$.
	    *   (Formally: $\forall a \in G, \quad \exists a^{-1} \in G \text{ s.t. } a * a^{-1} = a^{-1} * a = e$)
	
	---
	
	**Additional Terminology:**
	
	*   **Abelian Group (Commutative Group):** If, in addition to the four axioms above, the operation $*$ is also **commutative** (i.e., $a * b = b * a$ for all $a, b \in G$), then the group is called an Abelian group.
	
	*   **Order of a Group:** The number of elements in a group $G$ is called its **order**, denoted by $|G|$. If the number of elements is finite, it's a **finite group**; otherwise, it's an **infinite group**.
	
	**Examples of Groups:**
	
	*   The set of integers $\mathbb{Z}$ under addition $(+)$ is an Abelian group.
	    *   (Closure: $m+n \in \mathbb{Z}$)
	    *   (Associativity: $(m+n)+p = m+(n+p)$)
	    *   (Identity: $0 \in \mathbb{Z}$, $m+0=m$)
	    *   (Inverse: for $m$, $-m \in \mathbb{Z}$, $m+(-m)=0$)
	*   The set of non-zero rational numbers $\mathbb{Q}^*$ under multiplication $(\times)$ is an Abelian group.
	*   The set of all invertible $n \times n$ matrices under matrix multiplication is a non-Abelian group (for $n \ge 2$). This is called the general linear group $GL_n(\mathbb{R})$.

![Image/Class/Mathematics-for-AI/2.png](/img/user/Image/Class/Mathematics-for-AI/2.png)
![Image/Class/Mathematics-for-AI/3.png](/img/user/Image/Class/Mathematics-for-AI/3.png)
![Image/Class/Mathematics-for-AI/4.png](/img/user/Image/Class/Mathematics-for-AI/4.png)

### Continuation of Notes

*   **Vector Space (向量空间):**
    A **vector space** is a set of objects called **vectors** ($V$), along with a set of **scalars** (usually the real numbers $\mathbb{R}$), equipped with two operations: **vector addition** and **scalar multiplication**. These operations must satisfy ten axioms.

    **Axioms of a Vector Space:**
    Let $\mathbf{u}, \mathbf{v}, \mathbf{w}$ be vectors in $V$ and let $c, d$ be scalars in $\mathbb{R}$.

    1.  **Closure under Addition:** $\mathbf{u} + \mathbf{v}$ is in $V$.
    2.  **Commutativity of Addition:** $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$.
    3.  **Associativity of Addition:** $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$.
    4.  **Zero Vector (Additive Identity):** There is a vector $\mathbf{0}$ in $V$ such that $\mathbf{u} + \mathbf{0} = \mathbf{u}$.
    5.  **Additive Inverse:** For every vector $\mathbf{u}$, there is a vector $-\mathbf{u}$ in $V$ such that $\mathbf{u} + (-\mathbf{u}) = \mathbf{0}$.
        *   **Connection to Groups:** The first five axioms mean that the set of vectors $V$ with the addition operation $(V, +)$ forms an **Abelian Group**.

    6.  **Closure under Scalar Multiplication:** $c\mathbf{u}$ is in $V$.
    7.  **Distributivity:** $c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$.
    8.  **Distributivity:** $(c+d)\mathbf{u} = c\mathbf{u} + d\mathbf{u}$.
    9.  **Associativity of Scalar Multiplication:** $c(d\mathbf{u}) = (cd)\mathbf{u}$.
    10. **Scalar Identity:** $1\mathbf{u} = \mathbf{u}$.

*   **Subspace (子空间):**
    A **subspace** of a vector space $V$ is a subset $H$ of $V$ that is itself a vector space under the same operations of addition and scalar multiplication defined on $V$.
    *   **Subspace Test (子空间判别法):** To verify if a subset $H$ is a subspace, we only need to check three conditions:
        1.  **Contains the Zero Vector:** The zero vector of $V$ is in $H$ ($\mathbf{0} \in H$).
        2.  **Closure under Addition:** For any two vectors $\mathbf{u}, \mathbf{v} \in H$, their sum $\mathbf{u} + \mathbf{v}$ is also in $H$.
        3.  **Closure under Scalar Multiplication:** For any vector $\mathbf{u} \in H$ and any scalar $c$, the vector $c\mathbf{u}$ is also in $H$.
    *   **Key Examples:**
        *   Any line or plane in $\mathbb{R}^3$ that passes through the origin is a subspace of $\mathbb{R}^3$.
        *   The **null space** of an $m \times n$ matrix $A$, denoted $N(A)$, is a subspace of $\mathbb{R}^n$.
        *   The **column space** of an $m \times n$ matrix $A$, denoted $Col(A)$, is a subspace of $\mathbb{R}^m$.

*   **Linear Combination (线性组合):**
    Given vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_p$ in a vector space $V$ and scalars $c_1, c_2, \ldots, c_p$, the vector $\mathbf{y}$ defined by:
    $$\mathbf{y} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_p\mathbf{v}_p$$
    is called a **linear combination** of $\mathbf{v}_1, \ldots, \mathbf{v}_p$ with weights $c_1, \ldots, c_p$.

*   **Span (生成空间):**
    *   **Definition:** The **span** of a set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_p\}$, denoted $\text{Span}\{\mathbf{v}_1, \ldots, \mathbf{v}_p\}$, is the set of **all possible linear combinations** of these vectors.
    *   **Geometric Interpretation:**
        *   $\text{Span}\{\mathbf{v}\}$ (where $\mathbf{v} \neq \mathbf{0}$) is the line passing through the origin and $\mathbf{v}$.
        *   $\text{Span}\{\mathbf{u}, \mathbf{v}\}$ (where $\mathbf{u}, \mathbf{v}$ are not collinear) is the plane containing the origin, $\mathbf{u}$, and $\mathbf{v}$.
    *   **Property:** The span of any set of vectors is always a **subspace**.

*   **Linear Independence and Dependence (线性无关与线性相关):**
    *   **Linear Independence:** A set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_p\}$ is **linearly independent** if the vector equation
        $$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_p\mathbf{v}_p = \mathbf{0}$$
        has **only the trivial solution** ($c_1 = c_2 = \cdots = c_p = 0$).
    *   **Linear Dependence:** The set is **linearly dependent** if there exist weights $c_1, \ldots, c_p$, **not all zero**, such that the equation holds.
    *   **Intuitive Meaning:** A set of vectors is linearly dependent if and only if at least one of the vectors can be written as a linear combination of the others. Linearly independent vectors are non-redundant.

*   **Basis (基):**
    A **basis** for a vector space $V$ is a set of vectors $B = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ that satisfies two conditions:
    1.  The set $B$ is **linearly independent**.
    2.  The set $B$ **spans** the vector space $V$ (i.e., $\text{Span}(B) = V$).
    *   A basis is a "minimal" set of vectors needed to build the entire space.

*   **Dimension (维度):**
    *   **Definition:** The **dimension** of a non-zero vector space $V$, denoted $\text{dim}(V)$, is the **number of vectors in any basis** for $V$. The dimension of the zero vector space $\{\mathbf{0}\}$ is defined to be 0.
    *   **Uniqueness:** Although a vector space can have many different bases, all bases for a given vector space have the same number of vectors.
    *   **Connection to Rank-Nullity:**
        *   The dimension of the column space of a matrix $A$ is its rank: $\text{dim}(\text{Col}(A)) = \text{rank}(A)$.
        *   The dimension of the null space of a matrix $A$ is its nullity: $\text{dim}(N(A)) = \text{nullity}(A)$.
*   **Testing for Linear Independence using Gaussian Elimination:**

    To test if a set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is linearly independent:
    1.  Form a matrix $A$ using these vectors as its columns.
    2.  Perform Gaussian elimination on matrix $A$ to reduce it to **row echelon form**.

    *   The original vectors that correspond to **pivot columns** are linearly independent.
    *   The vectors that correspond to **non-pivot columns** can be written as a linear combination of the preceding pivot columns.

    **Example:** In the following row echelon form matrix:
    $$
    \begin{pmatrix}
      \mathbf{1} & 3 & 0 \\
      0 & 0 & \mathbf{1}
    \end{pmatrix}
    $$
    Column 1 and Column 3 are **pivot columns** (their corresponding original vectors are independent); Column 2 is a **non-pivot column** (its corresponding original vector is dependent).

    Therefore, the original set of vectors (the columns of matrix $A$) is **not linearly independent** because there is at least one non-pivot column. In other words, the set is **linearly dependent**.
*   **Linear Independence of Linear Combinations**

    Let's consider a set of $k$ linearly independent vectors $\{\mathbf{b}_1, \ldots, \mathbf{b}_k\}$, which can be seen as a basis for a $k$-dimensional space. We can form a new set of $m$ vectors $\{\mathbf{x}_1, \ldots, \mathbf{x}_m\}$ where each $\mathbf{x}_j$ is a linear combination of the base vectors:
    $$ \mathbf{x}_j = \sum_{i=1}^{k} \lambda_{ij} \mathbf{b}_i $$
    Each set of weights can be represented by a **coefficient vector** $\boldsymbol{\lambda}_j \in \mathbb{R}^k$.

    *   **Key Implication:** The set of new vectors $\{\mathbf{x}_1, \ldots, \mathbf{x}_m\}$ is linearly independent *if and only if* the set of their corresponding coefficient vectors $\{\boldsymbol{\lambda}_1, \ldots, \boldsymbol{\lambda}_m\}$ is linearly independent.
    
*   **The Dimension Theorem for Spanning Sets (A Fundamental Theorem)**
     The dimension of a vector space cannot exceed the number of vectors in any of its spanning sets. A direct consequence is that in a vector space of dimension $k$, any set containing more than $k$ vectors must be linearly dependent.

*   **Special Case: More New Vectors than Base Vectors ($m > k$)**

    **Theorem:** If you use $k$ linearly independent vectors to generate $m$ new vectors, and $m > k$, the resulting set of new vectors $\{\mathbf{x}_1, \ldots, \mathbf{x}_m\}$ is **always linearly dependent**.

    **Proof (using Matrix Rank):**

    1.  **Focus on the Coefficient Vectors:** As established, the linear independence of $\{\mathbf{x}_j\}$ is equivalent to the linear independence of their coefficient vectors $\{\boldsymbol{\lambda}_j\}$. We will prove that the set $\{\boldsymbol{\lambda}_1, \ldots, \boldsymbol{\lambda}_m\}$ must be linearly dependent.

    2.  **Construct the Coefficient Matrix:** Let's arrange these coefficient vectors as the columns of a matrix, $\Lambda$:
        $$ \Lambda = [\boldsymbol{\lambda}_1, \boldsymbol{\lambda}_2, \ldots, \boldsymbol{\lambda}_m] $$
        Since each coefficient vector $\boldsymbol{\lambda}_j$ is in $\mathbb{R}^k$, the matrix $\Lambda$ has $k$ rows and $m$ columns (it is a $k \times m$ matrix).

    3.  **Analyze the Rank of the Matrix:** The **rank** of a matrix has a fundamental property: it cannot exceed its number of rows or its number of columns. Specifically, we are interested in the fact that $\text{rank}(\Lambda) \le k$ (the number of rows).
        *   *(Justification via a more fundamental theorem: The rank is the dimension of the row space. The row space is spanned by $k$ row vectors. By [[Notion/Class/Proof/The Dimension Theorem for Spanning Sets\|The Dimension Theorem for Spanning Sets]], the dimension of this space cannot exceed $k$.)*

    4.  **Apply the Condition $m > k$:** We have established two key facts about the matrix $\Lambda$:
        *   The total number of columns is $m$.
        *   The rank, which represents the maximum number of linearly independent columns, is at most $k$. That is, $\text{rank}(\Lambda) \le k$.

    5.  **Connect Rank to Linear Dependence:** We are given that $m > k$. This leads to the crucial inequality:
        $$ \text{Total number of columns } (m) > \text{Maximum number of linearly independent columns } (\text{rank}(\Lambda)) $$
        This inequality means it is impossible for all $m$ columns of $\Lambda$ to be linearly independent. If you have more vectors ($m$) than the dimension of the space they can span (the rank, which is at most $k$), the set of vectors must be linearly dependent.

    6.  **Draw the Conclusion:** Because the columns of $\Lambda$ (which are the coefficient vectors $\{\boldsymbol{\lambda}_j\}$) form a linearly dependent set, the set of new vectors $\{\mathbf{x}_j\}$ that they define must also be **linearly dependent**. Q.E.D.

# Lecture 2: Linear Algebra: Basis and Rank, Linear Mappings, Affine Spaces

## Part I: Basis and Rank

*   **Generating Set (or Spanning Set)**
    *   **Definition:** A set of vectors $S$ is called a **generating set** for a vector space $V$ if $\text{Span}(S) = V$.
    *   **Key Idea:** A generating set can be **redundant**; it may contain linearly dependent vectors.
    *   **Example:** The set $S = \{(1, 0), (0, 1), (1, 1)\}$ is a generating set for $\mathbb{R}^2$. It is redundant because $(1, 1)$ is a linear combination of the other two vectors.

*   **Span (Additional Property)**
    *   **Connection to Linear Systems:** A system of linear equations $A\mathbf{x} = \mathbf{b}$ has a solution if and only if the vector $\mathbf{b}$ is in the span of the columns of matrix $A$. That is, $\mathbf{b} \in \text{Col}(A)$.

*   **Basis (Additional Properties)**
    *   **Unique Representation Theorem:** A key property of a basis is that every vector $\mathbf{v}$ in the space can be expressed as a linear combination of the basis vectors in **exactly one way**. The coefficients of this unique combination are called the **coordinates** of $\mathbf{v}$ with respect to that basis.
    *   **Example (The Standard Basis):** The most common basis for $\mathbb{R}^n$ is the **standard basis**, which consists of the columns of the $n \times n$ identity matrix $I_n$. For $\mathbb{R}^3$, the standard basis is $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\} = \left\{ \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} \right\}$.
    
*   **Characterizations of a Basis**
    For a non-empty set of vectors $B$ in a vector space $V$, the following statements are equivalent (meaning if one is true, all are true):
    1.  $B$ is a **basis** of $V$.
    2.  $B$ is a **minimal generating set** (i.e., it spans $V$, but no proper subset of $B$ spans $V$).
    3.  $B$ is a **maximal linearly independent set** (i.e., it's linearly independent, but adding any other vector from $V$ to it would make the set linearly dependent).

*   **Further Properties of Dimension**

    *   **Existence and Uniqueness of Size:** Every non-trivial vector space has a basis. While a space can have many different bases, all of them will have the same number of vectors. This makes the concept of dimension well-defined.
    *   **Subspace Dimension:** If $U$ is a subspace of a vector space $V$, then $\text{dim}(U) \le \text{dim}(V)$. Equality holds if and only if $U = V$.
    *   **Important Clarification:** The dimension of a space refers to the **number of vectors in its basis**, not the number of components in each vector. For example, the subspace spanned by the single vector $\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ is one-dimensional, even though the vector lives in $\mathbb{R}^3$.

*   **How to Find a Basis for a Subspace (Basis Extraction Method)**

    To find a basis for a subspace $U$ that is defined as the span of a set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_m\}$:
    1.  Create a matrix $A$ where the columns are the vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_m\}$.
    2.  Reduce the matrix $A$ to its **row echelon form**.
    3.  Identify the columns that contain **pivots**.
    4.  The basis for $U$ consists of the **original vectors** from the set $\{\mathbf{v}_1, \ldots, \mathbf{v}_m\}$ that correspond to these pivot columns.
    
*   **Rank of a Matrix**

    *   **Definition:** The **rank** of a matrix $A$, denoted $\text{rk}(A)$, is the number of linearly independent columns in $A$. A fundamental theorem of linear algebra states that this number is always equal to the number of linearly independent rows.
    *   **Key Property:** The rank of a matrix is equal to the rank of its transpose: $\text{rk}(A) = \text{rk}(A^T)$.

*   **Rank and its Connection to Fundamental Subspaces**

    *   **Column Space (Image/Range):** The rank of $A$ is the dimension of its column space.
        $\text{rk}(A) = \text{dim}(\text{Col}(A))$
    *   **Null Space (Kernel):** The rank determines the dimension of the null space through the Rank-Nullity Theorem. For an $m \times n$ matrix $A$:
        $\text{dim}(\text{Nul}(A)) = n - \text{rk}(A)$

*   **Properties and Applications of Rank**

    *   **Invertibility of Square Matrices:** An $n \times n$ matrix $A$ is invertible if and only if its rank is equal to its dimension, i.e., $\text{rk}(A) = n$. This is because a full-rank square matrix can be row-reduced to the identity matrix.
    *   **Solvability of Linear Systems:** The system $A\mathbf{x} = \mathbf{b}$ has at least one solution if and only if the rank of the coefficient matrix $A$ is equal to the rank of the augmented matrix $[A|\mathbf{b}]$.
        **Reasoning:** If $\text{rk}(A) < \text{rk}([A|\mathbf{b}])$, it means that the vector $\mathbf{b}$ is linearly independent of the columns of $A$. Therefore, $\mathbf{b}$ cannot be written as a linear combination of the columns of $A$, and no solution exists.
    *   **Full Rank and Rank Deficiency:**
        *   A matrix has **full rank** if its rank is the maximum possible for its dimensions: $\text{rk}(A) = \min(m, n)$.
        *   A matrix is **rank deficient** if $\text{rk}(A) < \min(m, n)$, indicating linear dependencies among its rows or columns.

*   **Why Rank is Important**

    The rank of a matrix is a core concept that reveals its fundamental structure. It tells us:
    *   The maximum number of linearly independent rows/columns.
    *   The dimension of the data (the dimension of the subspace spanned by the columns).
    *   Whether a linear system is consistent (has solutions).
    *   Whether a square matrix has an inverse.
    *   It is crucial for identifying redundancy and simplifying problems in data analysis, optimization, and machine learning.

*   **Summary: Tying Rank, Basis, and Pivots Together**
    1.  You start with a set of vectors.
    2.  You place them as columns in a matrix $A$.
    3.  You perform Gaussian elimination to find the **pivots**.
    4.  The **number of pivots** is the **rank** of the matrix $A$.
    5.  This rank is also the **dimension** of the subspace spanned by the original vectors.
    6.  The **original vectors** corresponding to the **pivot columns** form a **basis** for that subspace.

## Part II: Linear Mappings

*   **Linear Mappings (Linear Transformations)**
    *   **Definition:** A mapping (or function) $\Phi: V \to W$ from a vector space $V$ to a vector space $W$ is called **linear** if it preserves the two fundamental vector space operations:
        1.  **Additivity:** $\Phi(\mathbf{x} + \mathbf{y}) = \Phi(\mathbf{x}) + \Phi(\mathbf{y})$ for all $\mathbf{x}, \mathbf{y} \in V$.
        2.  **Homogeneity:** $\Phi(\lambda\mathbf{x}) = \lambda\Phi(\mathbf{x})$ for any scalar $\lambda$.
    *   **Matrix Representation:** Any linear mapping between finite-dimensional vector spaces can be represented by matrix multiplication: $\Phi(\mathbf{x}) = A\mathbf{x}$ for some matrix $A$.

*   **Properties of Mappings: Injective, Surjective, Bijective**
    *   **Injective (One-to-one):** A mapping is injective if distinct inputs always map to distinct outputs. Formally, if $\Phi(\mathbf{x}) = \Phi(\mathbf{y})$, then it must be that $\mathbf{x} = \mathbf{y}$.
    *   **Surjective (Onto):** A mapping is surjective if its range is equal to its codomain. This means every element in the target space $W$ is the image of at least one element from the starting space $V$.
    *   **Bijective:** A mapping is bijective if it is **both injective and surjective**. A bijective mapping has a unique inverse mapping, denoted $\Phi^{-1}$.

*   **Special Types of Linear Mappings**
    *   **Homomorphism:** For vector spaces, a homomorphism is simply another term for a **linear mapping**. It's a map that preserves the algebraic structure (addition and scalar multiplication).
    *   **Isomorphism:** A linear mapping that is also **bijective**. Isomorphic vector spaces are structurally identical, just with potentially different-looking elements.
    *   **Endomorphism:** A linear mapping from a vector space **to itself** ($\Phi: V \to V$). It does not need to be invertible.
    *   **Automorphism:** An endomorphism that is also **bijective**. It is an isomorphism from a vector space to itself (e.g., a rotation or reflection).
    *   **Identity Mapping:** The map defined by $\text{id}(\mathbf{x}) = \mathbf{x}$. It leaves every vector unchanged and is the simplest example of an automorphism.
    
*   **Isomorphism**
    *   **Isomorphism and Dimension:** A fundamental theorem states that two finite-dimensional vector spaces, $V$ and $W$, are **isomorphic** (structurally identical) if and only if they have the same dimension.
         $\text{dim}(V) = \text{dim}(W) \iff V \cong W$
    *   **Intuition:** This means any n-dimensional vector space is essentially a "re-labeling" of $\mathbb{R}^n$.
    *   **Properties of Linear Mappings:**
        *   The composition of two linear mappings is also a linear mapping.
        *   The inverse of an isomorphism is also an isomorphism.
        *   The sum and scalar multiple of linear mappings are also linear.

*   **Matrix Representation via Ordered Bases**
    The isomorphism between an abstract n-dimensional space $V$ and the concrete space $\mathbb{R}^n$ is made practical by choosing an **ordered basis**. The order of the basis vectors matters for defining coordinates.
    *   **Notation:** We denote an ordered basis with parentheses, e.g., $B = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$.

*   **Coordinates and Coordinate Vectors**
    *   **Definition:** Given an ordered basis $B = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ of $V$, every vector $\mathbf{x} \in V$ can be written uniquely as:
        $$ \mathbf{x} = \alpha_1\mathbf{b}_1 + \cdots + \alpha_n\mathbf{b}_n $$
        The scalars $\alpha_1, \ldots, \alpha_n$ are called the **coordinates** of $\mathbf{x}$ with respect to the basis $B$.
    *   **Coordinate Vector:** We collect these coordinates into a single column vector, which represents $\mathbf{x}$ in the standard space $\mathbb{R}^n$:
        $$ [\mathbf{x}]_B = \begin{pmatrix} \alpha_1 \\ \vdots \\ \alpha_n \end{pmatrix} \in \mathbb{R}^n $$

*   **Coordinate Systems and Change of Basis**
    *   **Concept:** A basis defines a coordinate system for the vector space. The familiar Cartesian coordinates in $\mathbb{R}^2$ are simply the coordinates with respect to the standard basis $(\mathbf{e}_1, \mathbf{e}_2)$. Any other basis defines a different, but equally valid, coordinate system.
    *   **Example:** A single vector $\mathbf{x} \in \mathbb{R}^2$ has different coordinates in different bases. For instance, its coordinate vector might be $\begin{pmatrix} 2 \\ 2 \end{pmatrix}$ with respect to the standard basis, but $\begin{pmatrix} 1.09 \\ 0.72 \end{pmatrix}$ with respect to another basis $B = (\mathbf{b}_1, \mathbf{b}_2)$. This means $\mathbf{x} = 2\mathbf{e}_1 + 2\mathbf{e}_2$ and also $\mathbf{x} = 1.09\mathbf{b}_1 + 0.72\mathbf{b}_2$.

*   **Importance for Linear Mappings**
    Once we fix ordered bases for the input and output spaces, we can represent any linear mapping as a concrete **matrix**. This matrix representation is entirely dependent on the chosen bases.

## Part III: Basis Change and Transformation Matrices

This section covers the representation of abstract linear mappings as matrices and the mechanics of changing coordinate systems.

### 1. The Transformation Matrix for a Linear Map

A transformation matrix provides a concrete computational representation for an abstract linear mapping, relative to chosen bases.

*   **Definition and Context:**
    We are given a linear map $\Phi: V \to W$, an ordered basis $B = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ for the vector space $V$, and an ordered basis $C = (\mathbf{c}_1, \ldots, \mathbf{c}_m)$ for the vector space $W$.

*   **Construction of the Transformation Matrix ($A_\Phi$):**
    The transformation matrix $A_\Phi$ is an $m \times n$ matrix whose columns describe how the input basis vectors are transformed by $\Phi$. It is constructed as follows:
    1.  For each input basis vector $\mathbf{b}_j$ (from $j=1$ to $n$):
    2.  Apply the linear map to get its image: $\Phi(\mathbf{b}_j) \in W$.
    3.  Express this image as a unique linear combination of the output basis vectors from $C$:
        $$ \Phi(\mathbf{b}_j) = \alpha_{1j}\mathbf{c}_1 + \alpha_{2j}\mathbf{c}_2 + \cdots + \alpha_{mj}\mathbf{c}_m $$
    4.  The coefficients from this combination form the $j$-th column of the matrix $A_\Phi$. This column is the coordinate vector of $\Phi(\mathbf{b}_j)$ with respect to basis $C$:
        $$ \text{Column } j \text{ of } A_\Phi = [\Phi(\mathbf{b}_j)]_C = \begin{pmatrix} \alpha_{1j} \\ \alpha_{2j} \\ \vdots \\ \alpha_{mj} \end{pmatrix} $$

*   **Usage and Interpretation:**
    This matrix maps the coordinate vector of any $\mathbf{v} \in V$ (relative to basis $B$) to the coordinate vector of its image $\Phi(\mathbf{v}) \in W$ (relative to basis $C$). The core operational formula is:
    $$ [\Phi(\mathbf{v})]_C = A_\Phi [\mathbf{v}]_B $$
    This formula translates the abstract function application into a concrete matrix-vector multiplication. The matrix $A_\Phi$ is the representation of the map $\Phi$ *with respect to the chosen bases*; changing either basis will result in a different transformation matrix for the same underlying linear map.

*   **Invertibility:**
    The linear map $\Phi$ is an invertible isomorphism if and only if its transformation matrix $A_\Phi$ is square ($m=n$) and invertible. [[Notion/Class/Proof/Non-Invertible Transformations and Information Loss\|Non-Invertible Transformations and Information Loss]]

### 2. The Change of Basis Matrix

This is a special application of the transformation matrix, used to convert a vector's coordinates from one basis to another within the *same* vector space. This process is equivalent to finding the transformation matrix for the **identity map** ($\text{id}: V \to V$, where $\text{id}(\mathbf{x}) = \mathbf{x}$).

*   **The Change of Basis Matrix ($P_{B \leftarrow B'}$):**
    This matrix converts coordinates from a new basis $B'$ to an old basis $B$. Its columns are the coordinate vectors of the new basis vectors, expressed in the old basis.
    $$ P_{B \leftarrow B'} = \Big[ \ [\mathbf{b}'_1]_B \ \ \ [\mathbf{b}'_2]_B \ \ \cdots \ \ [\mathbf{b}'_n]_B \ \Big] $$
    *Note: If the "old" basis $B$ is the standard basis in $\mathbb{R}^n$, the columns of this matrix are simply the vectors of the new basis $B'$ themselves.*

*   **Usage Formulas:**
    *   To convert coordinates **from new ($B'$) to old ($B$)**:
        $$ [\mathbf{x}]_B = P_{B \leftarrow B'} \ [\mathbf{x}]_{B'} $$
    *   To convert coordinates **from old ($B$) to new ($B'$)**:
        $$ [\mathbf{x}]_{B'} = (P_{B \leftarrow B'})^{-1} \ [\mathbf{x}]_B $$

*   **Example: Change of Basis in $\mathbb{R}^2$**
    *   **Old Basis (Standard):** $B = (\mathbf{e}_1, \mathbf{e}_2)$
    *   **New Basis:** $B' = \left( \mathbf{b}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \mathbf{b}_2 = \begin{pmatrix} -1 \\ 2 \end{pmatrix} \right)$
    *   **Change of Basis Matrix from $B'$ to $B$:**
        $$ P = \begin{pmatrix} 1 & -1 \\ 1 & 2 \end{pmatrix} $$
    *   To express a vector $\mathbf{x} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$ (whose coordinates are given in the standard basis $B$) in the new basis $B'$:
        $$ [\mathbf{x}]_{B'} = P^{-1} [\mathbf{x}]_B = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ -1 & 1 \end{pmatrix} \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 10/3 \\ 1/3 \end{pmatrix} $$

You are absolutely right, and I sincerely apologize. Your observation is spot on. I failed to integrate the clarifying distinction between "Domain/Codomain" (the spaces) and "Input/Output Basis" (the coordinate systems) into the formal note. You are correct that the note I provided, while technically accurate to the slides, lost the very explanation that resolved your earlier confusion.

### 3. The Theorem of Basis Change for Linear Mappings

[[Notion/Class/Proof/Change-of-Basis Theorem\|Change-of-Basis Theorem]]
This theorem provides a formula to calculate the new transformation matrix for a linear map when the bases (the coordinate systems) for its domain and codomain are changed.

*   **Theorem Statement:**
    Given a linear mapping $\Phi : V \to W$, with:
    *   An "old" input basis $B$ and a "new" input basis $B̃$, both for the **domain** $V$.
    *   An "old" output basis $C$ and a "new" output basis $C̃$, both for the **codomain** $W$.
    *   The original transformation matrix $A_\Phi$ (relative to the old bases $B$ and $C$).

    The new transformation matrix $Ã_\Phi$ (relative to the new bases $B̃$ and $C̃$) is given by:
    $$ Ã_\Phi = T^{-1}A_\Phi S $$
    Where the change-of-basis matrices are defined as:
    *   $S$: The matrix for the basis change **within the domain $V$**. It converts coordinates from the **new input basis $B̃$** to the **old input basis $B$**.
    *   $T$: The matrix for the basis change **within the codomain $W$**. It converts coordinates from the **new output basis $C̃$** to the **old output basis $C$**.

*   **Explanation of the Formula (The "Path" of the Transformation):**
    The formula represents a sequence of three operations on a coordinate vector. The path for the coordinates is from $B̃ \to B \to C \to C̃$.
    1.  **Step 1: `S` (from $B̃ \to B$ in the Domain)**: We start with a vector's coordinates in the new input basis, $[\mathbf{v}]_{B̃}$. We apply $S$ to translate these coordinates into the old input basis: $S[\mathbf{v}]_{B̃}=[\mathbf{v}]_B$.
    2.  **Step 2: `AΦ` (from Basis $B$ to Basis $C$)**: We apply the original transformation matrix $A_\Phi$ to the coordinates, which are now expressed in the old input basis $B$. This yields the image's coordinates in the old output basis $C$: $A_\Phi([\mathbf{v}]_B) = [\Phi(\mathbf{v})]_C$.
    3.  **Step 3: `T⁻¹` (from $C \to C̃$ in the Codomain)**: The result is in the old output basis $C$. To express it in the new output basis $C̃$, we must apply the inverse of $T$. Since $T$ converts from $C̃$ to $C$, $T^{-1}$ must be used to convert from $C$ to $C̃$: $T^{-1}[\Phi(\mathbf{v})]_C = [\Phi(\mathbf{v})]_{C̃}$.

### 4. Matrix Equivalence and Similarity

These concepts formalize the idea that different matrices can represent the same underlying linear map, just with different coordinate systems.

*   **Matrix Equivalence:**
    *   **Definition:** Two $m \times n$ matrices $A$ and $Ã$ are **equivalent** if there exist invertible matrices $S$ (in the domain) and $T$ (in the codomain) such that $Ã = T^{-1}AS$.
    *   **Interpretation:** Equivalent matrices represent the **exact same linear transformation** $\Phi: V \to W$. They are merely different numerical representations of $\Phi$ due to different choices of bases (coordinate systems) within the domain $V$ and the codomain $W$.

*   **Matrix Similarity:**
    *   **Definition:** Two **square** $n \times n$ matrices $A$ and $Ã$ are **similar** if there exists a single invertible matrix $S$ such that $Ã = S^{-1}AS$.
    *   **Interpretation:** Similarity is a special case of equivalence for **endomorphisms** ($\Phi: V \to V$), where the same space serves as both domain and codomain, and therefore the same basis change (i.e., $T=S$) is applied to both the input and output coordinates.

### 5. Composition of Linear Maps

*   **Theorem:** If $\Phi : V \to W$ and $\Psi : W \to X$ are linear mappings, their composition $(\Psi \circ \Phi) : V \to X$ is also a linear mapping.
*   **Matrix Representation:** The transformation matrix of the composite map is the product of the individual transformation matrices, in reverse order of application:
    $$ A_{\Psi \circ \Phi} = A_\Psi A_\Phi $$

## Part IV: Affine Spaces and Subspaces

While vector spaces and subspaces are fundamental, they are constrained by one critical requirement: they must contain the origin. Affine spaces generalize this idea to describe geometric objects like lines and planes that do *not* necessarily pass through the origin.

*   **Core Intuition: Vector Space vs. Affine Space**
    *   A **vector subspace** is a line or plane (or higher-dimensional equivalent) that **must pass through the origin**.
    *   An **affine subspace** is a line or plane (or higher-dimensional equivalent) that has been **shifted** or **translated** so it no longer needs to pass through the origin. It is a "flat" surface in the vector space.

*   **Formal Definition of an Affine Subspace**

    An **affine subspace** $L$ of a vector space $V$ is a subset that can be expressed as the sum of a specific vector (a point) and a vector subspace.

    $$ L = \mathbf{p} + U = \{ \mathbf{p} + \mathbf{u} \mid \mathbf{u} \in U \} $$

    Where:
    *   $\mathbf{p} \in V$ is a specific vector, often called the **translation vector** or **support point**. It acts as the "anchor" that shifts the space.
    *   $U$ is a **vector subspace** of $V$, often called the **direction space** or associated vector subspace. It defines the orientation and "shape" (line, plane, etc.) of the affine subspace.

    The **dimension** of the affine subspace $L$ is defined as the dimension of its direction space $U$.

*   **Geometric Examples:**
    *   **A Line in $\mathbb{R}^3$:** A line passing through point $\mathbf{p}$ with direction vector $\mathbf{d}$ is an affine subspace.
        $$ L = \mathbf{p} + t\mathbf{d} \quad (t \in \mathbb{R}) $$
        Here, the support point is $\mathbf{p}$, and the direction space is the 1D vector subspace $U = \text{Span}\{\mathbf{d}\}$.
    *   **A Plane in $\mathbb{R}^3$:** A plane containing point $\mathbf{p}$ and parallel to vectors $\mathbf{u}$ and $\mathbf{v}$ (which are linearly independent) is an affine subspace.
        $$ L = \mathbf{p} + s\mathbf{u} + t\mathbf{v} \quad (s, t \in \mathbb{R}) $$
        Here, the support point is $\mathbf{p}$, and the direction space is the 2D vector subspace $U = \text{Span}\{\mathbf{u}, \mathbf{v}\}$.

*   **Connection to Solutions of Linear Systems (Crucial Application)**

    Affine subspaces provide the perfect geometric description for the solution sets of linear systems.

    *   **Homogeneous System `Ax = 0`:** The set of all solutions to a homogeneous system is the **Null Space** of $A$, denoted $N(A)$. The null space is always a **vector subspace**.

    *   **Non-Homogeneous System `Ax = b`:** The set of all solutions to a non-homogeneous system (where $\mathbf{b} \ne \mathbf{0}$) is an **affine subspace**.
        Recall the general solution formula:
        $$ \mathbf{x} = \mathbf{x}_p + \mathbf{x}_h $$
        Let's map this to the definition of an affine subspace $L = \mathbf{p} + U$:
        *   The **particular solution** $\mathbf{x}_p$ serves as the **translation vector** $\mathbf{p}$.
        *   The set of all **homogeneous solutions** $\mathbf{x}_h$ is the **direction space** $U$. This is precisely the null space, $N(A)$.

        Therefore, the complete solution set for `Ax = b` is the affine subspace:
        $$ L = \mathbf{x}_p + N(A) $$
        This means the solution set is the null space $N(A)$ shifted by a particular solution vector $\mathbf{x}_p$.

*   **Summary of Key Differences**

| Feature | **Vector Subspace** (`U`) | **Affine Subspace** (`L = p + U`) |
| :--- | :--- | :--- |
| **Must Contain Origin?** | **Yes.** (`0 ∈ U`) | **No**, unless `p ∈ U`. |
| **Closure under Addition?** | **Yes.** If `u₁, u₂ ∈ U`, then `u₁ + u₂ ∈ U`. | **No.** In general, `l₁ + l₂ ∉ L`. |
| **Closure under Scaling?**| **Yes.** If `u ∈ U`, then `cu ∈ U`. | **No.** In general, `cl₁ ∉ L`. |
| **Geometric Example** | A line/plane through the origin. | Any line/plane, shifted. |
| **Linear System Example**| Solution set of `Ax = 0`. | Solution set of `Ax = b`. |

*   **Affine Combination**
    *   A related concept is an **affine combination**. It is a linear combination where the coefficients sum to 1.
        $$ \mathbf{y} = \alpha_1\mathbf{x}_1 + \alpha_2\mathbf{x}_2 + \cdots + \alpha_k\mathbf{x}_k \quad \text{where} \quad \sum_{i=1}^k \alpha_i = 1 $$
    *   An affine subspace is closed under affine combinations. The set of all affine combinations of a set of points forms the smallest affine subspace containing them (their "affine span").
[[如何用仿射子空间 (Affine Subspace) 的结构来理解线性方程组 `Aλ = b` 的通解]]
## Part V: Hyperplanes

A hyperplane is a generalization of the concept of a line (in 2D) and a plane (in 3D) to vector spaces of any dimension. It is an extremely important and common special case of an affine subspace.

### 1. Core Intuition

-   In a **2D** space ($\mathbb{R}^2$), a hyperplane is a **line** (which is 1-dimensional).
-   In a **3D** space ($\mathbb{R}^3$), a hyperplane is a **plane** (which is 2-dimensional).
-   In an **n-dimensional** space ($\mathbb{R}^n$), a hyperplane is an **(n-1)-dimensional** "flat" subspace.

Its key function is to "slice" the entire space into two half-spaces, making it an ideal **decision boundary** in classification problems.

#### 2. Two Equivalent Definitions of a Hyperplane

Hyperplanes can be defined in two equivalent ways: one algebraic and one geometric.

##### **Definition 1: The Algebraic Definition (via a Single Linear Equation)**

A **hyperplane** $H$ in $\mathbb{R}^n$ is the set of all points $\mathbf{x}$ that satisfy a single linear equation:
$$ a_1x_1 + a_2x_2 + \cdots + a_nx_n = d $$
where $a_1, \dots, a_n$ are coefficients that are not all zero, and `d` is a constant.

Using vector notation, this equation becomes much more compact:
$$ \mathbf{a}^T \mathbf{x} = d $$
-   **Normal Vector $\mathbf{a}$**: The vector $\mathbf{a} = (a_1, \dots, a_n)^T$ is called the **normal vector** to the hyperplane. Geometrically, it is **perpendicular** to the hyperplane itself.
-   **Offset `d`**: The constant `d` determines the hyperplane's offset from the origin.
    -   If `d = 0`, the hyperplane `aᵀx = 0` passes through the origin and is itself an **(n-1)-dimensional vector subspace**.
    -   If `d ≠ 0`, the hyperplane does not pass through the origin and is a true **affine subspace**.

##### **Definition 2: The Geometric Definition (via Affine Subspaces)**

A **hyperplane** $H$ in an n-dimensional vector space $V$ is an **affine subspace** of dimension **n-1**.
$$ H = \mathbf{p} + U $$
Where:
-   $\mathbf{p}$ is any specific point on the hyperplane (the support point).
-   $U$ is a vector subspace of dimension **n-1** (the direction space).

### 3. The Connection Between the Definitions

These two definitions are perfectly equivalent.

-   **From Algebraic to Geometric (`aᵀx = d` → `p + U`)**:
    1.  **Direction Space `U`**: The direction space `U` is the parallel hyperplane that passes through the origin. It is the set of all vectors `u` that satisfy `aᵀu = 0`. This set is the **orthogonal complement** of the normal vector `a` and has dimension n-1.
    2.  **Support Point `p`**: We can find a support point `p` by finding any **particular solution** to the equation `aᵀx = d`.

-   **Example**: Consider the plane $2x_1 + 3x_2 + 4x_3 = 12$ in $\mathbb{R}^3$.
    -   **Algebraic Form**: Normal vector $\mathbf{a} = (2, 3, 4)^T$, offset $d = 12$.
    -   **Geometric Form**:
        -   **Find a support point `p`**: Let $x_2=0, x_3=0$. Then $2x_1=12 \implies x_1=6$. So, a point on the plane is $\mathbf{p} = (6, 0, 0)^T$.
        -   **Find the direction space `U`**: `U` is the set of all vectors `u` such that $2u_1 + 3u_2 + 4u_3 = 0$. This is a 2-dimensional plane passing through the origin.
        -   The hyperplane can thus be written as $H = \begin{pmatrix} 6 \\ 0 \\ 0 \end{pmatrix} + \text{Span}\left\{\begin{pmatrix} -3/2 \\ 1 \\ 0 \end{pmatrix}, \begin{pmatrix} -2 \\ 0 \\ 1 \end{pmatrix}\right\}$, where the two vectors in the span form a basis for `U`.

### 4. Hyperplanes in Machine Learning

Hyperplanes are at the core of many machine learning algorithms, most famously the **Support Vector Machine (SVM)**.

-   **As a Decision Boundary**: In a binary classification problem, the goal is to find a hyperplane that best separates data points belonging to two different classes.
-   **The SVM Hyperplane**: An SVM seeks to find an optimal hyperplane defined by the equation:
    $$ \mathbf{w}^T\mathbf{x} - b = 0 $$
    -   $\mathbf{w}$ is the weight vector, which is equivalent to the **normal vector** `a`.
    -   `b` is the bias term, which is related to the **offset** `d`.
-   **The Classification Rule**:
    -   If a new data point $\mathbf{x}_{\text{new}}$ satisfies $\mathbf{w}^T\mathbf{x}_{\text{new}} - b > 0$, it is assigned to one class (e.g., the positive class).
    -   If it satisfies $\mathbf{w}^T\mathbf{x}_{\text{new}} - b < 0$, it is assigned to the other class (e.g., the negative class).
    -   This means the classification of a point is determined by which side of the hyperplane it lies on. The goal of an SVM is to find the `w` and `b` that make this separating "margin" as wide as possible.

## Part VI: Affine Mappings

We have established that linear mappings, of the form `φ(x) = Ax`, always preserve the origin (i.e., `φ(0) = 0`). However, many practical applications, especially in computer graphics, require transformations that include **translation**, which moves the origin. This more general class of transformation is called an affine mapping.

### 1. Core Idea: A Linear Map Followed by a Translation

An **affine mapping** is, in essence, a composition of a **linear mapping** and a **translation**.

-   **Linear Part:** Handles rotation, scaling, shearing, and other transformations that keep the origin fixed.
-   **Translation Part:** Shifts the entire result to a new location in the space.

### 2. Formal Definition

A mapping `f: V → W` from a vector space `V` to a vector space `W` is called an **affine mapping** if it can be written in the form:
$$ f(\mathbf{x}) = A\mathbf{x} + \mathbf{b} $$
Where:
-   `A` is an $m \times n$ matrix representing the **linear part** of the transformation.
-   `b` is an $m \times 1$ vector representing the **translation part**.

**Distinction from Linear Mappings:**
-   If the translation vector `b = 0`, the affine map degenerates into a purely linear map.
-   If `b ≠ 0`, then `f(0) = A(0) + b = b`, which means the origin is no longer mapped to the origin but is moved to the position defined by `b`.

### 3. Key Properties of Affine Mappings

While affine maps are generally not linear (since `f(x+y) ≠ f(x) + f(y)`), they preserve several crucial geometric properties.

1.  **Lines Map to Lines:** An affine map transforms a straight line into another straight line (or, in a degenerate case, a single point if the line's direction is in the null space of `A`).

2.  **Parallelism is Preserved:** If two lines are parallel, their images under an affine map will also be parallel.

3.  **Ratios of Lengths are Preserved:** If a point `P` is the midpoint of a line segment `QR`, then its image `f(P)` will be the midpoint of the image segment `f(Q)f(R)`. This property is vital for maintaining the relative structure of geometric shapes.

4.  **Affine Combinations are Preserved:** This is the most fundamental algebraic property of an affine map. If a point `y` is an affine combination of a set of points `xᵢ` (meaning `y = Σαᵢxᵢ` where `Σαᵢ = 1`), then its image `f(y)` is the **same affine combination** of the images `f(xᵢ)`:
    $$ f\left(\sum \alpha_i \mathbf{x}_i\right) = \sum \alpha_i f(\mathbf{x}_i), \quad \text{provided that} \quad \sum \alpha_i = 1 $$

### 4. Homogeneous Coordinates: The Trick to Unify Transformations

In fields like computer graphics, it is highly desirable to represent all transformations, including translations, with a **single matrix multiplication**. The standard form `Ax + b` requires both a multiplication and an addition, which is inconvenient for composing multiple transformations.

**Homogeneous Coordinates** elegantly solve this problem by adding an extra dimension, effectively turning an affine map into a linear map in a higher-dimensional space.

*   **How it Works:**
    1.  An n-dimensional vector `x = (x₁, ..., xₙ)ᵀ` is represented as an (n+1)-dimensional homogeneous vector:
        $$ \mathbf{x}_{\text{hom}} = \begin{bmatrix} x_1 \\ \vdots \\ x_n \\ 1 \end{bmatrix} $$
    2.  An affine map `f(x) = Ax + b` is represented by an `(n+1) × (n+1)` **augmented transformation matrix**:
        $$ T_f = \begin{bmatrix}
           & A & & \mathbf{b} \\
           \hline
           0 & \cdots & 0 & 1
           \end{bmatrix}
        $$
        Here, `A` is the $n \times n$ linear part, and `b` is the $n \times 1$ translation vector. The bottom row consists of zeros followed by a one.

*   **The Unified Operation:**
    The affine transformation can now be performed with a single matrix multiplication. The notation below shows the block matrix multiplication explicitly:

    $$
    T_f \mathbf{x}_{\text{hom}} =
    \begin{bmatrix}
    A & \mathbf{b} \\
    \mathbf{0}^T & 1
    \end{bmatrix}
    \begin{bmatrix}
    \mathbf{x} \\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    A\mathbf{x} + \mathbf{b}(1) \\
    \mathbf{0}^T\mathbf{x} + 1(1)
    \end{bmatrix}
    =
    \begin{bmatrix}
    A\mathbf{x} + \mathbf{b} \\
    1
    \end{bmatrix}
    $$
    The resulting vector's first `n` components are exactly the desired `Ax + b`, and the final component remains `1`.

*   **The Advantage:**
    This technique allows a sequence of transformations (e.g., a rotation, then a scaling, then a translation) to be composed by first multiplying their respective augmented matrices. The resulting single matrix can then be applied to all points, dramatically simplifying the computation and management of complex transformations.

### 5. Summary

| Concept                   | **Linear Mapping (`Ax`)**                                                    | **Affine Mapping (`Ax + b`)**                                                |
| :------------------------ | :--------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **Essence**               | Rotation / Scaling / Shearing                                                | Linear Transformation + Translation                                          |
| **Preserves Origin?**     | **Yes**, `f(0) = 0`                                                          | **No**, `f(0) = b` in general                                                |
| **Preserves Lin. Comb.?** | **Yes**                                                                      | **No**                                                                       |
| **What is Preserved?**    | Lines, parallelism, **linear combinations**                                  | Lines, parallelism, **affine combinations**                                  |
| **Representation**        | Matrix `A`                                                                   | Matrix `A` and vector `b`                                                    |
| **Homogeneous Form**      | $\begin{bsmallmatrix} A & \mathbf{0} \\ \mathbf{0}^T & 1 \end{bsmallmatrix}$ | $\begin{bsmallmatrix} A & \mathbf{b} \\ \mathbf{0}^T & 1 \end{bsmallmatrix}$ |
# Lecture 3: Analytic Geometry: Norms, Inner Products, and Lengths and Distances, Angles and Orthogonality

### Part I: Geometric Structures on Vector Spaces

In the previous parts, we established the algebraic framework of vector spaces and linear mappings. Now, we will enrich these spaces with **geometric structure**, allowing us to formalize intuitive concepts like the **length** of a vector, the **distance** between vectors, and the **angle** between them. These concepts are captured by norms and inner products.

#### 1. Norms

A norm is a formal generalization of the intuitive notion of a vector's "length" or "magnitude".

*   **Geometric Intuition:** The norm of a vector is its length, i.e., the distance from the origin to the point the vector represents.

*   **Formal Definition of a Norm:**
    A **norm** on a vector space $V$ is a function $\|\cdot\| : V \to \mathbb{R}$ that assigns a non-negative real value $\|\mathbf{x}\|$ to every vector $\mathbf{x} \in V$. This function must satisfy the following three axioms for all vectors $\mathbf{x}, \mathbf{y} \in V$ and any scalar $\lambda \in \mathbb{R}$:

    1.  **Positive Definiteness:** The length is positive, except for the zero vector.
        *   $\|\mathbf{x}\| \ge 0$
        *   $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$

    2.  **Absolute Homogeneity:** Scaling a vector scales its length by the same factor.
        *   $\|\lambda\mathbf{x}\| = |\lambda|\|\mathbf{x}\|$

    3.  **Triangle Inequality:** The length of one side of a triangle is no greater than the sum of the lengths of the other two sides.
        *   $\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$

    A vector space equipped with a norm is called a **normed vector space**.

*   **Examples of Norms on $\mathbb{R}^n$:**
    The idea of "length" can be defined in multiple ways. For a vector $\mathbf{x} = (x_1, \dots, x_n)^T$:

    *   **The $L_1$-norm (Manhattan Norm):** Measures the "city block" distance.
        $$ \|\mathbf{x}\|_1 := \sum_{i=1}^n |x_i| $$

    *   **The $L_2$-norm (Euclidean Norm):** The standard "straight-line" distance, derived from the Pythagorean theorem.
        $$ \|\mathbf{x}\|_2 := \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^T\mathbf{x}} $$
        **This is the default norm.** When we write $\|\mathbf{x}\|$ without a subscript, we almost always mean the Euclidean norm.

    *   **The $L_\infty$-norm (Maximum Norm):** The length is determined by the largest component of the vector.
        $$ \|\mathbf{x}\|_\infty := \max_{i=1}^n |x_i| $$

*   **Distance Derived from a Norm:**
    Any norm naturally defines a **distance** $d(\mathbf{x}, \mathbf{y})$ between two vectors as the norm of their difference vector:
    $$ d(\mathbf{x}, \mathbf{y}) := \|\mathbf{x} - \mathbf{y}\| $$

#### 2. Inner Products

An inner product is a more fundamental concept than a norm. It is a function that allows us to define not only the Euclidean norm but also the angle between vectors and the notion of orthogonality (perpendicularity).

*   **Motivation:** The inner product is a generalization of the familiar **dot product** in $\mathbb{R}^n$, which is defined as:
    $$ \langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T\mathbf{y} = \sum_{i=1}^n x_i y_i $$

*   **Formal Definition of an Inner Product:**
    An **inner product** on a real vector space $V$ is a function $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ that takes two vectors and returns a scalar. The function must be a **symmetric, positive-definite bilinear map**, satisfying the following axioms for all $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ and any scalar $\lambda \in \mathbb{R}$:

    1.  **Bilinearity:** The function is linear in each argument.
        *   *Linearity in the first argument:* $\langle \lambda\mathbf{x} + \mathbf{y}, \mathbf{z} \rangle = \lambda\langle\mathbf{x}, \mathbf{z}\rangle + \langle\mathbf{y}, \mathbf{z}\rangle$
        *   *Linearity in the second argument:* $\langle \mathbf{x}, \lambda\mathbf{y} + \mathbf{z} \rangle = \lambda\langle\mathbf{x}, \mathbf{y}\rangle + \langle\mathbf{x}, \mathbf{z}\rangle$

    2.  **Symmetry:** The order of the arguments does not matter.
        $$ \langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle $$

    3.  **Positive Definiteness:** The inner product of a vector with itself is non-negative, and is zero only for the zero vector.
        *   $\langle \mathbf{x}, \mathbf{x} \rangle \ge 0$
        *   $\langle \mathbf{x}, \mathbf{x} \rangle = 0 \iff \mathbf{x} = \mathbf{0}$

    A vector space equipped with an inner product is called an **inner product space**.

#### 3. The Bridge: From Inner Products to Geometry

The inner product is the foundation of Euclidean geometry within a vector space. All key geometric concepts can be derived from it.

*   **The Induced Norm:**
    **Every inner product naturally defines (or induces) a norm** given by:
    $$ \|\mathbf{x}\| := \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle} $$
    It can be proven that this definition satisfies all three norm axioms. The standard Euclidean norm $\|\mathbf{x}\|_2$ is precisely the norm induced by the standard dot product.

*   **The Cauchy-Schwarz Inequality:**
    This is one of the most important inequalities in mathematics. It relates the inner product of two vectors to their induced norms and provides the foundation for defining angles.
    $$ |\langle \mathbf{x}, \mathbf{y} \rangle| \le \|\mathbf{x}\| \|\mathbf{y}\| $$

*   **Geometric Concepts Defined by the Inner Product:**

    *   **Length:** $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$
    *   **Distance:** $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\| = \sqrt{\langle \mathbf{x}-\mathbf{y}, \mathbf{x}-\mathbf{y} \rangle}$
    *   **Angle:** The angle $\theta$ between two non-zero vectors $\mathbf{x}$ and $\mathbf{y}$ is defined via:
        $$ \cos\theta = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|} $$
        (The Cauchy-Schwarz inequality guarantees that the right-hand side is between -1 and 1, so $\theta$ is well-defined).

    *   **Orthogonality (Perpendicularity):**
        *   **Definition:** Two vectors $\mathbf{x}$ and $\mathbf{y}$ are **orthogonal** if their inner product is zero. We denote this as $\mathbf{x} \perp \mathbf{y}$.
            $$ \mathbf{x} \perp \mathbf{y} \iff \langle \mathbf{x}, \mathbf{y} \rangle = 0 $$
        *   **Geometric Meaning:** If the inner product is the standard dot product, orthogonality means the vectors are perpendicular (the angle between them is 90° or $\pi/2$ radians).
        *   **Pythagorean Theorem:** If $\mathbf{x} \perp \mathbf{y}$, then the familiar Pythagorean theorem holds:
            $$ \|\mathbf{x} + \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 $$


### Part II: Geometric Structures on Vector Spaces

#### 1. Symmetric, Positive Definite (SPD) Matrices and Inner Products

In finite-dimensional vector spaces like $\mathbb{R}^n$, the abstract concept of an inner product can be concretely represented and computed using a special class of matrices. These are **Symmetric, Positive Definite (SPD)** matrices. They are fundamental in machine learning, statistics, and optimization because they provide a way to define custom, yet valid, notions of distance, angle, and similarity, which are essential for algorithms like Support Vector Machines (with kernels), Gaussian models, and Newton's method in optimization.

*   **Definition of a Symmetric, Positive Definite Matrix:**
    A square matrix $A \in \mathbb{R}^{n \times n}$ is called **symmetric, positive definite** (SPD) if it satisfies two conditions:

    1.  **Symmetry:** The matrix is equal to its transpose.
        $$ A = A^T $$
    2.  **Positive Definiteness:** The quadratic form $x^T A x$ is strictly positive for every non-zero vector $x \in \mathbb{R}^n$.[[Notion/Class/Proof/Quadratic Form\|Quadratic Form]]
        $$ \mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n, \mathbf{x} \ne \mathbf{0} $$

*   **The Central Theorem: The Matrix Representation of Inner Products**
    The deep connection between algebra and geometry is captured by the following theorem, which states that inner products and SPD matrices are two sides of the same coin.

    **Theorem:** Let $V$ be an $n$-dimensional real vector space with an ordered basis $B$. Let $\hat{\mathbf{x}}$ and $\hat{\mathbf{y}}$ be the coordinate vectors of $\mathbf{x}, \mathbf{y} \in V$ with respect to basis $B$. A function defined by
    $$ \langle \mathbf{x}, \mathbf{y} \rangle := \hat{\mathbf{x}}^T A \hat{\mathbf{y}} $$
    is a valid **inner product** on $V$ if and only if the matrix $A \in \mathbb{R}^{n \times n}$ is **symmetric and positive definite**.

    **Explanation:**
    *   This theorem provides a universal recipe for all possible inner products on a finite-dimensional space.
    *   The standard **dot product** in $\mathbb{R}^n$ is the simplest case of this theorem, where the matrix $A$ is the identity matrix $I$:
        $$ \langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T I \mathbf{y} = \mathbf{x}^T \mathbf{y} $$
    *   More importantly, *any* SPD matrix $A$ can be used to define a new, perfectly valid inner product $\langle \cdot, \cdot \rangle_A$ on $\mathbb{R}^n$. This new inner product defines a new geometry on the space, with its own corresponding notions of length $(\|\mathbf{x}\|_A = \sqrt{\mathbf{x}^T A \mathbf{x}})$ and orthogonality $(\mathbf{x}^T A \mathbf{y} = 0)$.

*   **Properties of SPD Matrices**
    If a matrix $A$ is symmetric and positive definite, it has several important properties that follow directly from its definition:

    1.  **Invertibility (Trivial Null Space):** An SPD matrix is always invertible. Its null space (or kernel) contains only the zero vector.
        *   **Proof:** Suppose there exists a non-zero vector $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{0}$. Then multiplying by $\mathbf{x}^T$ gives $\mathbf{x}^T A \mathbf{x} = \mathbf{x}^T \mathbf{0} = 0$. This contradicts the positive definiteness condition, which states that $\mathbf{x}^T A \mathbf{x}$ must be strictly greater than 0 for any non-zero $\mathbf{x}$. Therefore, no such non-zero $\mathbf{x}$ can exist, and the only solution to $A\mathbf{x} = \mathbf{0}$ is $\mathbf{x} = \mathbf{0}$.

    2.  **Positive Diagonal Elements:** All the diagonal elements of an SPD matrix are strictly positive.
        *   **Proof:** To find the $i$-th diagonal element, $a_{ii}$, we can choose the standard basis vector $\mathbf{e}_i$ (which has a 1 in the $i$-th position and 0s elsewhere). Since $\mathbf{e}_i$ is a non-zero vector, we must have $\mathbf{e}_i^T A \mathbf{e}_i > 0$. But $\mathbf{e}_i^T A \mathbf{e}_i$ is precisely the element $a_{ii}$. Thus, $a_{ii} > 0$ for all $i$.

*   **Recap: Inner Product vs. Dot Product**
    It is crucial to distinguish between the general concept and its most common example:

    *   **Inner Product $\langle \mathbf{x}, \mathbf{y} \rangle$:** This is the **general concept**. It is any function that is bilinear, symmetric, and positive definite.
    *   **Dot Product $\mathbf{x}^T \mathbf{y}$:** This is a **specific example** of an inner product in $\mathbb{R}^n$. It is the inner product defined by the identity matrix, $A=I$.
    *   **Euclidean Norm $\|\mathbf{x}\|_2$:** This is the norm that is **induced by the dot product**: $\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^T\mathbf{x}}$. Other inner products (defined by other SPD matrices $A$) induce different norms.
You are absolutely right. My apologies for that oversight. I failed to maintain the English language consistency in the last response.

### Part III: Angles, Orthogonality, and Orthogonal Matrices

#### 1. Angles and Orthogonality

Inner products not only define lengths and distances but also enable us to define the **angle** between vectors, thereby generalizing the concept of "perpendicularity".

*   **Defining the Angle:**
    *   The Cauchy-Schwarz inequality guarantees that for any non-zero vectors $\mathbf{x}$ and $\mathbf{y}$:
        $$ -1 \le \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|} \le 1 $$
    *   This ensures that we can uniquely define an angle $\omega \in [0, \pi]$ (i.e., from 0° to 180°) such that:
        $$ \cos\omega = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|} $$
    *   This $\omega$ is defined as the **angle** between vectors $\mathbf{x}$ and $\mathbf{y}$. It measures the similarity of their orientation.

*   **Orthogonality:**
    *   **Definition:** Two vectors $\mathbf{x}$ and $\mathbf{y}$ are **orthogonal** if their inner product is zero. This is denoted as $\mathbf{x} \perp \mathbf{y}$.
        $$ \mathbf{x} \perp \mathbf{y} \iff \langle \mathbf{x}, \mathbf{y} \rangle = 0 $$
    *   **Geometric Meaning:** When $\langle \mathbf{x}, \mathbf{y} \rangle = 0$, it follows that $\cos\omega = 0$, which means the angle $\omega = \pi/2$ (90°). Therefore, orthogonality is a direct generalization of the geometric concept of "perpendicular".
    *   **Important Corollary:** The zero vector $\mathbf{0}$ is orthogonal to every vector, since $\langle \mathbf{0}, \mathbf{x} \rangle = 0$.

*   **Orthonormality:**
    *   Two vectors $\mathbf{x}$ and $\mathbf{y}$ are **orthonormal** if they are both **orthogonal** ($\langle \mathbf{x}, \mathbf{y} \rangle = 0$) and are **unit vectors** ($\|\mathbf{x}\| = 1$, $\|\mathbf{y}\| = 1$).

*   **Key Point: Orthogonality Depends on the Inner Product**
    *   Just like length, whether two vectors are orthogonal depends entirely on the chosen inner product.
    *   For example, the vectors $\mathbf{x}=[1, 1]^T$ and $\mathbf{y}=[-1, 1]^T$ are orthogonal under the standard dot product ($\mathbf{x}^T\mathbf{y} = 0$), but they may not be orthogonal under a different inner product defined by an SPD matrix $A$, such as $\langle \mathbf{x}, \mathbf{y} \rangle_A = \mathbf{x}^T A \mathbf{y}$.

#### 2. Orthogonal Matrices

An orthogonal matrix is a special type of square matrix whose corresponding linear transformation geometrically represents a shape-preserving transformation (like a rotation or reflection) and has excellent computational properties.

*   **Definition:**
    A square matrix $A \in \mathbb{R}^{n \times n}$ is called an **orthogonal matrix** if and only if its columns form an **orthonormal set**.

*   **Equivalent Properties:**
    The following statements are equivalent and are often used as practical tests for orthogonality:
    1.  The columns of $A$ are orthonormal.
    2.  $A^T A = I$
    3.  $A^{-1} = A^T$
    *   **Core Idea:** The inverse of an orthogonal matrix is simply its transpose. This makes the computationally expensive operation of inversion trivial.

*   **Geometric Properties: Preserving Lengths and Angles**
    A linear transformation $T(\mathbf{x}) = A\mathbf{x}$ defined by an orthogonal matrix $A$ is a **rigid transformation**, meaning it does not alter the geometry of the space.

    1.  **Preserves Lengths:**
        $$ \|A\mathbf{x}\|^2 = (A\mathbf{x})^T(A\mathbf{x}) = \mathbf{x}^T A^T A \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \mathbf{x}^T\mathbf{x} = \|\mathbf{x}\|^2 $$
        Therefore, $\|A\mathbf{x}\| = \|\mathbf{x}\|$.

    2.  **Preserves Inner Products and Angles:**
        $$ \langle A\mathbf{x}, A\mathbf{y} \rangle = (A\mathbf{x})^T(A\mathbf{y}) = \mathbf{x}^T A^T A \mathbf{y} = \mathbf{x}^T I \mathbf{y} = \langle \mathbf{x}, \mathbf{y} \rangle $$
        Since the inner product is preserved, the angle defined by it is also preserved.

*   **Practical Meaning and Applications:**
    *   **Geometric Models:** Orthogonal matrices perfectly model **rotations** and **reflections** in space.
    *   **Numerical Stability:** Algorithms involving orthogonal matrices (like QR decomposition) are typically very numerically stable.
    *   **Change of Coordinate Systems:** The transformation from one orthonormal basis to another is described by an orthogonal matrix.
    *   **Construction:** The **Gram-Schmidt process** can be used to construct an orthonormal basis from any set of linearly independent vectors, which can then be used as the columns of an orthogonal matrix.

## Part IV: Metric Spaces and the Formal Definition of Distance

Previously, we defined distance in an inner product space as `d(x, y) = ||x - y||`. This is a specific instance of a much more general concept called a **metric**. A metric formalizes the intuitive notion of "distance" between elements of any set, not just vectors.

### 1. The Metric Function (度量函数)

A **metric** (or **distance function**) on a set `V` is a function that quantifies the distance between any two elements of that set.

*   **Formal Definition:**
    A **metric** on a set `V` is a function `d : V × V → ℝ` that maps a pair of elements `(x, y)` to a real number `d(x, y)`, satisfying the following three axioms for all `x, y, z ∈ V`:

    1.  **Positive Definiteness (正定性):**
        *   `d(x, y) ≥ 0` (Distance is always non-negative).
        *   `d(x, y) = 0` if and only if `x = y` (The distance from an element to itself is zero, and it's the only case where the distance is zero).

    2.  **Symmetry (对称性):**
        *   `d(x, y) = d(y, x)` (The distance from x to y is the same as the distance from y to x).

    3.  **Triangle Inequality (三角不等式):**
        *   `d(x, z) ≤ d(x, y) + d(y, z)` (The direct path is always the shortest; going from x to z via y is at least as long).

*   **Metric Space (度量空间):**
    A set `V` equipped with a metric `d` is called a **metric space**, denoted as `(V, d)`.

### 2. The Connection: From Inner Products to Metrics

The distance we derived from the inner product is a valid metric. We can prove this by showing it satisfies all three metric axioms.

*   **Theorem:** The distance function `d(x, y) = ||x - y|| = √⟨x-y, x-y⟩` induced by an inner product is a metric.

*   **Proof:**
    1.  **Positive Definiteness:**
        *   `d(x, y) = ||x - y||`. By the positive definiteness of norms, `||x - y|| ≥ 0`.
        *   Equality `||x - y|| = 0` holds if and only if `x - y = 0`, which means `x = y`. The axiom holds.

    2.  **Symmetry:**
        *   `d(x, y) = ||x - y||`. Using the properties of norms, `||x - y|| = ||(-1)(y - x)|| = |-1| ||y - x|| = ||y - x|| = d(y, x)`. The axiom holds.

    3.  **Triangle Inequality:**
        *   `d(x, z) = ||x - z||`. We can rewrite the argument as `x - z = (x - y) + (y - z)`.
        *   By the triangle inequality for norms, `||(x - y) + (y - z)|| ≤ ||x - y|| + ||y - z||`.
        *   Substituting back the distance definition, we get `d(x, z) ≤ d(x, y) + d(y, z)`. The axiom holds.

### 3. Why is the Concept of a Metric Useful?

The power of defining a metric abstractly is that it allows us to measure "distance" in contexts far beyond standard Euclidean geometry.

*   **Generalization:** It applies to any set, including:
    *   **Strings:** The **edit distance** (or Levenshtein distance) between two strings (e.g., "apple" and "apply") is a metric. It counts the minimum number of edits (insertions, deletions, substitutions) needed to change one string into the other.
    *   **Graphs:** The shortest path distance between two nodes in a graph is a metric.
    *   **Functions:** We can define metrics to measure how "far apart" two functions are.

*   **Foundation for Other Fields:** The concept of a metric space is foundational for topology, analysis, and many areas of machine learning. For example, the **k-Nearest Neighbors (k-NN)** algorithm can work with any valid metric to find the "closest" data points, not just Euclidean distance.

### 4. Summary: Hierarchy of Spaces

This shows how these geometric concepts build upon each other:

`Inner Product Space` → `Normed Space` → `Metric Space` → `Topological Space`

*   Every **inner product** induces a **norm**.
*   Every **norm** induces a **metric**.
*   Every **metric** induces a **topology** (a notion of "open sets" and "closeness").

However, the reverse is not always true. There are metrics (like edit distance) that do not come from a norm, and norms (like the L1-norm) that do not come from an inner product.

## Part V: Orthogonal Projections

Orthogonal projection is a fundamental operation that finds the "closest" vector in a subspace to a given vector in the larger space. It is the geometric foundation for concepts like least-squares approximation.

### 1. The Concept of Orthogonal Projection

Let $U$ be a subspace of an inner product space $V$ (e.g., $\mathbb{R}^n$ with the dot product), and let $\mathbf{x} \in V$ be a vector. The **orthogonal projection** of $\mathbf{x}$ onto the subspace $U$, denoted $\pi_U(\mathbf{x})$, is the unique vector in $U$ that is "closest" to $\mathbf{x}$.

This projection, which we will call $\mathbf{p} = \pi_U(\mathbf{x})$, is defined by two fundamental properties:

1.  **Membership Property:** The projection $\mathbf{p}$ must lie within the subspace $U$.
    -   ($\mathbf{p} \in U$)

2.  **Orthogonality Property:** The vector connecting the original vector $\mathbf{x}$ to its projection $\mathbf{p}$ (the "error" vector $\mathbf{x} - \mathbf{p}$) must be orthogonal to the entire subspace $U$.
    -   ($(\mathbf{x} - \mathbf{p}) \perp U$)

This second property implies that $(\mathbf{x} - \mathbf{p})$ must be orthogonal to every vector in $U$. A key theorem states that this is equivalent to being orthogonal to all vectors in a **basis** for $U$.

### 2. Deriving the Projection Formula (The Normal Equation)

Our goal is to find an algebraic method to compute the projection vector $\mathbf{p}$. The strategy is to translate the two geometric properties above into a system of linear equations.

**Step 1: Express the Membership Property using a Basis**

First, we need a basis for the subspace $U$. Let's say we find a basis $\{\mathbf{b}_1, \mathbf{b}_2, \dots, \mathbf{b}_k\}$. We can arrange these basis vectors as the columns of a matrix $B$:
$$ B = \begin{bmatrix} | & | & & | \\ \mathbf{b}_1 & \mathbf{b}_2 & \cdots & \mathbf{b}_k \\ | & | & & | \end{bmatrix} $$
Since the projection $\mathbf{p}$ must be in the subspace $U$ (which is the column space of $B$), $\mathbf{p}$ must be a linear combination of the columns of $B$. This means there must exist a unique vector of coefficients $\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_k)^T$ such that:
$$ \mathbf{p} = \lambda_1\mathbf{b}_1 + \lambda_2\mathbf{b}_2 + \cdots + \lambda_k\mathbf{b}_k $$
In matrix form, this is written as:
$$ \mathbf{p} = B\boldsymbol{\lambda} $$
Our problem is now reduced from finding the vector $\mathbf{p}$ to finding the unknown coefficient vector $\boldsymbol{\lambda}$.

**Step 2: Express the Orthogonality Property as an Equation**

The orthogonality property states that $(\mathbf{x} - \mathbf{p}) \perp U$. This means the dot product of $(\mathbf{x} - \mathbf{p})$ with every basis vector of $U$ must be zero:
$$
\begin{cases}
    \mathbf{b}_1 \cdot (\mathbf{x} - \mathbf{p}) = 0 \\
    \mathbf{b}_2 \cdot (\mathbf{x} - \mathbf{p}) = 0 \\
    \quad \vdots \\
    \mathbf{b}_k \cdot (\mathbf{x} - \mathbf{p}) = 0
\end{cases}
\quad \iff \quad
\begin{cases}
    \mathbf{b}_1^T (\mathbf{x} - \mathbf{p}) = 0 \\
    \mathbf{b}_2^T (\mathbf{x} - \mathbf{p}) = 0 \\
    \quad \vdots \\
    \mathbf{b}_k^T (\mathbf{x} - \mathbf{p}) = 0
\end{cases}
$$
This system of equations can be written compactly in matrix form. Notice that the rows of the matrix are the transposes of our basis vectors, which is exactly the matrix $B^T$:
$$
\begin{bmatrix}
\text{--- } \mathbf{b}_1^T \text{ ---} \\
\text{--- } \mathbf{b}_2^T \text{ ---} \\
\vdots \\
\text{--- } \mathbf{b}_k^T \text{ ---}
\end{bmatrix}
(\mathbf{x} - \mathbf{p}) = \mathbf{0}
\quad \implies \quad
B^T(\mathbf{x} - \mathbf{p}) = \mathbf{0}
$$

**Step 3: Combine and Solve for λ**

We now have a system of two equations:
1.  $\mathbf{p} = B\boldsymbol{\lambda}$
2.  $B^T(\mathbf{x} - \mathbf{p}) = \mathbf{0}$

Substitute the first equation into the second:
$$ B^T(\mathbf{x} - B\boldsymbol{\lambda}) = \mathbf{0} $$
Distribute $B^T$:
$$ B^T\mathbf{x} - B^T(B\boldsymbol{\lambda}) = \mathbf{0} $$
Rearrange the terms to isolate the unknown $\boldsymbol{\lambda}$:
$$ (B^T B)\boldsymbol{\lambda} = B^T\mathbf{x} $$

This final result is known as the **Normal Equation**. It is a system of linear equations for the unknown coefficients $\boldsymbol{\lambda}$.

### 3. The Algorithm for Orthogonal Projection

Given a vector $\mathbf{x}$ and a subspace $U$:

1.  **Find a Basis:** Find a basis $\{\mathbf{b}_1, \dots, \mathbf{b}_k\}$ for the subspace $U$.
2.  **Form the Basis Matrix `B`:** Create a matrix $B$ whose columns are the basis vectors.
3.  **Set up the Normal Equation:** Compute the matrix `BᵀB` and the vector `Bᵀx`.
4.  **Solve for `λ`:** Solve the linear system `(BᵀB)λ = Bᵀx` to find the coefficient vector `λ`.
5.  **Compute the Projection `p`:** Calculate the final projection vector using the formula `p = Bλ`.

**Important Note:** The matrix `BᵀB` is square and is invertible if and only if the columns of `B` are linearly independent (which they are, because they form a basis).

### 4. Special Case: Orthonormal Basis

If the basis for $U$ is **orthonormal**, the columns of $B$ are orthonormal. In this case, the calculation simplifies dramatically:

*   $B^T B = I$ (the identity matrix).
*   The Normal Equation becomes: $I \boldsymbol{\lambda} = B^T\mathbf{x} \implies \boldsymbol{\lambda} = B^T\mathbf{x}$.
*   The projection formula becomes: $\mathbf{p} = B\boldsymbol{\lambda} = B(B^T\mathbf{x}) = (BB^T)\mathbf{x}$.

The matrix $P = BB^T$ is called the **projection matrix**. This simplified formula only works when the basis is orthonormal. For a general, non-orthogonal basis, one must solve the full Normal Equation.

# Lecture 4: Analytic Geometry: Orthonormal Basis, Orthogonal Complement, Inner Product of Functions, Orthogonal Projections, Rotations

## Part I: Orthonormal Basis and Orthogonal Complement

### 1. Orthonormal Basis

*   **Foundation:** In an n-dimensional vector space, a basis is a set of n linearly independent vectors. The inner product allows us to define geometric concepts like length and angle.
*   **Definition:** An **orthonormal basis** is a special type of basis where all basis vectors are mutually orthogonal (perpendicular) and each basis vector is a unit vector (has a length of 1).
    *   **Formally:** For a basis $\{\mathbf{b}_1, \dots, \mathbf{b}_n\}$ of a vector space $V$:
        *   **Orthogonality:** $\langle \mathbf{b}_i, \mathbf{b}_j \rangle = 0$ for all $i \neq j$.
        *   **Normalization:** $\langle \mathbf{b}_i, \mathbf{b}_i \rangle = \|\mathbf{b}_i\|^2 = 1$ for all $i$.
    *   **Orthogonal Basis:** If only the orthogonality condition holds (vectors are perpendicular but not necessarily unit length), the basis is called an **orthogonal basis**.
*   **Canonical Example:** The standard Cartesian basis in $\mathbb{R}^n$ (e.g., $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ in $\mathbb{R}^3$) is the most common example of an orthonormal basis.
*   **Example in $\mathbb{R}^2$:** The vectors $\mathbf{b}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$ and $\mathbf{b}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$ form an orthonormal basis because $\mathbf{b}_1^T \mathbf{b}_2 = 0$ and $\|\mathbf{b}_1\| = \|\mathbf{b}_2\| = 1$.

### 2. Gram-Schmidt Process: Constructing an Orthonormal Basis

The Gram-Schmidt process is a fundamental algorithm that transforms any set of linearly independent vectors (a basis) into an orthonormal basis for the same subspace.

*   **Goal:** Given a basis $\{\mathbf{a}_1, \dots, \mathbf{a}_n\}$, produce an orthonormal basis $\{\mathbf{q}_1, \dots, \mathbf{q}_n\}$.
*   **Algorithm Steps:**
    1.  **Initialize:** Start with the first vector. Normalize it to get the first orthonormal basis vector.
        $$ \mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|} $$
    2.  **Iterate and Orthogonalize:** For each subsequent vector $\mathbf{a}_k$ (from $k=2$ to $n$):
        a.  **Project and Subtract:** Calculate the projection of $\mathbf{a}_k$ onto the subspace already spanned by the previously found orthonormal vectors $\{\mathbf{q}_1, \dots, \mathbf{q}_{k-1}\}$. Subtract this projection from $\mathbf{a}_k$ to get a vector $\mathbf{v}_k$ that is orthogonal to that subspace.
            $$ \mathbf{v}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} \langle \mathbf{a}_k, \mathbf{q}_j \rangle \mathbf{q}_j $$
        b.  **Normalize:** Normalize the resulting orthogonal vector $\mathbf{v}_k$ to make it a unit vector.
            $$ \mathbf{q}_k = \frac{\mathbf{v}_k}{\|\mathbf{v}_k\|} $$
*   **Result:** The set $\{\mathbf{q}_1, \dots, \mathbf{q}_n\}$ is an orthonormal basis for the same space spanned by the original vectors $\{\mathbf{a}_1, \dots, \mathbf{a}_n\}$.

#### 3. Orthogonal Complement and Decomposition

The orthogonal complement generalizes the concept of perpendicularity from single vectors to entire subspaces.

*   **Definition (Orthogonal Complement):** Let $U$ be a subspace of a vector space $V$. The **orthogonal complement** of $U$, denoted $U^\perp$, is the set of all vectors in $V$ that are orthogonal to *every* vector in $U$.
    $$ U^\perp = \{ \mathbf{v} \in V \mid \langle \mathbf{v}, \mathbf{u} \rangle = 0 \text{ for all } \mathbf{u} \in U \} $$

*   **Space Decomposition (Direct Sum):** The entire space $V$ can be uniquely decomposed into the **direct sum** of the subspace $U$ and its orthogonal complement $U^\perp$. This implies that the intersection of $U$ and $U^\perp$ contains only the zero vector, and the sum of their dimensions equals the dimension of $V$.
    $$ V = U \oplus U^\perp \quad \text{and} \quad \dim(U) + \dim(U^\perp) = \dim(V) $$

*   **Vector Decomposition (Orthogonal Decomposition):** Based on the direct sum decomposition of the space, **any vector $\mathbf{x}$** in $V$ can be **uniquely** decomposed into the sum of a component in $U$ and a component in $U^\perp$.
    *   **Conceptually:**
        $$ \mathbf{x} = \mathbf{x}_U + \mathbf{x}_{U^\perp} \quad (\text{where } \mathbf{x}_U \in U, \mathbf{x}_{U^\perp} \in U^\perp) $$
    *   **Basis-Level Representation:** Computationally, this decomposition is expressed by finding the unique coordinates of $\mathbf{x}$ with respect to a basis for $U$ and a basis for $U^\perp$.
        $$ \mathbf{x} = \sum_{m=1}^{M} \lambda_m \mathbf{b}_m + \sum_{j=1}^{D-M} \psi_j \mathbf{b}_j^\perp $$
        where $\{\mathbf{b}_1, \ldots, \mathbf{b}_M\}$ is a basis for $U$, $\{\mathbf{b}_1^\perp, \ldots, \mathbf{b}_{D-M}^\perp\}$ is a basis for $U^\perp$, and $\lambda_m, \psi_j$ are the unique scalar coordinates.
    *   The component $\mathbf{x}_U = \sum \lambda_m \mathbf{b}_m$ is precisely the **orthogonal projection** of $\mathbf{x}$ onto the subspace $U$.

### 4. Applications and Examples of Orthogonal Complements

*   **Geometric Interpretation:** Orthogonal complements provide a powerful way to describe geometric objects.
    *   In $\mathbb{R}^3$, the orthogonal complement of a plane (a 2D subspace) is the line perpendicular to it (a 1D subspace), often called the **normal line**. The basis vector for this line is the plane's **normal vector**.
    *   More generally, in $\mathbb{R}^n$, the orthogonal complement of a hyperplane (an (n-1)-dimensional subspace) is a line, and vice-versa.
*   **Example in $\mathbb{R}^3$:**
    *   Let $U = \text{span}\{\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}\}$ be the xy-plane.
    *   Its orthogonal complement $U^\perp$ is the set of all vectors orthogonal to the xy-plane, which is the z-axis: $U^\perp = \text{span}\{\begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}\}$.
    *   We can see that $\dim(U) + \dim(U^\perp) = 2 + 1 = 3 = \dim(\mathbb{R}^3)$.

*   **Clarification on Dimension:** It's important to remember that if $U$ is a proper subspace of $V$, its dimension must be strictly less than the dimension of $V$ ($\dim(U) < \dim(V)$), even though the vectors in $U$ have the same number of coordinates as the vectors in $V$.

## Part II: Inner Product of Functions, Orthogonal Projections, and Rotations

### 1. Inner Product of Functions

The concept of an inner product can be extended from the familiar space $\mathbb{R}^n$ to vector spaces of functions, enabling us to apply geometric intuition to abstract objects like polynomials or signals.

*   **Definition:** For the vector space of continuous functions on an interval $[a, b]$, the standard inner product between two functions $f(x)$ and $g(x)$ is defined by the integral:
    $$ \langle f, g \rangle := \int_{a}^{b} f(x)g(x) \,dx $$
*   **Induced Geometry:** This definition allows us to measure:
    *   **Length (Norm):** $\|f\| = \sqrt{\int_{a}^{b} f(x)^2 \,dx}$
    *   **Distance:** $d(f, g) = \|f - g\|$
    *   **Orthogonality:** Two functions are orthogonal if $\langle f, g \rangle = 0$.
*   **Significance:** This generalization is crucial for fields like signal processing and quantum mechanics, and it forms the mathematical basis for **Fourier series**, which decompose functions into a sum of orthogonal sine and cosine functions.

### 2. Orthogonal Projections

Orthogonal projection is a core operation for finding the "best approximation" or "closest point" of a vector within a given subspace. It is the geometric foundation for solving least-squares problems.

*   **Concept:** The orthogonal projection of a vector $\mathbf{x}$ onto a subspace $U$ is the unique vector $\mathbf{p} \in U$ such that the error vector $(\mathbf{x} - \mathbf{p})$ is orthogonal to the entire subspace $U$.
*   **The Projection Formula:**
    1.  **Form a Basis Matrix:** Create a matrix $B$ whose columns are a basis for the subspace $U$.
    2.  **Solve the Normal Equation:** Find the coordinate vector $\boldsymbol{\lambda}$ by solving the system:
        $$ (B^T B)\boldsymbol{\lambda} = B^T\mathbf{x} $$
    3.  **Compute the Projection:** The projection vector is then given by:
        $$ \mathbf{p} = B\boldsymbol{\lambda} $$
*   **[[Notion/Class/Proof/Projection Matrix\|Projection Matrix]]:** The linear transformation that maps any vector $\mathbf{x}$ to its projection $\mathbf{p}$ is represented by the projection matrix $P = B(B^T B)^{-1}B^T$.
*   **Special Case (Orthonormal Basis):** If the columns of $B$ form an orthonormal basis, then $B^T B = I$, and the formulas simplify significantly:
    *   The coordinates are $\boldsymbol{\lambda} = B^T\mathbf{x}$.
    *   The projection is $\mathbf{p} = (BB^T)\mathbf{x}$.

### 3. Rotations

Rotations are a fundamental class of geometric transformations that preserve the shape and size of objects. In linear algebra, they are represented by a special type of matrix.

*   **Definition:** A rotation is a linear transformation, represented by a matrix $R$, that preserves lengths and angles.
*   **Matrix Properties:** A matrix $R$ represents a pure rotation if it satisfies two conditions:
    1.  **Orthogonality:** The matrix must be orthogonal, meaning $R^T R = I$. This ensures that lengths and angles are preserved.
    2.  **Orientation Preservation:** The determinant of the matrix must be `+1`, i.e., $\det(R) = 1$. (An orthogonal matrix with determinant `-1` represents a reflection, which is a "mirror image" transformation).
*   **2D Rotation:** The matrix for a counter-clockwise rotation by an angle $\theta$ in the 2D plane is:
    $$ R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} $$
*   **Group Structure:** The set of all $n \times n$ rotation matrices forms a mathematical group known as the **Special Orthogonal Group**, denoted $SO(n)$.

### Part III: Orthogonal Projections

Projections are a critical class of linear transformations, widely used in graphics, coding theory, statistics, and machine learning.

#### 1. The Importance and Concept of Orthogonal Projections

*   **Motivation in Machine Learning:** In machine learning, we often deal with high-dimensional data that is difficult to analyze or visualize. The key insight is that most of the relevant information is often contained within a much lower-dimensional subspace.
*   **The Goal (Dimensionality Reduction):** By projecting high-dimensional data onto a carefully chosen lower-dimensional "feature space", we can simplify the problem, reduce computational cost, and extract meaningful patterns. The objective is to perform this projection while **minimizing information loss**.
*   **What is an Orthogonal Projection?**
    *   It is a linear transformation that "drops" a vector from a higher-dimensional space onto a lower-dimensional subspace.
    *   It is "orthogonal" because it does so in a way that **retains as much information as possible** by **minimizing the error** (the distance) between the original data and its projected image.
    *   This property makes it central to linear regression, classification, and data compression.

#### 2. The Formal Definition and Properties of Projection

*   **Algebraic Definition (Idempotence):** A linear mapping $\pi: V \to U$ is called a **projection** if applying it twice is the same as applying it once. This is known as the **idempotent property**.
    $$ \pi^2 = \pi \quad (\text{or } \pi(\pi(\mathbf{x})) = \pi(\mathbf{x})) $$
    *   **Matrix Form:** A square matrix $P$ is a **projection matrix** if it satisfies $P^2 = P$.
*   **Geometric Definition (Closest Point):** The **orthogonal projection** $\pi_U(\mathbf{x})$ of a vector $\mathbf{x}$ onto a subspace $U$ is the unique point in $U$ that is **closest** to $\mathbf{x}$.
    *   This "closest point" condition is equivalent to the **orthogonality condition**: the difference vector $(\mathbf{x} - \pi_U(\mathbf{x}))$ must be orthogonal to every vector in the subspace $U$.

#### 3. Projection onto One-Dimensional Subspaces (Lines)
![Image/Class/Mathematics-for-AI/5.png](/img/user/Image/Class/Mathematics-for-AI/5.png)
We begin by deriving the projection formula for the simplest case: projecting a vector onto a line, assuming the standard dot product as the inner product unless stated otherwise.

*   **Setup:**
    *   Let $U$ be a one-dimensional subspace (a line through the origin).
    *   Let $\mathbf{b}$ be a non-zero basis vector that spans this line, so $U = \text{span}\{\mathbf{b}\}$.
*   **Derivation:**
    1.  **Membership Property:** The projection $\pi_U(\mathbf{x})$ must lie on the line, so it must be a scalar multiple of the basis vector $\mathbf{b}$.
        $$ \pi_U(\mathbf{x}) = \lambda\mathbf{b} $$
        The goal is to find the scalar coordinate $\lambda$.
    2.  **Orthogonality Property:** The error vector $(\mathbf{x} - \pi_U(\mathbf{x}))$ must be orthogonal to the basis vector $\mathbf{b}$.
        $$ \langle \mathbf{x} - \lambda\mathbf{b}, \mathbf{b} \rangle = 0 $$
    3.  **Solve for $\lambda$:** Using bilinearity, we get $\langle \mathbf{x}, \mathbf{b} \rangle - \lambda\langle \mathbf{b}, \mathbf{b} \rangle = 0$, which gives:
        $$ \lambda = \frac{\langle \mathbf{x}, \mathbf{b} \rangle}{\langle \mathbf{b}, \mathbf{b} \rangle} = \frac{\mathbf{b}^T\mathbf{x}}{\|\mathbf{b}\|^2} $$
        *(If $\|\mathbf{b}\|=1$, then $\lambda = \mathbf{b}^T\mathbf{x}$)*.
*   **Final Formulas for 1D Projection:**
    *   **Projection Vector:**
        $$ \pi_U(\mathbf{x}) = \left( \frac{\mathbf{b}^T\mathbf{x}}{\|\mathbf{b}\|^2} \right) \mathbf{b} $$
    *   **Length of Projection:** The length of the projection is $\|\pi_U(\mathbf{x})\| = |\lambda|\|\mathbf{b}\|$. If $\mathbf{b}$ is a unit vector, this simplifies to $\|\pi_U(\mathbf{x})\| = |\mathbf{b}^T\mathbf{x}| = |\cos\omega|\|\mathbf{x}\|$, where $\omega$ is the angle between $\mathbf{x}$ and $\mathbf{b}$.
    *   **Projection Matrix:** By rearranging the formula, $\pi_U(\mathbf{x}) = \left( \frac{\mathbf{b}\mathbf{b}^T}{\|\mathbf{b}\|^2} \right) \mathbf{x}$, we identify the projection matrix:
        $$ P_\pi = \frac{\mathbf{b}\mathbf{b}^T}{\|\mathbf{b}\|^2} $$
        This matrix is symmetric and has a rank of 1.

#### 4. Projection onto General Subspaces

The three-step procedure for 1D projection generalizes to any m-dimensional subspace $U \subseteq \mathbb{R}^n$.

*   **Setup:** Assume we have an ordered basis $\{\mathbf{b}_1, \dots, \mathbf{b}_m\}$ for $U$. We form the basis matrix $B = [\mathbf{b}_1, \dots, \mathbf{b}_m] \in \mathbb{R}^{n \times m}$.
*   **Derivation:**
    1.  **Membership Property:** The projection $\pi_U(\mathbf{x})$ is a linear combination of the basis vectors:
        $$ \pi_U(\mathbf{x}) = \sum_{i=1}^{m} \lambda_i \mathbf{b}_i = B\boldsymbol{\lambda} $$
    2.  **Orthogonality Property:** The error vector $(\mathbf{x} - \pi_U(\mathbf{x}))$ must be orthogonal to *all* basis vectors of $U$. This gives $m$ simultaneous equations:
        $$ \mathbf{b}_i^T (\mathbf{x} - B\boldsymbol{\lambda}) = 0 \quad \text{for } i=1, \dots, m $$
    3.  **Solve the Normal Equation:** The system of equations can be written compactly as $B^T(\mathbf{x} - B\boldsymbol{\lambda}) = \mathbf{0}$, which rearranges to the **Normal Equation**:
        $$ B^T B \boldsymbol{\lambda} = B^T \mathbf{x} $$
*   **Solving for the Projection:**
    *   Since the columns of $B$ form a basis, they are linearly independent, which guarantees that the $m \times m$ matrix $B^T B$ is **invertible**.
    *   **Coordinates $\boldsymbol{\lambda}$:** We can solve for the coordinate vector:
        $$ \boldsymbol{\lambda} = (B^T B)^{-1} B^T \mathbf{x} $$
        The matrix $(B^T B)^{-1} B^T$ is the **pseudoinverse** of $B$.
    *   **Projection Vector $\pi_U(\mathbf{x})$:**
        $$ \pi_U(\mathbf{x}) = B\boldsymbol{\lambda} = B(B^T B)^{-1} B^T \mathbf{x} $$
    *   **Projection Matrix $P_\pi$:**
        $$ P_\pi = B(B^T B)^{-1} B^T $$

#### 5. Remarks on Dimensions and Coordinates

*   **The Projection Vector:** A projected vector $\pi_U(\mathbf{x})$ is still an n-dimensional vector (it has $n$ components), but it is constrained to lie within the m-dimensional subspace $U$.
*   **The Power of Coordinates:** Crucially, this projected vector is fully determined by its **m coordinates** ($\lambda_1, \dots, \lambda_m$) with respect to the basis of $U$. This is the mathematical foundation of dimensionality reduction: we only need to store or use the $m$ coordinates and the $m$ basis vectors to perfectly represent the projected data, which is far more efficient than storing the original n-dimensional vectors.