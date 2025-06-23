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
        5.  **Proof:** [[Notion/Class/Proof/Compatibility-of-Matrix-Multiplication-with-Partitioned-Matrices\|Compatibility-of-Matrix-Multiplication-with-Partitioned-Matrices]]
	        $$C[A|I_n]=[CA|CI_n]=[I_n|C]\Rightarrow C=A^{-1}$$

    *   **Limitation:** For non-square matrices, the augmented matrix method (to find a traditional inverse) is not defined because non-square matrices do not have inverses.
* **Algorithms for Solving Linear Systems (`Ax = b`): Direct Methods**
	
	This section outlines various direct algorithms used to find solutions for the linear system `Ax = b`.
	
	*   **1. Direct Inversion:**
	    *   **Applicability:** This method is used if the coefficient matrix $A$ is **square and invertible** (i.e., non-singular).
	    *   **Formula:** The solution $x$ is directly computed as:
	        $$x = A^{-1}b$$
	    *   **Mechanism:** If $A$ is invertible, its inverse $A^{-1}$ exists, and multiplying both sides of $Ax = b$ by $A^{-1}$ from the left yields $A^{-1}Ax = A^{-1}b \implies I_nx = A^{-1}b \implies x = A^{-1}b$.
	
	*   **2. Pseudo-inverse ([[Notion/Class/Proof/Moore-Penrose-Pseudo-inverse\|Moore-Penrose-Pseudo-inverse]]):**
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
        *   *(Justification via a more fundamental theorem: The rank is the dimension of the row space. The row space is spanned by $k$ row vectors. By [[Notion/Class/Proof/The-Dimension-Theorem-for-Spanning-Sets\|The-Dimension-Theorem-for-Spanning-Sets]], the dimension of this space cannot exceed $k$.)*

    4.  **Apply the Condition $m > k$:** We have established two key facts about the matrix $\Lambda$:
        *   The total number of columns is $m$.
        *   The rank, which represents the maximum number of linearly independent columns, is at most $k$. That is, $\text{rank}(\Lambda) \le k$.

    5.  **Connect Rank to Linear Dependence:** We are given that $m > k$. This leads to the crucial inequality:
        $$ \text{Total number of columns } (m) > \text{Maximum number of linearly independent columns } (\text{rank}(\Lambda)) $$
        This inequality means it is impossible for all $m$ columns of $\Lambda$ to be linearly independent. If you have more vectors ($m$) than the dimension of the space they can span (the rank, which is at most $k$), the set of vectors must be linearly dependent.

    6.  **Draw the Conclusion:** Because the columns of $\Lambda$ (which are the coefficient vectors $\{\boldsymbol{\lambda}_j\}$) form a linearly dependent set, the set of new vectors $\{\mathbf{x}_j\}$ that they define must also be **linearly dependent**. Q.E.D.

# Lecture 2: Linear Algebra: Basis and Rank, Linear Mappings, Affine Space

