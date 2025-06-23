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

This section explains how to represent abstract linear mappings and changes of coordinate systems using concrete matrices.

### 1. The Transformation Matrix for a Linear Map

This matrix allows us to compute the result of a linear mapping by performing a simple matrix multiplication on coordinate vectors.

*   **Setup:**
    *   A linear mapping $\Phi: V \to W$.
    *   An ordered basis $B = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ for the input space $V$.
    *   An ordered basis $C = (\mathbf{c}_1, \ldots, \mathbf{c}_m)$ for the output space $W$.

*   **How to Construct the Transformation Matrix $A_\Phi$:**
    The matrix $A_\Phi$ is built column by column. The $j$-th column of $A_\Phi$ is the coordinate vector of the image of the $j$-th input basis vector, expressed in the output basis.

    1.  Take the $j$-th vector from the input basis, $\mathbf{b}_j$.
    2.  Apply the linear map to it to get its image in $W$: $\Phi(\mathbf{b}_j)$.
    3.  Express this image vector as a linear combination of the output basis vectors from $C$:
        $\Phi(\mathbf{b}_j) = \alpha_{1j}\mathbf{c}_1 + \alpha_{2j}\mathbf{c}_2 + \cdots + \alpha_{mj}\mathbf{c}_m$.
    4.  The coefficients form the $j$-th column of $A_\Phi$.
        $$ \text{Column } j \text{ of } A_\Phi = [\Phi(\mathbf{b}_j)]_C = \begin{pmatrix} \alpha_{1j} \\ \alpha_{2j} \\ \vdots \\ \alpha_{mj} \end{pmatrix} $$

*   **How to Use the Transformation Matrix:**
    To find the coordinates of the image vector $\Phi(\mathbf{v})$ in basis $C$, you multiply the transformation matrix $A_\Phi$ by the coordinate vector of $\mathbf{v}$ in basis $B$.
    $$ [\Phi(\mathbf{v})]_C = A_\Phi [\mathbf{v}]_B $$
    This formula translates the abstract operation $\Phi(\mathbf{v})$ into a concrete matrix-vector multiplication.

*   **Invertibility:** The mapping $\Phi$ can only be inverted if the transformation matrix $A_\Phi$ is **square and invertible**.

### 2. The Change of Basis Matrix

This is a special case of a transformation matrix where the linear mapping is the **identity map** ($\Phi(\mathbf{x}) = \mathbf{x}$), but we are changing the coordinate system from a "new" basis back to an "old" (often standard) basis.

*   **Setup:**
    *   An "old" basis $B$ for a vector space $V$.
    *   A "new" basis $B' = (\mathbf{b}'_1, \ldots, \mathbf{b}'_n)$ for the same space $V$.

*   **Change of Basis Matrix ($P_{B \leftarrow B'}$):**
    This matrix transforms coordinates from the new basis $B'$ to the old basis $B$. Its columns are simply the vectors of the new basis $B'$ expressed in the coordinates of the old basis $B$.
    $$ P_{B \leftarrow B'} = [[\mathbf{b}'_1]_B, [\mathbf{b}'_2]_B, \ldots, [\mathbf{b}'_n]_B] $$
    *If the "old" basis $B$ is the standard basis in $\mathbb{R}^n$, this is very simple: the columns of $P$ are just the vectors of the new basis $B'$ themselves.*

*   **How to Use the Change of Basis Matrix:**
    *   **From New to Old Coordinates:**
        $$ [\mathbf{x}]_B = P_{B \leftarrow B'} [\mathbf{x}]_{B'} $$
    *   **From Old to New Coordinates (more common):**
        $$ [\mathbf{x}]_{B'} = (P_{B \leftarrow B'})^{-1} [\mathbf{x}]_B $$

*   **Example:**
    Let the old basis be the standard basis $B$ in $\mathbb{R}^2$, and a new basis be $B' = \left( \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \begin{pmatrix} -1 \\ 2 \end{pmatrix} \right)$.
    The change of basis matrix from $B'$ to $B$ is:
    $$ P = \begin{pmatrix} 1 & -1 \\ 1 & 2 \end{pmatrix} $$
    To find the coordinates of the vector $\mathbf{x} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$ in the new basis $B'$:
    $$ [\mathbf{x}]_{B'} = P^{-1} \mathbf{x} = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ -1 & 1 \end{pmatrix} \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 10/3 \\ 1/3 \end{pmatrix} $$