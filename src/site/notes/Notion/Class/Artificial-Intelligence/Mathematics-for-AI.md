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
*   **Matrix Notation:**![Pasted image 20250616183101.png](/img/user/Pasted%20image%2020250616183101.png)
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
        5.  **Proof:**
	        $$C[A|I_n]=[CA|CI_n]=[I_n|C]\Rightarrow C=A^{-1}$$

    *   **Limitation:** For non-square matrices, the augmented matrix method (to find a traditional inverse) is not defined because non-square matrices do not have inverses.

---