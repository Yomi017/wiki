---
{"dg-publish":true,"permalink":"/wiki/notion/class/proof/compatibility-of-matrix-multiplication-with-partitioned-matrices/"}
---


"When you perform elementary row operations on the augmented matrix $[A | I_n]$, you are essentially applying the same operations simultaneously to both $A$ and $I_n$."

"From an algebraic perspective, this means you are left-multiplying the entire augmented matrix by the same sequence of elementary matrices:"
$$ E_k \cdots E_2 E_1 [A | I_n] $$

"According to the definition of matrix multiplication (specifically, how it applies to block matrices), this is equivalent to:"
$$ [ (E_k \cdots E_1)A \quad | \quad (E_k \cdots E_1)I_n ] $$

**How to prove this step:**

This step is crucial for understanding how the augmented matrix method for finding the inverse works. It is not a direct application of the distributive property (as matrix multiplication over matrices does not distribute in this sense for blocks), but rather based on the **definition of block matrix multiplication**.

Let's first prove the case for a **single elementary matrix** $E$.

**Proof of $E [A | I_n] = [EA | EI_n]$:**

Let $A$ be an $n \times n$ matrix, and $I_n$ be the $n \times n$ identity matrix.
The augmented matrix $[A | I_n]$ is an $n \times (n+n)$, i.e., $n \times (2n)$ matrix.
Let $E$ be an $n \times n$ elementary matrix (representing a single row operation).

We can consider the augmented matrix $[A | I_n]$ as a single $n \times (2n)$ matrix, whose columns are split into two blocks:
*   The first $n$ columns form matrix $A$.
*   The second $n$ columns form matrix $I_n$.

Let $C = [A | I_n]$. We can write $C$ in terms of its column vectors:
$C = [\mathbf{c}_1 \quad \mathbf{c}_2 \quad \cdots \quad \mathbf{c}_n \quad | \quad \mathbf{c}_{n+1} \quad \cdots \quad \mathbf{c}_{2n}]$
Where $\mathbf{c}_1, \ldots, \mathbf{c}_n$ are the column vectors of $A$, and $\mathbf{c}_{n+1}, \ldots, \mathbf{c}_{2n}$ are the column vectors of $I_n$.

Recall the fundamental definition of matrix multiplication: when a matrix $E$ left-multiplies another matrix $C$, each column of the resulting matrix $EC$ is $E$ multiplied by the corresponding column of $C$.
So,
$$ EC = [E\mathbf{c}_1 \quad E\mathbf{c}_2 \quad \cdots \quad E\mathbf{c}_n \quad | \quad E\mathbf{c}_{n+1} \quad \cdots \quad E\mathbf{c}_{2n}] $$

Now, let's examine the blocks of the resulting matrix $EC$:

1.  **The first $n$ columns of $EC$**: These are $E\mathbf{c}_1, E\mathbf{c}_2, \ldots, E\mathbf{c}_n$. By the definition of matrix multiplication, these are precisely the column vectors that form the matrix product $EA$.
    Therefore, the left block of the result is $EA$.

2.  **The last $n$ columns of $EC$**: These are $E\mathbf{c}_{n+1}, E\mathbf{c}_{n+2}, \ldots, E\mathbf{c}_{2n}$. Similarly, these are precisely the column vectors that form the matrix product $EI_n$.
    Therefore, the right block of the result is $EI_n$.

Combining these two observations, we have demonstrated that:
$$ E [A | I_n] = [EA | EI_n] $$

**Generalization to a Sequence of Elementary Matrices:**

This principle extends straightforwardly to a sequence of elementary matrices $E_1, E_2, \ldots, E_k$. We can apply the proven principle iteratively:

1.  Start with $E_1$: $E_1 [A | I_n] = [E_1 A | E_1 I_n]$
2.  Apply $E_2$ to the result: $E_2 (E_1 [A | I_n]) = E_2 [E_1 A | E_1 I_n] = [E_2 (E_1 A) | E_2 (E_1 I_n)]$
3.  Continue this process for all $k$ elementary matrices:
    $$ (E_k \cdots E_2 E_1) [A | I_n] = [ (E_k \cdots E_2 E_1)A \quad | \quad (E_k \cdots E_2 E_1)I_n ] $$

This is why, when you perform a sequence of elementary row operations on the augmented matrix $[A | I_n]$, the left block $A$ is transformed into the product $(E_k \cdots E_1)A$, and the right block $I_n$ is simultaneously transformed into $(E_k \cdots E_1)I_n$. If the process successfully transforms $A$ into $I_n$ (i.e., $(E_k \cdots E_1)A = I_n$), then the product $(E_k \cdots E_1)$ must be $A^{-1}$, and the right side becomes $A^{-1}I_n = A^{-1}$.