---
{"dg-publish":true,"permalink":"/wiki/notion/class/artificial-intelligence/mathematics-for-ai/concept/ref-rref/"}
---

## **Row Echelon Form (REF)**

*   **Definition:**
    A matrix is in **Row Echelon Form (REF)** if it satisfies the following three conditions:

    1.  **All non-zero rows are above any zero rows.**
        (If there are any rows consisting entirely of zeros, they must be at the bottom of the matrix.)
    2.  **The leading entry (the first non-zero element from the left) of each non-zero row is in a column to the right of the leading entry of the row above it.**
        (This creates a "stair-step" pattern.)
    3.  **All entries in a column below a leading entry are zero.**
        (This is a direct consequence of condition 2, ensuring the "steps" are clear.)

*   **Characteristics:**
    *   **Not Unique:** A given matrix can have multiple different Row Echelon Forms, depending on the sequence of elementary row operations performed.
    *   **Determining Rank:** The number of non-zero rows (which is equivalent to the number of leading entries/pivots) in an REF matrix gives the **rank** of the original matrix.
    *   **For Augmented Matrix `[A | b]`:**
        *   Used as an intermediate step in Gaussian elimination to simplify the system.
        *   Helps to quickly identify if a system is **inconsistent** (no solution). This occurs if an REF of `[A | b]` contains a row of the form `[0 0 ... 0 | c]`, where `c` is a non-zero constant. Such a row implies `0 = c`, which is a contradiction.
        *   Allows for **back-substitution** to find solutions, especially in the context of `Ax=b`.

*   **Example:**
    The following is a matrix in Row Echelon Form:
    $$
    \begin{bmatrix}
    \mathbf{1} & 2 & 3 & 4 \\
    0 & \mathbf{5} & 6 & 7 \\
    0 & 0 & \mathbf{8} & 9 \\
    0 & 0 & 0 & 0
    \end{bmatrix}
    $$
    *The leading entries (pivots) are marked in bold.*

---

## **Reduced Row Echelon Form (RREF)**

*   **Definition:**
    A matrix is in **Reduced Row Echelon Form (RREF)** if it satisfies all the conditions for Row Echelon Form (REF), and additionally meets these two stricter conditions:

    1.  **Each leading entry (pivot) must be `1`.**
        (All pivots are "normalized" to one.)
    2.  **Each column containing a leading entry (pivot) must have zeros everywhere else.**
        (Not only below the pivot but also *above* it, all other entries in that column must be zero.)

*   **Characteristics:**
    *   **Uniqueness:** The Reduced Row Echelon Form of any given matrix is **unique**. Regardless of the sequence of valid elementary row operations performed, the final RREF will always be the same.
    *   **Direct Solutions:** For a linear system `Ax = b`, transforming its augmented matrix `[A | b]` into RREF allows for direct reading of the **general solution**. The pivot variables can be immediately expressed in terms of the free variables and constants.
    *   **Identification of Variables:** RREF makes it straightforward to identify **pivot variables** (corresponding to pivot columns) and **free variables** (corresponding to non-pivot columns).
    *   **Null Space Basis:** RREF is the critical step for finding the basis vectors for the **null space** of a matrix, as it directly gives the relationships between pivot and free variables for the homogeneous system `Ax = 0`.

*   **Example (from your previous notes):**
    The matrix $A = \begin{bmatrix} 1 & 0 & 8 & -4 \\ 0 & 1 & 2 & 12 \end{bmatrix}$ is already in Reduced Row Echelon Form:
    $$
    \begin{bmatrix}
    \mathbf{1} & 0 & 8 & -4 \\
    0 & \mathbf{1} & 2 & 12
    \end{bmatrix}
    $$
    *The leading entries (pivots) are marked in bold.*
    *   All pivots are `1`.
    *   In the columns containing pivots (column 1 and column 2), all other entries are `0`.

    Here is another, more comprehensive example of an RREF matrix (which could be an augmented matrix `[A|b]`):
    $$
    \begin{bmatrix}
    \mathbf{1} & 0 & 0 & 5 & \bigm| & 10 \\
    0 & \mathbf{1} & 0 & -2 & \bigm| & 7 \\
    0 & 0 & \mathbf{1} & 3 & \bigm| & 0
    \end{bmatrix}
    $$
    From this RREF, you can directly read the solutions (e.g., $x_1 + 5x_4 = 10$, $x_2 - 2x_4 = 7$, $x_3 + 3x_4 = 0$).