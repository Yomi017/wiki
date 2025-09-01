---
{"dg-publish":true,"permalink":"/wiki/notion/class/proof/change-of-basis-theorem/"}
---

### 证明：基变换定理

#### 1. 设定与定义 (Setup and Definitions)

为了使证明清晰，我们首先要非常精确地定义所有涉及的对象。

*   **线性映射 (Linear Mapping):**
    $\Phi : V \to W$ 是一个从向量空间 $V$ 到向量空间 $W$ 的线性映射。

*   **向量空间与基 (Vector Spaces and Bases):**
    *   $V$ 是一个 $n$ 维向量空间 (定义域)。
        *   旧基 (Old basis): $B = \{\mathbf{b}_1, \mathbf{b}_2, \dots, \mathbf{b}_n\}$
        *   新基 (New basis): $B̃ = \{\mathbf{\tilde{b}}_1, \mathbf{\tilde{b}}_2, \dots, \mathbf{\tilde{b}}_n\}$
    *   $W$ 是一个 $m$ 维向量空间 (到达域)。
        *   旧基 (Old basis): $C = \{\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_m\}$
        *   新基 (New basis): $C̃ = \{\mathbf{\tilde{c}}_1, \mathbf{\tilde{c}}_2, \dots, \mathbf{\tilde{c}}_m\}$

*   **坐标向量 (Coordinate Vectors):**
    *   对于任意向量 $\mathbf{v} \in V$，其相对于基 $B$ 的坐标向量记为 $[\mathbf{v}]_B$。如果 $[\mathbf{v}]_B = (x_1, \dots, x_n)^T$，这意味着 $\mathbf{v} = x_1\mathbf{b}_1 + \dots + x_n\mathbf{b}_n$。
    *   同理，我们有 $[\mathbf{v}]_{B̃}$，$[\mathbf{w}]_{C}$ 和 $[\mathbf{w}]_{C̃}$。

*   **变换矩阵 (Transformation Matrices):**
    *   $A_\Phi$ 是 $\Phi$ 相对于**旧基** $B$ 和 $C$ 的变换矩阵。根据定义，对于任意 $\mathbf{v} \in V$，它满足：
      $$ [\Phi(\mathbf{v})]_C = A_\Phi [\mathbf{v}]_B \quad (*)$$
    *   $Ã_\Phi$ 是 $\Phi$ 相对于**新基** $B̃$ 和 $C̃$ 的变换矩阵。这是我们**要求解的矩阵**。根据定义，它必须满足：
      $$ [\Phi(\mathbf{v})]_{C̃} = Ã_\Phi [\mathbf{v}]_{B̃} \quad (**) $$

*   **基变换矩阵 (Change-of-Basis Matrices):**
    *   $S$ 是从**新基 $B̃$** 到**旧基 $B$** 的基变换矩阵。它的作用是将一个向量在新基下的坐标转换为在旧基下的坐标。对于任意 $\mathbf{v} \in V$：
      $$ [\mathbf{v}]_B = S [\mathbf{v}]_{B̃} \quad (1) $$
      （$S$ 的第 $j$ 列是向量 $\mathbf{\tilde{b}}_j$ 在基 $B$ 下的坐标，即 $[\mathbf{\tilde{b}}_j]_B$）。
    *   $T$ 是从**新基 $C̃$** 到**旧基 $C$** 的基变换矩阵。它的作用是将一个向量在新基下的坐标转换为在旧基下的坐标。对于任意 $\mathbf{w} \in W$：
      $$ [\mathbf{w}]_C = T [\mathbf{w}]_{C̃} \quad (2) $$
      （$T$ 的第 $j$ 列是向量 $\mathbf{\tilde{c}}_j$ 在基 $C$ 下的坐标，即 $[\mathbf{\tilde{c}}_j]_C$）。
      由于基变换矩阵总是可逆的，我们可以得到从**旧基 $C$** 到**新基 $C̃$** 的变换关系：
      $$ [\mathbf{w}]_{C̃} = T^{-1} [\mathbf{w}]_C \quad (3) $$

---

#### 2. 证明过程 (The Proof)

我们的目标是找到 $Ã_\Phi$。根据定义 $(* *)$，$Ã_\Phi$ 是连接 $[\mathbf{v}]_{B̃}$ 和 $[\Phi(\mathbf{v})]_{C̃}$ 的桥梁。我们将利用已知的关系，通过一条“绕行”的路径来建立这个连接。这个路径就是 $B̃ \to B \to C \to C̃$。

让我们从我们想要求解的表达式的左侧开始，即 $[\Phi(\mathbf{v})]_{C̃}$，然后一步步地将它与 $[\mathbf{v}]_{B̃}$ 联系起来。

**第一步：从新输出坐标到旧输出坐标 ( $C̃ \to C$ )**
我们有一个向量 $\Phi(\mathbf{v}) \in W$。利用关系式 (3)，我们可以将其在新基 $C̃$ 下的坐标用在旧基 $C$ 下的坐标来表示：
$$ [\Phi(\mathbf{v})]_{C̃} = T^{-1} [\Phi(\mathbf{v})]_C $$

**第二步：应用原始变换矩阵 ( $B \to C$ )**
现在我们有了 $[\Phi(\mathbf{v})]_C$。根据原始变换矩阵 $A_\Phi$ 的定义 (式 $*$ )，我们可以把它和输入向量在旧基 $B$ 下的坐标联系起来：
$$ [\Phi(\mathbf{v})]_C = A_\Phi [\mathbf{v}]_B $$
将这个代入第一步的结果中，我们得到：
$$ [\Phi(\mathbf{v})]_{C̃} = T^{-1} (A_\Phi [\mathbf{v}]_B) $$

**第三步：从旧输入坐标到新输入坐标 ( $B̃ \to B$ )**
最后，我们看到表达式中出现了 $[\mathbf{v}]_B$。利用关系式 (1)，我们可以将其与我们最终想要的输入坐标 $[\mathbf{v}]_{B̃}$ 联系起来：
$$ [\mathbf{v}]_B = S [\mathbf{v}]_{B̃} $$
将这个代入第二步的结果中，我们得到：
$$ [\Phi(\mathbf{v})]_{C̃} = T^{-1} (A_\Phi (S [\mathbf{v}]_{B̃})) $$

**第四步：得出结论**
利用矩阵乘法的结合律，我们可以去掉括号：
$$ [\Phi(\mathbf{v})]_{C̃} = (T^{-1} A_\Phi S) [\mathbf{v}]_{B̃} $$

现在，我们将这个结果与我们对 $Ã_\Phi$ 的定义 (式 $**$) 进行比较：
$$ [\Phi(\mathbf{v})]_{C̃} = Ã_\Phi [\mathbf{v}]_{B̃} \quad (\text{by definition of } Ã_\Phi) $$

我们已经证明，对于**任意**向量 $\mathbf{v} \in V$，它都满足 $[\Phi(\mathbf{v})]_{C̃} = (T^{-1} A_\Phi S) [\mathbf{v}]_{B̃}$。由于线性变换的矩阵表示是唯一的，我们必然得出结论：
$$ Ã_\Phi = T^{-1} A_\Phi S $$

**证明完毕。**

---

#### 3. 交换图 (Commutative Diagram)

这个证明过程可以用一个非常直观的**交换图**来表示，它完美地总结了整个逻辑。图中，箭头表示一个操作（矩阵乘法）。

```
                      Φ
      V ----------------------------> W
      |                               |
      | Coord. w.r.t. B̃              | Coord. w.r.t. C̃
      |                               |
      v                               v
   ℝⁿ (B̃-coords) -- Ã_Φ --> ℝᵐ (C̃-coords)
      |    ^                          ^    |
      | S  |                          | T⁻¹|
      v    |                          |    |
   ℝⁿ (B-coords) --- A_Φ --> ℝᵐ (C-coords)

```

*   **上方的直接路径**：从 $B̃$ 坐标直接到 $C̃$ 坐标，这是由我们想求的矩阵 $Ã_\Phi$ 完成的。
*   **下方的“绕行”路径**：
    1.  从 $B̃$ 坐标到 $B$ 坐标 (乘以 $S$)。
    2.  从 $B$ 坐标到 $C$ 坐标 (乘以 $A_\Phi$)。
    3.  从 $C$ 坐标到 $C̃$ 坐标 (乘以 $T^{-1}$)。

图的“交换性”意味着从任何一个起点到任何一个终点，所有路径都必须产生相同的结果。因此，直接路径必须等于间接路径，即：
$$ Ã_\Phi = T^{-1} A_\Phi S $$
这为我们的形式化证明提供了一个强大的视觉辅助。