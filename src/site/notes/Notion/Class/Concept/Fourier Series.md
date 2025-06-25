---
{"dg-publish":true,"permalink":"/notion/class/concept/fourier-series/"}
---

# 傅里叶级数：将函数分解为正交基

傅里叶级数是一种极其强大的数学工具，它允许我们将一个复杂的周期函数表示为一系列简单的正弦和余弦函数的无限和。其深刻的数学基础正是**函数空间中的内积和正交性**，这与我们在线性代数中处理向量的方式如出一辙。

## 一、核心思想：函数作为向量

在线性代数中，我们可以将一个向量分解到一组正交基上。例如，在 $\mathbb{R}^3$ 中，任何向量 $\mathbf{v}$ 都可以被分解到标准正交基 $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ 上。要找到向量在某个基方向上的分量，我们只需计算它与该基向量的内积（点积）。

傅里ಯೇ级数将这个思想从有限维的几何空间推广到了无限维的函数空间：

*   **向量空间 (Vector Space)**：在特定区间（如 $[-\pi, \pi]$）上平方可积的函数构成的空间。
*   **向量 (Vector)**：该空间中的每一个函数（如 $h(x)=x$）都可以被看作是一个“向量”。
*   **内积 (Inner Product)**：我们定义函数间的内积为积分：
    $$ \langle f, g \rangle = \int_{-\pi}^{\pi} f(x)g(x) \,dx $$
    这个定义完美地类比了向量点积（对应分量相乘再求和），只不过在这里，“求和”变成了连续的“积分”。
*   **正交基 (Orthogonal Basis)**：在这个函数空间中，存在一组非常特殊的“正交基向量”，它们就是三角函数族：
    $$ \{1, \cos(x), \sin(x), \cos(2x), \sin(2x), \dots, \cos(nx), \sin(nx), \dots\} $$

## 二、三角函数的正交性：为何它们是“垂直”的？

这组基的核心性质是它们在区间 $[-\pi, \pi]$ 上是**两两正交**的。这意味着任意两个不同的基函数，它们的内积（积分）都为零。

*   $\langle \sin(mx), \cos(nx) \rangle = \int_{-\pi}^{\pi} \sin(mx)\cos(nx) \,dx = 0$  (对所有 $m, n$)
*   $\langle \sin(mx), \sin(nx) \rangle = \int_{-\pi}^{\pi} \sin(mx)\sin(nx) \,dx = 0$  (当 $m \neq n$)
*   $\langle \cos(mx), \cos(nx) \rangle = \int_{-\pi}^{\pi} \cos(mx)\cos(nx) \,dx = 0$  (当 $m \neq n$)

这种正交性源于三角函数内在的**对称性与周期性**。当我们在一个对称的、完整的周期区间（如 $[-\pi, \pi]$）上积分时：
1.  **对称性抵消**：`sin`是奇函数，`cos`是偶函数，它们的乘积是奇函数，在对称区间上的积分为零。
2.  **周期性抵消**：不同频率的同名函数相乘后，会产生新的、更高频率的波形。在一个完整的周期内，这些波动的正负部分会精确地相互抵消，导致积分为零。

**重要提示：正交性依赖于区间**
两个函数是否正交，完全取决于内积定义中的积分区间。例如，$\sin(x)$ 和 $\cos(x)$ 在 $[-\pi, \pi]$ 上是正交的，但在 $[0, \pi]$ 上则不是。

## 三、傅里叶级数分解：将函数投影到基上

正如任何向量可以写成其在基向量上投影的和，任何（行为良好的）函数 $h(x)$ 也可以写成它在这些正交三角函数基上的投影之和。这就是**傅里叶级数**：

$$ h(x) = a_0 + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right] $$

这里的系数 $a_0, a_n, b_n$ 就是函数“向量” $h(x)$ 在每个基“向量”上的**坐标**或**投影分量**。它们的计算公式源于投影的定义（即内积除以基向量自身长度的平方）：

*   **常数项 (均值):**
    $$ a_0 = \frac{\langle h(x), 1 \rangle}{\langle 1, 1 \rangle} = \frac{\int_{-\pi}^{\pi} h(x) \,dx}{\int_{-\pi}^{\pi} 1^2 \,dx} = \frac{1}{2\pi} \int_{-\pi}^{\pi} h(x) \,dx $$
*   **余弦系数:**
    $$ a_n = \frac{\langle h(x), \cos(nx) \rangle}{\langle \cos(nx), \cos(nx) \rangle} = \frac{\int_{-\pi}^{\pi} h(x)\cos(nx) \,dx}{\int_{-\pi}^{\pi} \cos^2(nx) \,dx} = \frac{1}{\pi} \int_{-\pi}^{\pi} h(x)\cos(nx) \,dx $$
*   **正弦系数:**
    $$ b_n = \frac{\langle h(x), \sin(nx) \rangle}{\langle \sin(nx), \sin(nx) \rangle} = \frac{\int_{-\pi}^{\pi} h(x)\sin(nx) \,dx}{\int_{-\pi}^{\pi} \sin^2(nx) \,dx} = \frac{1}{\pi} \int_{-\pi}^{\pi} h(x)\sin(nx) \,dx $$

## 四、计算示例：$h(x) = x$ 在 $[-\pi, \pi]$ 上的傅里叶级数

让我们将函数 $h(x)=x$ 分解到这个正交基上。

#### 1. 计算系数 $a_0$ 和 $a_n$

我们利用**奇偶函数**的性质来简化计算：
*   $h(x)=x$ 是一个**奇函数**。
*   $\cos(nx)$ 是一个**偶函数**。
*   奇函数 × 偶函数 = 奇函数。
*   一个奇函数在对称区间 $[ -a, a ]$ 上的积分为零。

因此：
*   $a_0 = \frac{1}{2\pi} \int_{-\pi}^{\pi} \underbrace{x}_{\text{奇}} \,dx = 0$
*   $a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} \underbrace{x \cos(nx)}_{\text{奇}} \,dx = 0$

这意味着 $h(x)=x$ 这个函数在所有余弦基函数方向上都没有分量，它的傅里叶级数只包含正弦项。

#### 2. 计算系数 $b_n$

*   $h(x)=x$ 是**奇函数**。
*   $\sin(nx)$ 是**奇函数**。
*   奇函数 × 奇函数 = 偶函数。
*   一个偶函数在对称区间 $[ -a, a ]$ 上的积分等于它在 $[ 0, a ]$ 上积分的两倍。

$$ b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} \underbrace{x \sin(nx)}_{\text{偶}} \,dx = \frac{2}{\pi} \int_{0}^{\pi} x \sin(nx) \,dx $$

为了计算这个积分，我们使用**分部积分法 (Integration by Parts)**: $\int u \,dv = uv - \int v \,du$。
*   令 $u = x \implies du = dx$
*   令 $dv = \sin(nx)dx \implies v = -\frac{1}{n}\cos(nx)$

$$
\begin{aligned}
\int x \sin(nx) \,dx &= x \left(-\frac{1}{n}\cos(nx)\right) - \int \left(-\frac{1}{n}\cos(nx)\right) \,dx \\
&= -\frac{x}{n}\cos(nx) + \frac{1}{n^2}\sin(nx)
\end{aligned}
$$

现在，我们计算定积分：
$$
\begin{aligned}
\int_{0}^{\pi} x \sin(nx) \,dx &= \left[ -\frac{x}{n}\cos(nx) + \frac{1}{n^2}\sin(nx) \right]_{0}^{\pi} \\
&= \left( -\frac{\pi}{n}\cos(n\pi) + \frac{1}{n^2}\sin(n\pi) \right) - \left( 0 + 0 \right) \\
&= -\frac{\pi}{n}\cos(n\pi)
\end{aligned}
$$
因为 $\cos(n\pi) = (-1)^n$，所以上式等于 $-\frac{\pi}{n}(-1)^n = \frac{\pi}{n}(-1)^{n+1}$。

最后，代回到 $b_n$ 的表达式中：
$$ b_n = \frac{2}{\pi} \left( \frac{\pi}{n}(-1)^{n+1} \right) = \frac{2}{n}(-1)^{n+1} $$

#### 3. 最终的傅里叶级数

我们将计算出的系数代入傅里叶级数的总公式：
$$ x = \sum_{n=1}^{\infty} \frac{2}{n}(-1)^{n+1} \sin(nx) $$
展开前几项：
$$ x = 2\sin(x) - \sin(2x) + \frac{2}{3}\sin(3x) - \frac{1}{2}\sin(4x) + \dots $$
这个结果令人惊叹：一条简单的直线 $y=x$ 可以被精确地表示为无穷多个不同频率的正弦波的叠加。这充分展示了将函数视为向量并将其分解到正交基上的强大威力。