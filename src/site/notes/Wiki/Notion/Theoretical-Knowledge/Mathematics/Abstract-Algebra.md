---
{"dg-publish":true,"permalink":"/wiki/notion/theoretical-knowledge/mathematics/abstract-algebra/"}
---

# 1. Group

## 1. Basic Concept
### (0) Monoid

- ① $\forall x,y,z \in S,\; x * (y * z) = (x * y) * z$
- ② $\exists e \in S,\; \forall x \in S,\; e * x = x * e = x$

### (1) Group

**(G, ·) group $\iff$**
* ① 结合律: $\forall a,b,c \in G, a·(b·c)=(a·b)·c$  
* ② 单位元: $\exists e \in G, \forall a \in G, a·e=e·a=a$
* ③ 逆元: $\forall a \in G, \exists a^{-1} \in G, a·a^{-1}=a^{-1}·a=e$

#### Examples

* ① General Linear Group: $GL(n, \mathbb{R})=\{A \in M_n(\mathbb{R}) \mid \det(A) \neq 0\}$
* ② Special Linear Group: $SL(n, \mathbb{R})=\{A \in M_n(\mathbb{R}) \mid \det(A)=1\}$

其中 $M_n(\mathbb{R})$ 表示实数域上的 $n \times n$ 矩阵集合。二者均以矩阵乘法为运算，且 $SL(n, \mathbb{R})<GL(n, \mathbb{R})$。

### (2) Abel Group

**(G, ·) group $\iff$**
* ① 结合律: $\forall a,b,c \in G, a·(b·c)=(a·b)·c$  
* ② 单位元: $\exists e \in G, \forall a \in G, a·e=e·a=a$
* ③ 逆元: $\forall a \in G, \exists a^{-1} \in G, a·a^{-1}=a^{-1}·a=e$
* ④ 交换律: $\forall a,b \in G, a·b=b·a$

## 2. Subgroup

### (1) Subgroup

**$(H, ·)$ is a subgroup of $(G, ·)$, denoted $H<G$, $\iff$**
* ① 子集: $H \subseteq G$
* ② 单位元: $e \in H$
* ③ 封闭性: $\forall a,b \in H, a·b \in H$
* ④ 逆元: $\forall a \in H, a^{-1} \in H$

### (2) Submonoid

**$(N, ·)$ is a submonoid of $(M, ·)$ $\iff$**
* ① 子集: $N \subseteq M$
* ② 单位元: $M$ 的单位元 $e_M \in N$
* ③ 封闭性: $\forall a,b \in N, a·b \in N$

## 3. Homomorphism

### (1) Group Homomorphism

**For groups $(G, \cdot)$ and $(H, \ast)$, $f:G \to H$ is a group homomorphism $\iff$**
* ① 保持运算: $\forall a,b \in G, f(a \cdot b)=f(a) \ast f(b)$

#### Properties

设 $e_G$ 与 $e_H$ 分别为 $G$ 与 $H$ 的单位元。

* ① 保持单位元: $f(e_G)=e_H$
* ② 保持逆元: $\forall a \in G, f(a^{-1})=f(a)^{-1}$

#### Proof

①
$$
\begin{aligned}
e_H
&=f(e_G)^{-1} \ast f(e_G)\\
&=f(e_G)^{-1} \ast f(e_G \cdot e_G)\\
&=f(e_G)^{-1} \ast \bigl(f(e_G) \ast f(e_G)\bigr)\\
&=f(e_G).
\end{aligned}
$$

②
$$
\begin{aligned}
f(a) \ast f(a^{-1})
&=f(a \cdot a^{-1})=f(e_G)=e_H,\\
f(a^{-1}) \ast f(a)
&=f(a^{-1} \cdot a)=f(e_G)=e_H
\end{aligned}
\quad\Longrightarrow\quad
f(a^{-1})=f(a)^{-1}.
$$

### (2) Monoid Homomorphism

**For monoids $(M, \cdot)$ and $(N, \ast)$, $f:M \to N$ is a monoid homomorphism $\iff$**
* ① 保持运算: $\forall a,b \in M, f(a \cdot b)=f(a) \ast f(b)$
* ② 保持单位元: $f(e_M)=e_N$

## 4. Isomorphism

### (1) Group Isomorphism

**For groups $G$ and $H$, $f:G \to H$ is a group isomorphism $\iff$**
* ① 同态: $f$ 是群同态
* ② 双射: $f$ 是双射

若存在这样的 $f$，则称 $G$ 与 $H$ 同构，记作 $G \cong H$。

### (2) Monoid Isomorphism

**For monoids $M$ and $N$, $f:M \to N$ is a monoid isomorphism $\iff$**
* ① 同态: $f$ 是幺半群同态
* ② 双射: $f$ 是双射

若存在这样的 $f$，则称 $M$ 与 $N$ 同构，记作 $M \cong N$。
