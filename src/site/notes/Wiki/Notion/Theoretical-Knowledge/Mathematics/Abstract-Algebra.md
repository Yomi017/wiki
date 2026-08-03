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
