---
{"dg-publish":true,"permalink":"/notion/theoretical-knowledge/computer-science/information-and-entropy/"}
---

---
Textbook: [Syllabus | Information and Entropy | Electrical Engineering and Computer Science | MIT OpenCourseWare](https://ocw.mit.edu/courses/6-050j-information-and-entropy-spring-2008/pages/syllabus/)

Video: [MIT6.050J information and entropy 001_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1RS4y1z762/?spm_id_from=333.880.my_history.page.click&vd_source=516a94996ca902f85299544e0bf5de83)

### Information is a change of entropy

# 1. bits

### 1.1 The Boolean Bit
![Image/Information and entropy/1.png](/img/user/Image/Information%20and%20entropy/1.png)
![Image/Information and entropy/2.png](/img/user/Image/Information%20and%20entropy/2.png)
![Image/Information and entropy/3.png](/img/user/Image/Information%20and%20entropy/3.png)
### 1.2 The Circuit Bit

Combinational logic circuits graphically represent Boolean expressions. Each Boolean function (NOT, AND, XOR, etc.) corresponds to a "combinational gate" with inputs and an output. Gates are connected by lines, forming circuits. A key property is that combinational circuits **have no feedback loops**; an output never feeds back into an input earlier in its own causal chain. Circuits with loops are called sequential logic, which Boolean algebra alone cannot describe. For instance, a NOT gate with its output connected directly to its input creates a contradiction under Boolean analysis.

*   **Core:** Combinational logic circuits are gate-based, loop-free representations of Boolean operations.
*   **Distinction:** Unlike sequential logic circuits (which have loops), they are fully describable by Boolean algebra.

组合逻辑电路是布尔表达式的图形化表示，其中每个布尔函数（如NOT, AND, XOR）对应一个有输入和输出的“组合逻辑门”。这些门通过连线将一个门的输出连接到其他门的输入，形成电路。关键特性是组合电路中**没有反馈回路**（即输出不会反过来影响产生该输出的链条上的任何输入）。存在回路的电路称为时序逻辑电路，布尔代数不足以描述它们。例如，一个NOT门的输出直接连回其输入，用布尔代数分析会产生矛盾。

*   **核心**: 组合逻辑电路由门构成，无环路，用于表示布尔运算。
*   **区分**: 与有环路的时序逻辑电路不同，后者不能简单用布尔代数描述。

### 1.3 The Control Bit

In programming, Boolean expressions often control execution flow (i.e., which statements run). The algebra of control bits differs interestingly from pure Boolean algebra: **parts of a control expression not affecting the final result can be ignored (short-circuit evaluation)**. For example, in `(if (and (< x 0) (> y 0)) ...)`, if `x` is found to be non-negative, the `and` operation is false regardless of `y`, so `y` isn't evaluated. This speeds up programs and avoids potential side effects from evaluating `y`.

*   **Core:** Boolean logic used for program flow control.
*   **Key Difference:** Features "short-circuit evaluation" for efficiency and side-effect avoidance.

在计算机程序中，布尔表达式常用于控制执行流程（即决定哪些语句被执行）。控制位代数与布尔代数的一个有趣区别是：**表达式中不影响最终结果的部分可能被忽略（短路求值）**。例如，在 `(if (and (< x 0) (> y 0)) ...)` 中，如果 `x` 不小于0，`and` 表达式的结果就确定为假，此时 `y` 的值无需判断，程序运行更快，且避免了评估 `y` 可能带来的副作用。

*   **核心**: 程序中用于控制流程的布尔逻辑。
*   **关键区别**: 存在“短路求值”特性，可提高效率并避免副作用。

### 1.4 The Physical Bit

Storing or transporting a bit requires a physical medium. This medium must have two distinguishable states, representing 0 and 1. A bit is stored by setting the medium to one state and read by measuring its state. Communication occurs if the bit's state is transferred between locations unchanged; memory if it persists over time. Engineering aims for smaller, faster, etc., physical bits. Quantum mechanics defines the ultimate smallness limit for bit storage, leading to the concept of the quantum bit (qubit).

*   **Core:** Bits require a physical implementation with two distinct states.
*   **Development & Limit:** Engineering drives improvements; quantum mechanics sets fundamental limits on size.

比特的存储或传输需要物理载体，该载体具有两种可区分的状态，分别代表0和1。通过将物体置于特定状态来存储比特，通过测量其状态来读取比特。信息通信即比特状态在不同位置间的无损转移，而内存则是比特状态随时间的持续保持。工程上追求更小、更快等的物理比特。量子力学设定了物体能存储信息的最小尺寸极限，引出了量子比特（qubit）。

*   **核心**: 比特需要物理实现，依赖于物体的两种可区分状态。
*   **发展与极限**: 工程驱动物理比特的改进，量子力学揭示了其小型化的根本限制。

### 1.5 The Quantum Bit (Qubit)

A qubit is a quantum mechanical system capable of storing a single bit, typically represented by two basis states, $|0⟩$ and $|1⟩$. Three key quantum mechanical features distinguish qubits (or collections of them) from classical Boolean bits: reversibility, superposition, and entanglement.

*   **Core:** A bit based on quantum mechanical principles, denoted $|0⟩$ and $|1⟩$.
*   **Features:** Reversibility, superposition, and entanglement differentiate it from classical bits.

**Reversibility:**
Quantum evolution is inherently reversible; if a state can transition to another, the reverse is also possible. Thus, quantum functions are reversible, and outputs cannot be discarded. However, irreversibility arises from a quantum system's interaction with an unknown environment (losing information) and from the act of measurement itself.

*   **Core:** Quantum operations are fundamentally reversible.
*   **Exceptions:** Interaction with the environment and measurement introduce irreversibility.

**Superposition:**
A qubit can exist in a **superposition**, a combination of its $|0⟩$ and $|1⟩$ states. However, measuring a qubit in superposition forces it to **collapse** into one of the basis states (e.g., $|0⟩$ or $|1⟩$). The outcome is probabilistic; you get a definite "yes" or "no" for a specific state, never a "maybe" or a percentage. Crucially, after measurement, the qubit *is* in the measured state, so subsequent identical measurements yield the same result, offering no new information. While a single measurement's outcome is unpredictable, its probability can be calculated. This unique nature limits single-qubit information capacity but also enables novel system designs.
For example, a photon's polarization can store a bit (e.g., horizontal for $|0⟩$, vertical for $|1⟩$). A recipient (Bob) measures a specific polarization, getting a "yes"/"no" answer. If the sender (Alice) sends an arbitrarily polarized photon, Bob’s measurement is probabilistic, and the photon’s state changes to reflect the measurement outcome. Copying a qubit by measuring and recreating isn't possible due to this measurement disturbance.

*   **Core:** Qubits can exist in a combination of states simultaneously.
*   **Measurement Property:** Measurement yields a probabilistic, definite outcome, collapsing the superposition and altering the qubit's state. Exact superposition ratios cannot be known, and qubits cannot be perfectly copied via measurement.

**Entanglement:**
Two or more qubits can be **entangled**. In this state, they share a linked fate: their properties are correlated even when physically separated. Measuring one entangled qubit instantaneously influences the state of the other(s), regardless of distance.

*   **Core:** A strong, non-classical correlation between multiple qubits, where measuring one instantaneously affects others, even at a distance.

**Note:**
Importantly, if qubits are prepared independently (no entanglement) and restricted to basis states (no superposition), they behave like classical Boolean bits.

*   **Summary:** Under specific conditions, qubits can mimic classical bits.


量子比特（qubit）是能存储单个比特但受量子力学限制的微小物理模型，其两个基本状态通常表示为 $|0⟩$ 和 $|1⟩$。与经典布尔比特相比，量子比特或其集合具有三个显著特性：可逆性、叠加态和纠缠态。

*   **核心**: 基于量子力学原理的比特，用 $|0⟩$ 和 $|1⟩$ 表示。
*   **特性**: 可逆性、叠加态、纠缠态使其区别于经典比特。

**可逆性 (Reversibility):**
量子系统的状态演化是可逆的，因此量子运算函数也是可逆的，输出不能被丢弃。但量子系统与环境相互作用导致信息丢失，以及测量行为本身，是不可逆性的两个重要来源。

*   **核心**: 量子操作原则上可逆。
*   **例外**: 与环境交互和测量过程引入不可逆性。

**叠加态 (Superposition):**
量子比特可以处于 $|0⟩$ 和 $|1⟩$ 的某种组合（叠加态）中。然而，对量子比特进行测量时，它会**塌缩**到被测量的特定状态（例如，问“是否为$|1⟩$?”，答案总是“是”或“否”，而不是概率分布）。测量结果是概率性的，且测量后量子比特的状态会改变为测量结果对应的状态，后续测量不会提供更多信息。因此，无法精确知道叠加的具体组合，也无法通过多次测量求平均或复制量子比特来获取更多信息。
以光子偏振为例：Alice可以用水平偏振代表$|0⟩$，垂直偏振代表$|1⟩$。Bob测量时只能选择一个特定方向问“是否是这个方向偏振？”，得到“是”或“否”的单一比特结果。如果Alice发送的是任意角度偏振的光子，Bob的测量结果是概率性的（概率与实际偏振方向和测量方向夹角的余弦平方有关），且测量后光子偏振方向会变为Bob测量的方向或其垂直方向。

*   **核心**: 量子比特可同时处于多种状态的组合。
*   **测量特性**: 测量会迫使量子比特选择一个确定状态（概率性选择），并改变其原有状态。无法精确得知叠加比例，也无法复制。

**纠缠态 (Entanglement):**
两个或多个量子比特可以被制备成一种特殊关联状态，即使它们物理上分离遥远，其状态仍然是相互关联的。对其中一个量子比特的测量结果会瞬间影响到其他纠缠量子比特的状态。

*   **核心**: 多个量子比特间存在一种超越经典认知的强关联，即使分离遥远，测量一个会影响其他。

**注意:**
如果量子系统被独立制备（无纠缠）且其状态被限制在如水平和垂直偏振（无叠加），那么量子比特的行为就如同经典的布尔比特。

*   **总结**: 在特定条件下，量子比特可以表现得像经典比特。

### 1.6 The Classical Bit

Unlike quantum bits (qubits) which are altered by measurement, **classical bits can be measured repeatedly without being disturbed**. This is possible because a classical bit is typically represented by a large number of identical physical objects (e.g., ~60,000 electrons for a single bit in semiconductor memory, or many photons in radio communication).

Because many objects are involved, measurements can yield a **continuous range of values** (like voltage from 0V to 1V), not just a binary "yes" or "no." This allows for **error margins**: specific voltage ranges (e.g., 0-0.2V for logical 0, 0.8-1V for logical 1) define the bit's state, with an undefined region in between.
A key mechanism is **"restoring logic,"** where circuits actively correct small deviations from the ideal 0V and 1V values as information is processed. This ensures the robustness of modern computers.

Therefore, a **classical bit is an abstraction**: it's a model where a bit can be measured without perturbation, and as a result, **copies of it can be made**. This model works very well for circuits employing restoring logic.

While all physical systems are ultimately governed by quantum mechanics, the classical bit is an **excellent approximation for current technology**. However, as technology advances and components shrink to the scale of a few atoms or photons, the limiting role of quantum mechanics will become critical, and this classical approximation will eventually break down. 

**In essence:**
*   **Classical bits are robust and can be repeatedly measured and copied** because they are realized by many physical entities.
*   **Restoring logic** is crucial for maintaining signal integrity despite noise.
*   It's an **approximation of quantum reality** that holds well for current technology but will face limits as devices shrink.

与量子比特测量后会改变其状态不同，**经典比特可以被重复测量而不会受到扰动**。这是因为经典比特通常由大量具有相同属性的物理对象来表示（例如，半导体存储器中的一个比特可能由约6万个电子表示，或者无线电通信中使用大量光子）。

由于涉及大量对象，对其测量结果可以是一个**连续的值**（例如 $0\ V$ 到 $1\ V$ 的电压），而不仅仅是“是”或“否”的二元结果。这允许设定**容错区间**（error margins）：特定电压范围（如 $0-0.2\ V$ 代表逻辑 $0$ ， $0.8-1\ V$ 代表逻辑 $1$ ）定义比特状态，中间区域则不确定。
一种关键机制是 **“恢复逻辑”**（restoring logic），电路在信息处理过程中会主动纠正与理想 $0\ V$ 和 $1\ V$ 之间的微小电压偏差。这确保了现代计算机的**鲁棒性**（robustness）。

因此，**经典比特是一个抽象概念**：它是一个可以被测量而不受扰动的模型，因此**可以被复制**。这个模型非常适用于使用恢复逻辑的电路。

尽管所有物理系统最终都遵循量子力学，但经典比特对于当前技术而言是一个**极佳的近似**。然而，随着技术进步，当组件尺寸缩小到少量原子或光子级别时，量子力学的限制作用将变得至关重要，这种经典近似最终会失效。

**简而言之：**
*   **经典比特因由大量物理实体实现而具有鲁棒性，可以被重复测量和复制。**
*   **“恢复逻辑”对于抵抗噪声、维持信号完整性至关重要。**
*   它是**量子现实的一种近似**，目前适用性良好，但随着器件微型化将面临量子效应的挑战。

