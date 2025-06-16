---
{"dg-publish":true,"permalink":"/notion/theoretical-knowledge/computer-science/artificial-intelligence/information-and-entropy/"}
---

---
Textbook: [Syllabus | Information and Entropy | Electrical Engineering and Computer Science | MIT OpenCourseWare](https://ocw.mit.edu/courses/6-050j-information-and-entropy-spring-2008/pages/syllabus/)

Video: [MIT6.050J information and entropy 001_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1RS4y1z762/?spm_id_from=333.880.my_history.page.click&vd_source=516a94996ca902f85299544e0bf5de83)

### Information is a change of entropy

# Chapter 1: Bits

## 1.1 The Boolean Bit (布尔比特)
![Image/Information and entropy/1.png](/img/user/Image/Information%20and%20entropy/1.png)
![Image/Information and entropy/2.png](/img/user/Image/Information%20and%20entropy/2.png)
![Image/Information and entropy/3.png](/img/user/Image/Information%20and%20entropy/3.png)
## 1.2 The Circuit Bit (电路比特)

Combinational logic circuits graphically represent Boolean expressions. Each Boolean function (NOT, AND, XOR, etc.) corresponds to a "combinational gate" with inputs and an output. Gates are connected by lines, forming circuits. A key property is that combinational circuits **have no feedback loops**; an output never feeds back into an input earlier in its own causal chain. Circuits with loops are called sequential logic, which Boolean algebra alone cannot describe. For instance, a NOT gate with its output connected directly to its input creates a contradiction under Boolean analysis.

*   **Core:** Combinational logic circuits are gate-based, loop-free representations of Boolean operations.
*   **Distinction:** Unlike sequential logic circuits (which have loops), they are fully describable by Boolean algebra.

组合逻辑电路是布尔表达式的图形化表示，其中每个布尔函数（如NOT, AND, XOR）对应一个有输入和输出的“组合逻辑门”。这些门通过连线将一个门的输出连接到其他门的输入，形成电路。关键特性是组合电路中**没有反馈回路**（即输出不会反过来影响产生该输出的链条上的任何输入）。存在回路的电路称为时序逻辑电路，布尔代数不足以描述它们。例如，一个NOT门的输出直接连回其输入，用布尔代数分析会产生矛盾。

*   **核心**: 组合逻辑电路由门构成，无环路，用于表示布尔运算。
*   **区分**: 与有环路的时序逻辑电路不同，后者不能简单用布尔代数描述。

## 1.3 The Control Bit (控制比特)

In programming, Boolean expressions often control execution flow (i.e., which statements run). The algebra of control bits differs interestingly from pure Boolean algebra: **parts of a control expression not affecting the final result can be ignored (short-circuit evaluation)**. For example, in `(if (and (< x 0) (> y 0)) ...)`, if `x` is found to be non-negative, the `and` operation is false regardless of `y`, so `y` isn't evaluated. This speeds up programs and avoids potential side effects from evaluating `y`.

*   **Core:** Boolean logic used for program flow control.
*   **Key Difference:** Features "short-circuit evaluation" for efficiency and side-effect avoidance.

在计算机程序中，布尔表达式常用于控制执行流程（即决定哪些语句被执行）。控制位代数与布尔代数的一个有趣区别是：**表达式中不影响最终结果的部分可能被忽略（短路求值）**。例如，在 `(if (and (< x 0) (> y 0)) ...)` 中，如果 `x` 不小于0，`and` 表达式的结果就确定为假，此时 `y` 的值无需判断，程序运行更快，且避免了评估 `y` 可能带来的副作用。

*   **核心**: 程序中用于控制流程的布尔逻辑。
*   **关键区别**: 存在“短路求值”特性，可提高效率并避免副作用。

## 1.4 The Physical Bit (物理比特)

Storing or transporting a bit requires a physical medium. This medium must have two distinguishable states, representing 0 and 1. A bit is stored by setting the medium to one state and read by measuring its state. Communication occurs if the bit's state is transferred between locations unchanged; memory if it persists over time. Engineering aims for smaller, faster, etc., physical bits. Quantum mechanics defines the ultimate smallness limit for bit storage, leading to the concept of the quantum bit (qubit).

*   **Core:** Bits require a physical implementation with two distinct states.
*   **Development & Limit:** Engineering drives improvements; quantum mechanics sets fundamental limits on size.

比特的存储或传输需要物理载体，该载体具有两种可区分的状态，分别代表0和1。通过将物体置于特定状态来存储比特，通过测量其状态来读取比特。信息通信即比特状态在不同位置间的无损转移，而内存则是比特状态随时间的持续保持。工程上追求更小、更快等的物理比特。量子力学设定了物体能存储信息的最小尺寸极限，引出了量子比特（qubit）。

*   **核心**: 比特需要物理实现，依赖于物体的两种可区分状态。
*   **发展与极限**: 工程驱动物理比特的改进，量子力学揭示了其小型化的根本限制。

## 1.5 The Quantum Bit (Qubit) (量子比特)

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

## 1.6 The Classical Bit

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

# Chapter 2: Codes

In the previous chapter, we examined the fundamental unit of information, the bit, and its various abstract representations. This chapter concerns ways in which complex objects can be represented not by a single bit, but by arrays of bits.
A simple communication system model includes: Input (Symbols) -> Coder -> Channel (Arrays of Bits) -> Decoder -> Output (Symbols). This chapter will look into several aspects of the design of codes.

上一章我们探讨了信息的基本单位——比特，以及其各种抽象表示。本章关注如何用比特的数组来表示复杂的对象，而非单个比特。
一个简单的通信系统模型包括：输入（符号）-> 编码器 -> 信道（比特数组）-> 解码器 -> 输出（符号）。本章将探讨代码设计的几个方面。

Some objects for which codes may be needed include:
*   Letters: BCD, EBCDIC, ASCII, Unicode, Morse Code
*   Integers: Binary, Gray, 2’s complement
*   Numbers: Floating-Point
*   Proteins: Genetic Code
*   Telephones: NANP, International codes
*   Hosts: Ethernet, IP Addresses, Domain names
*   Images: TIFF, GIF, and JPEG
*   Audio: MP3
*   Video: MPEG

一些需要编码的对象包括：
*   字符: BCD, EBCDIC, ASCII, Unicode, 摩尔斯电码
*   整数: 二进制, 格雷码, 二进制补码
*   数字: 浮点数
*   蛋白质: 遗传密码
*   电话: NANP, 国际代码
*   主机: 以太网, IP地址, 域名
*   图像: TIFF, GIF, JPEG
*   音频: MP3
*   视频: MPEG

## 2.1 Symbol Space Size (符号空间大小)

**Core Point**: The number of symbols to be encoded (symbol space size) dictates the number of bits required and introduces different encoding challenges.
*   **2 symbols**: 1 bit is sufficient (e.g., coin toss).
*   **Integral power of 2 symbols**: The number of bits required equals the logarithm, base 2, of the symbol space size (e.g., 4 card suits need 2 bits, 32 students need 5 bits).
*   **Finite but not integral power of 2 symbols**: Use bits for the next higher power of 2, resulting in **unused bit patterns (spare capacity)**, which is a key design issue (e.g., 10 digits need 4 bits, leaving 6 unused patterns).
*   **Infinite, countable symbols**: Fixed-length bit strings can only represent a finite subset, leading to **overflow** for numbers outside the range (e.g., a 4-bit code for 0-15).
*   **Infinite, uncountable symbols**: Requires **discretization** to approximate continuous values with a finite set of selected values. This approximation is irreversible, but can be sufficiently accurate with enough bits (e.g., Floating-Point representation of real numbers).

**核心要点**: 需要编码的符号数量（符号空间大小）决定了所需比特的数量，并引出不同的编码挑战。
*   **符号数量为2**: 1比特即可编码（如硬币正反）。
*   **符号数量为2的整数次幂**: 所需比特数等于符号空间大小的以2为底的对数（如4种花色需2比特，32名学生需5比特）。
*   **符号数量有限但非2的整数次幂**: 使用能覆盖其数量的最小2的幂的比特数，会产生**未使用的比特模式（空闲容量）**，如何处理是设计关键（如10个数字需4比特，有6个未用模式）。
*   **符号数量无限但可数**: 固定长度的比特串只能表示有限范围内的符号，超出范围则出现**溢出(overflow)**情况，需特殊处理（如4比特只能表示0-15）。
*   **符号数量无限且不可数**: 必须进行**离散化(discretization)**，将连续值近似为有限个离散值。这种近似是不可逆的，但若比特数足够多，可实现足够精确的表示（如浮点数表示实数）。

## 2.2 Use of Spare Capacity (空闲容量的使用)

**Core Point**: In many situations, there are some unused code patterns because the number of symbols is not an integral power of 2. Handling these patterns is an important design issue.
*   **Strategies**: Ignore, Map to other values, Reserve for future expansion, Use for control codes, Use for common abbreviations.

**核心要点**: 当符号数量不是2的整数次幂时，会出现未使用的代码模式（空闲容量），处理这些模式是代码设计的重要环节。
*   **策略**: 忽略、映射到其他有效值、保留以供未来扩展、用作控制代码、用作常用缩写。

### 2.2.1 Binary Coded Decimal (BCD) (二进制编码的十进制)
*   **Core**: BCD uses ten four-bit patterns to represent digits 0-9, leaving six unused patterns (e.g., 1010).
*   **Handling**: These unused patterns can either be ignored (signaling an error if encountered) or mapped into legal values (e.g., converting all to 9).

*   **重点**: BCD 用4比特表示0-9的十个数字，留下6个未使用的模式（例如1010）。
*   **处理方式**: 这些未使用的模式可以被忽略（遇到则报错）或映射到现有合法值（如全部映射为9）。

### 2.2.2 Genetic Code (遗传密码)
*   **Core**: The Genetic Code is a biological encoding system that maps nucleotide sequences to amino acids, serving as a prime example of utilizing spare capacity and control codes.
*   **Encoding**: DNA and RNA are made of four nucleotide bases (A, C, G, T/U). A sequence of three nucleotides (a "codon") is needed to encode 20 different amino acids (4^3 = 64 codons, far more than 20 needed).
*   **Utilization of Spare Capacity**:
    *   **Redundancy/Robustness**: Most amino acids are specified by more than one codon (e.g., Alanine has 4 codes starting with GC, meaning the third nucleotide can vary without changing the amino acid – this is called "wobble"). This provides biological robustness.
    *   **Control Codes**: Three codons (UAA, UAG, UGA) serve as **stop codes** to terminate protein production. One codon (AUG) specifies Methionine and also acts as the **start code**, initiating protein synthesis.
*   **Significance**: The Genetic Code illustrates how redundancy can enhance code robustness and how reserved bit sequences can function as control information.

*   **核心**: 遗传密码是生物学中的一种编码，将核苷酸序列映射为氨基酸，是利用空闲容量和控制码的典范。
*   **编码方式**: DNA和RNA由四种核苷酸（碱基：A, C, G, T/U）组成。需要3个核苷酸序列（称为“密码子”，codon）来编码20种不同的氨基酸（4^3 = 64种密码子，远超20种需求）。
*   **空闲容量利用**:
    *   **冗余/健壮性**: 大多数氨基酸由多个密码子编码（例如，丙氨酸有4个密码子，都以GC开头，这意味着第三个核苷酸的变化不会影响氨基酸种类，这被称为“摆动效应”(wobble)）。这提供了生物功能的鲁棒性。
    *   **控制码**: 三个密码子（UAA, UAG, UGA）被用作**终止密码子** (stop code)，标志蛋白质合成的结束。一个密码子（AUG）既编码甲硫氨酸，也作为**起始密码子** (start code)，标志蛋白质合成的开始。
*   **重要性**: 遗传密码展现了如何通过冗余来增强代码的鲁棒性，以及如何利用保留的比特序列作为控制信息。

### 2.2.3 Telephone Area Codes (电话区号)
*   **Core**: Initial North American telephone Area Codes (1947) were designed with restrictions that limited them to 144 possible codes, with 58 reserved for future expansion.
*   **Future Expansion**: These reserved codes sufficed for over four decades until 1995, when new Area Codes were created by relaxing the middle-digit restriction to meet the exploding demand for telephone network usage.

*   **重点**: 北美电话区号的设计初期（1947年）通过限制数字（首位非0/1，中间位0/1，后两位不同）只产生144个区号，其中58个保留用于未来扩展。
*   **未来扩展**: 这些预留的区号支撑了四十多年，直到1995年通过放宽中间位限制来创建新的区号，以应对电话网络使用爆炸式增长的需求。

### 2.2.4 IP Addresses (IP地址)
*   **Core**: IPv4 addresses (32-bit) faced exhaustion due to the Internet's explosive growth.
*   **Future Expansion**: IPv6 (128-bit) was developed with a vastly larger address space, including large blocks reserved for future expansion (humorously, some blocks are set aside for other planets).

*   **重点**: IPv4地址（32比特）因互联网的爆炸式增长面临耗尽。
*   **未来扩展**: IPv6（128比特）被开发出来，它包含巨大的地址空间，其中大块地址保留用于未来扩展（甚至有幽默说法称有为其他行星保留的地址）。

### 2.2.5 ASCII (美国信息交换标准代码)
*   **Core**: ASCII (American Standard Code for Information Interchange) is a 7-bit code.
*   **Spare Capacity Use**: Of its 128 codes, 33 are explicitly reserved as **control characters** (e.g., CR, LF), with only 95 for printable characters.

*   **重点**: ASCII (American Standard Code for Information Interchange) 是7比特代码。
*   **空闲容量利用**: 128个代码中有33个显式保留为**控制字符**（如回车CR、换行LF），只有95个用于可打印字符。

## 2.3 Extension of Codes (代码的扩展性)

**Core Point**: Codes are often designed by humans, and sometimes, their initial design biases lead to difficulties when extended for purposes not originally envisioned.
*   **Examples**: ASCII's NUL (0000000) and DEL (1111111) were initially intended to be ignored, but their varying treatment across different systems (e.g., Unix) caused compatibility issues.
*   **Serious Bias**: The use of CR (carriage return) and LF (line feed) in ASCII, originally tied to teletype machine physical operations, resulted in severe file format incompatibilities across operating systems (Unix uses LF, Macs use CR, DOS/Windows requires both), a persistent source of frustration and errors.

**核心要点**: 代码设计者在最初可能未能预见其未来用途，导致代码扩展时出现问题和意外的偏见。
*   **例子**: ASCII的NUL (0000000) 和DEL (1111111) 最初被设计为可忽略字符，但在不同系统（如Unix）中处理方式不同，导致兼容性问题。
*   **严重偏见**: ASCII中CR (回车) 和LF (换行) 的使用，原本对应电传打字机物理操作，但在不同操作系统中（Unix用LF，Mac用CR，Windows用CR+LF）导致了文件格式的严重不兼容，是持续的痛点。

## 2.4 Fixed-Length and Variable-Length Codes (定长码和变长码)

**Core Point**: A key decision in code design is whether to represent all symbols with codes of the same number of bits (fixed-length) or to allow some symbols to use shorter codes than others (variable-length).
*   **Fixed-length codes**:
    *   **Advantages**: Usually easier to deal with because both the coder and decoder know the number of bits in advance.
    *   **Transmission**: Can be supported by parallel transmission. For serial transport, "framing errors" are a concern, often eliminated by sending "stop bits" between symbols to aid resynchronization.
*   **Variable-length codes**:
    *   **Advantages**: Can assign shorter codes to more frequent symbols, improving transmission efficiency.
    *   **Disadvantages**: The decoder needs a way to determine when the code for one symbol ends and the next one begins (e.g., by noting time intervals).

**核心要点**: 编码设计初期需决定所有符号使用相同比特数（定长码）还是不同比特数（变长码）。
*   **定长码**:
    *   **优点**: 编解码器预先知道比特数，处理简单。
    *   **传输**: 可并行传输。串行传输时需解决“帧错误”(framing error)问题，常通过发送“停止位”(stop bits)来帮助同步。
*   **变长码**:
    *   **优点**: 可以给常用符号分配更短的代码，提高传输效率。
    *   **缺点**: 解码器需要额外机制来判断一个符号的代码何时结束，下一个符号何时开始（如通过时间间隔）。

### 2.4.1 Morse Code (摩尔斯电码)
*   **Core**: Morse Code is a classic example of a variable-length code, developed for the telegraph.
*   **Encoding Principle**: It uses sequences of short and long pulses or tones (dots and dashes) separated by short periods of silence. The decoder determines the end of a character's code by noting the length of silence before the next dot or dash (intra-character, inter-character, and inter-word separations have different lengths).
*   **Efficiency**: Morse realized that some letters in the English alphabet are more frequently used than others, and he assigned them shorter codes. This allowed messages to be transmitted faster on average. He reportedly determined letter frequencies by counting movable type pieces in a print shop.
*   **Applications**: Widely used for telegraphy and later in radio communications before voice transmission was common. It was a required communication mode for ocean vessels until 1999.

*   **核心**: 摩尔斯电码是变长码的经典例子，为电报发明。
*   **编码原理**: 使用短促的“点”和较长的“划”组成，通过它们之间不同长度的静默间隔来区分字符内部和字符之间、单词之间的界限。
*   **效率**: 摩尔斯意识到英文字母的使用频率不同，因此为高频字母分配更短的代码，从而平均加快了消息传输速度。据说他通过统计印刷厂铅字数量来确定字母频率。
*   **应用**: 广泛用于电报和早期的无线电通信。直到1999年，仍是远洋船舶的强制通信模式。

## 2.5 Detail: ASCII (美国信息交换标准代码)

*   **Standard**: Introduced by ANSI in 1963, ASCII is a 7-bit code representing 33 control characters and 95 printing characters.
*   **8-bit Extensions**: In an 8-bit context, ASCII characters follow a leading 0. The 128 characters from HEX 80 to HEX FF (often called "extended ASCII") have been defined differently in various contexts (e.g., IBM PCs, Macs).
    *   ISO-8859-1 (ISO-Latin), common for web pages, uses HEX A0-FF for Western European letters and punctuation, reserving HEX 80-9F as control characters.
    *   Microsoft's Windows Code Page 1252 (Latin I) is similar but reassigns 27 of these 32 control codes to printed characters (e.g., HEX 80 for the Euro symbol), creating compatibility issues across platforms.
*   **Beyond 8-bits**: To represent Asian languages and more, additional characters are needed. Unicode is the strongest candidate for a 2-byte (16-bit) standard character code, aiming to represent less than 65,536 different characters globally.

*   **标准**: 1963年引入的7比特代码，包含33个控制字符和95个可打印字符。
*   **8比特扩展**: 在8比特环境中，ASCII被视为“下半部分”。HEX 80到HEX FF之间的128个字符（常被称为“扩展ASCII”）在不同操作系统（如IBM PC、Mac）中有不同定义。
    *   ISO-8859-1 (ISO-Latin) 是Web页面常用的8比特扩展，将HEX A0-FF用于西欧字符，将HEX 80-9F保留为控制字符。
    *   微软的Windows Code Page 1252 (Latin I) 则将ISO-8859-1中部分控制码（HEX 80-9F）重新分配给打印字符（如欧元符号），导致进一步的兼容性挑战。
*   **超越8比特**: 为表示亚洲语言等更多字符，需要更多比特。Unicode是目前最有力的16比特（2字节）字符编码标准，旨在支持全球范围内少于65,536个不同字符。

## 2.6 Detail: Integer Codes (整数编码)

**Core Point**: There are many ways to represent integers as bit patterns. All suffer from an inability to represent arbitrarily large integers in a fixed number of bits, potentially leading to an "overflow" condition.
*   **Binary Code**:
    *   **Use**: Nonnegative integers (e.g., memory addresses).
    *   **Property**: For code of length n, the 2^n patterns represent integers 0 through 2^n - 1. The LSB (least significant bit) is 0 for even and 1 for odd integers.
*   **Binary Gray Code**:
    *   **Use**: Nonnegative integers, useful for sensors where the integer being encoded might change while a measurement is in progress.
    *   **Property**: For code of length n, the 2^n patterns represent integers 0 through 2^n - 1. The two bit patterns of adjacent integers differ in exactly one bit.
*   **2’s Complement**:
    *   **Use**: Both positive and negative integers, widely used for ordinary arithmetic in computers.
    *   **Property**: For a code of length n, the 2^n patterns represent integers −2^(n-1) through 2^(n-1) - 1. The LSB is 0 for even and 1 for odd integers. Overlaps with binary code where applicable.
*   **Sign/Magnitude**:
    *   **Use**: Both positive and negative integers.
    *   **Property**: For code of length n, the 2^n patterns represent integers −(2^(n-1) − 1) through 2^(n-1) − 1. The MSB (most significant bit) is 0 for positive and 1 for negative integers; the other bits carry the magnitude. Conceptually simple but awkward in practice, with separate representations for +0 and -0.
*   **1’s Complement**:
    *   **Use**: Both positive and negative integers.
    *   **Property**: For code of length n, the 2^n patterns represent integers −(2^(n-1) − 1) through 2^(n-1) − 1. The MSB is 0 for positive integers; negative integers are formed by complementing each bit of the corresponding positive integer. Awkward and rarely used today, with separate representations for +0 and -0.

**核心要点**: 整数有多种比特表示方式，它们都受限于固定比特数，可能导致“溢出”(overflow)条件。
*   **二进制码 (Binary Code)**:
    *   **用途**: 非负整数 (如内存地址)。
    *   **特性**: n比特表示 0 到 2^n - 1。LSB (最低有效位) 决定奇偶。
*   **二进制格雷码 (Binary Gray Code)**:
    *   **用途**: 非负整数，尤其适用于测量变化的物理量（如传感器）。
    *   **特性**: n比特表示 0 到 2^n - 1。相邻整数的比特模式只相差一位。
*   **2的补码 (2’s Complement)**:
    *   **用途**: 有符号整数，广泛用于计算机普通算术。
    *   **特性**: n比特表示 -2^(n-1) 到 2^(n-1) - 1。LSB 决定奇偶。与二进制码重叠部分相同。
*   **符号-幅度码 (Sign/Magnitude)**:
    *   **用途**: 有符号整数。
    *   **特性**: n比特表示 -(2^(n-1) − 1) 到 2^(n-1) − 1。MSB (最高有效位) 为0表示正，1表示负；其余位表示幅度。概念简单但实际操作不便，存在+0和-0两种表示。
*   **1的补码 (1’s Complement)**:
    *   **用途**: 有符号整数。
    *   **特性**: n比特表示 -(2^(n-1) − 1) 到 2^(n-1) − 1。MSB为0表示正；负数通过对应正数每位取反得到。不常用且不便，存在+0和-0两种表示。

## 2.7 Detail: The Genetic Code (遗传密码的详细说明)

**Core Point**: The Genetic Code is a natural encoding system in living organisms used to translate genetic information (DNA/RNA) into proteins. It demonstrates redundancy and control mechanisms in code design.

**核心要点**: 遗传密码是生物体中用于将遗传信息（DNA/RNA）翻译成蛋白质的自然编码系统。它展示了编码设计中的冗余和控制机制。

*   **Information Flow**: DNA (genes) → mRNA (messenger RNA) → Ribosome/tRNA → Protein. Proteins are fundamental structural and functional units of cells and organisms.
*   **编码方式**: DNA (基因) → mRNA (信使RNA) → 核糖体/tRNA → 蛋白质。蛋白质是细胞和有机体的基本组成和功能单位。

*   **Encoding Unit**:
    *   DNA and RNA are composed of four different nucleotide bases (A, C, G, T/U).
    *   A **triplet codon** (three nucleotides) is required to encode the 20 different amino acids, resulting in 4^3 = 64 possible combinations.
*   **编码单位**:
    *   DNA和RNA由四种不同的核苷酸（碱基：A, C, G, T/U）组成。
    *   编码20种氨基酸需要**三联体密码子**(codon)，共 4^3 = 64 种组合。

*   **Redundancy and Robustness**:
    *   Most amino acids are specified by more than one codon (e.g., Alanine has 4, Isoleucine has 3).
    *   This redundancy enhances the code's robustness; even if a mutation occurs in the third nucleotide position, it may not change the encoded amino acid (known as the **wobble effect**).
*   **冗余性与鲁棒性**:
    *   大多数氨基酸有多个对应的密码子（例如，丙氨酸有4个，异亮氨酸有3个）。
    *   这种冗余增加了代码的健壮性，尤其是在第三个核苷酸位置，即使发生突变，也可能不会改变最终编码的氨基酸（称为**摆动效应**，wobble）。

*   **Control Information**:
    *   **Start Codon**: AUG codes for Methionine and also signals the beginning of protein synthesis.
    *   **Stop Codons**: UAA, UAG, and UGA do not code for any amino acid; instead, they signify the end of the protein chain, as proteins vary in length.
*   **控制信息**:
    *   **起始密码子**: AUG 编码甲硫氨酸，并标志蛋白质合成的开始。
    *   **终止密码子**: UAA, UAG, UGA 不编码任何氨基酸，而是标志蛋白质合成的结束，因为蛋白质长度不同。

*   **Value**: Although the coded description of a protein (DNA/RNA) may require more atoms than the protein itself, its standardized representation allows the same "assembly apparatus" (ribosome) to fabricate different proteins at different times, enabling versatility.
*   **价值**: 尽管蛋白质的编码描述（DNA/RNA）在原子数上可能比蛋白质本身更大，但其标准化表示使得同一个“组装设备”（核糖体）可以在不同时间制造不同的蛋白质，实现了通用性。

## 2.8 Detail: IP Addresses (IP地址的详细说明)

*   **Version 4 (IPv4)**: Each address is 32 bits, in the form x.x.x.x, where each x is a number between 0 and 255.
*   **管理**: IP地址由互联网号码分配机构 (IANA) 管理分配。
*   **Address Exhaustion & IPv6**: Due to the Internet's explosive growth, IPv4 addresses faced exhaustion.
*   **版本4 (IPv4)**: 每个地址由32比特组成，形式为x.x.x.x，每个x是0-255的数字。
*   **Management**: IP addresses are assigned by the Internet Assigned Numbers Authority (IANA).
*   **地址耗尽与IPv6**: 随着互联网的爆炸性增长，IPv4地址面临耗尽。

*   **Version 6 (IPv6)**:
    *   **Design**: Each address is expanded to 128 bits, dramatically increasing the address space.
    *   **Purpose**: To accommodate unique addresses for a vast number of devices, including appliances and sensors in vehicles.
    *   **Strategy**: New allocations include large blocks reserved for future expansion (humorously, some blocks are set aside for other planets).
*   **版本6 (IPv6)**:
    *   **设计**: 每个地址扩展为128比特，极大地增加了地址空间。
    *   **目的**: 能够为更多设备（如家用电器、汽车中的传感器）分配独立地址。
    *   **策略**: 新的分配方案包含了大量为未来扩展而保留的地址块（甚至有幽默说法称有为其他行星保留的地址）。

## 2.9 Detail: Morse Code (摩尔斯电码的详细说明)

*   **Invention Context**: Samuel F. B. Morse, motivated by his wife's sudden death and the inability to communicate rapidly, invented Morse Code.
*   **发明背景**: 塞缪尔·摩尔斯因妻子猝死未能及时赶回的经历，促使他意识到快速通信的重要性，从而发明了摩尔斯电码。

*   **Encoding Method**:
    *   A **variable-length code** consisting of short and long pulses or tones (dots and dashes) separated by short periods of silence.
    *   If a dot's duration is one unit of time, a dash is three units. The space between dots and dashes within one character is one unit, between characters is three units, and between words is seven units.
*   **编码方式**:
    *   一种**变长码**，由短促的“点”和较长的“划”组成，通过不同长度的静默间隔区分字符和单词。
    *   “点”的时长为1个时间单位，“划”为3个单位。字符内点划间间隔1单位，字符间间隔3单位，单词间间隔7单位。

*   **Design Advantage**: Morse realized that some letters in the English alphabet are more frequently used than others, and he assigned them shorter codes. This allowed messages to be transmitted faster on average. He reportedly determined letter frequencies by counting movable type pieces in a print shop.
*   **设计优点**: 摩尔斯根据英文字母的**使用频率**为它们分配了不同的代码长度，高频字母（如E、T）拥有更短的代码，从而提高了平均传输速度。据说他通过统计印刷厂的铅字数量来确定字母频率。

*   **Applications**: Widely used for telegraphy and later in radio communications before voice transmission was common. It remained a required communication mode for ocean vessels until 1999 and is still a requirement for some amateur radio licenses in the U.S.
*   **应用**: 广泛用于电报和早期的无线电通信。直到1999年，仍是远洋船舶的强制通信模式，在美国，某些业余无线电执照仍要求掌握摩尔斯电码。