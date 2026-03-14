---
title: Linear Approximation
headline: Provably Efficient Reinforcement Learning with Linear Function Approximation
---

## 1. Introduction

<!--
我先按原文顺序把第 1 节（含 1.1 Related Work）完整翻译出来；紧接着，我会把其中几个容易“读着顺、但没真正吃透”的概念补一层解释，帮助你把这节和后面算法/证明要用的逻辑链接上。

## 1 引言（Introduction）逐段翻译

强化学习（Reinforcement Learning, **RL**）是一个控制理论意义上的问题：智能体通过在时间上与一个未知环境交互，试图最大化其期望的累积回报 [41]。现代 RL 经常面对**状态数量极其巨大**的实际任务，这时就必须使用**函数逼近（function approximation）**去近似（动作）价值函数——从某个“状态–动作对”出发的期望累积回报——或者去近似策略——从状态映射到后续动作的函数。函数逼近（尤其是基于**深度神经网络**的函数逼近）是 RL 近年在 Atari 游戏 [30]、围棋 [38]、机器人 [23]、对话系统 [27] 等领域取得实践成功的核心原因之一。更进一步，深度神经网络还是许多通用深度 RL 算法的关键组成部分，例如 Deep Q-Network（**DQN**）[30]、Asynchronous Advantage Actor-Critic（**A3C**）[31]、Trust Region Policy Optimization（**TRPO**）[36]。

尽管函数逼近在经验上非常成功，但现有的大多数理论保证主要仍然局限在**表格型（tabular）RL**（例如 [20,33,8,22]）：状态与动作都是离散的，价值函数直接用一张表来表示。由于**维数灾难（curse of dimensionality）**，表格型 RL 只能处理相对较小的问题，因此研究者在理论与实践中都转向了函数逼近（例如 [40,12,43]）。然而，函数逼近（特别是在深度 RL 架构中）虽然显著扩展了 RL 的适用范围，却也带来一系列根本性的理论挑战。

例如，当使用函数逼近时，有效的状态与动作空间可以大得多，但在有限的学习回合里，大多数状态的“邻域”甚至一次都不会被访问到，从而很难对价值函数做出可靠估计（例如 [41,42,26]）。为了应对这种困难，人们常使用更简单的函数类（包括**线性函数类**）；但这又会引入偏差：即使训练数据趋于无穷，最优价值函数与最优策略也未必是线性的（例如 [10,11,43]）。因此，无论在理论还是实践中，RL 系统设计都必须在动态系统的背景下处理两个基础统计问题：**稀疏性（sparsity）**与**模型失配（misspecification）**。此外，RL 的一个核心区别性特征是必须处理**探索–利用（exploration/exploitation）权衡**。而要在算法上处理这种权衡，恰恰需要那些在 RL 场景中最难获得的统计估计——这正是因为稀疏性、失配与动态性共同造成的困难。于是，一个基本问题仍然悬而未决：

在函数逼近设定下，是否可能设计出具有可证明效率的 RL 算法？

这里所谓“高效（efficient）”，指的是在**运行时间**与**样本复杂度**两方面都高效：运行时间与样本复杂度不应依赖于状态数，而应当依赖于函数类的某种**内在复杂度度量（intrinsic complexity measure）**。

近期有若干工作尝试攻克这个问题，但它们要么需要访问一个“**模拟器（simulator）**”[49]（从而缓解探索难题），要么假设转移动态是确定性的 [47,48]、或具有低方差 [19]、或能由一个相对小的矩阵参数化 [50]（从而缓解转移动态估计的困难；更多细节见 1.1 节）。

本文聚焦于一个**线性设定**：转移动态与奖励函数都假设为线性的。在不需要额外 oracle 或更强假设的前提下，作者提出了第一个在运行时间与样本复杂度上都具有可证明效率的算法。更具体地，在一般的 episodic MDP 设定下，作者证明：对经典的 Least-Squares Value Iteration（**LSVI**）[12,33] 做一个**乐观（optimistic）**的改造，就能获得 (\widetilde{O}(\sqrt{d^{3}H^{3}T})) 的遗憾（regret），其中 (d) 是特征空间维度，(H) 是每个 episode 的长度，(T) 是总步数，而 (\widetilde{O}(\cdot)) 只隐藏绝对常数与多对数因子。重要的是，这个遗憾上界**不依赖于状态数与动作数**。算法的时间复杂度为 (O(d^{2}AKT))，空间复杂度为 (O(d^{2}H+dAT))，同样不依赖于状态数，因此在实践上也更可行。作者还强调结果对线性假设是“鲁棒”的：当真实转移模型不是严格线性的、但在全变差距离意义下与线性模型相差 (\zeta)（Assumption B）时，算法遗憾为 (\widetilde{O}(\sqrt{d^{3}H^{3}T}+\zeta dHT))。也就是说，除了标准的 (\sqrt{T}) 项之外，还会出现一个与 (\zeta) 成比例的**线性遗憾项**，它来自函数类失配导致的近似误差。

## 1.1 相关工作（Related Work）逐段翻译

表格型 RL：表格型 RL 在基于模型（model-based）[20,33,8,17] 与无模型（model-free）[39,22] 两条线上都已有充分研究。另一些工作考虑了一个更简化的设定：可以访问“模拟器”（也称生成模型 generative model）[24,6,7,25,37,45]。模拟器是一种很强的 oracle：它允许算法对任意状态–动作对进行查询，并返回奖励与下一状态。模拟器会显著缓解探索困难，因为一种朴素的探索策略——对所有状态–动作对均匀随机查询——就已经能得到寻找最优策略的最有效算法 [7]。在 episodic、转移非平稳且没有模拟器的设定中，已有基于模型与无模型算法能达到的最好遗憾分别为 (\widetilde{O}(\sqrt{H^{2}SAT})) [8] 与 (\widetilde{O}(\sqrt{H^{3}SAT})) [22]，它们都（几乎）达到了极小极大下界 (\Omega(\sqrt{H^{2}SAT})) [20,32,22]。这里 (S,A) 分别表示状态数与动作数。尽管这些算法（几乎）极小极大最优，但它们无法应对巨大状态空间，因为遗憾会随 (\sqrt{S}) 线性增长，而实践中的 (S) 往往呈指数规模（例如 [30,38,23,27]）。而且，这个下界也表明：从信息论角度看，除非利用额外的结构，否则巨大状态空间不可能被高效处理。与这一路线相比，本文利用奖励与转移函数的线性结构，证明乐观 LSVI 的遗憾随特征维度 (d) 多项式增长，而不是随状态数 (S) 增长。

线性 bandit：为了启用函数逼近，另一条相关路线研究随机线性 bandit 或随机线性上下文 bandit（例如 [5,16,28,35,14,2]）。这相当于本文“线性 MDP”（Assumption A）的一个特例：把 episode 长度设为 (H=1)。现有算法能达到的最好遗憾是：线性 bandit 为 (\widetilde{O}(d\sqrt{T})) [2]，线性上下文 bandit 为 (\widetilde{O}(\sqrt{dT})) [5,14]，都对维度 (d) 是多项式依赖。作者指出关键差异在于：MDP 有状态转移，而线性 bandit 没有；这种时间结构刻画了探索难度的根本差别。把现有线性 bandit 算法“天真地”迁移到线性 MDP，会得到对 (H) 指数级的遗憾（(H) 是每个 episode 长度）。

带函数逼近的 RL：在线性函数逼近设定下，经典文献中已有很长一条算法设计路线，但并未给出多项式样本效率保证（例如 [12,29,41,33,9]）。近期，Yang 与 Wang [49] 重新研究了线性转移与线性奖励的设定（Assumption A），并在可访问模拟器的前提下提出了样本高效算法；类似表格情形，模拟器显著缓解探索难题。作者还提到一项与本文独立发展的近期工作 [50]：它也在“线性 MDP”设定下给出探索的样本效率保证，但相比本文多需要一个关键假设——转移模型能由一个相对小的矩阵参数化。该假设把转移模型从“潜在无限自由度”（当状态空间无限时）降到“小且有限”，因此减轻了估计转移模型的难度；相应地，[50] 的主要机制是估计那个未知矩阵，这与本文方法不同。最后，在更广义语境下，如果不假设线性 MDP，也有人在其他假设下建立样本效率保证，例如转移完全确定 [47,48] 或低方差 [19]；这些假设在实践中可能偏强，甚至在表格设定下也未必成立。相较之下，本文结果直接覆盖标准表格情形而无需额外假设。

一般函数逼近：在更一般的函数逼近设定下，Jiang 等 [21] 提出了通用算法 Olive：当其定义的复杂度度量 **Bellman rank** 足够小时，可以获得样本效率。在 Assumption A 下可以证明 Bellman rank 至多为 (d)，因此 Olive 在本文设定里也具有样本效率。但与本文相比，Olive 通常**不具备计算效率**，并且不提供 (\sqrt{T}) 级别的遗憾界。与此同时，一条较新的工作线 [51,46] 研究了奖励与转移模型满足 **Hölder 光滑**的非参数设定，但其样本复杂度在最坏情况下对维度是指数级的。

## 必要的解释、补充与说明（帮助你把“引言”读成可用的技术地图）

这节引言其实在非常精确地“钉”三个难点：**覆盖不足（稀疏性）**、**可逼近性不足（失配）**、以及 RL 特有的**探索–利用耦合**。在监督学习里，你可以假设训练样本是某个固定分布的 i.i.d. 抽样；但在 RL 里，数据分布是由你的策略诱导出来的，而且策略还在学习中不断变化。于是你会同时遇到“很多状态根本没去过所以估不准”和“为了去那些状态你又必须先会探索”的闭环困难，这就是作者说的“需要的统计估计恰恰最难获得”的技术含义。

引言里反复提到的“**模拟器 / 生成模型（simulator / generative model）**”，你可以把它理解为一种极强的交互 oracle：允许你指定任意 ((s,a))，直接采样 (r, s')。这会把探索难度大幅“抹平”，因为你不必通过一条长轨迹去抵达想看的状态–动作对；在理论上它常用来隔离“估计误差”与“探索到达性”的混杂因素。作者强调“不需要模拟器”，是在强调他们处理的是更真实的在线交互设定。

最后，引言中关于 (\zeta) 的“近似线性”讨论，是在提前给你一个非常重要的结论：当你用**线性逼近**去做本质上非线性的环境时，除了常见的统计项（(\sqrt{T})）之外，必然会出现一个与失配程度成比例的**线性累计偏差**（(\zeta dHT) 这种量级）。这不是作者在“吓人”，而是在把“逼近误差会如何进入遗憾”说清楚：即使你把估计做得再好，函数类本身装不下真值时，误差会在长期交互里以线性方式累积。
-->

## 2. Preliminaries

考虑的是 Finite-horizon Episodic MDP $\mathrm{MDP}(\mathcal S,\mathcal A,H,\mathbb P,r)$，分别代表状态集合、动作集合、每个 episode 的长度。$\mathbb P = \{\mathbb P_h\}_{h=1}^H$ 和 $r = \{r_h\}_{h=1}^H$ 分别是分步/Time-inhomogeneous 的状态转移核与奖励函数。假设 $\mathcal S$ 是可测空间、$\mathcal A$ 是有限集合，且 $\lvert \mathcal A \rvert = A$。对每个 $h \in [H]$，$\mathbb P_h(\cdot \mid x,a)$ 表示在第 $h$ 步、状态为 $x$、采取动作 $a$ 时对下一状态的转移分布；$r_h: \mathcal S \times \mathcal A \to [0,1]$ 是第 $h$ 步的确定性奖励函数。这里奖励函数可以是随机的，本文的结果可以泛化到这种情形。

智能体与该分幕 MDP 的交互过程如下：在每一幕，初始状态 $x_1$ 对抗性地随机指定，随后对每个时间步 $h \in [H]$，智能体观察到状态 $x_h \in \mathcal S$，选择动作 $a_h \in \mathcal A$ 并获得奖励 $r_h(x_h,a_h)$，环境按照概率测度 $\mathbb P_h(\cdot \mid x_h,a_h)$ 采样产生新状态 $x_{h+1}$。当到达 $x_{H+1}$ 时本幕结束，不再获得奖励。

策略 $\pi$ 是函数 $\pi: \mathcal S \times [H] \to \mathcal A$，其中 $\pi(x,h)$ 表示在第 $h$ 步处于状态 $x$ 时所采取的动作。**由于在每一个时间步对应的动作价值函数和价值函数都不同，因此必须考虑每一个时间步对应的策略**。对应的是价值函数和动作价值函数

$$
\begin{aligned}
V_h^\pi(x) &\coloneqq \mathbb E \left[ \sum_{h' = h}^{H} r_{h'}(x_{h'}, \pi(x_{h'}, h')) \mid x_h = x \right], \quad \forall x \in \mathcal S, h \in [H]. \\
Q_h^\pi(x,a) &\coloneqq r_h(x,a) + \mathbb E \left[ \sum_{h' = h+1}^{H} r_{h'}(x_{h'}, \pi(x_{h'}, h')) \mid x_h = x, a_h = a \right], \quad \forall (x,a) \in \mathcal S \times \mathcal A, h \in [H].
\end{aligned}
$$

由于动作空间和幕长度都有限，因此一定存在一个最优策略 $\pi^\star$ 使得 $V_h^\star(x) = \sup_\pi V_h^\pi(x)$ 对所有 $x,h$ 成立。使用 $[\mathbb P_h V_{h+1}](x, a) \coloneqq \mathbb E_{x' \sim \mathbb P_h(\cdot \mid x,a)} [V_{h+1}(x')]$，策略 $\pi$ 的 Bellman 方程写作

$$
\begin{aligned}
Q_h^\pi(x,a) = (r_h + \mathbb P_h V_{h+1}^\pi)(x,a), \quad V_h^\pi(x) = Q_h^\pi(x, \pi_h(x)), \quad V_{H+1}^\pi(x) = 0 \\
Q^\star_h(x,a) = (r_h + \mathbb P_h V_{h+1}^\star)(x,a), \quad V_h^\star(x) = \max_{a \in \mathcal A} Q_h^\star(x,a), \quad V_{H+1}^\star(x) = 0
\end{aligned}
$$

在分幕 MDP 设置下，Agent 目标是在和环境的交互过程中学得最优策略：对于每一个 $k\geq 1$，在第 $k$ 幕的开始，对手会对抗挑选一个初始状态 $s_1^k$，Agent 挑选出策略 $\pi^k$，使用该策略与环境交互直至幕结束。使用 $V_1^\star(x_1^k) - V_1^{\pi^k}(x_1^k)$ 衡量当前策略的遗憾，总计遗憾定义为

$$
\mathrm{Regret}(K)=\sum_{k=1}^{K} \left[ V_1^\star(x_1^k) - V_1^{\pi^k}(x_1^k) \right]
$$

我们研究的核心是 Linear MDP，在这里 **状态转移和奖励函数** 被假设为 **在某一个特征映射上是线性的**，但是策略的形式并没有被假设为线性的。这样的假设可以推出一个关键性质，动作价值函数也是线性的。注意，这里的线性假设类似于统计建模中的数据生成机制的假设。

**Assumption: Linear MDP**：$\mathrm{MDP}(\mathcal S, \mathcal A, H, \mathbb P, r)$ 是线性的，当存在一个特征映射 $\phi: \mathcal S \times \mathcal A \to \mathbb R^d$，使得对于每一个 $h \in [H]$，存在 $d$ 个定义在 $\mathcal S$ 上的未知的符号测度 $\mathbf \mu_h = (\mu_h^{(1)}, \dots, \mu_h^{(d)})$ 以及一个未知的向量 $\mathbf \theta_h \in \mathbb R^d$，使得对于任意 $(x,a) \in \mathcal S \times \mathcal A$

$$
\mathbb P_h(\cdot \mid x,a) = \langle \phi(x,a), \mathbf \mu_h(\cdot) \rangle, \quad r_h(x,a) = \langle \phi(x,a), \mathbf \theta_h \rangle
$$

这里我们不失一般性地假设特征映射 $\phi$ 被归一化了，即对于所有 $(x,a)$，$\lVert \phi(x,a) \rVert \leq 1$，并且对于所有 $h$，$\max \{ \lVert \mathbf \mu_h(\mathcal S) \rVert, \lVert \mathbf \theta_h \rVert \} \leq \sqrt{d}$。

虽然这里面假设了线性，但是转移核 $\mathbb P_h(\cdot \mid x,a)$ 仍然可能有无限的自由度，因为 $\mathbf \mu_h$ 是一个未知的测度，而不是一个有限维的矩阵参数化的形式。熟悉的 Tabular MDP 就是一个 Linear MDP。Linear MDP 的最关键性质是其动作价值函数的线性性，因此在设计 RL 算法时只需要关注线性的 Q 函数就可以。

**Property: Linearity of Action-value Function in Linear MDP**：对于一个 Linear MDP 和其任意策略 $\pi$，存在未知的参数向量 $\{\mathbf w_h^\pi\}_{h=1}^H$，使得对于所有 $(x,a,h) \in \mathcal S \times \mathcal A \times [H]$，都有 $Q_h^\pi(x,a) = \langle \phi(x,a), \mathbf w_h^\pi \rangle$。


<!--

## 2.1 线性 Markov 决策过程（Linear Markov decision processes）

作者聚焦于**线性 MDP（Linear MDP）**：假设转移核与奖励函数都对某个特征映射是线性的。这个假设将推出（作者稍后证明）一个关键结构：对任意策略 $\pi$，对应的 $Q_h^\pi$ 都会在该特征上呈线性形式。作者强调这**不是**“策略是线性函数”的那种假设；这里更像统计建模里的“数据生成机制”假设：先规定环境如何由特征生成转移/奖励，再研究估计与探索。

**假设 A（Linear MDP）**：给定特征映射 $\phi:\mathcal S\times\mathcal A\to\mathbb R^d$，若对每个 $h\in[H]$，存在 $d$ 个未知的（可为符号的）测度组成的向量
$\mu_h=(\mu_h^{(1)},\dots,\mu_h^{(d)})$（每个 $\mu_h^{(i)}$ 都是定义在 $\mathcal S$ 上的测度），以及未知向量 $\theta_h\in\mathbb R^d$，使得对任意 $(x,a)$：
$$
P_h(\cdot\mid x,a)=\langle \phi(x,a),\mu_h(\cdot)\rangle,\qquad
r_h(x,a)=\langle \phi(x,a),\theta_h\rangle.
\tag{3}
$$
其中 $\langle \phi(x,a),\mu_h(\cdot)\rangle$ 表示把向量 $\phi(x,a)$ 与“测度向量”做内积得到一个测度；直观写开就是
$P_h(\cdot\mid x,a)=\sum_{i=1}^d \phi_i(x,a),\mu_h^{(i)}(\cdot)$，它最终必须对每个 $(x,a)$ 给出一个合法的概率测度。作者还给出一组归一化约束：对所有 $(x,a)$，$|\phi(x,a)|\le 1$，并且对所有 $h$，
$\max{|\mu_h(\mathcal S)|,|\theta_h|}\le \sqrt d$。

按定义，线性 MDP 中转移与奖励都在特征 $\phi$ 上线性。但作者特别提醒一个容易忽略的点：尽管它“线性”，转移核 $P_h(\cdot\mid x,a)$ 仍可能有**无限自由度**，因为 $\mu_h$ 是未知测度而不是有限维矩阵参数；这与 LQR 或某些“转移由小矩阵参数化”的工作不同。另一方面，由于他们已假设 $r_h\in[0,1]$，可推出 $V_h^\pi\in[0,H]$；在上述归一化下，下面两个例子都属于线性 MDP 的特例。

例 2.1（**表格型 MDP**）：当状态与动作都有限时，取 $d=|\mathcal S|\cdot|\mathcal A|$，用状态–动作对 $(x,a)$ 给 $\mathbb R^d$ 的坐标做索引，并令 $\phi(x,a)=e_{(x,a)}$ 为标准基。若令 $e_{(x,a)}^\top\mu_h(\cdot)=P_h(\cdot\mid x,a)$ 且 $e_{(x,a)}^\top\theta_h=r_h(x,a)$，就退化回普通的 tabular MDP。

例 2.2（**单纯形特征空间**）：若特征集合 ${\phi(x,a)}$ 是 $d$ 维单纯形 ${\psi\mid \sum_{i=1}^d \psi_i=1,\ \psi_i\ge 0}$ 的子集，则可通过令 $e_i^\top\mu_h$ 为任意定义在 $\mathcal S$ 上的概率测度，并令 $\theta_h$ 满足 $|\theta_h|_\infty\le 1$，来实例化一个线性 MDP。

作者最后回到线性 MDP 的核心性质：对所有策略，**动作价值函数始终在 $\phi$ 上是线性的**，因此设计 RL 算法时只需要关注线性 $Q$ 函数类即可。

## 算法 1：带 UCB 的最小二乘价值迭代（LSVI-UCB）

（该算法框在论文排版中紧接着第 2 节给出。）

```text
算法 1  Least-Squares Value Iteration with UCB (LSVI-UCB)

for episode k = 1, ..., K do
    接收初始状态 x_1^k
    for step h = H, ..., 1 do
        Λ_h ←  Σ_{τ=1}^{k-1} φ(x_h^τ, a_h^τ) φ(x_h^τ, a_h^τ)^⊤  +  λ I
        w_h ←  Λ_h^{-1} Σ_{τ=1}^{k-1} φ(x_h^τ, a_h^τ) [ r_h(x_h^τ, a_h^τ) + max_a Q_{h+1}(x_{h+1}^τ, a) ]
        Q_h(·,·) ← min{ w_h^⊤ φ(·,·) + β [ φ(·,·)^⊤ Λ_h^{-1} φ(·,·) ]^{1/2},  H }
    for step h = 1, ..., H do
        采取动作 a_h^k ← argmax_{a∈A} Q_h(x_h^k, a)，并观测到 x_{h+1}^k
```



## 命题 2.3（线性 MDP ⇒ $Q^\pi$ 线性）

对线性 MDP，任意策略 $\pi$ 都存在一组权重 ${w_h^\pi}_{h\in[H]}$，使得对任意 $(x,a,h)\in\mathcal S\times\mathcal A\times[H]$，
$$
Q_h^\pi(x,a)=\langle \phi(x,a),,w_h^\pi\rangle.
$$
该命题的证明在附录 A 中给出，同时附录也补充讨论了线性 MDP 的一些基本性质。

-->
