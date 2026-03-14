---
title: MAMBA
headline: Policy Improvement via Imitation of Multiple Oracles
---

> [!abstract]

> [!info] Contributions

## 1. Introduction

对于拥有单个专家的模仿学习，专家策略的汇报为其提供了一个天然的基准去匹配或超越，大多数现有的模仿学习技术都假设这样的设定。然而，在现实中，通常很难获得一个高质量的专家策略。相反，可能只有多个次优专家可用，每个专家在不同的情况下都有其优势。在本文，我们提出这样的问题：**如何利用编码在多个可能次优的专家策略中的领域知识来进行学习？**我们在交互式模仿学习的设定下研究这个问题。

直觉上，因为更多的专家策略可以提供关于问题领域的更多信息，强化学习智能体应该能够比使用单个专家更快地学到一个好的策略。然而，实际上智能体并不知道每个专家的特性，它看到的仅仅是来自不同专家策略的**相互冲突/Conflicting** 的演示。解决这种分歧/Disagreement 并非易事，因为可能不存在一个能在所有方面都全面超越其他专家的单一专家，而且每个专家策略的质量都是未知的。虽然最近很多工作，比如 InfoGAIL、AC-Teach 和 OIL，已经开始研究这个问题，但它们都回避了两个基本问题：

- 在多智能体模仿学习的设定下，什么是衡量策略性能的合理基准/Benchmark？这应类似于传统模仿学习中的单专家策略质量。
- 是否存在一种系统性的方法，将多个次优专家缝合/Stitch 成一个更强的基线/Baseline，并让我们能够在此基础上进一步提升？

这就是这篇文章的贡献，其提出一个新的 Max-Aggregated Baseline 基准，自然的将不同的专家策略组合在一起，在每一个状态下都比原来所有专家都表现更好，进而设计了一种新的模仿学习算法 MAMBA，使用 Roll-in/Roll-out 的 Interactive 范式与专家交互，基于 Generalized Advantage Estimation/GAE 思想设计 IL 梯度估计，并提供了基于遗憾的理论保证。

## 2. Preliminaries

### 2.1 Episodic Interactive Imitation Learning

考虑一个 Finite-horizon MDP，其状态空间为 $\mathcal{S}$，动作空间为 $\mathcal{A}$，时间视界为 $T$，初始状态分布为 $d_0(s)$，转移概率为 $\mathcal{P}(s' \mid s, a)$，奖励函数为 $r: \mathcal{S} \times \mathcal{A} \to [0, 1]$。我们假设 $d_0$、$\mathcal{P}$ 和 $r$ 是固定但未知的。给定一类状态相关的策略 $\Pi$，我们的目标是找到一个策略 $\pi \in \Pi$，使其最大化相对于初始状态分布 $d_0$ 的 $T$ 步回报。

为了处理非平稳过程，一般的做法是为每一个时间步 $t$ 都定义一个转移核、奖励函数以及对应的价值函数，这里我们使用比较紧凑的写法：假设状态空间 $\mathcal{S}$ 被构造为一个扩充状态空间 $\mathcal{S} = \bar{\mathcal{S}} \times \{0, \dots, T-1\}$，其中 $\bar{\mathcal{S}}$ 是某种基础状态空间，这样 $\mathcal{P}$ 和 $r$ 在 $\mathcal{S}$ 中可以是非平稳的，并且使用下标 $t$ 来强调时间索引，比如对于一个状态 $s_t$，其代表的是在时间 $t$ 时刻的状态，且一次状态转移会使得时间索引增加 1。

这样我们最大化的目标可以写为：

$$
V^{\pi}(d_0) \coloneqq \mathbb{E}_{s_0 \sim d_0} \mathbb{E}_{\xi_0 \sim \rho^\pi|s_0} \left[ \sum_{t=0}^{T-1} r(s_t, a_t) \right]
$$

这里面 $\rho^\pi(\xi_t \mid s_t)$ 表示在策略 $\pi$ 下，从时刻 $t$ 的状态 $s_t$ 开始直到问题结束所生成的轨迹 $\xi_t = s_t, a_t, \dots, s_{T-1}, a_{T-1}$ 的分布。

<!--
这是论文第二章（2. Background: Episodic Interactive Imitation Learning）的完整、专业翻译。
这一章至关重要，它不仅定义了数学符号，还引入了作者为了推导 MAMBA 算法而专门构造的**广义优势函数（Generalized Advantage Function）**概念，这是理解后续理论证明的核心前提。

2. 背景：情景式交互模仿学习
马尔可夫决策过程 (MDPs) 我们考虑具有状态空间 $\mathcal{S}$ 和动作空间 $\mathcal{A}$ 的有限视界（Finite-horizon）MDP。设 $T$、$d_0(s)$、$\mathcal{P}(s'|s,a)$ 和 $r: \mathcal{S} \times \mathcal{A} \to [0, 1]$ 分别表示问题的时间视界（Horizon）、初始状态分布、转移与动力学模型以及奖励函数 。我们假设 $d_0$、$\mathcal{P}$ 和 $r$ 是固定但未知的 。给定一类状态相关的策略 $\Pi$，我们的目标是找到一个策略 $\pi \in \Pi$，使其最大化相对于初始状态分布 $d_0$ 的 $T$ 步回报（Return）：
$$V^{\pi}(d_0) := \mathbb{E}_{s_0 \sim d_0} \mathbb{E}_{\xi_0 \sim \rho^\pi|s_0} \left[ \sum_{t=0}^{T-1} r(s_t, a_t) \right] \quad (1)$$
其中 $\rho^\pi(\xi_t|s_t)$ 表示通过运行策略 $\pi$ 从时刻 $t$ 的状态 $s_t$ 开始直到问题结束所生成的轨迹 $\xi_t = s_t, a_t, ..., s_{T-1}, a_{T-1}$ 的分布 。
为了紧凑地书写非平稳过程（Non-stationary processes），我们将状态空间 $\mathcal{S}$ 构造为 $\mathcal{S} = \bar{\mathcal{S}} \times \{0, \dots, T-1\}$，其中 $\bar{\mathcal{S}}$ 是某种基础状态空间；因此，$\mathcal{P}$ 和 $r$ 在 $\mathcal{S}$ 中可以是非平稳的 。我们允许 $\mathcal{S}$ 和 $\mathcal{A}$ 既可以是离散的也可以是连续的。我们使用下标 $t$ 来强调时间索引。当写成 $s_t$ 时，我们要么假设它处于时间 $t$，要么假设通过 $\mathcal{P}(s'|s,a)$ 从 $s$ 到 $s'$ 的每一次转移都会使时间索引增加 1 。
【解释与说明：扩充状态空间】
作者在这里使用了一个常见的数学技巧：将时间 $t$ 包含在状态定义中。
这意味着状态不仅仅是“车的位置”，而是“第 3 秒时车的位置”。
这样做的好处是，所有的策略和转移矩阵虽然本质上随时间变化，但在数学符号上可以写成不随时间变化的形式（Stationary），简化了推导公式。
状态分布与价值函数 我们用 $d^\pi_t$ 表示从 $d_0$ 开始运行策略 $\pi$ 所诱导出的在时刻 $t$ 的状态分布（即对于任何 $\pi$，都有 $d^\pi_0 = d_0$），并定义平均状态分布（Average State Distribution） 为 $d^\pi := \frac{1}{T} \sum_{t=0}^{T-1} d^\pi_t$ 。从 $d^\pi$ 中采样会返回一个 $s_t$，其中 $t$ 是均匀分布的 。因此，我们可以将 (1) 中策略的 $T$ 步回报重写为 $V^\pi(d_0) = T \mathbb{E}_{s \sim d^\pi} \mathbb{E}_{a \sim \pi|s} [r(s,a)]$ 。
稍微滥用一下符号，我们用 $V^\pi: \mathcal{S} \to \mathbb{R}$ 表示策略 $\pi$ 的价值函数，它满足 $V^\pi(d_0) = \mathbb{E}_{s \sim d_0}[V^\pi(s)]$ 。
给定一个满足 $f(s_T)=0$ 的函数 $f: \mathcal{S} \to \mathbb{R}$，我们定义关于 $f$ 的 Q 函数为 $Q^f(s,a) := r(s,a) + \mathbb{E}_{s' \sim \mathcal{P}|s,a}[f(s')]$，以及**关于 $f$ 的优势函数（Advantage Function）**为：
$$A^f(s,a) := Q^f(s,a) - f(s) = r(s,a) + \mathbb{E}_{s' \sim \mathcal{P}|s,a}[f(s')] - f(s) \quad (2)$$
当 $f = V^\pi$ 时，我们也记 $A^{V^\pi} =: A^\pi$ 和 $Q^{V^\pi} =: Q^\pi$，这些是策略 $\pi$ 的标准优势函数和 Q 函数 。我们将 $f$ 在某个策略 $\pi$ 下的优势函数记为 $A^f(s, \pi) := \mathbb{E}_{a \sim \pi|s}[A^f(s,a)]$，类似地有 $Q^f(s, \pi)$，以及给定状态分布 $d$ 时的 $f(d) := \mathbb{E}_{s \sim d}[f(s)]$ 。我们将索引 Q 或 A 函数的函数 $f$ 称为基线价值函数（Baseline Value Functions），因为我们的目标是改进它们在每个状态下所提供的价值 。
【关键点解析：广义优势函数 $A^f$】
这是本论文理论部分最关键的定义之一。
通常我们说的优势函数是 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$，这是相对于策略自身价值的优势。
这里作者定义了一个相对于任意函数 $f$ 的优势函数 $A^f$。
目的：后续作者会将 $f$ 替换为 $f^{max}$（多专家最大价值包络）。如果 $A^{f^{max}}(s, \pi) > 0$，就意味着策略 $\pi$ 比这个“最大包络”还要好。
定义 1. 如果 $A^f(s, \pi) \ge 0, \forall s \in \mathcal{S}$，我们称基线价值函数 $f$ 相对于 $\pi$ 是可改进的（Improvable） 。
多专家策略的策略优化 上述设定描述了一个通用的情景式 RL 问题，其中智能体面临着执行策略性探索和长期信用分配（Credit Assignment）的需求 。在实践中，规避探索挑战的一种常见方法是利用专家策略 。
在本文中，我们假设在训练期间可以访问多个（可能均为次优的）专家策略，并利用情景式交互模仿学习（Episodic Interactive IL）来改进它们 。我们假设学习者（即智能体）可以访问一组专家策略 $\Pi^e = \{\pi^k\}_{k \in [K]}$ 。
在训练期间，学习者可以在滚入-滚出（Roll-in-Roll-out, RIRO）范式下与专家交互以收集演示 。在每个回合（Episode）中，学习者从 $d_0$ 采样的初始状态开始，运行其策略 $\pi \in \Pi$ 直到切换时间 $t_e \in [0, T-1]$；然后学习者请求一个专家策略 $\pi^k \in \Pi^e$ 接管并完成剩余的轨迹 。最后，学习者记录整个轨迹，包括奖励信息 。
需要注意的是，我们不假设能观测到专家的动作（即不需要看到专家在每个状态下的具体动作概率分布，只需要看到执行后的结果轨迹）。此外，由于此时可以获得采样的奖励，学习者有潜力改进专家策略 。
【解释与说明：RIRO 范式】
RIRO 是交互式模仿学习的标准操作，用于解决**分布偏移（Distribution Shift）**问题。
Roll-in（滚入）：由学习者（Learner）自己走前半程。这确保了收集到的数据分布包含了学习者可能犯错的状态，而不是只包含专家走过的完美状态。
Switch（切换）：在 $t_e$ 时刻，控制权移交给专家。
Roll-out（滚出）：由专家（Oracle）走完后半程。这提供了从当前（可能是糟糕的）状态恢复并完成任务的“示范”价值（Value），即 $Q$ 值估计。
物理意义：想象学骑自行车，你自己先歪歪扭扭骑一段（Roll-in），快摔倒时教练扶住车把帮你骑到终点（Roll-out）。通过这个过程，你不仅知道正常怎么骑，还知道快摔倒时如果表现得好（后续由教练接管得到的奖励高），当下的状态其实是有救的。





-->
