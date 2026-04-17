---
title: "MAMBA"
headline: "Policy Improvement via Imitation of Multiple Oracles"
visibility: "public"
status: "drafting"
description: "Paper note on Policy Improvement via Imitation of Multiple Oracles."
---

> [!abstract] Contributions
>
> 本文研究多 Oracle 交互式模仿学习/Interactive Imitation Learning 问题：当学习者面对多个次优黑盒专家策略时，如何系统性地整合它们的领域知识以学到更好的策略。核心贡献有三：（1）提出最大聚合基线/Max-aggregated Baseline $f^{\max}(s) = \max_{k} V^k(s)$，即在每个状态取所有 Oracle 价值函数的逐状态最大值，作为多 Oracle IL 的自然性能基准；（2）设计 MAMBA（Max-Aggregation of Multiple BAselines）算法，通过将策略优化规约到在线学习/Online Learning，并引入类似广义优势估计/Generalized Advantage Estimation/GAE 的梯度估计器，实现对 $f^{\max}$ 的竞争性策略改进；（3）引入超参数 $\lambda$ 在单步策略改进（纯 IL）和完整回报优化（纯 RL）之间插值，理论证明更大的 $\lambda$ 允许更大的策略改进幅度但增加方差，并提供了基于遗憾/Regret 的性能保证。实验在四个连续控制任务上表明 MAMBA 能有效利用多个弱 Oracle 来加速策略优化，同时在单 Oracle 情形下也优于现有 IL 基线 AggreVaTeD。
>
> 本文的理论和算法均假设 Oracle 的价值函数可以通过 Monte-Carlo Roll-out 被合理近似，但在多 Oracle 情形下无法获得 $f^{\max}$ 的无偏估计（因为不知道每个状态下哪个 Oracle 最优），必须依赖函数近似器。当 Oracle 数量较多或状态空间维度较高时，学习多个价值函数近似器的开销显著增加，论文实验中也观察到了这一可扩展性瓶颈。此外，Oracle 选择采用均匀随机策略，未利用已有信息主动选择更优的 Oracle，这一问题在后续工作 MAPS 中被改进。

## 1. Introduction

强化学习/Reinforcement Learning/RL 在机器人、推荐系统等领域有广泛应用前景，但其核心瓶颈在于需要大量试错探索才能发现好的策略。模仿学习/Imitation Learning/IL 通过引入 Oracle 策略（专家演示）来引导探索，可以显著降低样本复杂度。在单 Oracle 设定下，Oracle 的回报提供了一个天然的性能基准——学习者的目标是匹配或超越这个基准，大多数现有 IL 技术都建立在这一假设之上。

然而在实际应用中，获得单个高质量专家策略往往很困难。更常见的情况是：存在多个次优的专家策略，每个在不同场景下各有优势。例如在负载均衡任务中，有多种人工设计的启发式策略；在自动驾驶中，PID 控制器和人类驾驶员各有所长。此时学习者面临的核心挑战是：来自不同 Oracle 的演示可能相互冲突，没有任何单个 Oracle 在所有状态下都优于其他 Oracle，且每个 Oracle 的质量事先未知。

虽然 InfoGAIL、AC-Teach、OIL 等工作已经开始研究多 Oracle 学习，但它们都回避了两个根本性问题：

- **性能基准**：在多 Oracle 设定下，什么是衡量学习者性能的合理基准？在单 Oracle IL 中，这个基准就是 Oracle 的回报，但多 Oracle 情形下并不存在一个显然的类比。
- **系统性整合**：是否存在一种有原则的方法，将多个次优 Oracle 缝合/Stitch 成一个更强的基线，并允许在此基础上进一步改进？

本文对这两个问题给出了肯定的回答。第一，作者提出 $f^{\max}(s) = \max_k V^k(s)$ 作为多 Oracle IL 的自然基准，它在每个状态下都不低于任何单个 Oracle 的价值。第二，作者设计了 MAMBA 算法，通过 Roll-in/Roll-out 交互范式收集数据，利用 GAE 风格的梯度估计器进行策略优化，并证明了基于遗憾的理论保证。MAMBA 实质上是单 Oracle IL 算法 AggreVaTeD 到多 Oracle 情形的推广，同时通过 $\lambda$ 参数在 IL 和 RL 之间建立了连续过渡。

## 2. Problem Setup

### 2.1 Episodic MDP

考虑有限时域马尔可夫决策过程/Markov Decision Process/MDP，其状态空间为 $\mathcal{S}$，动作空间为 $\mathcal{A}$，初始状态分布为 $d_0(s)$，转移概率为 $\mathcal{P}(s' \mid s, a)$，奖励函数为 $r: \mathcal{S} \times \mathcal{A} \to [0, 1]$，时间视界为 $T$。$d_0$、$\mathcal{P}$ 和 $r$ 均固定但未知。为了紧凑地处理非平稳性，将状态空间构造为扩充状态空间 $\mathcal{S} = \bar{\mathcal{S}} \times \{0, \dots, T-1\}$，其中 $\bar{\mathcal{S}}$ 为基础状态空间。这一技巧将时间信息编码进状态，使得转移和奖励可以统一用不含时间下标的形式书写。

给定策略类 $\Pi$，目标是找到 $\pi \in \Pi$ 最大化 $T$ 步回报：

$$
V^{\pi}(d_0) \coloneqq \mathbb{E}_{s_0 \sim d_0} \mathbb{E}_{\xi_0 \sim \rho^\pi \mid s_0} \left[ \sum_{t=0}^{T-1} r(s_t, a_t) \right], \tag{1}
$$

其中 $\rho^\pi(\xi_t \mid s_t)$ 为从 $s_t$ 出发按 $\pi$ 生成的轨迹 $\xi_t = s_t, a_t, \dots, s_{T-1}, a_{T-1}$ 的分布。

### 2.2 State Distribution and Value Functions

记 $d_t^\pi$ 为从 $d_0$ 出发运行 $\pi$ 在时刻 $t$ 的状态分布，定义平均状态分布/Average State Distribution $d^\pi \coloneqq \frac{1}{T} \sum_{t=0}^{T-1} d_t^\pi$，则 $V^\pi(d_0) = T \mathbb{E}_{s \sim d^\pi} \mathbb{E}_{a \sim \pi \mid s} [r(s,a)]$。

本文引入了一个关键的推广：给定满足 $f(s_T) = 0$ 的任意函数 $f: \mathcal{S} \to \mathbb{R}$，定义**关于 $f$ 的广义优势函数/Generalized Advantage Function**：

$$
A^f(s,a) \coloneqq r(s,a) + \mathbb{E}_{s' \sim \mathcal{P} \mid s,a}[f(s')] - f(s). \tag{2}
$$

当 $f = V^\pi$ 时退化为标准优势函数 $A^\pi$。$f$ 在策略 $\pi$ 下的期望优势记为 $A^f(s, \pi) \coloneqq \mathbb{E}_{a \sim \pi \mid s}[A^f(s,a)]$。

> [!tip] 广义优势函数的设计意图
> 标准优势函数 $A^\pi(s,a)$ 衡量的是动作 $a$ 相对于策略 $\pi$ 自身价值的优势。而 $A^f(s,a)$ 衡量的是动作 $a$ 相对于**任意基线价值函数** $f$ 的优势。后续作者会将 $f$ 替换为 $f^{\max}$（多 Oracle 最大价值包络），这样 $A^{f^{\max}}(s, \pi) > 0$ 就意味着策略 $\pi$ 在状态 $s$ 下的表现优于所有 Oracle 的"最优拼接"。

> [!note] Definition 1 (可改进性/Improvability)
> 若 $A^f(s, \pi) \geq 0, \, \forall s \in \mathcal{S}$，则称基线价值函数 $f$ 相对于策略 $\pi$ 是**可改进的/Improvable**。

### 2.3 Interactive Imitation Learning with Multiple Oracles

学习者可访问一组 Oracle 策略 $\Pi^e = \{\pi^k\}_{k \in [K]}$，在训练期间通过滚入-滚出/Roll-in-Roll-out/RIRO 范式与 Oracle 交互：每个回合中，学习者从 $d_0$ 采样的初始状态出发，先运行自己的策略 $\pi$ 至随机切换时间 $t_e \in [0, T-1]$（Roll-in），然后请求某个 Oracle $\pi^k$ 接管并完成剩余轨迹（Roll-out），最后记录完整轨迹及奖励。

这一范式有两个重要特点：（1）不需要观测 Oracle 的动作分布，只需看到执行后的轨迹和奖励；（2）由于可以获得奖励信号，学习者有潜力超越所有 Oracle。

> [!tip] RIRO 范式的作用
> Roll-in 阶段使用学习者自身策略，确保收集到的数据覆盖学习者实际会访问的状态分布，缓解了行为克隆/Behavior Cloning 中的分布偏移/Distribution Shift 问题。Roll-out 阶段由 Oracle 接管，提供了从当前状态出发的价值估计（$Q$ 值），作为策略改进的信号。

## 3. Algorithm

### 3.1 Conceptual Framework: Max-aggregation with Policy Improvement

在设计实际算法之前，本文首先建立了一个概念框架，回答"多 Oracle 的合理基准是什么"这一核心问题。

**朴素方案及其失败**。一种直接的做法是将多个 Oracle 合并为一个（如固定权重混合或动作概率相乘），但前者在存在差 Oracle 时表现很差，后者无法处理确定性策略。另一种做法是对每个 Oracle 分别运行单 Oracle IL 算法后选择最优结果，但附录 A 给出了反例：两个回报相同的 Oracle，选择其中任一个都是次优的，而在它们之间切换可以获得更高回报。

**Max-following 策略 $\pi^\bullet$**。如果已知每个 Oracle 的价值函数 $V^k \coloneqq V^{\pi^k}$，一个自然的想法是在每个状态跟随价值最高的 Oracle：

$$
\pi^\bullet(a \mid s) \coloneqq \pi^{k_s}(a \mid s), \quad k_s \coloneqq \arg\max_{k \in [K]} V^k(s). \tag{3}
$$

可以证明 $V^{\pi^\bullet}(s) \geq \max_{k \in [K]} V^k(s)$，即 $\pi^\bullet$ 在每个状态都不差于最优 Oracle。但 $\pi^\bullet$ 有一个退化问题：当某个 Oracle 在所有状态都占优时，$\pi^\bullet$ 退化为该 Oracle 本身，而我们已知通过单步策略改进可以构造更好的策略。

**Max-aggregation 策略 $\pi^{\max}$**。为克服 $\pi^\bullet$ 的退化，将逐状态最大值 $f^{\max}$ 与标准策略改进算子结合：

$$
f^{\max}(s) \coloneqq \max_{k \in [K]} V^k(s), \tag{4}
$$

$$
\pi^{\max}(a \mid s) \coloneqq \delta_{a = a_s}, \quad a_s \coloneqq \arg\max_{a \in \mathcal{A}} A^{\max}(s, a), \tag{5}
$$

其中 $A^{\max} \coloneqq A^{f^{\max}}$ 为关于 $f^{\max}$ 的广义优势函数。与 $\pi^\bullet$ 不同，$\pi^{\max}$ 不是简单地跟随某个 Oracle，而是利用所有 Oracle 的"价值包络" $f^{\max}$ 作为后继状态价值的估计，前瞻一步选择最优动作。

> [!note] Proposition 1
> $f^{\max}$ 相对于 $\pi^\bullet$ 是可改进的，即 $A^{\max}(s, \pi^\bullet) \geq 0, \, \forall s \in \mathcal{S}$。

> [!todo]- Proof of Proposition 1
> 不失一般性设 $k_s = 1$（状态 $s$ 的最优 Oracle 为 $\pi^1$），则
>
> $$
> A^{\max}(s, \pi^\bullet) = r(s, \pi^\bullet) + \mathbb{E}_{s' \sim \mathcal{P} \mid s, \pi^\bullet}[f^{\max}(s')] - f^{\max}(s) \geq r(s, \pi^1) + \mathbb{E}_{s'}[V^1(s')] - V^1(s) = A^{V^1}(s, \pi^1) = 0,
> $$
>
> 其中最后一步利用了 $\pi^\bullet(a \mid s) = \pi^1(a \mid s)$ 以及 $f^{\max}(s') \geq V^1(s')$。$A^{V^1}(s, \pi^1) = 0$ 是因为任何策略相对于自身价值函数的优势恒为零。

结合 Proposition 1 与下面的 Performance Difference Lemma，可以得到 $\pi^{\max}$ 也是一个有效的基准。

> [!note] Lemma 1 (Performance Difference Lemma)
> 设 $f: \mathcal{S} \to \mathbb{R}$ 满足 $f(s_T) = 0$，则对任意 MDP 和策略 $\pi$，
>
> $$
> V^\pi(d_0) - f(d_0) = T \mathbb{E}_{s \sim d^\pi}[A^f(s, \pi)]. \tag{6}
> $$

> [!note] Corollary 1
> 若 $f$ 相对于 $\pi$ 是可改进的，则 $V^\pi(s) \geq f(s), \, \forall s \in \mathcal{S}$。

由 Proposition 1 和 Corollary 1 立即得到 $V^{\pi^{\max}}(s) \geq f^{\max}(s) \geq V^k(s)$ 对所有 $k$ 和 $s$ 成立。因此 $\pi^{\max}$ 在每个状态下都至少与所有 Oracle 一样好，是多 Oracle IL 的合理基准。值得注意的是，$\pi^{\max}$ 与 $\pi^\bullet$ 的价值一般不可比——附录 A 给出了两者各自更优的反例。作者选择 $\pi^{\max}$ 作为基准而非 $\pi^\bullet$，是因为 $\pi^{\max}$ 不需要观测 Oracle 的动作，与交互式 IL 的设定一致。

### 3.2 Reduction to Online Learning

上一节的分析假设了对 MDP 和 Oracle 价值函数的完全知识。本节将算法设计规约为在线学习问题，以处理这些量未知的现实情况。

**在线损失定义**。记第 $n$ 轮学习者策略为 $\pi_n$，将 $d^{\pi_n}$ 视为对手选择的状态分布，定义在线损失：

$$
\ell_n(\pi) \coloneqq -T \mathbb{E}_{s \sim d^{\pi_n}}[A^{\max}(s, \pi)]. \tag{7}
$$

由 Lemma 1，使 $\ell_n(\pi_n)$ 较小等价于使 $V^{\pi_n}(d_0)$ 不低于 $f^{\max}(d_0)$。对 $N$ 轮在线学习取平均，利用 Regret 的定义可以得到：

$$
\frac{1}{N} \sum_{n \in [N]} V^{\pi_n}(d_0) = f^{\max}(d_0) + \Delta_N - \epsilon_N(\Pi) - \frac{\text{Regret}_N}{N}, \tag{8}
$$

其中 $\text{Regret}_N \coloneqq \sum_{n=1}^N \ell_n(\pi_n) - \min_{\pi \in \Pi} \sum_{n=1}^N \ell_n(\pi)$ 为在线学习的遗憾，$\Delta_N \coloneqq -\frac{1}{N} \sum_{n=1}^N \ell_n(\pi^{\max}) \geq 0$ 反映 $\pi^{\max}$ 本身的质量，$\epsilon_N(\Pi) \geq 0$ 反映策略类 $\Pi$ 的逼近误差。当 $\pi^{\max} \in \Pi$ 时 $\epsilon_N(\Pi) = 0$。

因此，只要采用无悔/No-regret 算法（$\text{Regret}_N = o(N)$），就能保证产生至少与 $f^{\max}$ 竞争的策略。这一规约将多 Oracle IL 问题等价转化为了一个在线优化问题。

**近似 Oracle 价值的影响**。在实践中 $f^{\max}$ 不可直接获取，需要用近似 $\hat{f}^{\max}(s) = \max_{k \in [K]} \hat{V}^k(s)$ 替代。对 $\ell_n(\pi)$ 的梯度可以写为：

$$
\nabla \hat{\ell}_n(\pi_n) = -T \mathbb{E}_{s \sim d^{\pi_n}} \mathbb{E}_{a \sim \pi \mid s} \left[ \nabla \log \pi(a \mid s) \hat{A}(s, a) \right] \Big|_{\pi = \pi_n}, \tag{9}
$$

其中 $\hat{A}(s,a) = r(s,a) + \mathbb{E}_{s'}[\hat{f}^{\max}(s')] - \hat{f}^{\max}(s)$。对于**单 Oracle** 情形，$\hat{f}^{\max}$ 可以用 Monte-Carlo 估计无偏地获得。但对于**多 Oracle** 情形，由于不知道每个状态下哪个 Oracle 最优，无法获得 $f^{\max}$ 的无偏 Monte-Carlo 估计——$\hat{f}^{\max}$ 必须是一个函数近似器。

> [!note] Theorem 1 (性能保证)
> 假设一阶在线算法的期望遗憾满足 $\mathbb{E}[\text{Regret}_N] \leq O(\beta N + \sqrt{\nu N})$，其中 $\beta$ 和 $\nu$ 分别为梯度估计的偏差和方差，则
>
> $$
> \mathbb{E}\left[\max_{n \in [N]} V^{\pi_n}(d_0)\right] \geq \mathbb{E}_{s \sim d_0}\left[\max_{k \in [K]} V^k(s)\right] + \mathbb{E}[\Delta_N - \epsilon_N(\Pi)] - O\left(\beta + \sqrt{\nu} N^{-1/2}\right). \tag{10}
> $$

Theorem 1 表明：只要梯度估计的偏差 $\beta$ 和方差 $\nu$ 足够小，MAMBA 产生的最优策略至少与 $f^{\max}$ 竞争。偏差 $\beta$ 来源于 $\hat{f}^{\max}$ 对 $f^{\max}$ 的近似误差，方差 $\nu$ 来源于 RIRO 数据收集的采样方差。

### 3.3 MAMBA: $\lambda$-weighted Online Loss

直接使用 $\ell_n(\pi)$ 的梯度估计式 (9) 面临一个实际困难：由于 RIRO 协议的限制，$\nabla \hat{\ell}_n$ 的方差比标准 Monte-Carlo 策略梯度高 $O(T)$ 倍。为解决这一问题，MAMBA 引入了一个基于几何加权/Geometric Weighting 的替代在线损失：

$$
\ell_n(\pi; \lambda) \coloneqq -(1-\lambda) T \mathbb{E}_{s \sim d^{\pi_n}}[A_\lambda^{\max, \pi}(s, \pi)] - \lambda \mathbb{E}_{s \sim d_0}[A_\lambda^{\max, \pi}(s, \pi)], \tag{11}
$$

其中 $\lambda \in [0, 1]$，$A_\lambda^{\max, \pi}$ 为 $\lambda$-加权优势函数/Lambda-weighted Advantage：

$$
A_\lambda^{\max, \pi}(s, a) \coloneqq (1-\lambda) \sum_{i=0}^{\infty} \lambda^i A_{(i)}^{\max, \pi}(s, a), \tag{12}
$$

$$
A_{(i)}^{\max, \pi}(s_t, a_t) \coloneqq \mathbb{E}_{\xi_i \sim \rho^\pi \mid s_t}\left[\sum_{j=0}^{i} r(s_{t+j}, a_{t+j}) + f^{\max}(s_{t+i+1})\right] - f^{\max}(s_t). \tag{13}
$$

这里 $A_{(i)}^{\max, \pi}$ 是 $i$ 步优势：执行 $i+1$ 步实际动作后用 $f^{\max}$ 估计剩余价值，与当前状态的 $f^{\max}$ 之差。$\lambda$-加权优势是对不同步长优势的指数加权平均。

**$\lambda$ 的作用及其与 IL/RL 的关系**：

- 当 $\lambda = 0$ 时，$A_\lambda^{\max, \pi} = A^{\max}$（单步优势），$\ell_n(\pi; 0) = \ell_n(\pi)$，退化为纯 IL 损失——仅基于 $f^{\max}$ 做单步策略改进。
- 当 $\lambda = 1$ 时，$A_\lambda^{\max, \pi} = Q^\pi - f^{\max}$，可以证明 $\ell_n(\pi; 1) = f^{\max}(d_0) - V^\pi(d_0)$，即标准 RL 目标——直接最大化累积回报。
- 当 $0 < \lambda < 1$ 时，在 IL 和 RL 之间插值。

> [!note] Lemma 2 (广义 Performance Difference Lemma)
> 对任意策略 $\pi$、任意 $\lambda \in [0, 1]$、任意基线价值函数 $f$，
>
> $$
> V^\pi(d_0) - f(d_0) = (1-\lambda) T \mathbb{E}_{s \sim d^\pi}\left[A_\lambda^{f, \pi}(s, \pi)\right] + \lambda \mathbb{E}_{s \sim d_0}\left[A_\lambda^{f, \pi}(s, \pi)\right]. \tag{14}
> $$

> [!todo]- Lemma 2 的推导
> 推导基于一个非均匀的 Performance Difference 分解。定义 $\Theta = V^\pi(d_0) - f(d_0)$，$A_{(i)} = A_{(i)}^{f, \pi}$。利用 Lemma 5（Non-even Performance Difference Lemma，将轨迹分段后应用 Lemma 1），可以将 $\Theta$ 写为不同步长优势的组合：
>
> $$
> \Theta = \sum_{t=0}^{T-1} \mathbb{E}_{d_t^\pi} \mathbb{E}_\pi [A_{(0)}],
> $$
>
> $$
> 2\Theta = \mathbb{E}_{d_0} \mathbb{E}_\pi [A_{(1)}] + \sum_{t=0}^{T-1} \mathbb{E}_{d_t^\pi} \mathbb{E}_\pi [A_{(1)}],
> $$
>
> 一般地，$(k+1)\Theta$ 可以表示为 $A_{(k)}$ 在 $d_0$ 和 $d_t^\pi$ 上的期望之和。对这些等式施加 $\lambda$-加权（系数为 $(1-\lambda)(1 + 2\lambda + 3\lambda^2 + \cdots) = \frac{1}{1-\lambda}$），并利用幂级数 $\lambda + 2\lambda^2 + 3\lambda^3 + \cdots = \frac{\lambda}{(1-\lambda)^2}$，化简后得到 Lemma 2 的结论。

Lemma 2 保证了 $\ell_n(\pi; \lambda)$ 与 $V^{\pi_n}(d_0) - f^{\max}(d_0)$ 之间的精确关系，因此对 $\ell_n(\pi_n; \lambda)$ 进行在线优化同样可以通过式 (8) 的类比建立性能保证。

> [!note] Theorem 2
> 对 $\ell_n(\pi; \lambda)$ 执行无悔在线学习，其性能保证与 Theorem 1 相同，但 $\epsilon_N(\Pi)$ 现在可以为负值（当 $\lambda > 0$ 时），这意味着可能获得更大的改进幅度。

Theorem 2 的关键含义是：$\pi^{\max}$ 可能不是多步优势 $A_\lambda^{\max, \pi}$ 意义下的最优策略（当 $\lambda > 0$ 时），因此 $\epsilon_N(\Pi) < 0$ 是可能的，这反而为学习者提供了超越 $f^{\max}$ 基准的空间。直觉上，更大的 $\lambda$ 使得算法考虑更多步的未来回报，相当于在 IL 的基础上融入了 RL 的优化能力，能够实现更大的策略改进。但代价是梯度方差 $\nu$ 也会增加，收敛更慢。

### 3.4 Gradient Estimation

尽管 $\ell_n(\pi; \lambda)$ 的表达式看起来复杂，但其梯度有一个简洁的形式。

> [!note] Lemma 3 (梯度表达式)
> 对任意 $\lambda \in [0, 1]$、任意基线价值函数 $f$ 和策略 $\pi$，定义
>
> $$
> h(\pi; \lambda) \coloneqq (1-\lambda) T \mathbb{E}_{s \sim d^\mu}\left[A_\lambda^{f, \pi}(s, \pi)\right] + \lambda \mathbb{E}_{s \sim d_0}\left[A_\lambda^{f, \pi}(s, \pi)\right],
> $$
>
> 则
>
> $$
> \nabla h(\pi; \lambda) \big|_{\mu = \pi} = T \mathbb{E}_{s \sim d^\pi} \mathbb{E}_{a \sim \pi \mid s}\left[\nabla \log \pi(a \mid s) A_\lambda^{f, \pi}(s, a)\right]. \tag{15}
> $$

利用 Lemma 3，以 $\hat{f}^{\max}$ 近似 $f^{\max}$，MAMBA 的梯度估计器为：

$$
\nabla \hat{\ell}_n(\pi_n; \lambda) = -T \mathbb{E}_{s \sim d^{\pi_n}} \mathbb{E}_{a \sim \pi \mid s}\left[\nabla \log \pi(a \mid s) \hat{A}_\lambda^{\pi}(s, a)\right] \Big|_{\pi = \pi_n}, \tag{16}
$$

其中 $\hat{A}_\lambda^\pi$ 是用 $\hat{f}^{\max}$ 替换 $f^{\max}$ 后的 $\lambda$-加权优势。

> [!note] Lemma 4 (GAE 风格的优势展开)
> 定义 $\hat{A}(s, a) \coloneqq r(s, a) + \mathbb{E}_{s' \mid s, a}[\hat{f}^{\max}(s')] - \hat{f}^{\max}(s)$，则
>
> $$
> \hat{A}_\lambda^\pi(s_t, a_t) = \mathbb{E}_{\xi_t \sim \rho^\pi \mid s_t}\left[\sum_{\tau=t}^{T-1} \lambda^{\tau-t} \hat{A}(a_\tau, s_\tau)\right]. \tag{17}
> $$

Lemma 4 表明 $\hat{A}_\lambda^\pi$ 可以表示为单步 TD 残差 $\hat{A}(s_\tau, a_\tau)$ 的 $\lambda$-折扣累积和，这与 RL 中 GAE 的形式完全一致。关键优势在于：给定 $\hat{f}^{\max}$ 作为函数近似器，式 (16) 的无偏采样估计只需要从 $\pi$ 采样完整轨迹，**不需要** RIRO 交互。RIRO 交互仅用于更新 $\hat{V}^k$（从而更新 $\hat{f}^{\max}$），而梯度计算使用独立的 on-policy 轨迹。

$\lambda$ 在这一梯度估计中扮演双重角色：（1）控制 IL/RL 的权衡（如 3.3 节所述）；（2）控制偏差-方差权衡——$\lambda$ 越大，对 $\hat{f}^{\max}$ 近似误差的依赖越小（因为更多使用实际奖励而非价值估计），但轨迹方差越大。当 $\hat{f}^{\max} = f^{\max}$ 时，所有 $\lambda$ 值的梯度都相等；仅在存在近似误差时 $\lambda$ 的选择才影响结果。

### 3.5 Complete Algorithm

综合上述分析，MAMBA 的完整流程如下：

> [!note] Algorithm 1: MAMBA
> **输入**：初始策略 $\pi_1$，Oracle 集 $\{\pi^k\}_{k \in [K]}$，价值函数近似器 $\{\hat{V}^k\}_{k \in [K]}$
>
> **for** $n = 1, \dots, N-1$ **do**
> 1. 均匀采样切换时间 $t_e \in [T-1]$ 和 Oracle 索引 $k \in [K]$
> 2. Roll-in $\pi_n$ 至 $t_e$，切换到 $\pi^k$ 完成轨迹，收集数据 $\mathcal{D}_n$
> 3. 用 $\mathcal{D}_n$ 更新 $\hat{V}^k$（Monte-Carlo 回归）
> 4. Roll-in $\pi_n$ 完整 $T$ 步，收集数据 $\mathcal{D}_n'$
> 5. 用 $\mathcal{D}_n'$ 和 $\hat{f}^{\max}(s) = \max_k \hat{V}^k(s)$ 计算梯度 $g_n$（式 (16)）
> 6. 以 $g_n$ 更新 $\pi_n \to \pi_{n+1}$（一阶在线学习，如镜像下降）

每轮迭代包含两类 rollout：一半用于 RIRO 数据收集以更新 Oracle 价值估计，一半用于 on-policy 数据收集以计算策略梯度。实现中 $\hat{V}^k$ 通过加权最小二乘 Monte-Carlo 回归训练，策略使用高斯分布参数化（均值由神经网络给出），更新算法采用 ADAM 或自然梯度下降。

## 4. Experiments

### 4.1 Setup

实验在四个 OpenAI Gym 连续控制任务上进行：CartPole（$T = 1000$，$|\mathcal{S}| = 4$，$|\mathcal{A}| = 1$）、DoubleInvertedPendulum/DIP（$T = 1000$，$|\mathcal{S}| = 8$，$|\mathcal{A}| = 1$）、HalfCheetah（$T = 500$，$|\mathcal{S}| = 17$，$|\mathcal{A}| = 6$）和 Ant（$T = 500$，$|\mathcal{S}| = 111$，$|\mathcal{A}| = 8$）。每个环境中 Oracle 为部分训练的次优神经网络策略。

比较算法：（1）PG-GAE（$\lambda = 0.9$）：纯 RL，使用 GAE 策略梯度；（2）AggreVaTeD：单 Oracle IL 基线，等价于 MAMBA 在 $\lambda = 0$ 时的特例；（3）MAMBA（$\lambda = 0.9$）：本文方法。为公平比较，三者使用相同的一阶优化器、相同的神经网络架构和相同的交互预算。在 HalfCheetah 和 Ant 上，IL 算法用最优 Oracle 进行行为克隆初始化。

### 4.2 $\lambda$-weighting 的效果

在单 Oracle 的 CartPole 实验中（Figure 2），AggreVaTeD（即 $\lambda = 0$ 的 MAMBA）比 PG-GAE 收敛更快，但最终性能不如后者——这符合理论预期，因为 $\lambda = 0$ 仅做单步改进，无法发现全局最优策略。$\lambda = 0.9$ 时 MAMBA 兼具了 IL 的快速收敛和 RL 的高渐近性能。中间值 $\lambda = 0.5$ 的表现反而最差，可能是因为其处于偏差-方差权衡的不利区域。

### 4.3 Multiple Oracles 的效果

<!-- TODO: insert Figure 3 from paper — Comparison of MAMBA with different number of oracles -->

在 DIP、HalfCheetah 和 Ant 上（Figure 3），MAMBA（$\lambda = 0.9$）使用不同数量的 Oracle（1, 2, 4, 8 个）进行测试。一个值得注意的现象是：即使引入**更弱**的 Oracle，MAMBA 的性能仍然得到提升。这验证了 $f^{\max}$ 基线的设计：即使新增的 Oracle 整体较差，它在某些状态下仍可能优于现有 Oracle，从而丰富价值包络。

MAMBA 在所有环境上都显著优于 PG-GAE 和 AggreVaTeD。但随着状态维度增加（从 DIP 到 Ant），多 Oracle 的边际收益递减——在 Ant 环境中，使用 8 个 Oracle 相比 2 个 Oracle 的提升不如在 DIP 中明显。作者将此归因于高维状态空间下学习多个价值近似器的困难：每个 $\hat{V}^k$ 独立训练且使用 Monte-Carlo 估计，当 Oracle 数量增加时，同等交互预算下每个 $\hat{V}^k$ 获得的训练数据减少，近似质量下降。

### 4.4 Additional Findings

附录 D 中的补充实验揭示了以下要点：

- **AggreVaTeD 的脆弱性**：在单 Oracle 设定下，AggreVaTeD 从不同 Oracle 出发的最终性能差异很大，与事后最优的 Oracle 选择结果不一致。MAMBA（$\lambda > 0$）对 Oracle 选择更鲁棒，可能是因为多步优势使算法能够超越 Oracle 本身的限制。
- **Oracle 排序的鲁棒性**：将 Oracle 随机排序（而非按回报降序）后，单 Oracle 设定下性能显著下降（因为可能选中极差的 Oracle），但使用多 Oracle 后 MAMBA 仍能恢复良好性能。

### 4.5 Critical Analysis

- **Oracle 选择策略**：MAMBA 采用均匀随机选择 Oracle 和切换时间，未利用已有信息进行自适应选择。在 Oracle 数量较多时，大量样本被浪费在已确认为次优的 Oracle 上。这一问题在后续工作 MAPS 中通过 UCB 策略得到改进。
- **可扩展性**：每个 Oracle 需要独立维护一个价值函数近似器，计算和存储开销随 $K$ 线性增长。实验中观察到的高维环境下边际收益递减印证了这一瓶颈，论文建议通过 off-policy 技术和参数共享来改善。
- **$\lambda$ 的选择**：论文在所有环境上统一使用 $\lambda = 0.9$，未提供系统的调优指导。$\lambda$ 的最优值可能随环境和 Oracle 质量变化。
- **理论与实践的差距**：理论保证基于在线凸优化框架，但实际使用的神经网络策略类是非凸的。实验虽然验证了算法的实际有效性，但理论保证的实际松紧程度不明。

## 5. Related Work & Future Work

### Related Work

**单 Oracle 交互式 IL**：DAgger 开创了在线收集专家数据纠正分布偏移的范式。AggreVaTe 引入了 cost-to-go 概念，AggreVaTeD 进一步推广为可微分的策略梯度形式。MAMBA 直接推广了 AggreVaTeD：当只有一个 Oracle 且 $\lambda = 0$ 时，MAMBA 的在线损失与 AggreVaTeD 相同；当 $\lambda > 0$ 时，MAMBA 额外利用了多步优势信息。

**多 Oracle 学习**：InfoGAIL 通过隐变量建模不同 Oracle 的演示风格，AC-Teach 用贝叶斯方法对 Oracle 属性建模，OIL 在每个状态选择最优 Oracle 的策略值但缺乏理论分析。MAMBA 与这些方法的根本区别在于：它不试图模仿任何单个 Oracle 或它们的混合，而是通过价值函数的逐状态最大值构建更强的基线，并在此基础上进行策略改进。

**GAE 与策略梯度**：MAMBA 的梯度估计器在形式上与 GAE 策略梯度一致，但两者优化的是**不同的**目标函数——GAE 策略梯度是 $-\nabla V^\pi(d_0)$ 的近似，而 MAMBA 的梯度是 $\nabla \ell_n(\pi; \lambda)$ 的近似。仅在单 Oracle 且 $\lambda = 1$ 的特殊情况下两者才完全等价。

### Future Work

**论文明确提出的方向**：

> 可以通过 off-policy 技术和跨 Oracle 的参数共享来改善多 Oracle 价值函数估计的可扩展性。

**从论文局限性延伸的方向**：

- 用自适应的 Oracle 选择策略替代均匀随机选择，例如基于 UCB 的主动选择（这正是后续工作 MAPS 的核心改进）。
- 研究自适应的切换时间选择，将 Oracle 查询集中在真正需要信息的状态上（后续 MAPS-SE 的 ASE 组件解决了这一问题）。
- 探索 $\lambda$ 的自适应调整策略，随训练进展从偏 IL（快速启动）过渡到偏 RL（精细优化）。
- 在更大规模、更高维度的任务上验证方法的有效性和可扩展性。
