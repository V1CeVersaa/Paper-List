---
title: "RPI"
headline: "Blending Imitation and Reinforcement Learning for Robust Policy Improvement"
visibility: "public"
status: "drafting"
description: "Paper note on Blending Imitation and Reinforcement Learning for Robust Policy Improvement."
---

> [!abstract] Contributions
>
> 本文提出了 max⁺ 框架及其实例化算法 Robust Policy Improvement/RPI，用于在拥有多个次优黑盒 oracle 的情况下进行鲁棒的策略改进。核心技术创新在于：(1) 构建扩展 oracle 集合/Extended Oracle Set，将学习者策略本身纳入 oracle 集合，从而使得基线价值函数 $f^+$ 能够同时反映 oracle 和学习者的最优状态级表现；(2) 提出 Robust Active Policy Selection/RAPS，利用 UCB/LCB 置信度感知机制高效选择最优 oracle 进行价值函数估计；(3) 提出 Robust Policy Gradient/RPG，基于新设计的 $A^{\text{GAE+}}$ 优势函数在 actor-critic 框架下执行策略梯度更新。RPI 在训练早期通过模仿 oracle 快速启动学习，随着学习者策略改善逐渐过渡到自我改进的强化学习模式。理论分析证明 RPI 的性能下界不弱于先前方法 MAMBA，实验在 DeepMind Control Suite 和 Meta-World 的 8 个任务上验证了 RPI 在稠密和稀疏奖励环境中均优于现有方法。
>
> 本文的关键假设在于：oracle 的价值函数需要通过有限的 rollout 数据进行估计，估计质量直接影响策略选择和梯度计算的准确性。此外，置信度阈值 $\Gamma_s$ 作为超参数需要调优，虽然实验表明 $\Gamma_s = 0.5$ 在大多数环境下表现良好，但在 Pendulum 等特定环境中并非最优。实验中 oracle 均由 PPO/SAC 在不同训练阶段保存的策略充当，尚未验证面对真正异质的、来源各异的 oracle 时方法的表现。

## 1. Introduction

强化学习/Reinforcement Learning/RL 在围棋、视频游戏等领域取得了超越人类的表现，但其高样本复杂度限制了在机器人控制、医疗等需要大量真实交互的领域中的应用。模仿学习/Imitation Learning/IL 通过利用 oracle 的示范来替代部分环境交互，从而提升样本效率，但其效果严重依赖于 oracle 的质量。传统的交互式 IL 方法（如 *DAgger*、*AggreVaTeD*）假设 oracle 是近最优的；当拥有奖励信号时，学习者有潜力超越 oracle 的表现，例如 *THOR* 利用近最优 oracle 进行代价塑形来优化相对于 oracle 价值函数的 $k$-步优势。

然而，现实场景中通常只能获得次优/suboptimal 且黑盒/black-box 的 oracle——它们可能在不同状态下表现各异，且无法提供量化的性能指标。*LOKI* 和 *TGRL* 尝试结合 IL 与 RL，但仅针对单 oracle 设定。*MAMBA* 和 *MAPS* 支持多 oracle 学习，但它们的基本假设是在任意给定状态下至少有一个 oracle 能提供最优动作。当所有 oracle 在某些状态下都表现不佳时，这些方法仍会尝试模仿最差的 oracle，从而阻碍策略提升。

本文的核心洞察是：应当自适应地融合 IL 与 RL——当存在优于学习者的 oracle 时进行模仿学习，当学习者已超越所有 oracle 时切换到自我改进的强化学习。这一思路的实现依赖于将学习者策略纳入 oracle 集合（构成扩展 oracle 集），并据此设计新的基线价值函数和策略梯度方法，使算法能够在状态级别上自动判断应当模仿还是自主探索。

## 2. Problem Setup

上述动机要求一个统一的形式化框架来描述学习者、oracle 集合、以及策略改进的目标。本文考虑有限时间步/finite-horizon 的马尔可夫决策过程/Markov Decision Process/MDP $\mathcal{M}_0 = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, r, H \rangle$，其中 $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$\mathcal{P}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ 为未知的随机转移动态，$r: \mathcal{S} \times \mathcal{A} \to [0, 1]$ 为未知奖励函数，$H$ 为每个 episode 的时间步长。训练总轮数为 $N$。

学习者可以访问一组（可能为空的）$K$ 个黑盒 oracle $\Pi = \{\pi^k\}_{k=1}^K$，其中 $\pi_k: \mathcal{S} \to \Delta(\mathcal{A})$。"黑盒"意味着学习者只能查询 oracle 在给定状态下的动作，但无法获得 oracle 的价值函数或内部参数。

给定初始状态分布 $d_0 \in \Delta(\mathcal{S})$，策略 $\pi$ 在时间步 $t$ 的状态访问分布为 $d_t^\pi$，平均状态访问分布为 $d^\pi := \frac{1}{H} \sum_{t=0}^{H-1} d_t^\pi$。策略 $\pi$ 在初始分布 $d_0$ 下的价值函数为：

$$
V^\pi(d_0) = \mathbb{E}_{s_0 \sim d_0}[V^\pi(s)] = \mathbb{E}_{s_0 \sim d_0}\left[\mathbb{E}_{\tau_t \sim \rho^\pi(\tau_t | s_t)}\left[\sum_{t=0}^{H-1} r(s_t, a_t)\right]\right]
$$

学习目标是找到策略 $\pi = \arg\max_\pi J(\pi)$，其中：

$$
J(\pi) = \mathbb{E}_{s \sim d_0}[V^\pi(s)] \tag{1}
$$

关于一般函数 $f: \mathcal{S} \to \mathbb{R}$ 的广义 Q 函数/generalized Q-function 和广义优势函数/generalized advantage function 定义为：

$$
Q^f(s, a) := r(s, a) + \mathbb{E}_{s' \sim \mathcal{P}|s,a}[f(s')]
$$

$$
A^f(s, a) = Q^f(s, a) - f(s) = r(s, a) + \mathbb{E}_{s' \sim \mathcal{P}|s,a}[f(s')] - f(s)
$$

当 $f(s) = V^\pi(s)$ 时，$Q^f$ 恢复为标准的 $Q^\pi(s, a)$。这些广义定义允许使用任意基线函数 $f$ 来衡量优势，这是后续 max⁺ 框架的基础。

## 3. Algorithm

在上述问题设定下，论文的核心挑战是：如何在不知道 MDP 转移动态和 oracle 价值函数的条件下，自适应地决定在每个状态上应当模仿 oracle 还是依赖学习者自身的经验进行强化学习。为此，论文首先在理想条件下建立 max⁺ 理论框架，然后将其放松到实际的黑盒在线学习设定，最终形成 RPI 算法。

### 3.1 Method Overview

RPI 的整体思路可以概括为：在每一轮训练中，学习者策略首先 roll-in（即按自身策略执行）到某个随机切换点，然后通过 RAPS 选择当前状态下最有前途的 oracle 进行 roll-out（即从该状态起按所选 oracle 策略执行），收集数据用于更新价值函数估计。同时，学习者策略也进行完整的 roll-in 以收集自身轨迹数据。最后，RPG 利用基于 $A^{\text{GAE+}}$ 优势函数的策略梯度对学习者策略进行更新。这一过程使得 RPI 在训练早期能够充分利用 oracle 的知识（因为此时 oracle 通常优于学习者），而随着学习者改善，逐渐转向自我驱动的强化学习。

<!-- TODO: insert Figure 1 from paper (Method Overview) -->

### 3.2 The max⁺ Framework: Policy Improvement with Perfect Knowledge

在介绍实际算法之前，论文首先在理想条件下（已知 MDP 和所有 oracle 的价值函数）建立了 max⁺ 框架的理论基础。

**扩展 oracle 集合。** 先前工作（如 MAMBA）的 max-following 策略会在每个状态下选择价值函数最高的 oracle 进行模仿。但这种策略有一个根本缺陷：当所有 oracle 在某些状态下都表现差于学习者时，它仍然会模仿最差的 oracle。为此，论文提出将学习者策略 $\pi_n$（第 $n$ 轮的策略）加入 oracle 集合，形成扩展 oracle 集合/extended oracle set：

$$
\Pi^\mathcal{E} = \Pi^o \cup \{\pi_n\} = \{\pi^1, \ldots, \pi^K, \pi_n\} \tag{2}
$$

**$A^+$ 优势函数与基线价值函数 $f^+$。** 基于扩展 oracle 集，定义 $A^+$ 优势函数：

$$
A^+(s, a) := r(s, a) + \mathbb{E}_{s' \sim \mathcal{P}|s,a}[f^+(s')] - f^+(s) \tag{3}
$$

其中基线价值函数 $f^+$ 取扩展 oracle 集中所有策略价值函数的逐状态最大值：

$$
f^+(s) = \max_{k \in [|\Pi^\mathcal{E}|]} V^k(s), \quad \text{where } [V^k]_{k \in [|\Pi^\mathcal{E}|]} := [V^{\pi^1}, \ldots, V^{\pi^K}, V^{\pi_n}] \tag{4}
$$

$f^+$ 的含义是：在每个状态 $s$ 上，取所有 oracle 和学习者中价值最高者的价值作为基线。这使得优势函数 $A^+$ 能够自动区分两种情况——当某个 oracle 在状态 $s$ 上优于学习者时，$f^+(s)$ 由该 oracle 的价值决定，此时学习者被引导去模仿；当学习者在状态 $s$ 上优于所有 oracle 时，$f^+(s) = V^{\pi_n}(s)$，此时 $A^+$ 退化为标准优势函数，学习者进行自我改进。

**max⁺-following 策略 $\pi^\circ$。** 定义 max⁺-following 策略为在每个状态下模仿扩展 oracle 集中价值最高的策略：

$$
\pi^\circ(a|s) := \pi^{k^\star}(a|s), \quad k^\star := \arg\max_{k \in [|\Pi^\mathcal{E}|]} V^k(s), \quad V^{K+1} = V^{\pi_n} \tag{5}
$$

> [!note] Proposition 4.5
> $\pi^\circ$ 的表现至少不劣于扩展 oracle 集 $\Pi^\mathcal{E}$ 中的任何单一最优策略。即 $V^{\pi^\circ}(s) \geq f^+(s) = \max_{k \in [|\Pi^\mathcal{E}|]} V^k(s)$。

> [!todo]- Proof sketch
> 关键在于证明 $A^+(s, \pi^\circ) \geq 0$ 对所有状态 $s$ 成立。设状态 $s$ 下最优 oracle 为 $\pi^1$，则 $\pi^\circ(a|s) \geq \pi^{\bullet}(a|s) = \pi^1(a|s)$（因为 $\pi^\circ$ 在扩展集上选择，而 $\pi^{\bullet}$ 只在原始 oracle 集上选择）。因此 $A^+(s, \pi^\circ) \geq r(s, \pi^{\bullet}) + \mathbb{E}_{s'}[f^+(s')] - f^+(s) \geq r(s, \pi^1) + \mathbb{E}_{s'}[V^1(s')] - V^1(s) = A^{V^1}(s, \pi^1) \geq 0$。结合 Lemma C.1（性能差引理），可得 $V^{\pi^\circ}(s) \geq f^+(s)$。

虽然 $\pi^\circ$ 已经改进了 max-following 策略，但它仍然只是在模仿——没有自主探索、寻找更优动作的能力。当某个 oracle $\pi^k$ 在所有状态上的价值函数始终最高时，$\pi^\circ$ 简单退化为模仿该单一 oracle，而不会尝试在某些状态上做得更好。

**max⁺-aggregation 策略 $\pi^\oplus$。** 为了实现真正的策略改进，论文提出 max⁺-aggregation 策略，它在每个状态上不是简单模仿，而是执行关于 $f^+$ 的一步贪心改进：

$$
\pi^\oplus(a|s) = \delta_{a = a^\star}, \quad a^\star = \arg\max_{a \in \mathcal{A}} A^+(s, a) \tag{6}
$$

其中 $\delta$ 为 Dirac delta 分布。直观理解：$\pi^\oplus$ 在每个状态上选择使得 $A^+$ 最大化的动作。由于 $A^+(s, \pi^\oplus) \geq A^+(s, \pi^\circ) \geq 0$，$\pi^\oplus$ 的表现不劣于 $\pi^\circ$，因此也不劣于扩展 oracle 集中的任何单一策略。与 $\pi^\circ$ 的区别在于，$\pi^\oplus$ 不仅可以模仿最好的 oracle，还可以找到比任何 oracle 都更好的动作。

> [!tip] $f^+$ 的语义
> $f^+(s)$ 对应的是在状态 $s$ 选择扩展 oracle 集中的最优策略，并从该状态起一直执行该策略到 episode 结束的价值。而 $\pi^\circ$ 和 $\pi^\oplus$ 则是在每个时间步上都重新选择最优策略或最优动作。这就是为什么 $\pi^\oplus$ 能超越 $f^+$。

> [!info] 空 oracle 集的退化
> 当 oracle 集为空时，$\Pi^\mathcal{E}$ 只包含学习者策略，$f^+ \equiv V^{\pi_n}$，$\pi^\circ$ 不再改进，$\pi^\oplus$ 退化为纯强化学习的一步策略改进。

### 3.3 Online Learning with Black-Box Oracles

在实际中，MDP 和 oracle 的价值函数都是未知的。论文将策略改进问题转化为在线学习问题。

**已知 oracle 价值函数的情况。** 当 MDP 未知但价值函数已知时，将 $d^{\pi_n}$ 视为对手，建立在线损失：

$$
\ell_n(\pi) := -H\mathbb{E}_{s \sim d^{\pi_n}} \mathbb{E}_{a \sim \pi|s}[A^+(s, a)] \tag{7}
$$

对 $N$ 轮在线学习取平均，学习者策略的平均性能为：

$$
\frac{1}{N} \sum_{n \in [N]} V^{\pi_n}(d_0) = f_m^+(d_0) + \Delta_N - \epsilon_N(\Pi^\mathcal{L}) - \text{Regret}_N^\mathcal{L} \tag{8}
$$

其中 $f_m^+(s) := \max_{k \in [|\Pi^o \cup \{\pi_m\}|]} V^k(s)$（$m \ll N$）为稍弱的静态基线，$\text{Regret}_N^\mathcal{L}$ 为在线算法的学习遗憾，$\Delta_N := -\frac{1}{N}\sum_{n=1}^N \ell_n(\pi_m^\oplus)$ 为基线 max⁺-aggregation 策略的损失（由 Proposition 4.5 保证 $\Delta_N \geq 0$），$\epsilon_N(\Pi^\mathcal{L})$ 衡量学习者策略类的质量。

**未知 oracle 价值函数的情况。** 此时 $f^+$ 和 $A^+$ 需要通过估计来近似。梯度的样本估计为：

$$
\nabla \hat{\ell}_n(\pi_n) = -H\mathbb{E}_{s \sim d^{\pi_n}} \mathbb{E}_{a \sim \pi_n|s}\left[\nabla \log \pi_n(a|s) \hat{A}^+(s, a)\right] \tag{9}
$$

> [!note] Proposition 5.1（性能下界）
> 定义 $f_m^+ := \max_{k \in [|\Pi^o \cup \{\pi_m\}|]} V^k(s)$，则：
>
> $$
> \mathbb{E}\left[\max_{n \in [N]} V^{\pi_n}(d_0)\right] \geq \mathbb{E}_{s \sim d_0}[f_m^+(s)] + \mathbb{E}\left[\Delta_N - \epsilon_N(\Pi^\mathcal{L}) - \text{Regret}_N^\mathcal{L}\right]
> $$
>
> 由于 $\mathbb{E}_{s \sim d_0}[f_m^+(s)] \geq \mathbb{E}_{s \sim d_0}[f_0^+(s)]$（学习者策略的加入只会提高基线），RPI 的性能下界不弱于 MAMBA。

这一结论的意义在于：即使在最坏情况下（学习者从未超越任何 oracle），RPI 也不会比 MAMBA 差；而在学习者部分超越 oracle 的常见情况下，RPI 的基线 $f_m^+$ 严格高于 MAMBA 的 $f_0^+$，从而获得更好的性能保证。

### 3.4 RAPS: Robust Active Policy Selection

RAPS 的目标是在每一轮训练中，为当前状态高效地选择最优的 oracle 进行 roll-out，以改善价值函数估计 $\hat{f}^+$ 的质量，降低 Proposition 5.1 中 $\text{Regret}_N^\mathcal{L}$ 的偏差项。

**价值函数集成估计。** 对每个策略（oracle 和学习者）使用集成/ensemble 的预测模型来估计价值函数。具体而言，为每个策略训练若干个独立初始化的价值预测网络，对给定状态 $s$，输出均值 $\hat{V}_\mu^k(s)$ 和不确定性 $\sigma_k(s)$。据此定义 UCB 和 LCB：

$$
\overline{V}^k(s) = \hat{V}_\mu^k(s) + \sigma_k(s), \quad \underline{V}^k(s) = \hat{V}_\mu^k(s) - \sigma_k(s)
$$

**oracle 选择策略。** 对 oracle 使用 UCB（乐观估计），对学习者使用 LCB（保守估计），以此选择最佳 oracle $\pi^{k_\star}$：

$$
k_\star = \arg\max_{k \in [|\Pi^\mathcal{E}|]} \left\{\overline{V}^1(s), \overline{V}^2(s), \ldots, \overline{V}^K(s), \underline{V}^{K+1}(s)\right\} \tag{10}
$$

> [!tip] UCB/LCB 非对称设计的直觉
> 对 oracle 使用 UCB 是为了鼓励探索——当不确定某个 oracle 是否优秀时，倾向于乐观地尝试它。对学习者使用 LCB 则是保守策略——只有当我们有充分信心学习者确实优于所有 oracle 时，才让学习者自行探索而非模仿 oracle。这种非对称性确保了 oracle 指导的探索在学习早期占主导，随着学习者改善和估计不确定性降低，自我改进逐渐接管。

与前身 MAPS 相比，RAPS 的关键改进在于将学习者策略纳入选择候选集（通过扩展 oracle 集），使得在学习者已超越 oracle 的状态上，算法能识别出这一点并让学习者进行自主 roll-out。

### 3.5 RPG: Robust Policy Gradient

RPG 基于一种新设计的优势函数 $A^{\text{GAE+}}$ 在 actor-critic 框架下进行策略梯度更新。

**$A^{\text{GAE+}}$ 优势函数。** 论文将 GAE（广义优势估计/Generalized Advantage Estimation）推广到 max⁺ 基线下。定义 TD 残差：

$$
\hat{\delta}_t = r_t + \gamma \hat{f}^+(s_{t+1}) - \hat{f}^+(s_t)
$$

则 GAE+ 优势估计为：

$$
\hat{A}_t^{\text{GAE+}} = \hat{\delta}_t + (\gamma\lambda)\hat{\delta}_{t+1} + \ldots + (\lambda\gamma)^{T-t+1}\hat{\delta}_{T-1} \tag{11}
$$

其中 $T \ll H$ 为截断长度，$\gamma$ 和 $\lambda$ 控制偏差-方差权衡。实验中使用 $\lambda = 0.9$，$\gamma = 1$。

**带置信度阈值的 $\hat{f}^+$ 估计。** 为了控制对 oracle 价值函数估计的可靠性，引入置信度阈值 $\Gamma_s$：

$$
\hat{f}^+(s) = \begin{cases} \hat{V}_\mu^{\pi_n}(s), & \text{if } \sigma_k(s) > \Gamma_s, \text{ where } k = \arg\max_{k \in [|\Pi^\mathcal{E}|]} \hat{V}_\mu^k(s) \\ \max_{k \in [|\Pi^\mathcal{E}|]} \hat{V}_\mu^k(s), & \text{otherwise} \end{cases} \tag{12}
$$

当被选中 oracle 的不确定性过高（$\sigma_k > \Gamma_s$）时，不信任该 oracle 的价值估计，退回到使用学习者自身的价值函数。较低的 $\Gamma_s$ 意味着更高的置信要求（更保守），实验中使用 $\Gamma_s = 0.5$。

**在线损失与梯度。** 基于 $A^{\text{GAE+}}$，第 $n$ 轮的在线损失和梯度估计为：

$$
\hat{\ell}_n(\pi_n) := -H\mathbb{E}_{s \sim d^{\pi_n}} \mathbb{E}_{a \sim \pi|s}\left[\hat{A}_t^{\text{GAE+}}(s, a)\right]\bigg|_{\pi = \pi_n} \tag{13}
$$

$$
\hat{g}_n = \nabla \hat{\ell}_n(\pi_n) = -H\mathbb{E}_{s \sim d^{\pi_n}} \mathbb{E}_{a \sim \pi_n|s}\left[\nabla \log \pi(a|s) \hat{A}_t^{\text{GAE+}}(s, a)\right]\bigg|_{\pi = \pi_n} \tag{14}
$$

> [!tip] RPI 中 IL 与 RL 的自适应融合
> $\hat{f}^+$ 的实例化自然实现了 IL 和 RL 的融合：训练早期学习者策略较弱，$\hat{f}^+$ 主要由 oracle 的价值函数决定，此时策略梯度方向等价于模仿学习（学习者被引导向 oracle 的高价值状态靠拢）；随着学习者改善，$\hat{f}^+$ 越来越多地由学习者自身的价值函数决定，此时 RPG 退化为标准的 actor-critic 自我改进。当 $\lambda = 0$ 且 $\gamma = 1$ 时，公式 (13) 退化为 max⁺-aggregation 策略下的损失公式 (7)，性能下界等同于 Proposition 5.1；而当 $\lambda > 0$ 时，RPI 优化的是多步优势，相比 max⁺-aggregation 的一步优势能获得更小的 $\epsilon_N(\Pi^\mathcal{L})$ 项，从而改善性能下界。

**整体算法流程（Algorithm 1）：**

1. 构建扩展 oracle 集 $\Pi^\mathcal{E} = [\pi^1, \ldots, \pi^K, \pi_n]$
2. 均匀随机采样切换点 $t_e \in [H-1]$
3. Roll-in 学习者策略 $\pi_n$ 到 $t_e$，通过 RAPS（公式 10）选择最优 oracle $k_\star$，roll-out $\pi^{k_\star}$ 收集数据 $\mathcal{D}^k$
4. 用 $\mathcal{D}^k$ 更新所选 oracle 的价值函数估计 $\hat{V}^{k_\star}$
5. Roll-in 学习者策略 $\pi_n$ 完整 $H$ 步，收集数据 $\mathcal{D}_n'$
6. 用 $\mathcal{D}_n'$ 更新学习者价值函数估计 $\hat{V}_n$
7. 基于 $\mathcal{D}_n'$ 计算 $\hat{A}^{\text{GAE+}}$（公式 11）和梯度 $\hat{g}_n$（公式 14）
8. 使用 PPO 风格的策略更新将 $\pi_n$ 更新为 $\pi_{n+1}$

## 4. Experiments

上述理论框架和算法设计提出了若干实证问题：RPI 能否在不同奖励结构和 oracle 质量下稳定地超越纯 IL 和纯 RL 方法？RAPS 和 RPG 各自贡献了多少？置信度机制是否真正有效？以下实验逐一回答这些问题。

### 4.1 Setup

**环境。** 论文在 8 个连续状态-动作空间的任务上进行评估：DeepMind Control Suite 的 Cheetah-run、CartPole-swingup、Pendulum-swingup、Walker-walk，以及 Meta-World 的 Window-close、Faucet-open、Drawer-close、Button-press。Meta-World 还测试了修改后的稀疏奖励版本（成功时奖励 1，否则奖励 0）。

**Oracle 构造。** Oracle 由 PPO（配合 GAE）和 SAC 在不同训练阶段保存的策略权重构成，每个环境提供 3 个多样化的 oracle，它们在不同状态下具有不同的专长。价值函数集成大小为 5。

**基线方法。** 包括 (1) PPO-GAE（纯 RL 基线），(2) Max-Aggregation（纯 IL 基线，AggreVaTeD 的多 oracle 变体），(3) LOKI-variant（先 IL 后 RL 的两阶段方法），(4) MAMBA，(5) MAPS（当时的 SOTA）。所有方法使用相同数量的环境交互。

### 4.2 Main Results

RPI 在所有 8 个基准任务上均超越了所有基线方法，展现了在不同 oracle 质量和奖励结构下的鲁棒性。

在稠密奖励环境（Cheetah、Walker-walk、Cartpole）中，一个值得注意的现象是纯 RL 基线 PPO-GAE 在训练后期能超越纯 IL 方法（因为 IL 方法受制于次优 oracle 而性能饱和），但 IL 方法在早期学习速度更快。RPI 兼得两者优势——早期利用 oracle 快速启动学习，后期通过自我改进超越 oracle 的性能天花板。

在稀疏奖励环境（Pendulum-swingup、Window-close）中，纯 RL 方法因奖励信号稀疏而难以有效探索，IL 方法虽然样本效率更高但同样因 oracle 次优而性能饱和。RPI 先从 oracle 引导中获得初始学习信号，再通过 RL 自我改进突破 oracle 的局限，在两类任务上都取得最优表现。这一结果直接验证了 IL 与 RL 自适应融合的设计动机。

<!-- TODO: insert Figure 3 from paper (Main results) -->

### 4.3 Ablation Studies

**多 oracle 聚合能力。** 在 Cartpole 上，RPI 使用 3 个 oracle 时达到 645 的回报，而使用单一最好 oracle 时回报低于 600，验证了 RPI 对 oracle 状态级专长的有效聚合。

**RAPS vs APS。** 将 RAPS 与 MAPS 中的 APS（不含学习者策略的版本）对比，RAPS 因能在学习者优于 oracle 的状态上选择学习者自行 roll-out 而获得更好的结果。

**置信度感知 RPG。** 移除 $\hat{f}^+$ 中的置信度阈值 $\Gamma_s$ 后，RPG 在 oracle 不确定性高的状态上更容易受到噪声干扰，性能下降。这验证了 $\Gamma_s$ 对于在高不确定性下从模仿学习平滑过渡到强化学习的重要性。

**UCB/LCB vs MEAN 策略选择。** RPI-LCB/UCB 在所有 DeepMind Control Suite 基准上以总计约 40% 的优势超越不考虑不确定性的 RPI-MEAN，凸显了将置信度纳入策略选择的价值。

**IL 与 RL 的动态转换可视化。** 在 Pendulum 任务上可视化了 RPI 梯度估计器中 oracle 选择的频率变化：训练初期 RPI 主要模仿 oracle（oracle 被频繁选择），随着学习者策略提升，学习者自身被选择的频率逐渐增加，最终以自我改进为主。

<!-- TODO: insert Figure 5 from paper (IL and RL transition) -->

### 4.4 Critical Evaluation

**优势：** 实验覆盖了稠密/稀疏奖励、操控/运动控制等多种场景，基线方法较为全面（纯 RL、纯 IL、两阶段、SOTA 混合方法均有覆盖），消融设计有针对性地验证了 RAPS 和 RPG 的各个组件。空 oracle 和单 oracle 的附加实验进一步验证了方法的鲁棒性。

**可能的不足：**
- Oracle 全部由 PPO/SAC 在不同训练阶段保存而成，本质上是同源的策略，不同 oracle 之间的差异较为有限。面对真正异质的 oracle（如来自不同算法族、不同任务迁移、或人类示范）时，方法的表现尚未验证。
- 所有基线均使用相同的环境交互次数，但 RPI 将部分交互分配给 oracle roll-out 和价值函数预训练，实际用于学习者策略更新的数据少于 PPO-GAE。这种比较虽然公平，但也意味着 RPI 的优势部分来源于 oracle 信息本身，而非纯粹的算法设计。
- $\Gamma_s$ 在多数环境上 0.5 表现良好，但在 Pendulum 上 $\Gamma_s = 3$ 显著更优。论文未提供自适应选择 $\Gamma_s$ 的方法。
- 实验报告的是 5-10 次试验的均值和标准差，部分结果（如 Cartpole）的标准差较大（$670.4 \pm 110.1$），方法的稳定性存在一定疑问。

## 5. Related Work & Future Work

RPI 的设计建立在多条研究线索的交汇点上。以下按与本文关系的紧密程度组织相关工作。

### Related Work

**从次优 oracle 中学习。** MAMBA 提出了 max-aggregation 基线和几何加权泛化的优势函数，为多 oracle 策略改进提供了理论保证，但样本复杂度较高（需要均匀采样 oracle 来确定最优者）。MAPS 在 MAMBA 基础上引入主动策略选择/Active Policy Selection 和主动状态探索/Active State Exploration，改善了样本效率，但仍受限于 oracle 质量——即使 oracle 集合整体较差，仍会执行模仿学习。RPI 的核心区别在于通过扩展 oracle 集将学习者纳入候选，仅在 oracle 确实优于学习者的状态上执行模仿。

**在线选择次优专家。** CAMS 在无状态的在线学习环境中从多个黑盒专家中选择模型，不适用于 MDP 设定。SAC-X 学习多个意图策略（各自优化辅助奖励函数），再推理执行哪个意图策略。与 CAMS 和 SAC-X 依赖多个 oracle 执行子任务不同，RPI 训练一个独立的学习者策略，每个 episode 只进行一次 oracle 查询，并通过全局探索实现超越。

**结合 IL 和 RL。** LOKI 采用两阶段策略（先 IL 后 RL），但仅针对单 oracle 且两阶段的切换是固定的。TGRL 也是单 oracle 设定。RPI 支持多 oracle，且 IL 与 RL 的混合是在状态级别上自适应的，无需预设切换时间。

### Future Work

论文明确提到的未来方向：处理更具挑战性的鲁棒设定，如缺失状态信息或 oracle 信息不完整的场景。

可自然推断的扩展方向包括：(1) 将 RAPS 中的置信度阈值 $\Gamma_s$ 从固定超参数改为自适应调整机制；(2) 将方法扩展到离线 oracle 数据（而非需要在线查询 oracle 策略）的设定；(3) 验证面对真正异质的 oracle 来源（如人类示范、不同算法族的策略）时的表现。
