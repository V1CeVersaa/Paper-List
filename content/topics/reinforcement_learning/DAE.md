---
title: "DAE"
headline: "Direct Advantage Estimation"
visibility: "public"
status: "drafting"
description: "Paper note on Direct Advantage Estimation."
---

> [!abstract] Contributions
>
> 本文从因果推断/Causal Inference 的视角重新审视了强化学习中的优势函数/Advantage Function，证明优势函数可以被解释为动作对期望回报的因果效应/Causal Effect。基于这一理论洞察，作者证明了优势函数是在 $\pi$-centered 约束下最小化回报方差的最优解，并据此提出了直接优势估计/Direct Advantage Estimation/DAE——一种无需预先学习值函数或 Q 函数、直接从 on-policy 轨迹数据中建模和估计优势函数的方法。DAE 还可以与值函数的自举/Bootstrapping 机制无缝结合，形成类似于 $n$-step TD 的多步更新。实验在合成环境、MinAtar 和 Arcade Learning Environment/ALE 三类离散控制任务上表明，DAE 在大多数环境中优于广义优势估计/Generalized Advantage Estimation/GAE，且在网络容量增大时表现出更好的可扩展性。
>
> DAE 的核心假设是动作的因果效应是局部的（即动作仅影响有限步内的未来状态分布），这一条件在"树状"环境（动作导致完全不同的后续分支）中可能不成立。此外，DAE 目前仅适用于离散动作空间，因为 $\pi$-centered 约束在连续空间中难以精确施加。方法本身是 on-policy 的，且在训练初期由于函数逼近的偏差可能落后于 GAE。

## 1. Introduction

强化学习的核心目标是找到使累积回报最大化的策略。在这一过程中，一个根本性的困难是信用分配/Credit Assignment 问题：在一条长轨迹中，大量决策共同影响了最终结果，如何判断哪些决策是关键的？最直接的方式是根据期望回报来分配信用，这也是标准 RL 目标函数的做法。然而，期望回报将动作与所有后续状态的策略评估绑定在一起，即使动作本身的"效果"仅是即时的，其 Q 值也会随策略更新而剧烈波动。

<!-- TODO: insert Figure 1 from paper — 链式环境示意图，每个状态有 u/d 两个动作，u 得奖励 1，d 得奖励 0，转移与动作无关 -->

以论文 Figure 1 的环境为例：每个状态下选择 u 得到奖励 1、选择 d 得到奖励 0，且状态转移与动作无关。在这个环境中，动作的效果仅是即时奖励的差异，但 Q 函数却包含了所有后续状态上策略的累积期望，导致 Q 值随策略变化而大幅波动。这种不稳定性在 off-policy 学习中被称为分布偏移/Distribution Shift，在 on-policy 学习中同样会造成高方差和学习速度下降。

本文的核心洞察源自因果推断领域：如果我们关心的不是"在某状态下采取某动作后的期望回报是多少"，而是"采取某动作相比于正常情况会产生多大的效果"，那么自然的度量就是因果效应。作者证明，优势函数 $A(s,a) = Q(s,a) - V(s)$ 恰好就是动作 $a$ 对期望回报的因果效应。这一联系不仅在概念上更加清晰，更重要的是它揭示了优势函数相比 Q 函数具有更好的稳定性：优势函数仅依赖于动作因果地影响到的局部区域内的策略，而非整条轨迹上的策略。

基于此，作者提出 DAE，直接对优势函数进行建模和回归，绕过了先学 $V$ 或 $Q$ 再间接得到 $A$ 的传统路径。同时，作者还展示了 DAE 如何与自举机制结合，并在 PPO 框架中给出了完整的集成方案。

## 2. Problem Setup

考虑折扣马尔可夫决策过程/Markov Decision Process/MDP $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$，其中 $\mathcal{S}$ 是有限状态空间，$\mathcal{A}$ 是有限动作空间，$P(s'|s,a)$ 是转移概率，$r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 是期望奖励函数，$\gamma \in [0,1)$ 是折扣因子。策略 $\pi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 给出在状态 $s$ 下选择动作 $a$ 的概率 $\pi(a|s)$。轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 按 $a_t \sim \pi(\cdot|s_t)$, $s_{t+1} \sim P(\cdot|s_t, a_t)$ 采样。轨迹回报定义为：

$$
G(\tau) = \sum_{t \geq 0} \gamma^t r_t, \quad r_t = r(s_t, a_t)
$$

RL 的目标是找到 $\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}[G(\tau)]$。给定策略 $\pi$，定义：

- 值函数/Value Function：$V^\pi(s) = \mathbb{E}_{\tau \sim \pi}[G(\tau) | s_0 = s]$
- 动作-值函数/Action-Value Function：$Q^\pi(s,a) = \mathbb{E}_{\tau \sim \pi}[G(\tau) | s_0 = s, a_0 = a]$
- 优势函数/Advantage Function：$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$

优势函数刻画了在状态 $s$ 下选择动作 $a$ 相比于按当前策略行动的期望优势。

**估计方法的背景。** 蒙特卡洛/Monte Carlo/MC 方法通过采样完整轨迹来无偏地估计 $V(s)$，但方差较高。时序差分/Temporal Difference/TD 方法通过自举目标 $r + \gamma V(s')$ 来更新，方差更低但引入了偏差。$n$-step 方法和 $\text{TD}(\lambda)$ 在两者之间进行插值：$n$ 越大越接近 MC（低偏差高方差），$n$ 越小越接近 TD（高偏差低方差）。

在策略优化中，策略梯度的标准形式为：

$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)] = \sum_{s \in \mathcal{S}} d^{\pi_\theta}(s) \sum_{a \in \mathcal{A}} (Q(s,a) - b(s)) \nabla_\theta \pi_\theta(a|s)
\tag{1}
$$

其中 $d^{\pi_\theta}(s) = \sum_{t \geq 0} \gamma^t p(s_t = s)$ 是折扣状态访问分布，$b(s)$ 是任意基线函数。当 $b(s) = V(s)$ 时，策略梯度中使用的恰好就是优势函数。

## 3. Methods

### 3.1 Causal Interpretation of the Advantage Function

本文首先从因果推断的角度重新理解优势函数。借鉴 Neyman-Rubin 因果模型的思想，动作 $a$ 在状态 $s$ 对某个量 $X$ 的因果效应定义为：

$$
\mathbb{E}[X | s, a] - \mathbb{E}[X | s]
$$

当 $X$ 取为轨迹回报 $G(\tau)$ 时：

$$
\mathbb{E}[G(\tau) | s_t = s, a_t = a] - \mathbb{E}[G(\tau) | s_t = s] = Q(s,a) - V(s) = A(s,a)
\tag{2}
$$

这表明优势函数恰好就是动作对期望回报的因果效应。

> [!tip] 因果效应视角的意义
> 这一联系不仅是概念上的重新包装。因果表征/Causal Representation 被认为在分布偏移下具有更好的稳定性（这与稀疏机制偏移/Sparse Mechanism Shift 假说一致）。如果优势函数确实只编码了动作的局部因果效应，那么它在策略更新过程中应比 Q 函数更稳定。

**优势函数的局部性。** 为了形式化"优势函数只依赖局部策略"这一直觉，作者给出了如下命题：

> [!note] Proposition 1
> 给定 $t' > t$，若 $\mathbb{E}[V(s_{t'}) | s_t, a_t] = \mathbb{E}[V(s_{t'}) | s_t]$（即从某个时间步 $t'$ 开始，未来状态的值函数不再依赖于当前动作 $a_t$），则：
>
> $$
> A(s_t, a_t) = \mathbb{E}\left[\sum_{k=t}^{t'-1} \gamma^{k-t} r_k \;\middle|\; s_t, a_t\right] - \mathbb{E}\left[\sum_{k=t}^{t'-1} \gamma^{k-t} r_k \;\middle|\; s_t\right]
> $$

> [!todo]- Proof
> 由 $Q$ 函数的定义展开：
>
> $$
> Q^\pi(s_t, a_t) = \mathbb{E}\left[\sum_{k=t}^{t'-1} \gamma^{k-t} r_k \;\middle|\; s_t, a_t\right] + \gamma^{t'-t} \mathbb{E}[V(s_{t'}) | s_t, a_t]
> $$
>
> 类似地：
>
> $$
> V^\pi(s_t) = \mathbb{E}\left[\sum_{k=t}^{t'-1} \gamma^{k-t} r_k \;\middle|\; s_t\right] + \gamma^{t'-t} \mathbb{E}[V(s_{t'}) | s_t]
> $$
>
> 当条件 $\mathbb{E}[V(s_{t'}) | s_t, a_t] = \mathbb{E}[V(s_{t'}) | s_t]$ 成立时，两式相减即得 Proposition 1 的结论。$\square$

Proposition 1 说明优势函数是"局部的"：它仅依赖于动作因果地影响到的那段有限时间窗口内的奖励和策略，而不涉及更远的未来。这与 Q 函数形成鲜明对比——Q 函数的值会因为遥远未来的策略变化而波动，而优势函数不会。

Proposition 1 的条件 $\mathbb{E}[V(s_{t'}) | s_t, a_t] = \mathbb{E}[V(s_{t'}) | s_t]$ 要求未来状态分布在某一步之后完全不依赖当前动作，这在实际中可能过强。一个更宽松的假设是 $p(s_{t'}|s_t, a_t) \approx p(s_{t'}|s_t)$ 对某些 $t' > t$ 近似成立，例如当马尔可夫链趋近于平稳分布时，或环境包含"瓶颈状态"（如门廊）时。作者在 MinAtar Breakout 环境中实证验证了这一点：在 PPO 训练过程中，优势函数的变化量 $\Delta f = |f^{\pi_i} - f^{\pi_{i-1}}|$ 始终显著小于 Q 函数的变化量。

<!-- TODO: insert Figure 2 from paper — Breakout 环境中 Q(s,a) 和 A(s,a) 在策略更新过程中的变化幅度对比，A 的变化始终更小更稳定 -->

<!-- TODO: insert Figure 3 from paper — 树状环境反例，状态 s 通过 u/d 转移到 s_a 和 s_d，说明优势函数不能通过自举学习 -->

**优势函数不能通过自举学习。** 既然 $Q$ 和 $V$ 都可以通过 TD 自举来学习，优势函数是否也可以？作者给出了一个反例：在一个树状环境中，状态 $s$ 有两个动作 u 和 d 分别转移到 $s_\text{a}$（奖励 1）和 $s_\text{d}$（奖励 0），且 $s_\text{a}$ 和 $s_\text{d}$ 各只有一个动作。此时 $s_\text{a}$ 和 $s_\text{d}$ 的即时奖励为 0，优势也为 0。如果试图用 TD 方式从下一步的即时奖励和优势来自举当前的优势，则无法恢复 $s$ 处的优势值。这是因为优势函数只编码因果效应，而值函数包含关于所有未来奖励的信息——优势函数丢弃了非因果信息，因此无法像值函数那样自举。

### 3.2 Theoretical Foundation of DAE

DAE 的推导从一个关键观察出发：优势函数满足 $\pi$-centered 性质，即 $\sum_{a \in \mathcal{A}} \pi(a|s) A^\pi(s,a) = 0$。

> [!note] Definition 1 ($\pi$-centered 函数)
> 给定策略 $\pi$，函数 $f: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 是 $\pi$-centered 的，如果对所有 $s \in \mathcal{S}$：
>
> $$
> \sum_{a \in \mathcal{A}} \pi(a|s) f(s,a) = 0
> \tag{3}
> $$

$\pi$-centered 函数有一个重要性质：如果用任意 $\pi$-centered 函数 $\hat{A}$ 对奖励进行变换 $r'_t = r_t - \hat{A}(s_t, a_t)$，则变换后的回报 $G'(\tau) = \sum_{t \geq 0} \gamma^t r'_t$ 的期望不变，即 $\mathbb{E}[G'(\tau)] = \mathbb{E}[G(\tau)]$。这可以理解为一种奖励塑形/Reward Shaping。由于期望不变，最小化 $G'(\tau)$ 的方差等价于最小化：

$$
\mathbb{E}\left[\left(\sum_{t=0}^{\infty} \gamma^t (r_t - \hat{A}(s_t, a_t))\right)^2\right]
\tag{4}
$$

直觉上，$\hat{A}$ 的作用是从回报中"减去"每个时间步上动作的因果效应，使剩余部分（即与动作选择无关的回报分量）的方差尽可能小。当 $\hat{A}$ 恰好等于真实优势函数 $A$ 时，这个方差最小化问题达到最优。

> [!note] Theorem 1
> 给定策略 $\pi$ 和时间步 $t \geq 0$，记 $F_\pi$ 为所有 $\pi$-centered 函数的集合，$\hat{A}_t = \hat{A}(s_t, a_t)$。若 $(s,a) \in \mathcal{S} \times \mathcal{A}$ 或 $s \in \mathcal{S}$ 在时间步 $t$ 内可达（即存在 $0 \leq t' \leq t$ 使得 $p(s_{t'} = s, a_{t'} = a) > 0$ 或 $p(s_{t'} = s) > 0$），则：
>
> $$
> \hat{A}^* = \arg\min_{\hat{A} \in F_\pi} \mathbb{E}\left[\left(G(\tau) - \sum_{t'=0}^{t} \gamma^{t'} \hat{A}_{t'}\right)^2\right]
> $$
>
> 对所有可达的 $(s,a)$ 满足 $\hat{A}^*(s,a) = A^\pi(s,a)$。

> [!todo]- Proof
> 记 $f \in F_\pi$，即 $\sum_{a \in \mathcal{A}} \pi(a|s) f(s,a) = 0$ 对所有 $s$ 成立。简记 $f_{t'} = f(s_{t'}, a_{t'})$。构造 Lagrangian：
>
> $$
> \mathcal{L} = \mathbb{E}_{\tau \sim \pi}\left[\left(G(\tau) - \sum_{t'=0}^{t} \gamma^{t'} f_{t'}\right)^2\right] + \sum_{s \in \mathcal{S}} \lambda_s \sum_{a \in \mathcal{A}} \pi(a|s) f(s,a)
> $$
>
> 对 $f(s', a')$ 求偏导。注意 $\frac{\partial f_k}{\partial f(s',a')} = \mathbb{I}(s_k = s', a_k = a')$，因此：
>
> $$
> \frac{\partial \mathcal{L}}{\partial f(s', a')} = -2 \sum_{k=0}^{t} \gamma^k \mathbb{E}_{\tau \sim \pi}\left[\left(G(\tau) - \sum_{t'=0}^{t} \gamma^{t'} f_{t'}\right) \mathbb{I}(s_k = s', a_k = a')\right] + \lambda_{s'} \pi(a'|s')
> $$
>
> 利用 $p(s_k = s', a_k = a') = p(s_k = s') \pi(a'|s')$，上式可改写为：
>
> $$
> = -2 \sum_{k=0}^{t} \gamma^k p(s_k = s', a_k = a') \mathbb{E}\left[G(\tau) - \sum_{t'=0}^{t} \gamma^{t'} f_{t'} \;\middle|\; s_k = s', a_k = a'\right] + \lambda_{s'} \pi(a'|s')
> $$
>
> 对所有 $a' \in \mathcal{A}$ 求和（利用 $\pi$-centered 约束消去 $f$ 的贡献），得到：
>
> $$
> \lambda_{s'} = 2 \sum_{k=0}^{t} \gamma^k p(s_k = s') \mathbb{E}\left[G(\tau) - \sum_{t'=0}^{t} \gamma^{t'} f_{t'} \;\middle|\; s_k = s'\right]
> $$
>
> 将 $\lambda_{s'}$ 代回一阶条件并令其为零，利用马尔可夫性将条件期望化简：
>
> $$
> \mathbb{E}\left[G(\tau) - \sum_{t'=0}^{t} \gamma^{t'} f_{t'} \;\middle|\; s_k = s', a_k = a'\right] - \mathbb{E}\left[G(\tau) - \sum_{t'=0}^{t} \gamma^{t'} f_{t'} \;\middle|\; s_k = s'\right]
> $$
>
> $$
> = \gamma^k \left(Q(s', a') - f(s', a') - V(s')\right) = 0
> $$
>
> 因此，对所有可达的 $(s', a')$，$f(s', a') = Q(s', a') - V(s') = A(s', a')$。$\square$

Theorem 1 有两个重要推论：

1. **解纠缠性/Disentanglement**：如果 $(s,a)$ 在时间步 $t$ 内可达，则 $\hat{A}^*(s,a) = A^\pi(s,a)$ 对任意 $\hat{t} > t$ 也成立，即增加更多时间步的 $\hat{A}$ 不会干扰已有结果。
2. **可加性/Additivity**：$\sum_{t'} \gamma^{t'} \hat{A}_{t'}$ 可理解为各动作因果效应的线性叠加。

### 3.3 Practical Implementation

由 Theorem 1，优势函数可以通过最小化以下约束损失从采样轨迹 $\tau_1, \ldots, \tau_N$ 中直接估计：

$$
L(\theta) = \frac{1}{N} \sum_{\tau = \tau_1, \ldots, \tau_N} \left(\sum_{t=0}^{\infty} \gamma^t \left(r_t - \hat{A}_\theta(s_t, a_t)\right)\right)^2, \quad \text{s.t.} \sum_{a} \pi(a|s) \hat{A}_\theta(s,a) = 0
\tag{5}
$$

在实践中，$\pi$-centered 约束通过构造函数形式来自动满足：令 $f_\theta$ 为一般的神经网络输出，则定义：

$$
\hat{A}_\theta(s,a) = f_\theta(s,a) - \sum_{a' \in \mathcal{A}} \pi(a'|s) f_\theta(s,a')
$$

这一减去期望的操作确保了 $\sum_a \pi(a|s) \hat{A}_\theta(s,a) = 0$ 恒成立，无需额外处理约束。

> [!tip] DAE 与 V/Q 学习的根本区别
> 学习 $V$ 或 $Q$ 是在寻找状态（或状态-动作对）到期望回报的映射。DAE 则试图将回报"分配"到各个时间步的动作上——它学习的是每个动作对回报的贡献量，而不是"预测"回报。

**与自举的结合。** DAE 在基本形式下需要完整轨迹，类似 MC 方法。为克服这一限制，Theorem 2 展示了如何引入值函数的自举目标。

> [!note] Theorem 2
> 记 $V_\text{target}(s)$ 为自举目标，$\hat{V}(s)$ 为待学习的值函数。定义损失：
>
> $$
> L(\hat{A}, \hat{V}) = \mathbb{E}\left[\left(\sum_{t=0}^{n-1} \gamma^t (r_t - \hat{A}_t) + \gamma^n V_\text{target}(s_n) - \hat{V}(s_0)\right)^2\right]
> \tag{6}
> $$
>
> 若 $\hat{V}^* = \arg\min_{\hat{V}} L(\hat{A}, \hat{V})$ 且 $p(s_0 = s) > 0$，则对任意 $\hat{A} \in F_\pi$：
>
> $$
> \hat{V}^*(s) = \mathbb{E}\left[\sum_{t=0}^{n-1} \gamma^t r_t + \gamma^n V_\text{target}(s_n) \;\middle|\; s_0 = s\right]
> $$
>
> 若 $\hat{A}^* = \arg\min_{\hat{A} \in F_\pi} L(\hat{A}, \hat{V})$ 且 $(s,a)$ 在 $t = 0$ 到 $t = n-1$ 内可达，则：
>
> $$
> \hat{A}^*(s,a) = \sum_{t=0}^{n-1} \frac{w_t(s)}{W_{n-1}(s)} \left(\mathbb{E}[r_t + \cdots + \gamma^{n-t-1} r_{n-1} + \gamma^{n-t} V_\text{target}(s_n) | s_t = s, a_t = a] - \mathbb{E}[\cdots | s_t = s]\right)
> $$
>
> 其中 $w_t(s) = \gamma^{2t} p(s_t = s)$，$W_{n-1}(s) = \sum_{t=0}^{n-1} w_t(s)$。

Theorem 2 的关键在于：最优 $\hat{A}^*$ 仅依赖于自举目标 $V_\text{target}$ 而不依赖于 $\hat{V}$（反之亦然）。这意味着可以通过迭代更新来交替优化 $\hat{A}$ 和 $\hat{V}$：令 $V_\text{target} = \hat{V}_{k-1}$，更新 $\hat{V}_k$，这恰好复现了多步 TD 学习。

虽然引入 $V_\text{target}$ 看似重新引入了优势函数对策略的依赖，但作者论证了：只要使用足够长的 backup horizon $n$（使 $n \gg 1$），且动作的因果效应是局部的（$p(s_{t'}|s_t, a_t) \approx p(s_{t'}|s_t)$），则当 $n$ 足够大时 $p(s_{t+n}|s_t, a_t) \approx p(s_{t+n}|s_t)$，从而 $\mathbb{E}[V_\text{target}(s_{t+n})|s_t, a_t] - \mathbb{E}[V_\text{target}(s_{t+n})|s_t] \approx 0$，对 $V_\text{target}$ 的依赖被削弱。

**实际损失函数。** 在实践中，采样 $n$-step 轨迹后，将每条子轨迹 $(s_i, a_i, \ldots, s_n)$（$i = 0, \ldots, n-1$）也视为合法轨迹，从而充分利用数据。最终损失为：

$$
L_A(\theta, \phi) = \mathbb{E}\left[\sum_{t=0}^{n-1} \left(\sum_{t'=t}^{n-1} \gamma^{t'-t} (r_{t'} - \hat{A}_\theta(s_{t'}, a_{t'})) + \gamma^{n-t} \hat{V}_\text{target}(s_n) - \hat{V}_\phi(s_t)\right)^2\right]
\tag{7}
$$

在实现中，$\hat{A}$ 和 $\hat{V}$ 共享网络主干（参数 $\theta$），输出三个分支分别对应 $\hat{A}(s,a)$、$\pi(a|s)$ 和 $\hat{V}(s)$。

### 3.4 Integration with PPO

DAE 与 PPO 的集成如 Algorithm 1 所示。每轮迭代中：
1. 用当前策略 $\pi$ 采样 $N_\text{actors}$ 条 $n$-step 轨迹。
2. 冻结一份网络参数 $\phi \leftarrow \text{copy}(\theta)$，用 $V_\phi$ 和 $\pi_\phi$ 作为目标。
3. 对每个 mini-batch：
   - 计算 $\hat{A}(s,a)$（按 $\pi_\phi$ 做 centering）。
   - 用 $\hat{A}$（stop gradient）计算 PPO clipping loss $L_\pi$。
   - 计算 DAE loss $L_A$。
   - 联合优化 $L = L_\pi + \beta_V L_A$。

PPO 的 clipping loss 为：

$$
L_\pi = \mathbb{E}\left[\min\left(\frac{\pi_\theta(a|s)}{\mu(a|s)} \hat{A}(s,a),\; \text{clip}\left(\frac{\pi_\theta(a|s)}{\mu(a|s)}, 1-\epsilon, 1+\epsilon\right) \hat{A}(s,a)\right)\right]
$$

其中 $\mu$ 是采样策略（即 $\pi_\phi$）。注意 DAE 不需要 GAE 中的 $\lambda$ 超参数，因为它不依赖 $\text{TD}(\lambda)$。

## 4. Experiments

实验在三类离散控制任务上比较 DAE 与 GAE：(1) 合成环境（Figure 1 的有限版本），(2) MinAtar（5 个 Atari 简化环境），(3) ALE（49 个完整 Atari 游戏）。

**合成环境。** 状态空间 $\mathcal{S} = \{s_1, \ldots, s_{128}\}$，使用 actor-critic 框架。评估指标为优势估计的 MSE 和策略的期望回报。结果（Figure 4）表明 DAE 能准确逼近真实优势函数，而 GAE 则受限于 $n$-step 回报的高方差。在策略优化方面，GAE 在早期表现更好（因为 GAE 的 $n$-step 回报包含无偏的有用信号，而 DAE 的函数逼近在初期偏差较大），但后期被 DAE 超越——随着 DAE 的函数逼近变得更准确，低方差的优势带来了更大的性能提升。

**MinAtar & ALE。** 使用 PPO 作为基础算法，DAE 调优的超参数包括值函数损失系数 $\beta_V$ 和每轮 epoch 数，均在 MinAtar 上选定后固定用于 ALE。在 MinAtar 上训练 10M 帧，ALE 上训练 40M 帧。

> 网络架构方面，除了 Baseline 网络外，还测试了 Wide（将各层通道/宽度扩大 4 倍或 2 倍）和 Deep（基于 IMPALA 的残差网络）变体。

结果总结（Table 1）：
- 在 MinAtar 的所有 5 个环境上，DAE 在 Baseline 和 Wide 架构下均优于 GAE（Overall 和 Last 两项指标）。
- 在 ALE 的 49 个环境中，DAE 在 Baseline 下赢得 32/49（Overall）和 30/49（Last），Wide 下赢得 35/49 和 34/49，Deep 下赢得 35/49 和 32/49。
- 增大网络容量对 DAE 的提升更为显著，尤其在 MinAtar 上切换到 Wide 网络后 DAE 的提升远超 GAE。这是因为 DAE 的优势估计直接依赖于网络表征能力，而 GAE 的优势估计主要来自 $n$-step 回报的加权平均，网络容量仅间接通过值函数影响。

**消融实验（Appendix D）。** 在 MinAtar 上比较 DAE 与两个额外 baseline：
1. **Indirect**：分别学习 $\hat{Q}$ 和 $\hat{V}$，再通过 $\hat{A} = \hat{Q} - \hat{V}$ 间接得到优势。
2. **Duel**：基于 Dueling Network 的 $n$-step 变体，使用学习到的策略（而非均匀策略）进行 centering。

结果表明 Indirect 表现最差，说明直接建模优势函数（而非间接经由 $Q - V$）是有益的。Duel 优于 Indirect 但不及 DAE，说明在时间步之间联合回归优势函数（DAE 的 loss 对沿轨迹的优势求和）是关键。

**实验局限性分析：**
- 所有实验均限于离散动作空间，连续控制任务（如 MuJoCo）缺席，限制了对方法通用性的评估。
- DAE 相比 GAE 需要额外调优 $\beta_V$（值函数损失系数），且由于按轨迹采样 mini-batch，batch 中的有效样本数取决于轨迹长度 $n$。
- 在 ALE 的部分环境（如困难探索环境 Montezuma's Revenge）中，DAE 和 GAE 的 baseline 均无法有效学习，这些环境在归一化比较中被排除。
- 论文未报告 DAE 相对于 GAE 的额外计算开销，尤其是 centering 操作和沿轨迹求和的代价。

## 5. Related Work & Future Work

**Related Work.** 优势函数最早在 *Advantage Updating* 中被引入，用于解决细粒度时间步上 Q-Learning 的困难。在策略优化中，优势函数被广泛用作 Q 函数的低方差替代。*Dueling Network Architectures for Deep Reinforcement Learning* 提出了联合估计 $V$ 和 $A$ 的网络架构，本文的工作在理论上为使用 $\pi$-centered 函数估计优势函数提供了正当性，并展示了如何与多步学习结合。*High-Dimensional Continuous Control Using Generalized Advantage Estimation* 通过 $\text{TD}(\lambda)$ 加权 TD 误差来估计优势函数，是当前最主流的方法。DAE 与 GAE 的根本区别在于：GAE 先学 $V$、再用 TD 误差间接得到 $A$，而 DAE 直接对 $A$ 建模。

因果推断与 RL 信用分配的联系也被其他工作探索过。*Counterfactual Credit Assignment in Model-Free Reinforcement Learning* 利用反事实的思想构建了条件于未来的值函数来降低策略梯度方差。*Disentangling Causal Effects for Hierarchical Reinforcement Learning* 在分层 RL 中用不同的优势函数定义来建模层次化的因果效应。

**Future Work.** 作者提出了三个扩展方向：
1. **部分可观测和含混淆因子的环境**：当前 DAE 假设完全可观测的 MDP。在部分可观测环境中，因果推断技术可能有助于处理未观测混淆因子。
2. **连续动作空间**：$\pi$-centered 约束在连续空间中需要对策略进行积分。一种可能的方案是使用采样方法来近似 centering 操作。
3. **Off-policy 扩展**：当前 DAE 是 on-policy 的。使用重要性采样技术（如 Retrace 或 V-Trace）可能将 DAE 扩展到 off-policy 设置。
