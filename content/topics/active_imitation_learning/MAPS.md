---
title: MAPS
headline: Active Policy Improvement from Multiple Black-box Oracles
---

> [!abstract]

> [!info] Contributions

## 1. Introduction

## 2. Related Work

## 3. Preliminaries

### 3.1 Reinforcement Learning Setup

本文考虑的是一个有限时域的 MDP $\mathcal{M}_0 = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, r, H \rangle$，策略 $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ 将当前状态映射为动作分布。我们有一组 $K$ 个黑盒 Oracle $\Pi = \{\pi_k\}_{k=1}^K$，总集/Episode 数为 $N$。对一个给定的函数 $f: \mathcal{S} \to \mathbb{R}$，定义相对于 $f$ 的广义 $Q$ 函数：

$$
Q^f(s, a) \doteq r(s, a) + \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)}[f(s')]
$$

当 $f(s)$ 取为某个策略 $\pi$ 的价值函数 $V^\pi(s)$ 时，上式就退化为该策略的标准 $Q$ 函数 $Q^\pi(s, a)$。令 $d_t^\pi \in \Delta(\mathcal{S})$ 表示在初始状态分布 $d_0 \in \Delta(\mathcal{S})$ 下，执行策略 $\pi$ 于时间步 $t$ 的状态分布。则策略 $\pi$ 的平均状态分布可以写为

$$
d^\pi \doteq \frac{1}{H} \sum_{t=0}^{H-1} d_t^\pi.
$$

因此，在初始分布 $d_0$ 下，策略 $\pi$ 的价值函数为

$$
V^\pi(d_0) \doteq \mathbb{E}_{s_0 \sim d_0}[V^\pi(s_0)] \doteq \mathbb{E}_{s_0 \sim d_0} \left[ \mathbb{E}_{\tau_0 \sim \rho^\pi(\cdot \mid s_0)} \left[ \sum_{t=0}^{H-1} r(s_t, a_t) \right] \right],
$$

这里 $\rho^\pi(\tau_t \mid s_t)$ 表示从状态 $s_t$ 出发、按策略 $\pi$ 生成的后续轨迹分布。本文目标是找到一个策略函数 $\pi$，最大化相对于初始分布 $d_0$ 的 $H$ 步累计回报。与之相关的优势函数定义为

$$
A^f(s, a) \doteq Q^f(s, a) - f(s) \doteq r(s, a) + \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)}[f(s')] - f(s).
$$

### 3.2 Algorithms for Learning from Multiple Oracles

考虑智能体可以访问一组黑盒 Oracles $\Pi = \{\pi_k\}_{k=1}^K$，并讨论若干种从该集合学习的思路。

**Single-best Oracle $\pi^\star$**：最简单最基础的 baseline，选择一个整体上最好的 Oracle $\pi^\star$，其定义为事后最优/Hindsight Optimal：$\pi^\star \coloneqq \arg\max_{\pi \in \Pi} V^\pi(d_0)$。但这个 baseline 不能体现算法优越性，因为它没有利用不同 Oracle 在不同状态下各有所长的**逐状态最优性**。

**Max-following $\pi^\bullet$**：由于最优 Oracle 会随状态变化，可以使用每个 Oracle 在状态 $s$ 的价值 $V_k(s)$ 来表达其在该状态下的专业程度。 Max-following 策略在每个状态独立选择价值最大的 Oracle：

$$
\pi^\bullet(a \mid s) \doteq \pi^{k^\star}(a \mid s), \quad k^\star \doteq \operatorname*{\arg\max}_{k \in [K]} V_k(s). \tag{1}
$$

Max-following 策略可以被理解为一种贪心策略：在任意状态都跟随当前看起来最强的 Oracle。

**Max-aggregation $\pi_{\max}$**：本文使用 Max-aggregation 技术作为 benchmark，其相对于 Max-following 策略进行一步的策略改进。定义基线价值函数为

$$
f^{\max}(s) \doteq \max_{k \in [K]} V_k(s).
$$

则 Max-aggregation 策略为

$$
\pi^{\max} (a \mid s) \doteq \delta_{a=a^\star}, \quad a^\star \doteq \operatorname*{\arg\max}_{a \in \mathcal{A}} A^{f^{\max}}(s, a).
$$

在单策略的情况下，这就是一步策略改进，因此可以保证不差于 Max-following 策略。在多策略的情况下，Max-aggregation 策略和 Max-following 策略一般不可直接比较；除非存在某个 Oracle 在所有状态都统一优于其他 Oracle。

算法的关键是高效地计算出 $f^{\max}(s)$，需要知道每一个专家在每个状态的价值函数 $V_k(s)$。当前我们的设定是分幕式交互模仿学习，无法访问专家的价值函数，但是可以遵循前面 AggreVaTe/AggreVaTeD 的思路，将模仿学习规约为一个在线学习问题，从而估计出每个专家在每个状态的价值函数。

我们将

<!--

因此，我们将第 $n$ 轮的在线模仿学习损失（loss）定义如下：

$$l_{n}^{IL}(\pi) \doteq -H \mathbb{E}_{s\sim d^{\pi_{n}}}[A^{f^{max}}(s,\pi)] \quad (5)$$


💡 补充说明：为什么要视为“对手 (Adversary)”？
在 Interactive IL (如 DAgger) 中，训练数据的分布是由当前策略 $\pi_n$ 诱导产生的 ($d^{\pi_n}$)。
如果我们假设数据是独立同分布（i.i.d.）的，一旦策略更新变成 $\pi_{n+1}$，数据分布就会发生漂移（Distribution Shift），导致原来的误差界失效。
通过将 $d^{\pi_n}$ 视为“对手”选择的分布，我们就不需要假设分布是固定的。通过使用无悔（No-Regret）在线学习算法，我们可以保证无论对手（分布）如何变化，策略的平均表现都能收敛到最优。
公式 (5) 的含义是：我们要找一个策略 $\pi$，使其在当前分布 $d^{\pi_n}$ 下，尽可能地去最大化相对于“最大价值包络” $f^{max}$ 的优势函数。



-->

## 4. Algorithm

## 5. Theoretical Analysis
