---
title: MAPS
headline: Active Policy Improvement from Multiple Black-box Oracles
---

> [!abstract]
> 

> [!info] Contributions
>
> 

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
\pi^\bullet(a \mid s) \doteq \pi^{k^\ast}(a \mid s), \quad k^\ast \doteq \operatorname*{\arg\max}_{k \in [K]} V_k(s). \tag{1}
$$


Max-following 策略可以被理解为一种贪心策略：在任意状态都跟随当前看起来最强的 Oracle。

**Max-aggregation $\pi_{\max}$**：本文使用 Max-aggregation 技术作为 benchmark，


<!--  
### （3）Max-aggregation (\pi_{\max})：在“最大值基线”上做一步前瞻改进

本文用 max-aggregation（Cheng et al., 2020）作为主要 benchmark：它基于 max-following 的思想，但会“向前看一步”。先定义一个自然的价值基线
[
f_{\max}(s)\triangleq \max_{k\in[K]} V_k(s). \tag{2}
]
再定义 max-aggregation 策略为
[
\pi_{\max}(a\mid s)\triangleq \delta_{a=a^\ast},\qquad
a^\ast=\arg\max_{a\in A} A_{f_{\max}}(s,a), \tag{3}
]
其中 (\delta) 是 Dirac delta 分布（补充：意味着 (\pi_{\max}) 在该定义下是**确定性**地选择使优势最大的动作）。 

当 oracle 集合 (\Pi) 只有一个 oracle（记为 (\pi_e)）时，一个标准做法是对 (\pi_e) 做**一步策略改进**得到 (\pi_e^+)：
[
\pi_e^+(s)=\arg\max_{a\in A}\left[r(s,a)+\mathbb{E}*{s'\sim P(\cdot\mid s,a)}\big[V^{\pi_e}(s')\big]\right]. \tag{4}
]
由于可保证 (V^{\pi_e^+}(s)\ge V^{\pi_e}(s))，(\pi_e^+) 在所有状态上都不差于 (\pi_e)。在单 oracle 情况下，(\pi^\bullet) 退化为 (\pi_e)，而 (\pi*{\max}) 退化为 (\pi_e^+)，因此 (\pi_{\max}) 会优于 (\pi^\bullet)。多 oracle 情况下，(\pi_{\max}) 和 (\pi^\bullet) 一般不可直接比较；除非存在某个 oracle 在所有状态都统一优于其他，此时同样可退化回上述单 oracle 结论。 

为了做**在线（online）模仿学习**并逼近 max-aggregation 策略，关键在于 (f_{\max})：它编码了“每个状态哪个 oracle 最强”的信息（见式(3)）。但 (f_{\max}) 需要知道每个 oracle 的价值函数；而在 episodic 的交互式 IL 设定中，oracle 是黑盒的，无法直接访问其价值函数。为解决这个问题，作者沿用以往工作，将 IL **约化为在线学习（online learning）问题**。 

由于 MDP 的转移与奖励未知，作者把第 (n) 轮策略 (\pi_n) 诱导的状态分布 (d^{\pi_n}) 视为在线学习中的“对手/环境”（即可能是任意分布）。于是定义第 (n) 轮的在线模仿学习损失为：
[
\ell^{\mathrm{IL}}*n(\pi)\triangleq -H\ \mathbb{E}*{s\sim d^{\pi_n}}\Big[ A_{f_{\max}}(s,\pi)\Big]. \tag{5}
]
（补充解释：这里 (A_{f_{\max}}(s,\pi)) 通常可理解为对 (a\sim \pi(\cdot\mid s)) 的优势期望，即 (\mathbb{E}*{a\sim\pi(\cdot\mid s)}[A*{f_{\max}}(s,a)])。） 

在本文中，作者进一步改写 Cheng et al. (2020) 的在线损失 (\ell_n(\pi;\lambda))，用来平衡：

* **纯 RL**：只用 learner 策略探索环境；
* **IL**：模仿 (\pi_{\max})。
  得到第 (n) 轮损失：
  [
  \ell_n(\pi;\lambda)\triangleq
  -(1-\lambda)H\ \mathbb{E}*{s\sim d^{\pi_n}}!\Big[A^{\lambda}*{f_{\max},\pi}(s,\pi)\Big]
  -\lambda\ \mathbb{E}*{s\sim d_0}!\Big[A^{\lambda}*{f_{\max},\pi}(s,\pi)\Big]. \tag{6}
  ]
  其中 (A^{\lambda}*{f*{\max},\pi}(s,a)) 是一个 (\lambda)-加权的优势：
  [
  A^{\lambda}*{f*{\max},\pi}(s,a)
  \triangleq (1-\lambda)\sum_{i=0}^{\infty}\lambda^i\ A^{(i)}*{f*{\max},\pi}(s,a). \tag{7}
  ]
  它把不同步数 (i) 的优势组合起来，而
  [
  A^{(i)}*{f*{\max},\pi}(s_t,a_t)\triangleq
  \mathbb{E}*{\tau_t\sim \rho^\pi(\cdot\mid s_t)}
  \Big[r(s_t,a_t)+\cdots+r(s*{t+i},a_{t+i})+f_{\max}(s_{t+i+1})\Big]-f_{\max}(s_t).
  ]
  （补充：这是把“未来 (i) 步累计奖励 + 末端基线 (f_{\max})”与当前基线 (f_{\max}(s_t)) 做差，形成多步优势信号。） 

接着，考虑在某个状态 (s_t) 处反复让第 (k) 个 oracle 起步 roll-out。收集到 (N_k(s_t)) 条轨迹 (\tau_{1,k},\ldots,\tau_{N_k,k}) 后，用这些轨迹的平均回报估计该状态的价值：
[
\hat V_k(s_t)\triangleq \hat V^{\pi_k}(s_t)
\triangleq
\frac{1}{N_k(s_t)}\sum_{i=1}^{N_k(s_t)}\sum_{j=0}^{H-1}\lambda^j r(s_j,a_j). \tag{8}
]
其中 (N_k(s_t)) 是从初始状态 (s_t) 出发、由 oracle (k) 产生的轨迹条数。 

---

## 3.2. Estimator for the Policy Gradient（策略梯度的估计器）

作者把 (\nabla \ell_n(\pi_n;\lambda)) 的经验估计定义为：
[
\widehat{\nabla \ell_n}(\pi_n;\lambda)
======================================

-H\ \mathbb{E}*{s\sim d^{\pi_n},\ a\sim \pi_n(\cdot\mid s)}
\Big[\nabla \log \pi_n(a\mid s)\ A^{\lambda}*{\hat f_{\max},\pi_n}(s,a)\Big]. \tag{9}
]
其中偏导数是对策略参数（记作 (\pi_n) 的参数）求的。由于每个 oracle 的真实价值函数未知，作者为每个 oracle (k) 用一个单独的函数逼近器 (\hat V_k(\cdot)) 来表示其价值函数。这样带来的逼近误差会进一步影响 (\hat f_{\max}(\cdot)) 的估计，而 (\hat f_{\max}(\cdot)) 又是式(9) 中计算策略梯度所必需的；因此，**(\hat f_{\max}) 的误差下降速度（learning speed）**会显著影响整体样本效率。 

（图 1 为方法概览。）作者指出先前 SOTA 方法 MAMBA（Cheng et al., 2020）的一个局限是样本复杂度高：它基于 (\hat f_{\max}(s)) 来估计策略梯度，但为了在某个状态识别最优 oracle，它采用“对 oracle 做均匀随机采样”的策略，往往需要很长的 episode 才能确定哪个 oracle 更好；一旦识别失败，误差（从而 regret）会大量累积（见文中 Theorem 1）。此外，MAMBA 在选择 roll-out 的状态时并不控制梯度估计的逼近误差。本文则旨在通过**主动选择 oracle**以及通过**主动状态探索**来控制逐状态不确定性，从而降低估计误差。 
 -->

## 4. Algorithm

## 5. Theoretical Analysis

