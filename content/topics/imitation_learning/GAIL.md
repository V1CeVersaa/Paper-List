---
title: "GAIL"
headline: "Generative Adversarial Imitation Learning"
visibility: "public"
status: "drafting"
description: "Paper note on Generative Adversarial Imitation Learning."
---

> [!abstract] Contributions
>
> 本文解决的核心问题是：传统的模仿学习/Imitation Learning/IL 方法要么依赖行为克隆/Behavioral Cloning/BC 而遭受复合误差/Compounding Error，要么通过逆强化学习/Inverse Reinforcement Learning/IRL 间接地先恢复代价函数再用强化学习/Reinforcement Learning/RL 提取策略，计算代价高昂且绕了弯路。GAIL 的关键技术洞察是：将 IRL 与后续 RL 步骤合并分析后，整个过程可以被刻画为一个直接在策略空间上的优化问题——最小化学习策略与专家策略的占用度量/Occupancy Measure 之间的差异，并以策略的因果熵/Causal Entropy 作为正则化。当使用特定的代价函数正则化器 $\psi_\text{GA}$ 时，该差异恰好等价于 Jensen-Shannon 散度/JS Divergence，从而自然地与生成对抗网络/Generative Adversarial Network/GAN 建立类比：策略充当生成器，判别器区分学习策略与专家策略的状态-动作对。由此得到的 GAIL 算法是 model-free 的，能够在高维连续控制任务上以少量专家演示数据实现显著优于 BC、特征期望匹配/Feature Expectation Matching/FEM 和博弈论学徒学习/Game-Theoretic Apprenticeship Learning/GTAL 的表现。
>
> 该方法的主要限制在于：GAIL 在专家数据方面较为高效，但在环境交互方面效率较低（训练所需的环境交互量与直接用 RL 从奖励信号训练相当）；此外，作为 model-free 方法，它天然比 model-based 方法需要更多环境交互。论文的理论分析基于有限状态-动作空间和所有函数类 $\mathcal{C} = \mathbb{R}^{\mathcal{S} \times \mathcal{A}}$ 的设定，向连续空间和有限函数类的推广需要额外的技术处理。

## 1. Introduction

模仿学习关注的是仅从专家演示中学习执行任务的策略，而不需要与专家交互、也不需要获取强化信号。在这一设定下，现有方法主要分为两条路径：

- **行为克隆**将模仿学习简化为监督学习，直接从状态-动作对拟合策略。它简单直接，但由于协变量偏移/Covariate Shift，单步预测误差会随时间步累积，导致策略在长时间段后严重偏离专家行为。这要求大量的训练数据来缓解。
- **逆强化学习**从演示中恢复一个代价函数/Cost Function，使得专家策略在该代价下是最优的，然后在恢复的代价函数上运行 RL 来提取策略。IRL 学习的是全局的轨迹级目标，不受复合误差困扰，但计算代价高昂——许多 IRL 算法需要在内循环中反复运行 RL。更根本的问题是：IRL 学习的是代价函数而非策略，但学习者的最终目标是学会如何行动；如果运行 RL 提取策略的过程本身并不能从代价函数中获益（即最优策略的质量并不依赖于代价函数的表达形式），那么为什么要花费大量计算去学习一个代价函数？

本文的核心动机正是回答这一问题：能否直接从演示数据中提取策略，跳过 IRL 这一中间步骤？为此，论文从最大因果熵 IRL/Maximum Causal Entropy IRL 出发，分析「先做 IRL 再做 RL」这一复合过程的数学结构。关键发现是：这一复合过程等价于在策略的占用度量空间上求解一个凸优化问题（Proposition 3.2），其目标是找到一个占用度量接近专家的、同时具有高熵的策略。不同的 IRL 正则化器 $\psi$ 对应不同的「接近」度量，从而统一了多种已有的模仿学习算法。

在此框架下，论文提出了一种新的正则化器 $\psi_\text{GA}$，其对应的占用度量差异恰好是 JS 散度——一个真正的度量，不像线性学徒学习/Apprenticeship Learning 方法那样受限于有限维特征空间。由此导出的算法与 GAN 有深刻的结构类比：策略（生成器）试图产生与专家不可区分的状态-动作对，判别器试图将二者区分开。论文在 9 个基于物理仿真的控制任务上验证了该算法，包括高维连续控制任务（如 Humanoid），结果显示 GAIL 在多种数据量设定下均显著优于 BC、FEM 和 GTAL。

## 2. Problem Setup

考虑 $\gamma$-折扣无限时域设定。状态空间 $\mathcal{S}$ 和动作空间 $\mathcal{A}$ 均为有限集（理论分析的基础设定，实验中扩展到连续空间）。环境动力学由转移概率 $P(s'|s,a)$ 描述。$\Pi$ 是所有在 $\mathcal{S}$ 上取动作于 $\mathcal{A}$ 的平稳随机策略的集合。策略 $\pi \in \Pi$ 下的期望代价定义为：

$$
\mathbb{E}_\pi[c(s,a)] \triangleq \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t c(s_t, a_t)\right]
$$

其中 $s_0 \sim p_0$，$a_t \sim \pi(\cdot|s_t)$，$s_{t+1} \sim P(\cdot|s_t, a_t)$。专家策略记为 $\pi_E$，在实际中仅以一组轨迹样本 $\tau_E \sim \pi_E$ 的形式提供。

**最大因果熵 IRL** 是本文理论推导的起点。给定代价函数族 $\mathcal{C}$，IRL 求解以下优化问题：

$$
\max_{c \in \mathcal{C}} \left(\min_{\pi \in \Pi} -H(\pi) + \mathbb{E}_\pi[c(s,a)]\right) - \mathbb{E}_{\pi_E}[c(s,a)]
\tag{1}
$$

其中 $H(\pi) \triangleq \mathbb{E}_\pi[-\log \pi(a|s)]$ 是策略 $\pi$ 的 $\gamma$-折扣因果熵。直觉上，IRL 寻找一个代价函数 $c$，使得专家策略的期望代价低、其他策略的期望代价高（在熵正则化意义下）。内层最小化定义了 RL 过程：

$$
\text{RL}(c) = \arg\min_{\pi \in \Pi} -H(\pi) + \mathbb{E}_\pi[c(s,a)]
\tag{2}
$$

即在给定代价函数下找到最小化期望代价同时最大化熵的策略。

**占用度量**是连接策略空间与凸优化的桥梁。对于策略 $\pi \in \Pi$，其占用度量 $\rho_\pi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 定义为：

$$
\rho_\pi(s,a) = \pi(a|s) \sum_{t=0}^{\infty} \gamma^t P(s_t = s | \pi)
$$

它可以理解为智能体在策略 $\pi$ 下访问状态-动作对 $(s,a)$ 的（折扣）频率分布。占用度量的一个关键性质是：它允许将策略上的期望改写为占用度量上的线性函数，即 $\mathbb{E}_\pi[c(s,a)] = \sum_{s,a} \rho_\pi(s,a) c(s,a)$。

合法占用度量的集合 $\mathcal{D}$ 可以写成仿射约束定义的凸集：

$$
\mathcal{D} = \left\{\rho : \rho \geq 0 \text{ and } \sum_a \rho(s,a) = p_0(s) + \gamma \sum_{s',a} P(s|s',a)\rho(s',a) \;\; \forall s \in \mathcal{S}\right\}
$$

> [!note] Proposition 3.1（策略与占用度量的一一对应）
> 若 $\rho \in \mathcal{D}$，则 $\rho$ 是策略 $\pi_\rho$ 的占用度量，其中 $\pi_\rho(a|s) \triangleq \rho(s,a) / \sum_{a'} \rho(s,a')$，且 $\pi_\rho$ 是唯一以 $\rho$ 为占用度量的策略。

这一命题意味着策略空间 $\Pi$ 与占用度量空间 $\mathcal{D}$ 之间存在双射，因此在策略上的优化可以等价地转化为在占用度量上的优化。后者的优势在于 $\mathcal{D}$ 是凸集，且期望代价在 $\rho$ 上是线性的，从而使整个问题具有凸优化结构。

## 3. Methods

### 3.1 From IRL to Occupancy Measure Matching

论文的理论核心是分析「IRL 后接 RL」这一复合过程（$\text{RL} \circ \text{IRL}_\psi$）在最大表达力的代价函数族 $\mathcal{C} = \mathbb{R}^{\mathcal{S} \times \mathcal{A}}$ 下的行为。为防止在有限数据上过拟合，引入代价函数正则化器 $\psi: \mathbb{R}^{\mathcal{S} \times \mathcal{A}} \to \bar{\mathbb{R}}$（闭的、proper 的凸函数），其中 $\bar{\mathbb{R}} = \mathbb{R} \cup \{+\infty\}$ 是扩展实数。$\psi$-正则化的 IRL 定义为：

$$
\text{IRL}_\psi(\pi_E) = \arg\max_{c \in \mathbb{R}^{\mathcal{S} \times \mathcal{A}}} -\psi(c) + \left(\min_{\pi \in \Pi} -H(\pi) + \mathbb{E}_\pi[c(s,a)]\right) - \mathbb{E}_{\pi_E}[c(s,a)]
\tag{3}
$$

> [!note] Proposition 3.2（IRL 后接 RL 的等价刻画）
>
> $$
> \text{RL} \circ \text{IRL}_\psi(\pi_E) = \arg\min_{\pi \in \Pi} -H(\pi) + \psi^*(\rho_\pi - \rho_{\pi_E})
> \tag{4}
> $$
>
> 其中 $\psi^*$ 是 $\psi$ 的凸共轭/Convex Conjugate，定义为 $f^*(x) = \sup_{y \in \mathbb{R}^{\mathcal{S} \times \mathcal{A}}} x^T y - f(y)$。

> [!todo]- Proof of Proposition 3.2
> 证明利用了最优代价函数和最优策略构成一个鞍点。定义 $\tilde{L}(\rho, c) = -\bar{H}(\rho) + \sum_{s,a} c(s,a)(\rho(s,a) - \rho_E(s,a))$，其中 $\bar{H}(\rho) = -\sum_{s,a} \rho(s,a) \log(\rho(s,a)/\sum_{a'}\rho(s,a'))$ 是占用度量上的因果熵。设 $\tilde{c} \in \text{IRL}_\psi(\pi_E)$，$\tilde{\pi} \in \text{RL}(\tilde{c})$，$\pi_A \in \arg\min_\pi -H(\pi) + \psi^*(\rho_\pi - \rho_{\pi_E})$。
>
> 定义 $\bar{L}(\rho, c) = -H(\rho) - \psi(c) + \sum_{s,a} (\rho(s,a) - \rho_{\pi_E}(s,a)) c(s,a)$。由 Proposition 3.1 和 Lemma 3.1（$H(\pi) = \bar{H}(\rho_\pi)$），可以在策略和占用度量之间自由切换。关键步骤：
>
> 1. $\rho_A \in \arg\min_{\rho \in \mathcal{D}} \max_c \bar{L}(\rho, c)$（$\pi_A$ 的占用度量）
> 2. $\tilde{c} \in \arg\max_c \min_{\rho \in \mathcal{D}} \bar{L}(\rho, c)$（IRL 的输出）
> 3. $\tilde{\rho} \in \arg\min_{\rho \in \mathcal{D}} \bar{L}(\rho, \tilde{c})$（RL 的输出的占用度量）
>
> 由于 $\mathcal{D}$ 紧凸、$\mathbb{R}^{\mathcal{S} \times \mathcal{A}}$ 凸，$\bar{L}(\cdot, c)$ 因 $-\bar{H}$ 和 $\psi$ 的凸性而对 $\rho$ 凸，$\bar{L}(\rho, \cdot)$ 对 $c$ 凹，由 minimax 定理得 $\min_\rho \max_c \bar{L} = \max_c \min_\rho \bar{L}$。因此 $(\rho_A, \tilde{c})$ 是鞍点，从 (1) 和 (2) 推出 $\rho_A \in \arg\min_\rho \bar{L}(\rho, \tilde{c})$。又因 $\bar{L}(\cdot, c)$ 严格凸（Lemma 3.1 保证 $-\bar{H}$ 严格凸），最小化点唯一，故 $\rho_A = \tilde{\rho}$，由 Proposition 3.1 得 $\pi_A = \tilde{\pi}$。

Proposition 3.2 揭示了一个深刻的等价关系：$\psi$-正则化的 IRL 后接 RL，本质上是在寻找一个占用度量接近专家的（由 $\psi^*$ 度量）、同时具有高熵的策略。正则化器 $\psi$ 的选择决定了「接近」的含义，不同的 $\psi$ 对应不同的模仿学习算法。

这一视角还揭示了 IRL 的对偶结构：**IRL 是占用度量匹配问题的对偶**。经典 IRL 算法在内循环中反复运行 RL（如价值迭代），本质上是在求解对偶问题的一个坐标；而 RL 步骤恢复原始问题的解。传统上 IRL 被定义为寻找使专家最优的代价函数，但现在可以等价地将其视为诱导一个匹配专家占用度量的策略的过程。

**$\psi$ 为常函数的特殊情况**（Corollary 3.2.1）：若 $\psi$ 是常函数（即不对代价函数施加任何正则化），则恢复的策略 $\tilde{\pi}$ 满足 $\rho_{\tilde{\pi}} = \rho_{\pi_E}$，即精确匹配专家的占用度量。此时优化问题退化为：

$$
\min_{\rho \in \mathcal{D}} -\bar{H}(\rho) \quad \text{subject to} \quad \rho(s,a) = \rho_E(s,a) \;\; \forall s \in \mathcal{S}, a \in \mathcal{A}
\tag{5}
$$

> [!todo]- Proof of Corollary 3.2.1
> 当 $\psi$ 为常函数时，其凸共轭 $\psi^*$ 是原点的指示函数：$\psi^*(x) = 0$ 若 $x = 0$，否则 $\psi^*(x) = +\infty$（因为 $\psi^*(x) = \sup_y x^T y - \text{const}$，仅当 $x = 0$ 时有限）。代入 Proposition 3.2 的 $(4)$ 式，优化问题退化为在约束 $\rho_\pi = \rho_{\pi_E}$ 下最大化熵。
>
> 等价地，由 Lemma 3.2（$L(\pi, c) = \bar{L}(\rho_\pi, c)$），将 IRL 目标改写为：
>
> $$
> \tilde{c} \in \arg\max_c \min_{\rho \in \mathcal{D}} -\bar{H}(\rho) + \sum_{s,a} \rho(s,a)c(s,a) - \sum_{s,a} \rho_E(s,a)c(s,a) + \text{const}
> $$
>
> $$
> = \arg\max_c \min_{\rho \in \mathcal{D}} \tilde{L}(\rho, c)
> $$
>
> 这是一个以 $c(s,a)$ 为拉格朗日乘子、$\rho(s,a) = \rho_E(s,a)$ 为等式约束的对偶问题。$\tilde{c}$ 是对偶最优解；由 $\mathcal{D}$ 紧凸、$-\bar{H}$ 严格凸（Lemma 3.1），强对偶成立，原始最优解 $\tilde{\rho} = \rho_E$，因此 $\rho_{\tilde{\pi}} = \tilde{\rho} = \rho_E$。

### 3.2 Unifying Existing Algorithms: Entropy-Regularized Apprenticeship Learning

在推导 GAIL 之前，论文首先展示 Proposition 3.2 的框架如何统一已有的学徒学习算法。学徒学习的目标是：

$$
\min_\pi \max_{c \in \mathcal{C}} \mathbb{E}_\pi[c(s,a)] - \mathbb{E}_{\pi_E}[c(s,a)]
\tag{6}
$$

其中 $\mathcal{C}$ 是一族代价函数。经典方法通过基函数 $f_1, \ldots, f_d$ 定义特征向量 $f(s,a) = [f_1(s,a), \ldots, f_d(s,a)]$，并在此基础上构造线性代价函数族。两种典型选择：

- $\mathcal{C}_\text{linear} = \{\sum_i w_i f_i : \|w\|_2 \leq 1\}$，对应特征期望匹配（最小化 $\ell_2$ 距离 $\|\mathbb{E}_\pi[f(s,a)] - \mathbb{E}_{\pi_E}[f(s,a)]\|_2$）
- $\mathcal{C}_\text{convex} = \{\sum_i w_i f_i : \sum_i w_i = 1, w_i \geq 0\}$，对应 MWAL 和 LPAL（最小化各基函数上的最大超额代价）

论文指出，学徒学习目标 $(6)$ 是 Proposition 3.2 框架中 $\psi = \delta_\mathcal{C}$（代价函数族 $\mathcal{C}$ 的指示函数）的特殊情况。具体地：

$$
\max_{c \in \mathcal{C}} \mathbb{E}_\pi[c(s,a)] - \mathbb{E}_{\pi_E}[c(s,a)] = \max_{c \in \mathbb{R}^{\mathcal{S} \times \mathcal{A}}} -\delta_\mathcal{C}(c) + \sum_{s,a} (\rho_\pi(s,a) - \rho_{\pi_E}(s,a)) c(s,a) = \delta_\mathcal{C}^*(\rho_\pi - \rho_{\pi_E})
$$

因此熵正则化学徒学习等价于在 $\psi = \delta_\mathcal{C}$ 下的 $\text{RL} \circ \text{IRL}_\psi$。

**学徒学习的优缺点**：优点在于受限的 $\mathcal{C}$ 可以通过策略函数逼近扩展到大规模状态-动作空间，且可以利用策略梯度公式进行高效优化。*Model-Free Imitation Learning with Policy Optimization* 利用以下策略梯度公式并结合 TRPO 步进行优化：

$$
\nabla_\theta \max_{c \in \mathcal{C}} \mathbb{E}_{\pi_\theta}[c(s,a)] - \mathbb{E}_{\pi_E}[c(s,a)] = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q_{c^*}(s,a)]
\tag{7}
$$

其中 $c^* = \arg\max_{c \in \mathcal{C}} \mathbb{E}_{\pi_\theta}[c(s,a)] - \mathbb{E}_{\pi_E}[c(s,a)]$，$Q_{c^*}(\bar{s},\bar{a}) = \mathbb{E}_{\pi_\theta}[c^*(\bar{s},\bar{a}) | s_0 = \bar{s}, a_0 = \bar{a}]$。该公式恰好是以 $c^*$ 为代价函数的 RL 策略梯度，使得算法可以交替进行代价拟合和策略更新。

缺点在于：线性代价函数族的表达力有限。如果 $\mathcal{C}$ 不包含使专家策略唯一最优的代价函数，那么即使 $\pi$ 在 $\mathcal{C}$ 上的所有元素上都优于 $\pi_E$，也无法保证 $\pi = \pi_E$。从 Proposition 3.2 的角度理解，学徒学习强制 $\pi_E$ 的代价函数必须可以编码为 $\mathcal{C}$ 的元素；如果 $\mathcal{C}$ 不包含这样的编码，就无法恢复专家行为。

### 3.3 GAIL: From a New Regularizer to Adversarial Training

学徒学习的问题在于受限的代价函数族无法保证精确匹配占用度量，而常函数正则化器虽然能精确匹配但在大环境中不可行（因为需要对每个状态-动作对施加约束）。论文提出了一种新的正则化器 $\psi_\text{GA}$，兼顾两者的优点：

$$
\psi_\text{GA}(c) \triangleq \begin{cases} \mathbb{E}_{\pi_E}[g(c(s,a))] & \text{if } c < 0 \\ +\infty & \text{otherwise} \end{cases}, \quad g(x) = \begin{cases} -x - \log(1-e^x) & \text{if } x < 0 \\ +\infty & \text{otherwise} \end{cases}
\tag{8}
$$

$\psi_\text{GA}$ 对赋予专家状态-动作对负代价（接近零）的代价函数施加较低惩罚；如果代价函数对专家赋予过大的代价，则 $\psi_\text{GA}$ 会严重惩罚。与学徒学习使用的指示函数正则化器 $\delta_\mathcal{C}$ 相比，关键差异在于：$\delta_\mathcal{C}$ 将代价函数限制在有限维基函数张成的子空间中，而 $\psi_\text{GA}$ 允许任意代价函数，只要它对专家样本的取值是负的。

> [!note] Corollary A.1.1（$\psi_\text{GA}^*$ 的显式形式）
>
> $$
> \psi_\text{GA}^*(\rho_\pi - \rho_{\pi_E}) = \max_{D \in (0,1)^{\mathcal{S} \times \mathcal{A}}} \mathbb{E}_\pi[\log(D(s,a))] + \mathbb{E}_{\pi_E}[\log(1 - D(s,a))]
> \tag{9}
> $$
>
> 其中最大化是在所有判别分类器 $D: \mathcal{S} \times \mathcal{A} \to (0,1)$ 上进行的。

> [!todo]- Proof of Corollary A.1.1
> 证明利用了 Proposition A.1（附录中的一般性结果）的特殊情况。取 logistic 损失 $\phi(x) = \log(1 + e^{-x})$（严格递减凸函数），定义：
>
> $$
> g_\phi(x) = \begin{cases} -x + \phi(-\phi^{-1}(-x)) & \text{if } x \in T \\ +\infty & \text{otherwise} \end{cases}, \quad \psi_\phi(c) = \begin{cases} \sum_{s,a} \rho_{\pi_E}(s,a) g_\phi(c(s,a)) & \text{if } c(s,a) \in T \; \forall s,a \\ +\infty & \text{otherwise} \end{cases}
> $$
>
> 其中 $T$ 是 $-\phi$ 的值域。由 Proposition A.1，$\text{RL} \circ \text{IRL}_{\psi_\phi}(\pi_E) = \arg\min_\pi -H(\pi) - R_\phi(\rho_\pi, \rho_{\pi_E})$，其中 $R_\phi$ 是基于 $\phi$ 的最小期望风险。对于 logistic 损失，验证 $g_\phi$ 化简后恰好等于 $\psi_\text{GA}$ 中的 $g$，且 $\psi_\phi = \psi_\text{GA}$。
>
> 计算 $\psi_\text{GA}^*$：
>
> $$
> \psi_\text{GA}^*(\rho_\pi - \rho_{\pi_E}) = \max_c \sum_{s,a} (\rho_\pi(s,a) - \rho_{\pi_E}(s,a)) c(s,a) - \sum_{s,a} \rho_{\pi_E}(s,a) g_\phi(c(s,a))
> $$
>
> 逐 $(s,a)$ 最大化，令 $c \to -\phi(\gamma)$（变量替换 $\gamma \in \mathbb{R}$），利用 $\sigma(\gamma) = 1/(1+e^{-\gamma})$，最终化简为：
>
> $$
> = \sum_{s,a} \max_{d \in (0,1)} \rho_\pi(s,a) \log d + \rho_{\pi_E}(s,a) \log(1-d) = \max_{D \in (0,1)^{\mathcal{S}\times\mathcal{A}}} \mathbb{E}_\pi[\log D(s,a)] + \mathbb{E}_{\pi_E}[\log(1-D(s,a))]
> $$

$(9)$ 式的右端恰好是二分类问题的最优负对数损失——判别器 $D$ 试图区分来自 $\pi$ 和 $\pi_E$ 的状态-动作对。其最优值（差一个常数）就是 JS 散度 $D_\text{JS}(\rho_\pi, \rho_{\pi_E}) \triangleq D_\text{KL}(\rho_\pi \| (\rho_\pi + \rho_E)/2) + D_\text{KL}(\rho_E \| (\rho_\pi + \rho_E)/2)$。

将 $(9)$ 代入 Proposition 3.2 的框架 $(4)$，并将因果熵 $H$ 作为由 $\lambda \geq 0$ 控制的正则化器，得到 GAIL 的优化目标：

$$
\min_\pi \psi_\text{GA}^*(\rho_\pi - \rho_{\pi_E}) - \lambda H(\pi) = D_\text{JS}(\rho_\pi, \rho_{\pi_E}) - \lambda H(\pi)
\tag{10}
$$

这意味着 GAIL 寻找一个占用度量在 JS 散度意义下最接近专家的策略。由于 JS 散度是一个真正的度量（对称且满足三角不等式的平方根形式），不像线性学徒学习方法仅在有限维特征子空间上匹配，GAIL 可以精确模仿专家策略。

### 3.4 Algorithm

$(10)$ 式与 GAN 之间的结构类比是直接的：GAN 训练生成模型 $G$ 使其生成的数据分布骗过判别器 $D$；GAIL 训练策略 $\pi$ 使其占用度量 $\rho_\pi$ 骗过判别器 $D$。具体地，GAIL 寻找表达式

$$
\mathbb{E}_\pi[\log(D(s,a))] + \mathbb{E}_{\pi_E}[\log(1 - D(s,a))] - \lambda H(\pi)
\tag{11}
$$

的鞍点 $(\pi, D)$。引入参数化策略 $\pi_\theta$ 和判别网络 $D_w: \mathcal{S} \times \mathcal{A} \to (0,1)$，算法交替执行：

1. 从当前策略 $\pi_{\theta_i}$ 采样轨迹 $\tau_i$
2. 对判别器参数 $w$ 进行 Adam 梯度上升以增大 $(11)$，梯度为：

$$
\hat{\mathbb{E}}_{\tau_i}[\nabla_w \log D_w(s,a)] + \hat{\mathbb{E}}_{\tau_E}[\nabla_w \log(1 - D_w(s,a))]
$$

3. 对策略参数 $\theta$ 进行 TRPO 步以减小 $(11)$，具体为 KL 约束的自然梯度步：

$$
\hat{\mathbb{E}}_{\tau_i}[\nabla_\theta \log \pi_\theta(a|s) Q(s,a)] - \lambda \nabla_\theta H(\pi_\theta)
\tag{12}
$$

其中 $Q(\bar{s},\bar{a}) = \hat{\mathbb{E}}_{\tau_i}[\log(D_{w_{i+1}}(s,a)) | s_0 = \bar{s}, a_0 = \bar{a}]$。

> [!tip] 判别器作为动态代价函数
> 判别器 $D$ 可以被理解为为策略提供局部代价信号的动态代价函数：$c(s,a) = \log D(s,a)$。策略更新减小该代价的期望值，即让策略向判别器认为「像专家」的状态-动作区域移动。随着策略改进，判别器也随之更新以保持区分能力，形成交替优化的对抗过程。

TRPO 步的作用与学徒学习算法中相同：通过 KL 散度约束确保策略更新步长不会过大（防止因策略梯度中的高方差导致性能崩溃）。$(12)$ 式的结构与 $(7)$ 式完全对应——策略梯度的形式一致，唯一不同的是代价函数 $c^*$ 被替换为判别器输出 $\log D$。

## 4. Experiments

### 4.1 Experimental Setup

论文在 9 个基于物理仿真的控制任务上评估 GAIL（Algorithm 1），涵盖从经典低维控制任务（Cartpole、Acrobot、Mountain Car）到高维连续控制任务（HalfCheetah、Hopper、Walker、Ant、Humanoid）。除经典任务外，其余环境均使用 MuJoCo 仿真器。专家策略通过在真实代价函数上运行 TRPO 生成，专家数据集由不同数量的轨迹组成（每条约 50 个状态-动作对）。

对比的三个基线：

- **Behavioral Cloning**：监督学习，70/30 训练/验证划分，minibatch 大小 128
- **FEM**（Feature Expectation Matching）：使用 $\mathcal{C}_\text{linear}$ 的学徒学习
- **GTAL**（Game-Theoretic Apprenticeship Learning）：使用 $\mathcal{C}_\text{convex}$ 的学徒学习

所有算法使用相同的两层全连接网络（100 单元，tanh 激活）。GAIL 的判别网络也采用相同架构。FEM、GTAL 和 GAIL 使用相同的环境交互量。

### 4.2 Main Results

<!-- TODO: insert Figure 1(a) from paper，展示 9 个任务上不同数据量下各方法的性能对比 -->

**经典控制任务**（Cartpole、Acrobot、Mountain Car）：BC 的数据效率较差；FEM 和 GTAL 在大多数数据量下能达到接近专家的性能；GAIL 始终优于或持平。但在 Reacher 任务上，BC 表现优异且比 GAIL 更数据高效。

**MuJoCo 高维任务**：GAIL 相对于所有基线有显著优势。关键观察：

- GAIL 在几乎所有数据量设定下都达到了至少 70% 的专家性能
- FEM 和 GTAL 在 Ant 任务上表现极差，甚至不如随机策略——这可以从 Proposition 3.2 的角度理解：线性代价函数族 $\mathcal{C}_\text{linear}$ 和 $\mathcal{C}_\text{convex}$ 无法编码使专家在 Ant 这样的高维任务上最优的代价函数
- BC 在 HalfCheetah、Hopper、Walker、Ant 上需要大量数据才能达到可接受的性能，在 Humanoid（376 维观测）上即使数据量最大也无法超过 60% 的专家性能；而 GAIL 在所有测试数据量下都达到了近乎完美的专家性能

**因果熵正则化的效果**：在 Reacher 的 4 条轨迹设定下，从 $\lambda = 0$ 到 $\lambda = 10^{-3}$ 的提升在 Wilcoxon 秩和检验下具有统计显著性（$p = 0.05$）。其余任务未使用因果熵正则化（$\lambda = 0$）。

### 4.3 Analysis of Experimental Limitations

- **基线范围有限**：仅对比了 BC 和两种线性学徒学习方法（FEM、GTAL），缺少与其他 IRL 方法（如基于神经网络的 IRL）和在线模仿学习方法（如 DAgger）的对比。这使得 GAIL 的优势是否来自对抗训练框架本身、还是仅来自更强的代价函数族，难以完全解耦。
- **专家数据来源单一**：所有专家策略均通过 TRPO 在真实代价函数上训练获得，即专家本身是 RL 策略。对于来自人类专家或次优专家的演示数据，GAIL 的表现尚不清楚。
- **环境交互效率未充分分析**：论文在 Discussion 中承认 GAIL 的环境交互效率与直接用 TRPO 从奖励训练相当，但正文实验中并未量化这一对比或讨论其实际影响。
- **因果熵正则化的作用未充分探索**：仅在 Reacher 上展示了 $\lambda > 0$ 的效果，其余任务均使用 $\lambda = 0$，未系统分析 $\lambda$ 对不同任务的影响。

## 5. Related Work & Future Work

### Related Work

GAIL 建立在两条研究线之上。第一条是逆强化学习，特别是最大因果熵 IRL 框架——GAIL 的理论分析以此为起点。与经典 IRL 方法需要在内循环中运行 RL 不同，GAIL 将 IRL 和 RL 步骤合并为一个直接的策略优化问题。第二条是学徒学习/Apprenticeship Learning，特别是 *Model-Free Imitation Learning with Policy Optimization* 提出的基于策略梯度的学徒学习框架——GAIL 的算法结构（交替进行代价拟合和 TRPO 策略更新）直接继承自该工作，核心改进在于用神经网络判别器替代了线性代价函数族。

与 GAN 的联系是本文的核心贡献之一：策略对应生成器，判别器对应鉴别器，专家占用度量对应真实数据分布。但与 GAN 不同的是，策略的优化不能通过反向传播实现（因为环境不可微），而必须依赖策略梯度方法。

与 DAgger 等需要在训练中与专家交互的方法不同，GAIL 与 IRL 一样不需要专家在训练时在线提供动作。GAIL 通过随机探索来确定哪些动作能使策略的占用度量更接近专家。

### Future Work

论文明确指出的方向：

> GAIL 在专家数据方面高效，但在环境交互方面效率较低。通过行为克隆初始化策略参数，可以显著加速训练（因为 BC 不需要环境交互）。

> 作为 model-free 方法，GAIL 天然比 model-based 方法需要更多环境交互。结合良好的环境模型，有望在专家数据效率和环境交互效率两方面同时获得优势。

从论文的局限性可以自然推断的方向包括：(1) 将理论分析从有限状态-动作空间推广到连续空间；(2) 探索更稳定的对抗训练技术（如 Wasserstein 距离）以改善训练稳定性；(3) 将 GAIL 与 model-based 方法结合以提升环境交互效率；(4) 处理多模态专家演示（即专家策略包含多种行为模式的情况，后续 *InfoGAIL* 正是针对这一方向的扩展）。
