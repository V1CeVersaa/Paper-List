---
title: "TRPO"
headline: "Trust Region Policy Optimization"
visibility: "public"
status: "complete"
description: "Paper note on Trust Region Policy Optimization."
---

> [!abstract]
> 

## 1. Introduction

## 2. Preliminaries

考虑一个 Infinite-Horizon 的带折扣 MDP $(\mathcal{S}, \mathcal{A}, P, r, \rho_0, \gamma)$，其中 $\mathcal{S}$ 是有限状态集，$\mathcal{A}$ 是有限动作集，$P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$ 是转移概率分布，$r: \mathcal{S} \rightarrow \mathbb{R}$ 是奖励函数，$\rho_0: \mathcal{S} \rightarrow \mathbb{R}$ 是初始状态 $s_0$ 的分布，并且 $\gamma \in (0, 1)$ 是折扣因子。

对于策略 $\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$，其期望折扣奖励定义为
$$
\eta(\pi) = \mathbb{E}_{s_0, a_0, \ldots} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t) \right]
$$

这里 $s_0, a_0, \ldots$ 为 $\pi$ 生成的一个无限长的轨迹，类似可以定义价值函数 $V_\pi$、动作-价值函数 $Q_\pi$ 和优势函数 $A_\pi$。可以证明，对于任意两个策略 $\pi$ 和 $\tilde{\pi}$，我们可以使用优势函数估计两个策略期望回报的差距：
$$
\eta(\tilde{\pi}) - \eta(\pi) = \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t A_\pi(s_t, a_t) \right] \tag{1}
$$

> [!proof] Proof
>
> 注意到 $A_\pi (s, a) = \mathbb{E}_{s' \sim P(\cdot \mid s, a)} [r(s) + \gamma V_\pi (s') - V_\pi (s)]$。我们可以有下面估计
> $$
> \begin{aligned}
> \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t A_\pi (s_t, a_t) \right] &= \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t (r(s_t) + \gamma V_\pi (s_{t+1}) - V_\pi (s_t)) \right] \\
> &= \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ -V_\pi (s_0) + \sum_{t=0}^{\infty} \gamma^t r(s_t) \right] \\
> &= -\mathbb{E}_{s_0} [V_\pi (s_0)] + \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t) \right] \\
> &= -\eta(\pi) + \eta(\tilde{\pi})
> \end{aligned}
> $$

如果定义 $\bar{A}(s)$ 为状态 $s$ 下 $\tilde{\pi}$ 相对于 $\pi$ 的预期优势：
$$
\bar{A}(s) = \mathbb{E}_{a \sim \tilde{\pi}(\cdot \mid s)} [A_\pi (s, a)].
$$

那么，式 (1) 可以写成如下形式：
$$
\eta(\tilde{\pi}) = \eta(\pi) + \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t \bar{A}(s_t) \right] \tag{2}
$$

如果给出一个（未归一化的）折扣状态访问频率 $\rho_\pi$，我们就可以重写期望 $\mathbb{E}_{\tau \sim \tilde{\pi}} [\sum \gamma^t A_\pi (s_t, a_t)]$：
$$
\begin{aligned}
\rho_\pi (s) &= P(s_0 = s) + \gamma P(s_1 = s) + \gamma^2 P(s_2 = s) + \ldots, \\
\eta(\tilde{\pi}) &= \eta(\pi) + \sum_{t=0}^{\infty} \sum_s P(s_t = s \mid \tilde{\pi}) \sum_a \tilde{\pi}(a \mid s) \gamma^t A_\pi (s, a) \\
&= \eta(\pi) + \sum_s \sum_{t=0}^{\infty} \gamma^t P(s_t = s \mid \tilde{\pi}) \sum_a \tilde{\pi}(a \mid s) A_\pi (s, a) \\
&= \eta(\pi) + \sum_s \rho_{\tilde{\pi}} (s) \sum_a \tilde{\pi}(a \mid s) A_\pi (s, a).\tag{3}
\end{aligned} 
$$

如果对于一个策略更新 $\pi \rightarrow \tilde{\pi}$，$\tilde{\pi}$ 在每个状态 $s$ 处都具有非负的期望优势 $\sum_a \tilde{\pi}(a \mid s) A_\pi (s, a) \geq 0$，那么这个策略更新就一定可以保证可以提升策略性能 $\eta$，就算这个期望更新在所有状态都为零也可以保持策略性能恒定。这就可以保证经典的策略迭代成立了，如果使用确定性策略 $\tilde{\pi}(s) = \operatorname*{\arg\max}_a A_\pi (s, a)$，且至少存在一个状态-动作对 $(s, a)$ 使得 $A_\pi (s, a) > 0$ 且 $P(s \mid \tilde{\pi}) > 0$，那么策略就一定会提升，否则算法就已经收敛到最优策略了。

但是，在近似情形/Approximate Setting 下，由于目标和更新都不是精确的，通常会出现估计误差和近似误差，要么是通过采样/时序差分得到的 $A_\pi$ 不精确，要么是 $\tilde{\pi}$ 不是贪心的，或者两者兼而有之，因此对某些状态 $s$ 来说，有可能出现 $\sum_a \tilde{\pi}(a \mid s) A_\pi (s, a) < 0$，这是不可避免的。因此我们需要考虑使用策略梯度。

另一方面，式 (3) 中的 $\rho_{\tilde{\pi}}$ 对 $\tilde{\pi}$ 的依赖过于复杂，如果直接使用策略梯度，那就必须要对 $\rho_{\tilde{\pi}}$ 进行求导，还得对新分布采样，这就极其复杂。因此我们将 (3) 内的 $\rho_{\tilde{\pi}}$ 冻结为旧分布 $\rho_\pi$，忽略由于策略变化而引起的状态访问密度的变化，从而切断上述复杂依赖，得到 $\eta$ 的局部近似
$$
L_\pi (\tilde{\pi}) = \eta(\pi) + \sum_s \rho_\pi (s) \sum_a \tilde{\pi}(a \mid s) A_\pi (s, a). \tag{4}
$$

很容易知道，如果 $\pi_\theta$ 是一个参数的策略，$\pi_\theta (a \mid s)$ 可微，其实 $L_\pi$ 和 $\eta$ 在旧点处是一阶等价的，也就是对于任意的参数 $\theta_0$，都有
$$
\begin{aligned}
L_{\pi_{\theta_0}} (\pi_{\theta_0}) &= \eta(\pi_{\theta_0}), \\
\left.\nabla_\theta L_{\pi_{\theta_0}} (\pi_\theta)\right|_{\theta = \theta_0} &= \left.\nabla_\theta \eta(\pi_\theta) \right|_{\theta = \theta_0}.
\end{aligned} \tag{5}
$$

> [!info] Supplementary
>
> 第一个是显然的，第二个可以直接证明：
> $$
> \begin{aligned}
> \nabla_\theta L_{\pi_{\theta_0}} (\pi_\theta) &= \nabla_\theta \left( \eta(\pi_{\theta_0}) + \sum_s \rho_{\pi_{\theta_0}} (s) \sum_a \pi_{\theta}(a \mid s) A_{\pi_{\theta_0}} (s, a) \right) \\
> &= \sum_s \rho_{\pi_{\theta_0}} (s) \sum_a \nabla_\theta \pi_{\theta}(a \mid s) A_{\pi_{\theta_0}} (s, a) \\
> \nabla_\theta \eta(\pi_{\theta}) &= \nabla_\theta \left( \eta(\pi_{\theta_0}) + \sum_s \rho_{\pi_{\theta}} (s) \sum_a \pi_{\theta}(a \mid s) A_{\pi_{\theta_0}} (s, a) \right) \\
> &= \sum_s \left( \nabla_\theta \rho_{\pi_{\theta}} (s) \sum_a \pi_{\theta}(a \mid s) A_{\pi_{\theta_0}} (s, a) + \rho_{\pi_{\theta}} (s) \sum_a \nabla_\theta \pi_{\theta}(a \mid s) A_{\pi_{\theta_0}} (s, a) \right) \\
> &= \sum_s \rho_{\pi_{\theta}} (s) \sum_a \nabla_\theta \pi_{\theta}(a \mid s) A_{\pi_{\theta_0}} (s, a) \\
> \end{aligned}
> $$

这就表明，可以改善 $L_{\pi_{\theta_{old}}}$ 的一个充分小的更新 $\pi_{\theta_0} \rightarrow \tilde{\pi}$ 

<!-- 
公式 (4) 表明，充分小的步骤 π_{θ_0} → ˜ππθ0→˜ππ_{θ_0} → ˜ππθ0​​→˜π 能够改善 L_{π_{θ_{old}}}LπθoldL_{π_{θ_{old}}}Lπθold​​​ 也会改善 η，但是没有为我们提供有关采取多大步骤的指导。


为了解决这个问题，Kakade & Langford (2002) 提出了一种称为保守策略迭代 (conservative policy iteration) 的策略更新方案，他们可以为 η 的改进提供明确的下限。
为了定义保守策略迭代更新，令 π_{old}πoldπ_{old}πold​ 表示当前策略，并令 π' = \text{arg max}_{π'} L_{π_{old}}(π')π′=arg maxπ′Lπold(π′)π' = \text{arg max}_{π'} L_{π_{old}}(π')π′=arg maxπ′​Lπold​​(π′)。
新的策略 π_{new}πnewπ_{new}πnew​ 定义为以下混合：


π_{new}(a|s) = (1 − α)π_{old}(a|s) + απ'(a|s).πnew(a∣s)=(1−α)πold(a∣s)+απ′(a∣s).π_{new}(a|s) = (1 − α)π_{old}(a|s) + απ'(a|s).πnew​(a∣s)=(1−α)πold​(a∣s)+απ′(a∣s).


Kakade 和 Langford 推导了以下下限：


η(π_{new}) ≥ L_{π_{old}}(π_{new}) − \frac{2γ}{(1 − γ)^2} α^2 其中  = \text{max}_s |E_{a∼π'(a|s)} [A^π(s, a)]|.


（我们对其进行了修改，使其稍微弱一些但更简单。）但是请注意，到目前为止，此边界仅适用于公式 (5) 生成的混合策略。
这种策略类在实践中是笨拙且具有限制性的，并且对于一种实用的策略更新方案，希望适用于所有通用随机策略类。

 -->

## Appendix: Conservative Policy Iteration
