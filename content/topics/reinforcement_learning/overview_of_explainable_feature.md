---
title: Explainable Feature Selection
headline: Overview of Explainable Feature Selection in Reinforcement Learning
draft: true
---

## Budget Aware Feature Selection

- [x] RLC 2025 Finding the Frame Workshop: Budget-Aware Feature Selection for Reinforcement Learning, [OpenReview](https://openreview.net/forum?id=rYhK13RkA5)

这篇文章给 RL 内的 Feature Selection 问题给出了一个 naive 的解决方案，引入了 Constrained MDP 框架，并且扩展到了其新定义的 BAFS-CMDP 框架上，联合训练两个策略，一个是原本的和环境交互的策略，目标是达到最大回报，另一个是特征选择策略，目标是在有限的预算下，一方面保持最大化回报，一方面减少特征选择的费用。但是本文并未提出新的算法，而对任意环境做一个包装，扩展动作空间，使用 Safe RL 的方法来进行训练。

标准 MDP 是一个五元组：$\mathcal{M} = (S, A, T, r, \gamma)$，分别代表状态空间、动作空间、转移概率、奖励函数和折扣因子。CMDP 在 MDP 的基础上，加入了一个或者多个约束函数和整体的预算，单约束情形可以将其写成一个七元组：$\mathcal{M}_c = (S, A, T, r, \gamma, C, d)$，分别代表状态空间、动作空间、转移概率、奖励函数和折扣因子、约束函数和预算。约束函数/成本函数 $C: S \times A \rightarrow \mathbb{R}$ 表示在状态 $s$ 下，选择动作 $a$ 的约束/成本/风险，预算 $d$ 限制整体成本不能太大。CMDP 的目标是找到一个策略，使得在整体约束下，最大化期望回报。
$$
\begin{aligned}
\max_\pi \quad & J_r(\pi) = \mathbb{E}_\pi\Big[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\Big] \\
\text{s.t.} \quad &
J_c(\pi) = \mathbb{E}_\pi\Big[\sum_{t=0}^{\infty} \gamma^t C(s_t, a_t)\Big] \le d.
\end{aligned}
$$

在这个问题下，BAFS-CMDP 基本和 CMDP 一致，唯一就是扩展了动作空间了，加入了特征选择动作 $A_{\text{feat}} = \{u_t \in \{0,1\}^n\}$，其中 $n$ 是特征数量，$u_t$ 是一个二进制向量，表示在时间 $t$ 选择的特征，我们可以将每一个特征对应一个传感器，那么 $u_t$ 的第 $i$ 个分量表示在当前时间步第 $i$ 传感器是否被开启。于是这个智能体的决策由两个策略 $\pi = (\pi_u, \pi_m)$ 共同决定：

- $\pi_u$：特征选择策略，它决定在每一步获取哪些特征，从而产生特征使用成本；
- $\pi_m$：环境交互策略，根据当前收到的观测选择控制动作。

每一个特征向量 $f = (f_1, f_2, \dots, f_n)$ 的每一个分量 $f_i$ 都和一个非负成本 $c(f_i) \geq 0$ 相对应，在任何时间步，如果这个智能体选择了某一个特征子集 $u_t$，则该时间步的总特征成本定义为：
$$
C(s_t, u_t) = c(f)^\top u_t.
$$

这样我们只需要求解下面优化问题就可以：
$$
\begin{aligned}
\max_{\pi = (\pi_u, \pi_m)} \quad & \mathbb{E}_{\pi_m}\Big[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\Big] \\
\text{s.t.} \quad & \mathbb{E}_{\pi_u}\Big[\sum_{t=0}^{\infty} \gamma^t C(s_t, u_t)\Big] \le d.
\end{aligned}
$$

在方法上，文章使用多头 Actor 结构，在同一个网络中并行训练两个子策略，前几层共享参数，后面分裂成两个独立的 Head，每个 Head 负责一个具体任务。对于连续动作部分，使用高斯策略，对于离散动作部分，使用离散概率分布。本质上任何一个可以处理混合动作空间的 Safe RL 算法都可以解决这个问题，文章参考 [MERL Framework (NeurIPS 2019)](https://arxiv.org/abs/1909.11939) 使用了 CPO 算法来进行训练。

但是本文的方法还有很多内在的局限与矛盾：一方面，引入专门的特征选择策略，特征选择是为了降低运行成本，但是在原有的主策略之外添加一个策略会提升整体问题的维度。并且 BAFS-CMDP 本质上扩大了动作空间，复杂度的增加会削弱学习效率，尤其在特征数量比较庞大的环境下，引入的复杂性反而可能导致策略质量下降。将 Mask 当做动作的做法，在高维特征下会遇到 Scalability 问题，多头 Actor 的结构只是一个工程技巧，而不是从理论上解决维度爆炸的方法。

Reference: Constraned Markov Decision Processes, E. Altman, 1999, [Source](https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf)

## Mask Selection in Visual RL/IL

- IROS 2023, **MIL**: Masked Imitation Learning: Discovering Environment-Invariant Modalities in Multimodal Demonstrations, [arXiv](https://arxiv.org/abs/2209.07682)

- CoRL 2024, **MaIL**: MaIL: Improving Imitation Learning with Selective State Space Models, [OpenReview](https://openreview.net/forum?id=IssXUYvVTg), [GitHub](https://github.com/ALRhub/MaIL)

- ICRL 2017: Learning to Gather Information via Imitation, [arXiv](https://arxiv.org/abs/1611.04180)

- AAMAS 2024 Oral: MaDi: Learning to Mask Distractions for Generalization in Visual Deep Reinforcement Learning, [arXiv](https://arxiv.org/abs/2312.15339)

## Privilieged-teacher Imitation Learning

- NIPS 2015: Data Generation as Sequential Decision Making, [arXiv](https://arxiv.org/abs/1506.03504)
- ICLR 2025 Spotlight: Student-Informed Teacher Training, [OpenReview](https://openreview.net/forum?id=Dzh0hQPpuf)

## Safe Reinforcement Learning



## Related Work

- ICLR 2018: Learning Sparse Neural Networks through L0 Regularization, [arXiv](https://arxiv.org/abs/1712.01312), [OpenReview](https://openreview.net/forum?id=H1Y8hhg0b)
- arXiv 2023: Effective Neural Network L0 Regularization With BinMask, [arXiv](https://arxiv.org/abs/2304.11237), [OpenReview](https://openreview.net/forum?id=CqZvqyqusY)
