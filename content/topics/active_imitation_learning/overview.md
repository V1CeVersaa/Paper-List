---
title: Overview
headline: Overview of Active Imitation Learning
---

## 1. Single Expert Active Imitation Learning

- [ ] JMLR 2014: Active Imitation Learning: Formal and Practical Reductions to I.I.D. Learning, [JMLR](http://jmlr.org/papers/v15/judah14a.html)

## 2. Multiple Expert Active Imitation Learning

当前我们考虑的设置主要有如下四点，有几点和单专家的设定重合：

- 多个黑盒的专家/Expert/预言机/Oracle：专家可能在能力水平、偏好和专业领域上存在差异。
- 主动查询：Learner 需要主动选择在哪个状态向哪一个专家进行查询；
- 超越所有专家的稳健性能改进：Learner 需要博采众专家所长，并在此基础上实现性能提升；
- Learner 可以与环境交互，决定何时何地进行查询，但是具体是否可以获得奖励信号取决问题设定。

目标是一样的：

- 尽可能达到低 Regret，也就是高性能；
- 尽可能减少对专家查询的次数，因为查询专家往往是昂贵的。

一些设定限制了我们访问专家的方式，例如只能查询专家在某个状态下的动作，不能大批量获得专家的轨迹数据，也不能访问专家的价值函数或者梯度信息。一般的奖励函数公式是未知的，但是允许智能体和环境进行交互，采样获得即时奖励信号与环境动态，从而估计专家的价值函数和优势函数。

- [ ] NeurIPS 2024: Contextual Active Model Selection, [arXiv](https://arxiv.org/abs/2207.06030)
- [ ] NeurIPS 2020, **MAMBA**: Policy Improvement via Imitation of Multiple Oracles, [arXiv](https://arxiv.org/abs/2007.00795), [Note](./MAMBA.md)
- [ ] ICML 2023, **MAPS**: Active Policy Improvement from Multiple Black-box Oracles, [arXiv](https://arxiv.org/abs/2306.10259), [Note](./MAPS.md)
- [ ] ICLR 2024, **RPI**: Blending Imitation and Reinforcement Learning for Robust Policy Improvement, [arXiv](https://arxiv.org/abs/2310.01737)

在 **MAMBA**、**MAPS** 以及 **RPI** 的设定下，我们考虑的是一个有限 MDP，智能体可以访问一组专家策略，记为 $\Pi = \{\pi^k\}_{k=1}^K$，这些专家都不是全局最优的，一般情况下互有优劣，不存在一个专家在所有状态下都优于其他专家的情况，这意味着智能体不能简单的只模仿一个专家，同时不同专家在不同状态下的表现也一般是不同的。专家/Oracle 对学习者来讲是一个黑盒，学习者可以**查询专家的动作**，或者可以连续让**专家和环境交互产生轨迹**，但是无法获得**专家内部的价值函数或者梯度信息**。但是，虽然奖励函数 $r$ 是未知的，智能体在开始的时候不能知道奖励函数的数学公式，但是在实际的交互过程中，**环境会反馈即时的奖励信号 $r(s,a)$**。因此可以通过采样的方式估计出**黑盒专家的价值函数 $\hat{V}^k(s)$** 以及优势函数 $A(s, a)$。

在这种设定下，一般的流程（以 MAPS 为例）是：先让学习者在 Horizon 内与环境交互几步，然后专家上号，一直控制到本集结束（到了 Horizon 限制或者到达终止状态），这样产生了一个混合轨迹，后半段是专家跑出来的（带奖励数据），利用这部分数据可以更新估计专家的价值函数，然后学习者再完整跑一集，利用收集的数据以及根据专家价值函数的的估计计算策略梯度，进行对学习者参数的更新。这样的算法结合了模仿学习和强化学习的思想，一方面可以利用专家来进行一部分蒸馏，可以节省样本复杂度，另一方面可以使用强化学习的方法来进一步优化回报，比较当前学习者和专家的差距，博采众长，实现超越所有专家的性能提升。

在另一些设定下，环境限制智能体获得奖励信号，只可以获得环境动态，训练期间学习者永远不会观察到 $r_t$，只可以通过专家的轨迹或者查询信息来进行学习。

- [ ] arXiv 2020, **APIL**: Active Imitation Learning from Multiple Non-Deterministic Teachers: Formulation, Challenges, and Algorithms, [arXiv](https://arxiv.org/abs/2006.07777)
- [ ] NeurIPS 2023, **RAVIOLI**: Selective Sampling and Imitation Learning via Online Regression, [arXiv](https://arxiv.org/abs/2307.04998)
- [ ] NeurIPS 2023, **AURORA**: Contextual Bandits and Imitation Learning with Preference-Based Active Queries, [arXiv](https://arxiv.org/abs/2307.12926), [Note](./AURORA.md)
- [ ] NeurIPS 2025, **WARM-STAGGER**: Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning, [OpenReview](https://openreview.net/forum?id=sT1U2enBh0)
- [ ] ICLR 2025, **RND-Dagger**: Efficient Active Imitation Learning with Random Network Distillation, [arXiv](https://arxiv.org/abs/2411.01894)

这些文章的算法结构高度相似，一个主要的 Learner 将 Oracle/Expert 的反馈当做监督信号，基于监督信号在线回归/结构预测一个模型，一个查询策略基于不确定性的估计来设计选择性采样，但是不确定性度量方法各不相同：

<!-- - APIL 指出，Expert 行为的不确定性来自于两部分，一部分是 Intrinsic Uncertainty，指同一个状态下专家可能会给出不同的动作（专家是非确定性的），另一部分是 Extrinsic Uncertainty，这源于多个专家的行为差异性。对于单个确定性的 Expert 来讲，使用不确定性来作为查询的依据是合理的，因为这时候不确定性是零，但是对于多个非确定性的专家来讲，基于不确定性的查询策略就会引起灾难，因为这时候专家的不确定性就是一个非零的未知数，无法确定查询阈值。APIL 的状态空间是结构化的，是一个度量空间，其将状态分为两类，我们的目标是到达一个目标状态集 $S_G$ 内的状态，并且我们也有一个举例函数 $d(s)$ 来衡量当前状态和目标状态集的距离，只有当 $s$ 在 $S_G$ 内时，$d(s) = 0$。我们使用这个度量来替代不确定性，衡量边际贡献，设计查询策略。  
    APIL 构建一个 Teacher Persona 框架，将专家的策略建模为一个由身份分布 $\rho_\psi(k \mid s)$、persona 模型 $h_\phi(k)$ 以及 persona-conditioned 策略 $\hat{\pi}_{\theta, h}(a \mid s)$ 的随机过程。其先在这个框架内学习一个 Policy Distribution 模型，然后针对查询选择，构造一个预测未来表现的模型，估计在该状态下查询并且跟随专家行为，最后距离到达目标状态集的距离 $d(s)$ 会有多大，对比如果不查询专家，继续跟随当前学习者策略，距离目标状态集的距离 $d(s)$ 会有多大，两者的差值作为查询的收益，只有当收益大于一个阈值时才进行查询。
- **RAVIOLI**：RAVIOLI 把主动 IL 抽象成一个带边际条件的在线回归与选择性采样问题。它假设专家可以由某个函数类 $F$ 中的 $f_\star$ 表示，对每个 $(x,a)$ 输出一个 score，score 越大动作越优。Learner 持续调用一个 online regression oracle 得到当前预测 $f_t$，并利用以往只在查询时才得到的标签构造一个高置信度版本空间
    $$
    F_t=\left\{f\in F:\sum_{s\le t-1}Z_s|f(x_s)-f_s(x_s)|^2\le \Psi(F,T)\right\},
    $$

    这里 $Z_s\in\{0,1\}$ 表示第 $s$ 轮是否查询过。然后定义两个量：一是 **$\text{margin}(f_t(x))$**，即当前预测下最优动作和第二优动作 score 的差；二是一个“置信半径”
    $$
    \Delta_t(x)=\max_{f\in F_t}|f(x)-f_t(x)|,
    $$

    表示在版本空间里所有候选函数在这个点上的最大偏差。RAVIOLI 的查询规则是：只有当 $\text{margin}(f_t(x))\le 2\gamma\Delta_t(x)$ 时才向专家要标签，也就是只有当版本空间内部对于谁是最优动作还没定论时才查询，这时候 $\text{margin}$ 相对置信半径太小。在 IL 版本中，它为每个时间步 $h$ 各维护一个回归 oracle $f_{t,h}$ 和版本空间 $F_{t,h}$，同样用 margin–radius 规则对 on-policy 访问到的 state 决定是否需要 expert 的动作标签，并证明 regret 和查询复杂度主要由 online regression regret 与函数类的 eluder dimension 决定，而只在 optimal policy 在小 margin 区域出现的次数上付代价。
- **AURORA**：AURORA 在结构上几乎沿用了 RAVIOLI 的框架，但把监督信号从“动作标签 / cost”换成了**偏好比较**，并在 IL 中把一个 $H$-步 MDP 拆成 $H$ 个 contextual bandit 来处理。具体地，它假设存在一个偏好函数
    $$
    f^\star_h(x,a,b)=Q^{\pi_e}_h(x,a)-Q^{\pi_e}_h(x,b),
    $$

    expert 在被查询时给出 noisy preference $y\in\{+1,-1\}$，其分布由 $f^\star_h$ 经 link function（比如 logistic）诱导；Learner 只接触到这些偏好标签，看不到 reward 和 expert 动作。算法对每个 step $h$ 维护一个函数类 $F_h$ 和 regression oracle，基于已查询的 pairwise 比较构造版本空间 $F_{t,h}$，在每个 $(x,a,b)$ 上给出当前估计 $f_{t,h}(x,a,b)$ 及其“置信半径” $\Delta_{t,h}(x,a,b)$。**查询策略可以理解为：如果在 $F_{t,h}$ 里既有函数认为“$a\succ b$”（$f(x,a,b)$ 明显为正），又有函数认为“$b\succ a$”（明显为负），即偏好区间跨过 $0$ 或 margin 相对半径太小，就对这对动作发起一次偏好查询；否则，不再问，只根据当前估计选择动作。**AURORAE 在 IL 中为每个时间步各开一个 AURORA 实例，把一条 trajectory 看成 $H$ 次 contextual bandit 交互，最终在 regret 上可以做到和需要 reward+action 的交互 IL 算法同一量级，同时显著减少 query 数，并且在 expert 本身次优时有机会学到比 expert 更好的策略。
- **WARM-STAGGER / STAGGER**：和前面几篇都围绕“不确定度阈值来决定查不查询”不同，STAGGER/WARM-STAGGER 的主动部分极其“朴素”：**查询策略基本被硬编码为“每轮 on-policy 轨迹里只标一个状态”**，几乎不显式估计不确定度。论文首先改变了代价模型：不再按整条 trajectory 计一次成本，而是按**每个被 expert 标注的 state–action 对计一次成本**；在这个 state-wise 成本模型下，多轮交互的目标变成“用尽量少的 state-label 把 policy 拉到 expert 水平”。  
    STAGGER 的做法是：第 $n$ 轮从当前 policy $\pi_n$ 出发 roll out 一条完整轨迹，得到 induced 分布 $d^{\pi_n}$，然后**从这条 on-policy 轨迹中均匀随机采样一个状态–时间步 $(s_n,h_n)$**，调用 state-wise oracle $O_{\text{State}}$ 拿到 expert 动作 $a^\star_n=\pi_{E,h_n}(s_n)$，用这一条样本在线更新一个 no-regret learner（如 exponentiated weights）上的“policy-level” loss，得到下一轮 policy $\pi_{n+1}$。整个过程中，每轮必定查询一次，而且总是在“Learner 当前真正会访问到的状态分布”上抽点，这相当于把不确定性假设为“on-policy 访问频率 + 在线学习器自身的误差”，用极简的 uniform-sampling 近似掉了显式 uncertainty estimate。WARM-STAGGER 在此基础上增加一个 offline warm-start：先用一批 offline expert trajectories $D_{\text{off}}$ 收缩 policy class，构造与所有 offline $(s,a)$ 对一致的子类 $B_{\text{bc}}$，再在这个更小的类上运行和 STAGGER 完全相同的“每轮一状态”交互阶段。理论上他们证明：在 expert 具有低恢复成本的 MDP 中，在**同样的总 state-wise 注释预算**下，STAGGER/WARM-STAGGER 能严格优于纯 BC，其核心优势来自“on-policy + 单状态标注立即在线更新”，而不是更复杂的不确定度推断。

- **RND-DAgger（RND-Dagger）**：RND-DAgger 依然是一个“主 Learner + selective querying Expert”的结构，但它把“何时需要 expert 接管 / 给演示”的决策，几乎完全交给一个**基于 Random Network Distillation（RND）的状态级 OOD 分数**来做，而不是像 Lazy/Ensemble-DAgger 那样比较 agent 动作和 expert 动作的差异。具体地，它维护一个随机初始化且冻结的 target 网络 $f_{\text{target}}$ 和一个可训练的 predictor 网络 $f_{\text{pred}}$：对每个状态 $s$ 定义新颖度分数
    $$
    \text{err}(s)=|f_{\text{pred}}(s)-f_{\text{target}}(s)|^2.
    $$

    在训练过程中，只要 agent（由当前 policy 控制）访问到某个状态，就把它加入 RND 的训练数据，让 $f_{\text{pred}}$ 尽量拟合 $f_{\text{target}}$。结果是：对**频繁出现的 in-distribution 状态**，$\text{err}$ 会逐渐变小；对**少见或未见过的状态**，$\text{err}$ 会显著偏大，于是 $\text{err}$ 就可以当作一个不依赖 expert 的 OOD/novelty 指标。RND-DAgger 的查询规则是一个带“迟滞”的双阈值机制：当 agent 控制时，如果当前 $\text{err}(s)$ 超过上阈值 $\tau_{\text{high}}$，认为已经进入了高风险的分布外区域，便**切换为 expert 接管控制**，并把这一段 expert 轨迹加入 IL 训练数据；在 expert 控制阶段，如果连续 $W$ 步都满足 $\text{err}(s)$ 低于下阈值 $\tau_{\text{low}}$，就认为 expert 已经把系统带回熟悉区域，再把控制权交还给 Learner。这样一来，查询触发完全依赖**无监督的 state 新颖度**而非“实时比对 expert 动作”，既避免了每一步都要向 expert 要动作再比较的高负担，也能在真正危险 / 分布外的状态上集中获取高价值的演示，从而在保持最终 performance 的同时显著减少 expert intervention 时间和总 query 数量。
 -->
