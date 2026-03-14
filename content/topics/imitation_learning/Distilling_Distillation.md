---
title: Distilling Policy Distillation
headline: "Distilling Policy Distillation"
---

> [!abstract] Contributions
>
> 本文对强化学习/Reinforcement Learning/RL 中策略蒸馏/Policy Distillation 的各种变体进行了系统性的理论与实验分析。核心发现有三：（1）许多被广泛使用的 on-policy 蒸馏更新规则并不构成合法的梯度向量场/Gradient Vector Field，因而可能导致不收敛的振荡行为；（2）通过引入基于奖励的修正项，可以恢复梯度向量场性质并保证收敛；（3）综合理论分析与大规模合成 MDP 实验，论文提出期望熵正则化蒸馏/Expected Entropy Regularised Distillation 作为最可靠的蒸馏方法——它同时满足梯度向量场性质、低方差估计、以及直接最大化学生轨迹下教师行为概率这三项优势。此外，论文还分析了 Actor-Critic 设定下如何利用教师的价值函数来处理不完美教师，并提供了一套基于决策树的蒸馏方法选择指南。
>
> 主要局限：所有理论结果和实验均在有限状态空间的表格/Tabular 设定下完成，未涉及深度网络函数逼近器；实验环境为随机生成的 grid world MDP，尚未在 Atari 等高维视觉任务上验证；部分收敛保证依赖强随机性（所有状态-动作对概率大于零）假设，在实际深度 RL 中难以满足。

## 1. Introduction

策略蒸馏是深度 RL 中知识迁移的核心技术：将教师策略 $\pi$ 的行为迁移到学生策略 $\pi_\theta$，广泛用于加速训练、构建更强策略、模型压缩和多任务学习。尽管其高层形式简单——让学生在每个状态下匹配教师的动作分布——但实践中存在数十种不同的数学实现，它们在控制策略/Control Policy（谁收集轨迹）、损失函数、是否引入奖励信号等方面各不相同，且这些细微差异会显著影响最终性能。

本文的出发点正是这一混乱现状：现有蒸馏方法之间缺乏统一的比较框架。论文提出一个统一的更新规则视角，将所有主流蒸馏方法纳入同一公式框架（Equation 1），通过改变控制策略 $q_\theta$、辅助损失 $\ell$ 和伪奖励 $\hat{r}$ 三个"旋钮"来实例化不同方法。在此基础上，论文从梯度向量场这一数学性质出发，系统鉴别了哪些方法具有理论收敛保证、哪些可能振荡发散，并据此提出了改进方案。

论文的关键洞察在于：**更新规则与损失函数是不同的概念**。许多蒸馏方法的更新方向并不对应于任何损失函数的梯度，这意味着它们不构成梯度向量场，从而可能出现循环行为而永远不收敛。这一发现不仅解释了实践中某些蒸馏方法的不稳定性，也为设计更可靠的蒸馏方法提供了理论基础。

## 2. Problem Setup

### MDP 与策略蒸馏框架

考虑有限状态空间 $\mathcal{S}$、有限动作空间 $\mathcal{A}$ 的马尔可夫决策过程/Markov Decision Process/MDP。智能体策略 $\pi: \mathcal{S} \to \Delta^{|\mathcal{A}|}$ 输出每个状态下动作的概率分布。在时间步 $t$，智能体处于状态 $\tau_t \in \mathcal{S}$，采样动作 $a_t \sim \pi(\tau_t)$，环境转移到 $\tau_{t+1} \sim T(\tau_t, a_t)$ 并产生奖励 $r_t \sim r(\tau_t, a_t)$。轨迹记为 $\tau = \{\tau_1, a_1, r_1, \ldots, \tau_{|\tau|}, a_{|\tau|}, r_{|\tau|}\}$。RL 的目标是找到最优策略：

$$
\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=1}^{|\tau|} \gamma^{t-1} r_t \right]
$$

其中 $\gamma \in [0, 1]$ 为折扣因子。论文在理论分析中多取 $\gamma = 1$（无折扣），实验中取 $\gamma = 0.99$。

### 统一的蒸馏更新规则

蒸馏的目标是从教师策略 $\pi$ 向学生策略 $\pi_\theta$ 迁移知识。论文将各种蒸馏方法统一为如下参数更新方向：

$$
\mathbb{E}_{q_\theta} \left[ \sum_{t=1}^{|\tau|} -\nabla_\theta \log \pi_\theta(a_t|\tau_t) \widehat{R}_t + \nabla_\theta \ell(\pi_\theta, V_{\pi_\theta}, \tau_t) \right]
\tag{1}
$$

其中：
- $q_\theta$ 为控制策略/Control Policy，决定谁来收集轨迹（教师 $\pi$、学生 $\pi_\theta$、或其他策略）
- $\ell(\pi_\theta, V_{\pi_\theta}, \tau_t)$ 为辅助损失项，负责当前时间步的策略对齐
- $\widehat{R}_t = \sum_{i=t}^{|\tau|} \hat{r}_i$ 是基于伪奖励 $\hat{r}$ 的累积回报，负责长期对齐
- $V_{\pi_\theta}$ 为学生策略的价值函数

通过不同的 $(q_\theta, \ell, \hat{r})$ 组合，可以实例化所有主流蒸馏方法。下表总结了各方法的具体实例化及其关键性质（$\mathrm{H}^\times(p(s) \| q(s))$ 表示 Shannon 交叉熵 $-\mathbb{E}_{a \sim p(s)}[\log q(a|s)]$）：

| 方法 | $q_\theta$ | $\ell$ | $\hat{r}_i$ | 是否梯度向量场 |
|---|---|---|---|---|
| Teacher distill | $\pi$ | $\mathrm{H}^\times(\pi(\tau_t) \| \pi_\theta(\tau_t))$ | 0 | 是 |
| On-policy distill | $\pi_\theta$ | $\mathrm{H}^\times(\pi(\tau_t) \| \pi_\theta(\tau_t))$ | 0 | **否** |
| Entropy regularised | $\pi_\theta$ | 0 | $\log \pi(a_t|\tau_t)$ | 是 |
| N-distill | $\pi_\theta$ | $\mathrm{H}^\times(\pi(\tau_t) \| \pi_\theta(\tau_t))$ | $-\mathrm{H}^\times(\pi(\tau_{t+1}) \| \pi_\theta(\tau_{t+1}))$ | 是 |
| Exp. entropy regularised | $\pi_\theta$ | $\mathrm{H}^\times(\pi_\theta(\tau_t) \| \pi(\tau_t))$ | $\log \pi(a_{t+1}|\tau_{t+1})$ | 是 |
| Teacher V reward | $\pi_\theta$ | 0 | $r_i + V_\pi(\tau_{i+1}) - V_\pi(\tau_i)$ | 是 |

> [!tip] 更新规则 vs. 损失函数
> 论文特别强调：**更新规则和损失函数是不同的概念**。一个更新规则 $g(\theta)$ 是梯度向量场，当且仅当存在某个函数 $f$ 使得 $g(\theta) = \nabla_\theta f(\theta)$——等价地，$g$ 的 Jacobian 矩阵必须对称。许多常用的蒸馏更新方向（如 on-policy distill）并不满足这一条件，因此它们不对应于任何损失函数的梯度下降，可能产生循环而不收敛。

## 3. Methods

### 3.1 控制策略的选择：教师驱动 vs. 学生驱动

蒸馏中第一个关键设计选择是：由谁来收集训练轨迹？

**教师驱动蒸馏/Teacher Distill** 使用教师策略 $\pi$ 收集轨迹，然后通过监督学习让学生逐状态匹配教师的动作分布：

$$
\mathbb{E}_\pi \left[ \sum_t \mathrm{H}^\times(\pi(\tau_t) \| \pi_\theta(\tau_t)) \right]
$$

这是一个标准的监督学习问题，对应合法的梯度向量场，在表格设定下保证收敛。

**学生驱动蒸馏/On-policy Distill** 使用学生策略 $\pi_\theta$ 收集轨迹，更新方向为：

$$
\mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \mathrm{H}^\times(\pi(\tau_t) \| \pi_\theta(\tau_t)) \right]
$$

直觉上这更合理：蒸馏完成后学生将独立执行策略，因此在学生自身轨迹分布下训练可以减少分布偏移/Distribution Shift。实验也普遍显示学生驱动蒸馏在收敛速度和最终回报方面优于教师驱动，平均快约 3 倍。

然而，论文证明了一个重要的负面结果：

> [!note] Theorem 1（On-policy 蒸馏不是梯度向量场）
> 设 $g(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \ell(\tau|\theta)]$ 可微，且不存在 $\alpha_\tau \in \mathbb{R}$ 使得 $\nabla_\theta \ell(\tau|\theta) = \alpha_\tau \nabla \pi_\theta(\tau)$ 几乎处处成立，则 $g(\theta)$ 不是任何函数的梯度向量场。

> [!todo]- Proof sketch
> 证明的核心思路是：若 $g(\theta)$ 是某函数的梯度，则其 Jacobian 矩阵必须对称，即 $\frac{\partial}{\partial \theta_j} g(\theta)[i] = \frac{\partial}{\partial \theta_i} g(\theta)[j]$。利用 log-derivative trick 展开 $g(\theta)$ 的偏导数后，可以得到：
>
> $$
> \frac{\partial}{\partial \theta_j} g(\theta)[i] - \frac{\partial}{\partial \theta_i} g(\theta)[j] = \mathbb{E}_{\pi_\theta} \left[ \frac{\partial}{\partial \theta_j} \log \pi_\theta(\tau) \frac{\partial}{\partial \theta_i} \ell(\tau, \theta) - \frac{\partial}{\partial \theta_i} \log \pi_\theta(\tau) \frac{\partial}{\partial \theta_j} \ell(\tau, \theta) \right]
> $$
>
> 该表达式一般不为零，除非 $\nabla_\theta \ell(\tau|\theta) = \alpha_\tau \nabla \pi_\theta(\tau)$，即损失梯度与策略梯度处处共线。对于交叉熵损失（以及常用的熵惩罚项）均不满足此条件，因此 on-policy 蒸馏和 Actor-Critic 中的熵正则化更新都不是梯度向量场。

这一定理的实际含义是 on-policy 蒸馏的更新动态可能产生循环振荡。论文通过一个具体的 7 状态 MDP 反例（Appendix C）展示了这种振荡现象：学生策略的参数在一个闭合曲线上循环运动，永远不收敛到教师策略。

尽管如此，论文也证明了一个正面结果：

> [!note] Proposition 1（表格设定下的收敛保证）
> 对于强随机性（$\forall s, a: \pi_\theta(s)[a] > 0$）的学生策略，在有限状态空间的 episodic MDP 和表格策略参数化下，使用更新规则 $\mathbb{E}_{\pi_\theta}[\sum_t \nabla_\theta \ell(\pi(\tau_t), \pi_\theta(\tau_t))]$ 保证在教师策略可达的所有状态上收敛到教师策略。

> [!todo]- Proof sketch
> 在表格设定下，不同状态的参数更新相互独立。将更新规则 $g(\theta)$ 按状态分解为 $g(\theta) = [p_\theta(s_1) \nabla_{\theta_1} \ell_\theta^1 \quad \ldots \quad p_\theta(s_N) \nabla_{\theta_N} \ell_\theta^N]^T$，其中 $p_\theta(s)$ 是学生策略下访问状态 $s$ 的概率。同时，教师策略下期望损失的梯度为 $g^*(\theta) = [p(s_1) \nabla_{\theta_1} \ell_\theta^1 \quad \ldots \quad p(s_N) \nabla_{\theta_N} \ell_\theta^N]^T$。由于权重 $p_\theta(s_i), p(s_i) > 0$，两者的内积 $\langle g(\theta), g^*(\theta) \rangle = \sum_i p(s_i) p_\theta(s_i) \|\nabla_{\theta_i} \ell_\theta^i\|^2 \geq 0$，且为零当且仅当所有可达状态上教师与学生策略一致。因此更新方向始终是期望损失的严格下降方向。

这意味着在表格设定下 on-policy 蒸馏虽然不是梯度下降，但仍然收敛。然而一旦引入环境奖励（将蒸馏嵌入 RL 训练），收敛可能被打破。

### 3.2 控制策略的实验比较

论文在 1000 个随机生成的 $20 \times 20$ grid world MDP 上进行大规模实验，比较教师驱动、学生驱动和均匀随机三种控制策略。主要发现：

- **学生驱动蒸馏在回报和 KL 损失上均显著优于教师驱动**：收敛到完全教师性能的速度约快 3 倍（教师驱动需约 10 倍于均匀策略的步数）
- **直觉解释**：蒸馏完成后学生将独立执行策略，在学生自身分布下训练减少了训练-测试分布偏移。此外，教师如果近乎确定性，它只会访问状态空间中很小的子集，学生在这些状态之外未见过的状态上缺乏泛化能力
- **均匀控制策略**在全状态空间覆盖方面最优，但收敛极慢，不适用于大状态或动作空间
- **唯一例外**：当评估指标是教师分布下的 KL 散度（而非学生分布下）时，教师驱动蒸馏略优——但这一场景在实际应用中很少出现

### 3.3 更新规则的设计与分析

确定使用学生驱动（$q_\theta = \pi_\theta$）后，关键问题变为：如何设计具体的更新规则？论文分析了两大类方法。

**方法一：蒸馏即监督学习**

将蒸馏视为逐状态的监督学习，损失取每步交叉熵 $\ell(\pi(\tau_t) \| \pi_\theta(\tau_t))$，没有伪奖励（$\hat{r} = 0$）。这对应 on-policy distill。问题在于 Theorem 1 已表明其不是梯度向量场。

**方法二：蒸馏即 RL（熵正则化）**

将蒸馏视为一个 RL 问题，其中奖励为 $\hat{r}_t = \log \pi(a_t|\tau_t)$——即教师对学生所选动作赋予的对数概率。此时没有辅助损失（$\ell = 0$），更新方向完全是标准策略梯度的形式。这对应 Entropy regularised 方法，等价于将教师的对数概率作为奖励信号进行策略优化。

该方法虽然是合法的梯度向量场（因为它就是标准策略梯度），但存在一个严重的实际问题：**方差随动作空间增大而急剧增长**。这是因为整个蒸馏信号被打包在奖励通道中，而策略梯度的奖励估计方差本身就很高。论文实验（Figure 4）显示，当动作空间从 4 增加到 1000 时，entropy regularised 的性能几乎完全崩塌。

> [!tip] 交叉熵方向的含义
> 论文指出蒸馏中交叉熵的方向至关重要。使用 $\mathrm{H}^\times(\pi_\theta \| \pi)$（学生作为先验、教师作为后验）倾向于均值寻找/Mean Seeking——学生会尝试覆盖教师的整个动作分布。使用 $\mathrm{H}^\times(\pi \| \pi_\theta)$（教师作为先验、学生作为后验）倾向于模式寻找/Mode Seeking——学生会集中复制教师最可能的动作。当学生容量不足以完美复制教师时，这一差异尤为关键：mean seeking 产生平均化行为，mode seeking 产生专注于某一模式的行为。

### 3.4 期望熵正则化蒸馏/Expected Entropy Regularised Distillation

论文的核心方法贡献是将上述两种思路结合。关键观察是：on-policy distill 的梯度可以分解为两个期望项：

$$
\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{\pi_\theta(\tau|\theta)} \left[ \nabla_\theta \log \pi_\theta(\tau) \ell(\tau, \theta) \right] + \mathbb{E}_{\pi_\theta(\tau|\theta)} \left[ \nabla_\theta \ell(\tau, \theta) \right]
$$

其中第一项对应标准 RL 目标（以 $-\ell$ 为奖励），第二项对应 1-step on-policy 蒸馏更新。这一分解直接启发了恢复梯度向量场性质的方法：

> [!note] Theorem 2（通过奖励修正恢复梯度向量场）
> 对于任意损失 $\ell(\pi(\tau_t) \| \pi_\theta(\tau_t))$ 的 1-step on-policy 蒸馏更新，添加额外伪奖励 $\hat{r}_t = -\ell(\pi(\tau_{t+1}) \| \pi_\theta(\tau_{t+1}))$ 即可恢复梯度向量场性质。类似地，若损失形如 $\mathbb{E}_{a \sim \pi_\theta} \hat{\ell}(\pi(\tau_t))$，则修正项为 $-\hat{\ell}(\pi(\tau_{t+1}))$。

> [!todo]- Proof
> 考虑损失 $\mathcal{L}(\theta) = \mathbb{E}_{\pi_\theta}[\ell(\tau, \theta)]$ 及其梯度：
>
> $$
> \nabla_\theta \mathcal{L}(\theta) = \nabla_\theta \int \pi_\theta(\tau|\theta) [\ell(\tau, \theta)] \, d\tau = \int [\nabla_\theta \pi_\theta(\tau|\theta)] \ell(\tau, \theta) + \pi_\theta(\tau|\theta) [\nabla_\theta \ell(\tau, \theta)] \, d\tau
> $$
>
> 利用 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$，得到：
>
> $$
> \nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{\pi_\theta(\tau|\theta)} \left[ \nabla_\theta \log \pi_\theta(\tau) \ell(\tau, \theta) \right] + \mathbb{E}_{\pi_\theta(\tau|\theta)} \left[ \nabla_\theta \ell(\tau, \theta) \right]
> $$
>
> 第二项正是 1-step on-policy 蒸馏的更新。第一项的结构为 $\mathbb{E}[\nabla_\theta \log \pi_\theta(\tau) \cdot \ell(\tau, \theta)]$，这与策略梯度中的 REINFORCE 项形式完全一致，其中 $-\ell$ 扮演奖励的角色。因此，只要在蒸馏更新中加上这个"奖励"项，就恢复了完整的梯度 $\nabla_\theta \mathcal{L}(\theta)$，自然是梯度向量场。
>
> 具体而言，对于逐步损失 $\ell(\tau, \theta) = \sum_t \ell(\pi(\tau_t) \| \pi_\theta(\tau_t))$，对应的伪奖励在时间步 $t$ 为 $\hat{r}_t = -\ell(\pi(\tau_{t+1}) \| \pi_\theta(\tau_{t+1}))$（将未来损失的负值作为当前步的"奖励"）。

将 Theorem 2 应用于交叉熵损失 $\ell = \mathrm{H}^\times(\pi_\theta(\tau_t) \| \pi(\tau_t))$（注意这里是反向交叉熵），得到的方法就是 **Expected Entropy Regularised Distillation**：

- 辅助损失：$\ell = \mathrm{H}^\times(\pi_\theta(\tau_t) \| \pi(\tau_t))$
- 伪奖励：$\hat{r}_t = \log \pi(a_{t+1}|\tau_{t+1})$

该方法具有三项关键优势：
1. **构成合法的梯度向量场**，保证收敛
2. **低方差**：蒸馏信号主要通过 $\ell$ 项传递（逐状态的交叉熵），仅部分信号通过奖励通道，避免了纯 entropy regularised 方法在大动作空间下的方差爆炸
3. **直接最大化学生轨迹下教师行为概率**：与 N-distill（最大化教师和学生处于相同状态的概率）不同，expected entropy regularised 优化的目标更直接地对应蒸馏的最终目的

实验（Figure 4）证实：随着动作空间从 4 增大到 1000，entropy regularised 的性能急剧下降，而 expected entropy regularised 始终保持稳定且接近 on-policy distill 的最优性能。

### 3.5 N-distill 与 N-distill+R

N-distill 是另一种恢复梯度向量场的方式，它将 Theorem 2 应用于标准方向的交叉熵 $\ell = \mathrm{H}^\times(\pi(\tau_t) \| \pi_\theta(\tau_t))$，得到伪奖励 $\hat{r}_t = -\mathrm{H}^\times(\pi(\tau_{t+1}) \| \pi_\theta(\tau_{t+1}))$。N-distill+R 进一步加入环境奖励 $r_i$。实验中 N-distill 的表现也较稳定，但相比 expected entropy regularised 缺少直接最大化学生轨迹概率的优势。

### 3.6 蒸馏方法选择决策树

论文基于理论和实验结论，提供了一个实用的蒸馏方法选择决策树（Figure 1）。主要决策逻辑如下：

- 是否在教师环境外评估学生？若否 → Teacher distill
- 是否需要收敛保证？若是且不需加速 → Exp. entropy regularised
- 是否需要超越蒸馏速度（即结合环境奖励）？若是 →
  - 教师是否足够强？若是且有价值函数 → Teacher V reward
  - 否则 → On-policy distill+R 或 Exp. entropy regularised+R

## 4. Actor-Critic 设定下的蒸馏

当教师除了策略 $\pi$ 之外还提供价值函数 $V_\pi$ 时，蒸馏可以利用这一额外信息来处理不完美教师。

### 利用价值函数进行门控

论文提出基于价值函数的门控蒸馏损失：

$$
\ell(\theta, s) = \mathrm{H}^\times(\pi(s) \| \pi_\theta(s)) \cdot [V_\pi(s) - V_{\pi_{\theta^*}}(s)]_{>0}
$$

其中 $[x]_{>0} = 1$ 当且仅当 $x > 0$。直觉是：仅在教师比学生当前策略表现更好的状态上执行蒸馏。这相当于广义策略改进/Generalised Policy Improvement 的一个无动作依赖版本。

> [!note] Proposition 2
> 对于初始状态分布 $\mathcal{S}_1$，若在所有状态 $s \in \mathcal{S}$ 上 $\ell(\theta^*, s) \leq \mathrm{H}(\pi(s))$，则收敛后学生的价值不低于教师：$\mathbb{E}_{s \sim \mathcal{S}_1}[V_{\pi_{\theta^*}}(s)] \geq \mathbb{E}_{s \sim \mathcal{S}_1}[V_\pi(s)]$。

这意味着即使教师不完美，使用价值门控的蒸馏学生在收敛后至少与教师一样好。实验（Figure 5）验证：在教师存在 25% 动作噪声时，纯 policy cloning 方法最终复制了教师的错误并导致回报下降，而 On-policy distill+R+V cond 通过价值门控避免了这一问题。

### 利用教师的 Critic 进行 Bootstrapping

另一种利用 $V_\pi$ 的方式是将其用于 TD bootstrapping：将标准 Actor-Critic 的 TD(1) 更新中的 $V_{\pi_\theta}(\tau_{t+1})$ 替换为 $V_\pi(\tau_{t+1})$（记为 TD+$V_\pi$）。

> [!note] Proposition 3
> 设教师的真实价值函数 $V_\pi$ 已知，对于有限状态 MDP，使用更新 $\mathbb{E}_{\pi_\theta}[\sum_t \nabla \log \pi_\theta(a_t|\tau_t)[r(a_t, \tau_t) + V_\pi(\tau_{t+1})]]$ 收敛到一个策略，其价值不低于教师：$\mathbb{E}_{s \sim \mathcal{S}_1} V_{\pi_\theta}(s) \geq \mathbb{E}_{s \sim \mathcal{S}_1} V_\pi(s)$。

这一方法的优势是能**超越**教师——它不仅仅复制教师的行为，而是利用教师的价值估计来指导自身的策略优化。但如果教师很弱（$V_\pi$ 接近 0），bootstrapping 的效果也很有限。

### 基于教师 Critic 的内在奖励

定义基于教师价值函数的塑形奖励/Shaping Reward：

$$
\hat{r}_t^V := V_\pi(\tau_{t+1}) - V_\pi(\tau_t) + r_t
$$

这是标准的基于势函数的奖励塑形。其累积和 $\sum_t \hat{r}_t^V = \mathbb{E}_{\pi_\theta}[\sum_t V_\pi(\tau_{t+1}) - V_\pi(\tau_t) + r_t] = \mathbb{E}_{\pi_\theta}[\sum_t r_t] + \text{const}$，因此不改变最优策略。

> [!note] Proposition 4
> 若教师是最优策略，则对于学生执行的任何偏离最优路径的动作 $a_t$，塑形奖励给出即时惩罚 $\hat{r}_t^V < r_t$；而沿最优路径的动作则 $\hat{r}_t^V = r_t$。

该性质使得 Teacher V reward 方法能在教师强的情况下显著加速学习——偏离教师路径的动作立即受到惩罚。实验显示（Figure 6），对于最优教师，Teacher V reward 与 policy cloning 方法均快速收敛。但对于对抗性教师（故意最小化回报），所有 policy cloning 方法完全失败，而 Teacher V reward 仍然失败——因为它只保证改进到教师水平，而非解决原始任务。

## 4. Experiments

### 实验设置

所有实验在随机生成的 $20 \times 20$ grid world MDP 上进行：
- 4 个移动动作（上下左右），转移含 10% 噪声（以等概率执行其他动作）
- 状态空间约 400 个格子，部分为墙壁（不可达）、部分为终止状态（带正/负奖励）
- 教师由 Q-Learning 训练 30k 步，或由 A2C 训练
- 每个实验在 1000 个随机 MDP 上平均，报告 0.95 置信区间
- 蒸馏使用表格策略参数化（每个状态-动作对一个 logit 参数），学生通过 softmax 输出概率

### 核心实验结果

**控制策略比较**（Figure 3）：学生驱动蒸馏在几乎所有设定下均为最优。在学生分布下的 KL 散度、教师分布下的匹配度、以及实际获得的回报三个指标上均领先。教师越确定性（温度越低），学生驱动蒸馏的优势越显著，因为确定性教师只覆盖状态空间的极小子集。

**更新规则比较**（Figure 4、5）：
- **大动作空间下的稳定性**：当动作空间从 4 扩大到 1000 时，entropy regularised 完全崩塌（方差爆炸），而 expected entropy regularised 保持稳定
- **不完美教师下的鲁棒性**：纯 policy cloning（on-policy distill、entropy regularised 等）在教师有 25% 噪声时最终复制教师错误导致回报下降；加入环境奖励（+R）的方法和价值门控方法表现更好
- **最优教师 vs. 对抗教师**（Figure 6）：在链式 MDP（corridor）上，最优教师下所有方法都能快速收敛；对抗教师下只有利用环境奖励的方法（On-policy distill+R、Exp. entropy regularised+R）才能学到正向回报，纯蒸馏方法完全失败

### 实验局限性

- **仅限表格设定**：所有实验使用表格策略参数化（每个状态-动作对独立参数），未验证深度神经网络函数逼近下的行为。论文自身指出函数逼近器的作用是重要的开放问题
- **合成 MDP 环境**：$20 \times 20$ grid world 与 Atari 等高维视觉任务差距巨大，结论的可迁移性未知
- **收敛保证的强假设**：Proposition 1 要求强随机性（所有动作概率大于零），实践中深度 RL 的策略往往近乎确定性
- **缺少与已有深度 RL 蒸馏方法的直接对比**：如 *Policy Distillation*、*Kickstarting*、*Mix&Match* 等方法均未在深度 RL 实验中与本文方法比较
- **Proposition 2 的实用性**：价值门控依赖对 $V_{\pi_{\theta^*}}$ 的准确估计，而这在训练过程中本身就是难题

## 5. Related Work & Future Work

### Related Work

**策略蒸馏方法族**：本文统一分析的方法涵盖了多种已有工作。*Policy Distillation* 和 *Actor-Mimic* 使用教师驱动蒸馏（教师策略收集轨迹）；*Kickstarting* 和 *Mix&Match* 使用学生驱动蒸馏（学生策略收集轨迹）；*Distral* 和 *Equivalence between Policy Gradients and Soft Q-learning* 使用 KL 正则化的 RL 目标。本文将这些方法纳入统一框架，首次系统地比较了它们的数学性质。

**更新规则与梯度向量场**：论文指出 Q-Learning 的更新同样不是梯度向量场，但 Q-Learning 在表格设定下仍然收敛。这与 Proposition 1 的结论一致——表格设定下的独立参数化提供了额外的收敛保证。不同的是，本文进一步指出引入奖励后表格设定也可能不收敛（振荡反例），因此仅有表格参数化并不足以保证安全。

**模仿学习的分布偏移**：学生驱动蒸馏与 *DAgger* 思想一脉相承——在学生自身分布下训练以解决分布偏移。但 DAgger 需要在每轮迭代后重新查询教师，而策略蒸馏中教师的响应（动作概率分布）在任何状态都可以即时计算。

### Future Work

论文自身提及的方向：

> - 将理论结果从表格设定推广到函数逼近器（深度网络）设定，理解函数逼近如何影响收敛性质
> - 在大规模深度 RL 任务（如 Atari、连续控制）上验证 expected entropy regularised distillation 的实际效果
> - 探索不同蒸馏方法在非平稳教师（教师仍在训练中）场景下的行为

自然延伸的方向包括：将统一框架扩展到连续动作空间（此时交叉熵需替换为连续分布间的散度度量）；研究多教师蒸馏和渐进式蒸馏中不同更新规则的影响；以及将梯度向量场分析应用于其他 RL 中的非梯度更新（如各类 Actor-Critic 变体的熵项）。
