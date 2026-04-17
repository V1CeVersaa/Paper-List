---
title: "Scaling CRL Depth"
headline: "1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities"
visibility: "public"
status: "drafting"
description: "Paper note on 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities."
---

> [!abstract] Contributions
>
> 本文研究了自监督强化学习/Self-Supervised RL 中网络深度的扩展问题。作者以对比强化学习/Contrastive RL/CRL 为基础算法，结合残差连接/Residual Connections、层归一化/Layer Normalization 和 Swish 激活函数，将网络深度从传统 RL 中常用的 2–5 层扩展到最多 1024 层。在无监督目标条件/Goal-Conditioned 设定下（无奖励函数、无演示），深度扩展在 10 个仿真运动、导航和操控任务上带来了 2×–50× 的性能提升，超过了 SAC+HER、TD3+HER、GCBC、GCSL 等基线。更关键的是，性能提升并非渐进式的，而是在特定的临界深度/Critical Depth 处出现跃迁，伴随着质变的策略行为（如 Humanoid 从摔倒到直立行走再到翻墙）。
>
> 本文的主要局限在于：（1）深度扩展带来的计算开销显著（depth 64 的训练时间约为 depth 4 的 3–5 倍）；（2）深度扩展的有效性高度依赖 CRL 算法本身——对 SAC、TD3+HER 等基于时序差分/Temporal Difference 的方法，增加深度几乎无效；（3）在离线/Offline 设定下，深度扩展尚未展现出一致的收益。
 
## 1. Introduction

扩展模型规模在自然语言处理和计算机视觉中已经催生了一系列突破性能力，并且一般只有在模型规模超过一个临界点时，模型才能学到解决特定问题的能力，但在强化学习/Reinforcement Learning/RL 领域，类似的规模扩展进展一直缓慢，是否可以通过扩展 RL 网络的方式来实现类似的飞跃仍然是一个开放问题。传统观点认为，大型的 AI 系统应该通过自监督方式进行训练，而 RL 只应被用来对这些模型进行微调。因此，若希望扩展强化学习的规模，自监督方法可能就是答案的关键。

<!--

在本文中，我们研究的是**扩展强化学习规模所需的基本构件**。第一步，是重新思考上述传统观点：“强化学习”和“自监督学习”并不是彼此对立的学习规则，相反，它们可以被结合为一种**自监督强化学习系统**；这种系统无需参考奖励函数或示范数据，就能够通过探索来学习策略（Eysenbach et al., 2021, 2022; Lee et al., 2022）。在这项工作中，我们采用的是最简单的自监督 RL 算法之一，即 **Contrastive RL（CRL）**（Eysenbach et al., 2022）。第二步，是认识到提高数据可获得性的重要性。为此，我们建立在近期的 **GPU 加速 RL 框架** 之上（Makoviychuk et al., 2021; Rutherford et al., 2023; Rudin et al., 2022; Bortkiewicz et al., 2024）。第三步，则是增加网络深度，使用比以往工作中通常采用的网络**深 100 倍**的模型。为了稳定训练如此深的网络，我们需要吸收先前工作中的一些架构技术，包括 **残差连接**（He et al., 2015）、**层归一化**（Ba et al., 2016）以及 **Swish 激活函数**（Ramachandran et al., 2018）。我们的实验还将研究 **batch size** 和 **网络宽度** 的相对重要性。 

本文的主要贡献在于表明：一种将上述构件整合进单一 RL 方法中的做法，能够表现出很强的**可扩展性**。

**经验可扩展性（Empirical Scalability）**：我们观察到了显著的性能提升；在一半的环境中，性能提升超过 **20 倍**，并且优于其他标准的目标条件（goal-conditioned）基线方法。这些性能提升还对应于随着规模增大而出现的、**定性上不同的策略行为**。 

**网络架构中的深度扩展（Scaling Depth in Network Architecture）**：尽管许多先前的 RL 工作主要关注增加网络宽度，但它们往往报告说，增加深度只能带来有限收益，甚至会带来负收益（Lee et al., 2024; Nauman et al., 2024b）。相比之下，我们的方法打通了沿着**深度维度**进行扩展的能力，并获得了超过单纯增加宽度的性能提升（见第 4 节）。 

**经验分析（Empirical Analysis）**：我们对这一扩展方法中的关键组成部分进行了广泛分析，揭示了其中的关键因素，并提供了新的见解。 

我们预计，未来的研究可以在这一基础上继续挖掘出更多这样的基本构件。 

## **2 Related Work**

近来，自然语言处理（NLP）和计算机视觉（CV）在方法上逐渐趋同：它们采用了相似的架构（例如 **Transformer**），也共享了相似的学习范式（即 **自监督学习**）；正是这两者的结合，使得大规模模型获得了变革性的能力（Vaswani et al., 2017; Srivastava et al., 2023; Zhai et al., 2021; Dehghani et al., 2023; Wei et al., 2022）。相比之下，在强化学习（RL）中实现类似的进展仍然具有挑战性。已有若干研究考察了扩展大规模 RL 模型所面临的障碍，包括：**参数利用不足**（Obando-Ceron et al., 2024）、**可塑性与容量损失**（Lyle et al., 2024, 2022）、**数据稀疏性**（Andrychowicz et al., 2017; LeCun, 2016），以及 **训练不稳定性**（Ota et al., 2021; Henderson et al., 2018; Van Hasselt et al., 2018; Nauman et al., 2024a）。因此，当前对 RL 模型进行规模扩展的尝试，大多仍局限于某些特定问题领域，例如 **模仿学习**（Tuyls et al., 2024）、**多智能体博弈**（Neumann and Gros, 2022）、**语言引导的 RL**（Driess et al., 2023; Ahn et al., 2022），以及 **离散动作空间**（Obando-Ceron et al., 2024; Schwarzer et al., 2023）。 

近期的一些方法提出了若干有前景的方向，包括新的**架构范式**（Obando-Ceron et al., 2024）、**分布式训练方法**（Ota et al., 2021; Espeholt et al., 2018）、**分布式 RL**（distributional RL）（Kumar et al., 2023），以及**蒸馏**（Team et al., 2023）。与这些方法相比，我们的方法只是对一个现有的自监督 RL 算法做了一个简单扩展。与我们最接近的近期工作包括 Lee et al. (2024) 和 Nauman et al. (2024b)：它们利用**残差连接**来促进更宽网络的训练。这些工作主要聚焦于**网络宽度**，并指出增加深度只能带来有限收益，因此两篇工作都只使用了 **4 层 MLP** 架构。而在我们的方法中，我们发现增加宽度确实可以提升性能（第 4.4 节）；但与此同时，我们的方法也使得沿着**深度方向**的扩展成为可能，而且这一路径比单纯增加宽度更强。 

另一项值得注意的关于训练更深网络的工作，是 Farebrother et al. (2024) 提出的做法：他们将**基于价值的 RL** 转化为一个**分类问题**，具体做法是将 TD 目标离散化，并写成一个类别交叉熵损失。该方法背后的推测是：**基于分类的方法可能比基于回归的方法更鲁棒、更稳定，因此也可能表现出更好的扩展性质**（Torgo and Gama, 1996; Farebrother et al., 2024）。而我们所使用的 CRL 算法，本质上也使用了一种交叉熵损失（Eysenbach et al., 2022）。它的 **InfoNCE 目标** 可以被看作交叉熵损失的一种推广，因此，它实际上是通过判断当前状态和动作是否与一个通向目标状态的轨迹属于同一条轨迹，来完成 RL 任务的。沿着这一思路，我们的工作构成了第二项证据，说明**分类**——就像交叉熵在 NLP 规模化成功中所扮演的角色那样——也可能是 RL 中一个潜在的关键基本构件。 

如果你愿意，下一条我可以继续给你做 **Preliminaries** 的准确翻译，并且把其中的公式一起规范地整理成 LaTeX。
 -->

当前大多数 RL 工作使用的网络仅有 2–5 层 MLP，在扩展深度时常遭遇参数利用不足、可塑性损失/Plasticity Loss、数据稀疏、训练不稳定等问题。已有的扩展尝试主要集中在增大网络宽度上，且往往报告深度增加带来的收益有限甚至为负。

本文的核心洞察是：要实现 RL 中的深度扩展，需要同时满足三个条件——（1）采用自监督学习范式而非传统的奖励驱动学习，因为大规模 AI 系统的训练主要依赖自监督，RL 的稀疏奖励信号不足以支撑大模型训练；（2）充分利用数据，借助 GPU 加速的 RL 框架提升数据吞吐量；（3）采用经过验证的深度网络架构技术（残差连接、层归一化、Swish 激活）来稳定深层网络的训练。

作者选择对比强化学习/CRL 作为基础算法。CRL 是一种 actor-critic 自监督 RL 方法，其 critic 使用 InfoNCE 损失进行训练——这本质上是一种交叉熵/分类损失。已有研究指出，基于分类的方法可能比基于回归的方法更具鲁棒性和扩展性（如 *Stop Regressing* 一文所论证的），CRL 的 InfoNCE 目标恰好是交叉熵损失的推广形式。这为 CRL 的深度扩展提供了一个可能的解释：分类范式本身就具备更好的 scaling 特性。

## 2. Problem Setup

本文的实验在在线目标条件强化学习/Online Goal-Conditioned RL 设定下进行。形式化地，定义目标条件 MDP 为元组 $\mathcal{M}_g = (\mathcal{S}, \mathcal{A}, p_0, p, p_g, r_g, \gamma)$，其中：

- $\mathcal{S}$：状态空间；$\mathcal{A}$：动作空间
- $p_0(s_0)$：初始状态分布
- $p(s_{t+1} \mid s_t, a_t)$：状态转移概率
- $\mathcal{G}$：目标空间，通过映射 $f: \mathcal{S} \to \mathcal{G}$ 与状态空间关联（$\mathcal{G}$ 可以是 $\mathcal{S}$ 的子维度）
- $p_g(g)$：目标先验分布
- $r_g(s_t, a_t) \triangleq (1-\gamma) p(s_{t+1} = g \mid s_t, a_t)$：奖励函数，定义为下一步到达目标 $g$ 的概率密度乘以 $(1-\gamma)$
- $\gamma$：折扣因子

目标条件策略 $\pi(a \mid s, g)$ 同时接收当前状态和目标作为输入。定义折扣状态访问分布为：

$$
p_\gamma^{\pi(\cdot \mid \cdot, g)}(s) \triangleq (1-\gamma) \sum_{t=0}^{\infty} \gamma^t p_t^\pi(s)
$$

其中 $p_t^\pi(s)$ 是策略 $\pi$ 在第 $t$ 步访问状态 $s$ 的概率。对应的 Q 函数为 $Q_g^\pi(s,a) \triangleq p_\gamma^{\pi(\cdot \mid \cdot, g)}(g \mid s, a)$，即以 $(s,a)$ 为起点、在策略 $\pi$ 下到达目标 $g$ 的折扣概率。优化目标为最大化期望累积奖励：

$$
\max_\pi \mathbb{E}_{p_0(s_0), p_g(g), \pi(\cdot \mid \cdot, g)} \left[ \sum_{t=0}^{\infty} \gamma^t r_g(s_t, a_t) \right]
\tag{1}
$$

需要注意的是，这里的实验设定是**无监督**的：不提供人工设计的奖励函数，也不提供专家演示。智能体需要从零开始探索环境，学习如何到达被指定的任意目标状态。评估指标为在 1000 步 episode 中智能体停留在目标附近的时间步数。

## 3. Methods

### 3.1 Contrastive RL

CRL 采用 actor-critic 架构。记 critic 为 $f_{\phi,\psi}(s, a, g)$，actor 为 $\pi_\theta(a \mid s, g)$。

Critic 由两个神经网络参数化：状态-动作对编码器 $\phi(s,a)$ 和目标编码器 $\psi(g)$，critic 的输出定义为两个嵌入之间的 $l^2$ 范数：

$$
f_{\phi,\psi}(s, a, g) = \|\phi(s,a) - \psi(g)\|_2
$$

Critic 使用 InfoNCE 目标进行训练。在每个 batch $\mathcal{B}$ 中，$(s_i, a_i, g_i)$ 表示来自同一条轨迹的状态、动作和未来状态（作为目标），而 $g_j$ 是从不同轨迹中采样的目标。优化目标为：

$$
\min_{\phi, \psi} \mathbb{E}_{\mathcal{B}} \left[ -\sum_{i=1}^{|\mathcal{B}|} \log \left( \frac{e^{f_{\phi,\psi}(s_i, a_i, g_i)}}{\sum_{j=1}^{K} e^{f_{\phi,\psi}(s_i, a_i, g_j)}} \right) \right]
\tag{2}
$$

这个目标的直觉是：critic 学会判断当前的状态-动作对是否处在通往特定目标的轨迹上。正样本 $(s_i, a_i, g_i)$ 来自同一条轨迹，负样本 $g_j$ 来自其他轨迹。InfoNCE 的 softmax 结构使得正样本对应的 $f$ 值需要高于负样本才能降低损失——换言之，$f$ 在此充当对数似然意义上的评分函数/scoring function，而非直觉上的"距离越小越好"。训练使得嵌入空间被组织为：处于通向目标 $g$ 的轨迹上的 $(s,a)$ 对应更高的 $f$ 值。

Actor 的训练目标是最大化 critic 的值：

$$
\max_{\pi_\theta} \mathbb{E}_{p_0(s_0), p(s_{t+1} | s_t, a_t), p_g(g), \pi_\theta(a | s, g)} \left[ f_{\phi,\psi}(s, a, g) \right]
\tag{3}
$$

> [!tip] CRL 与传统 RL 的关键区别
> CRL 的核心优势在于其损失函数是 InfoNCE——本质上是一种分类/交叉熵损失，而非传统 TD 方法中的回归损失。这使得 CRL 避免了 TD 学习中常见的 bootstrapping 不稳定性，同时也可能是其在深度扩展上表现优异的关键因素。

### 3.2 Deep Residual Architecture

为使深层网络能够稳定训练，本文在 actor 和 critic（两个编码器 $\phi$、$\psi$）中均引入残差连接。每个残差块/Residual Block 包含 4 个 Dense 层，每层之后依次接 Layer Normalization 和 Swish 激活函数。残差连接在块的最后一层激活之后施加：

$$
\mathbf{h}_{i+1} = \mathbf{h}_i + F_i(\mathbf{h}_i)
$$

其中 $F_i$ 是残差块内的 4 层变换。本文定义网络深度为所有残差块中 Dense 层的总数，因此深度 $D = 4N$（$N$ 为残差块数）。Actor 和两个 critic 编码器的深度同步扩展（除了个别消融实验外）。

<!-- TODO: insert Figure 2 from paper — 残差架构示意图 -->

> [!info] 三个架构组件缺一不可
> 消融实验（附录 A.5）显示，移除 Layer Normalization 或将 Swish 替换为 ReLU 都会显著削弱深度扩展的效果。结合残差连接本身的必要性（Figure 5 右侧），这三个组件——残差连接、Layer Normalization、Swish 激活——共同构成深度扩展的基础。

### 3.3 Training Pipeline

整体训练流程保持标准在线 RL 框架：智能体在 512 个并行环境中与环境交互，采集的经验存入 replay buffer（容量 10,000 个 episode），critic 和 actor 按 1:40 的 UTD（update-to-data）比率更新。两者的学习率均为 $3 \times 10^{-4}$，表示维度为 64，batch size 默认 512。

> 具体超参数：episode 长度 1000 步，折扣因子 $\gamma = 0.99$，logsumexp penalty 为 0.1。完整超参数列表见原文 Table 7。

## 4. Experiments

### 4.1 Experimental Setup

实验在 JaxGCRL 基准上进行，基于 Brax 和 MJX 物理引擎，涵盖 10 个任务：

- **运动任务**：Humanoid、Humanoid U-Maze、Humanoid Big Maze
- **导航任务**：Ant Big Maze、Ant Hardest Maze、Ant U4-Maze、Ant U5-Maze
- **操控任务**：Arm Push Easy、Arm Push Hard、Arm Binpick Hard

所有环境使用稀疏奖励：仅当智能体位于目标附近时 $r=1$。评估指标为 1000 步 episode 中停留在目标附近的时间步数，取最后 5 个 epoch 的平均值。基线方法使用 4 层 MLP（与 JaxGCRL 基准及先前工作一致），本文测试深度 8、16、32、64，最高至 1024。

<!-- TODO: insert Figure 1 from paper — 10 个任务上不同深度的学习曲线，展示 depth 扩展带来的性能提升 -->

### 4.2 Core Scaling Results

深度扩展在所有 10 个环境中均带来显著的性能提升（Table 1）：

| 任务类别            | 观测维度 | Depth 4 → 64 提升倍数 |
| ------------------- | -------- | --------------------- |
| 操控任务（dim=17）  | 17       | 2.4×–5.7×             |
| 导航任务（dim=29）  | 29       | 1.8×–63×              |
| 类人任务（dim=268） | 268      | 50×–1051×             |

提升幅度与观测维度高度正相关：在 268 维的 Humanoid 环境中，depth 64 相比 depth 4 实现了 52× 的提升（从 12.6 到 649），而 Humanoid Big Maze 甚至达到 1051×（从 0.06 到 59）。

与其他基线的比较（Figure 12，附录 A.1）：Scaled CRL（depth 64）在 10 个环境中的 8 个超越了所有基线（SAC、SAC+HER、TD3+HER、GCBC、GCSL）。唯一的例外是 Humanoid Maze 系列中 SAC 在早期表现更好，但 Scaled CRL 最终追平。

### 4.3 Critical Depth and Emergent Policies

性能提升并非随深度平滑增长，而是在特定的临界深度处出现跃迁。不同任务的临界深度不同：Ant Big Maze 约 8 层，Humanoid U-Maze 约 64 层，最深至 1024 层仍可在最难的任务上持续提升。

在 Humanoid 环境中，不同深度对应质变的行为：

- **Depth 4**：智能体摔倒后将身体扔向目标
- **Depth 16**：智能体学会直立行走到达目标
- **Depth 64**：在 U-Maze 中智能体试图绕墙但仍倒下
- **Depth 256**：智能体学会蜷缩身体翻越中间墙壁，展现出类似"攀爬"的复杂行为

<!-- TODO: insert Figure 3 from paper — Humanoid 在不同深度下的行为对比：摔倒、行走、挣扎、翻墙 -->

这种行为的质变表明，深度扩展不仅是量的积累，而是在表征能力突破特定阈值后涌现出新的技能。

### 4.4 Scaling Analysis

**深度 vs. 宽度**：在固定参数量的对比下（Figure 4），深度扩展比宽度扩展更为高效。例如在 Humanoid 环境中，将宽度从 256 增加到 2048（depth=4）的效果不如简单地将深度从 4 加倍到 8（width=256）。由于每层的参数量与宽度的平方成正比，参数总量随宽度二次增长、随深度线性增长——例如 4 层 width-2048 的网络约有 35M 参数，而 32 层 width-256 的网络仅约 2M。因此在固定 FLOP 或内存预算下，深度扩展是更参数高效的方式。这一效应在高维观测任务中更为突出。

**Actor vs. Critic 的扩展**：Figure 6 的消融显示，两者的扩展具有互补性。在 Arm Push Easy 和 Humanoid 中，扩展 critic 更为重要；在 Ant Big Maze 中，扩展 actor 更为关键。整体而言，同时扩展两者的效果最佳。

**深层网络解锁 batch size 扩展**：一个重要发现是，batch size 增大仅在网络足够深时才有效。在 depth 4 时，batch size 从 128 到 2048 几乎没有影响；但在 depth 64 时，增大 batch size 带来显著的额外提升（Figure 7）。这表明，传统浅层 RL 网络的容量不足，无法利用更大 batch size 提供的信息，而深层网络打破了这一瓶颈。

**探索与表达能力的协同效应**：通过一个精巧的实验设计（Figure 8），作者分离了"更好探索"和"更强表达能力"两个因素。设置三个网络并行训练：一个 collector 负责与环境交互并填充共享 replay buffer，另外两个 learner（一深一浅）仅从 buffer 中学习。结果表明：

- 当 collector 为深层网络（depth 32）时，深层 learner 显著优于浅层 learner → 表达能力确实重要
- 当 collector 为浅层网络（depth 4）时，深浅 learner 性能接近且均较差 → 仅有表达能力不够，需要高质量的数据覆盖

这说明深度扩展的收益来自探索能力与表达能力的协同：更强的学习能力驱动更广泛的探索，而更好的数据覆盖又反过来充分释放深层网络的学习潜力。

**更丰富的对比表征**：在 Ant U4-Maze 中可视化 Q 值，浅层网络（depth 4）的 Q 值主要依赖欧几里得距离作为代理，即使目标被墙隔开也给出高 Q 值；而深层网络（depth 64）学到的 Q 值正确地反映了迷宫的拓扑结构，沿着内侧路径逐渐衰减。

<!-- TODO: insert Figure 9 from paper — U4-Maze 中 depth 4 vs depth 64 的 Q 值热力图对比 -->

在 Humanoid 环境中可视化轨迹的 state-action 嵌入（Figure 10），浅层网络将接近目标的状态紧密聚集，而深层网络将它们展开到一个弯曲表面上——这表明深层网络能将更多的表征容量分配给任务关键的状态区域。

**部分经验拼接/Partial Experience Stitching**：在一个修改版的 Ant U-Maze 中，训练时只暴露距离 $\leq 3$ 单位的起点-目标对，测试时评估跨越 6 单位的远距离目标。Depth 4 仅能到达最近的目标，depth 16 偶尔成功，depth 64 能到达最远的目标——说明深层网络具备将短距离经验组合拼接为长距离规划的泛化能力。

### 4.5 The CRL Algorithm is Key

深度扩展的效果高度依赖算法选择。附录 A.2 的实验表明，SAC、SAC+HER、TD3+HER 在深度超过 4 层后性能饱和甚至下降，无法从深度增加中获益。这与先前工作的发现一致，说明基于 TD 的方法目前不具备深度扩展的条件。GCSL 在 Humanoid 和 Ant 任务上完全失效。GCBC 是一个例外——在 Arm Push Easy 上随深度增加有所提升——但总体效果远不如 CRL。

这一结果暗示 CRL 的 InfoNCE（分类）损失可能是使其适合深度扩展的核心因素，与 NLP 中交叉熵损失在扩展中的关键角色形成类比。

### 4.6 Offline Setting and Computational Cost

在离线目标条件 RL 设定（OGBench）中，CRL 的深度扩展效果不一致：3 个环境中有 2 个出现性能下降。作者尝试了仅扩展 critic 深度以及 cold initialization 等策略，但未能稳定改善。不过，附录 A.3 的补充实验表明，离线 GCBC 在 antmaze-medium-stitch 任务上随深度增加确实取得了提升，而 BC 和 QRL 则未能获益。这说明离线设定下深度扩展的可行性可能取决于算法选择，而非一律无效；但至少对于 CRL 而言，在线探索似乎是深度扩展发挥作用的关键条件之一。

计算代价方面，wall-clock time 大致随深度线性增长（超过某一阈值后）。Depth 64 的训练时间约为 depth 4 的 3–5 倍。在 Humanoid U-Maze 上训练 1024 层网络需要约 134 小时。但以达到特定性能水平所需的时间来衡量（Table 6），Scaled CRL 在 7/10 环境中比 SAC 更快。

### 4.7 Critical Assessment

本文实验设计的优点在于系统性的消融和多角度分析，尤其是 collector-learner 分离实验和经验拼接实验具有较高的洞察价值。但也存在以下可指出的弱点：

- **环境多样性有限**：所有任务均为仿真中的连续控制，缺乏离散动作空间、视觉观测、多智能体等更广泛的设定
- **基线深度比较不完全对称**：基线方法使用标准 4 层 MLP，而 Scaled CRL 使用了完全不同的架构（残差+LayerNorm+Swish）。性能提升中有多少来自深度本身、多少来自架构改进，虽有 Figure 5 的部分消融，但未完全分离
- **1024 层实验仅限 Humanoid Maze 的 2 个任务**，且 1024 层时 actor loss 发散，作者不得不将 actor 深度限制在 512 层，这暗示当前方案在极端深度下的训练稳定性仍有不足
- **离线设定的负面结果**说明该方法的适用性尚窄，但作者对此的讨论较为简短
- **缺少与 Transformer 架构的比较**：考虑到 Transformer 在其他领域的扩展成功，这一对比的缺失值得注意

## 5. Related Work & Future Work

### Related Work

**RL 中的扩展障碍**：已有研究指出 RL 模型扩展面临参数利用不足、可塑性和容量损失、数据稀疏、训练不稳定等挑战。现有的扩展努力主要限于模仿学习、多智能体博弈、语言引导 RL 和离散动作空间等特定领域。

**网络宽度扩展**：*SimBa* 和 *Bigger, Regularized, Optimistic (BRO)* 是最新的扩展工作，两者均使用残差连接但主要聚焦于宽度扩展，且仅使用 4 层 MLP，报告深度增加收益有限。本文的方法在此基础上打开了深度扩展这一新维度。

**分类视角的 RL**：*Stop Regressing* 一文将基于价值的 RL 转化为分类问题，使用类别交叉熵损失替代 TD 回归目标，并据此训练更深的网络。CRL 的 InfoNCE 目标本质上也是交叉熵损失的推广，两者从不同角度共同支持了"分类范式在 RL 扩展中具有优势"这一假说。

**准度量架构/Quasimetric Architectures**：近期工作发现时间距离满足三角不等式的不变性，据此提出准度量网络。附录 A.4 测试了在 CMD-1 算法上的深度扩展，结果表明扩展收益同样存在，但一致性不如 CRL。

### Future Work

作者明确提出的方向：

- 探索分布式训练以利用更多计算资源
- 研究剪枝和蒸馏技术以降低深层网络的推理成本
- 将深度扩展方法适配到离线 RL 设定
- 调查其他自监督 RL 算法是否也能从深度扩展中获益

此外，从论文的局限性可以自然推断的方向包括：将方法扩展到视觉观测和更复杂的任务设定，理解临界深度出现的理论机制，以及探索 Transformer 等替代架构在自监督 RL 中的扩展特性。附录 A.6 已初步验证 Simba-v2 的超球面归一化/Hyperspherical Normalization 可以进一步提升样本效率，说明该框架能够自然地整合新兴的 RL 扩展技术。
