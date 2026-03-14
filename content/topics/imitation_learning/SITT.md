---
title: SITT
headline: "Student-Informed Teacher Training"
---

> [!abstract] Contributions
>
> 本文针对特权模仿学习/Privileged Imitation Learning 中教师-学生信息不对称/Information Asymmetry 的问题，提出了一种联合训练框架 SITT/Student-Informed Teacher Training。核心思路是：不再单纯调整学生策略以接近教师，而是反过来约束教师策略，使其学习到学生可以模仿的行为。具体地，基于模仿学习中学生-教师性能差距的上界，SITT 在教师训练中引入两项修改：（1）将教师与学生的近似动作差异作为惩罚项加入奖励函数，引导教师避免访问学生无法推断正确动作的状态；（2）通过 KL 散度梯度直接监督教师网络权重，推动教师表征向学生对齐。此外，通过引入代理学生网络/Proxy Student Network，避免了在教师交互时渲染高维学生观测的计算开销。在迷宫导航、视觉四旋翼避障和视觉机械臂操作三个任务上，SITT 均显著提升了学生成功率。
>
> 该方法的核心假设是教师和学生可以联合训练（即训练过程在同一仿真环境中交替进行），且依赖代理学生网络对真实学生行为的近似质量。代理学生仅使用教师观测来预测学生行为，其近似精度受限于对齐数据的子集大小和分布覆盖。此外，实验场景均在仿真中完成，缺少 sim-to-real 的验证。

## 1. Introduction

在强化学习/Reinforcement Learning/RL 中，智能体通过与环境的交互最大化累积奖励来学习任务，但当输入为高维观测（如图像）时，策略需要同时学习感知和控制，导致训练效率低下。特权模仿学习通过两阶段流程加速训练：先用特权信息（如环境中所有障碍物的精确位置）训练教师策略，再让学生策略从有限观测（如前向摄像头图像）中模仿教师。这种范式避免了学生从零开始探索的高成本。

然而，这一框架面临一个根本性困难：**信息不对称**。教师可以访问完整的环境信息，因此倾向于学习依赖全局可观测性的行为，而这些行为可能恰恰是学生无法从其有限观测中推断出的。例如，在机器人导航任务中，教师接收到所有障碍物的相对距离，而学生仅有前向摄像头——当障碍物不在视野内时，学生根本无法推断应该如何避障。

现有方法主要从学生端入手，例如在学生训练中混合 RL 目标（如 *COSIL*、*TGRL*、*Real-World Humanoid Locomotion with RL*），或通过 DAgger 式的在线数据聚合来缓解分布偏移。但这些方法都没有改变教师本身的行为——教师依然可能给出学生无法模仿的"目标动作"。

SITT 的关键洞察在于：与其让学生努力追赶一个不考虑学生能力的教师，不如让教师主动适应学生的限制。具体而言，作者从模仿学习的性能上界出发，将学生性能的上界嵌入教师的优化目标。这一思路自然导出两个机制：一是通过惩罚项影响教师的探索行为（避免高分歧状态），二是通过梯度直接对齐教师的表征。由此，教师不仅优化任务奖励，还被引导学习"对学生友好"的行为，从而从根本上缩小信息不对称造成的性能差距。

## 2. Problem Setup

### RL and IL Foundations

问题在标准马尔可夫决策过程/Markov Decision Process/MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \mu_0)$ 框架下定义。策略 $\pi$ 的期望回报为：

$$
J(\pi) = \mathbb{E}_{s \sim \mu_0}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 \sim \mu_0, a_t \sim \pi(\cdot|s_t), s_{t+1} \sim P(\cdot|s_t, a_t)\right]
$$

在特权模仿学习中，教师策略 $\pi_T$ 利用特权观测通过 RL 训练得到，学生策略 $\pi_S$ 使用有限观测通过模仿教师来学习。

### Performance Gap Upper Bound

*Error Bounds of Imitating Policies and Environments* 和 *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* 建立了教师与学生性能差距的上界：假设奖励函数有界 $|r(s,a)| \le r_{\max}$，则

$$
J(\pi_T) - J(\pi_S) \le \frac{2\sqrt{2}\, r_{\max}}{(1-\gamma)^2}\sqrt{\epsilon} \tag{1}
$$

其中 $\epsilon$ 是在教师折扣平稳状态分布 $d_{\pi_T}(s) = (1-\gamma)\sum_{t=0}^{\infty}\gamma^t \Pr(s_t = s; \pi_T)$ 下，教师与学生动作分布的期望 KL 散度的上界：

$$
\mathbb{E}_{s \sim d_{\pi_T}}[D_{\mathrm{KL}}(\pi_T(\cdot|s), \pi_S(\cdot|s))] \le \epsilon \tag{2}
$$

这一上界是 SITT 的理论出发点：减小 $\epsilon$ 既可以通过调整学生 $\pi_S$（传统模仿学习的做法），也可以通过调整教师 $\pi_T$ 的状态分布 $d_{\pi_T}$ 和动作分布来实现——后者正是 SITT 的切入点。

### Core Difficulty

在特权模仿学习中，教师和学生的观测空间存在根本差异：教师观测 $o_T$ 包含特权信息（如障碍物精确位置），学生观测 $o_S$ 仅包含受限信息（如图像）。由于学生的部分可观测性，存在某些状态 $s$，教师可以确定最优动作，但学生从 $o_S$ 中无法推断该动作。如果教师频繁访问这类状态，$\epsilon$ 必然很大，学生性能就会受到严重影响。

## 3. Algorithm

### 3.1 Objective Derivation

SITT 的核心思路是：不再仅通过调整 $\pi_S$ 来最小化公式 $(2)$ 中的 $\epsilon$，而是同时优化教师 $\pi_T$，使其在最大化任务奖励的同时考虑与学生的对齐。具体地，寻找最大化以下目标的教师策略：

$$
\tilde{J}(\pi_T) = \mathbb{E}_{s \sim d_{\pi_T}, a \sim \pi_T(\cdot|s)}[r(s,a)] - \mathbb{E}_{s \sim d_{\pi_T}}[D_{\mathrm{KL}}(\pi_T(\cdot|s), \pi_S(\cdot|s))] \tag{3}
$$

第一项是标准的任务奖励最大化，第二项则要求教师在其访问的状态上与学生的动作分布保持一致。将状态分布转化为对轨迹 $\tau \sim p_\theta$ 的期望后（$\theta$ 为教师网络参数），定义轨迹回报 $R(\tau) = \sum_{s_t, a_t \in \tau} \gamma^t r(s_t, a_t)$ 和轨迹 KL 散度和 $D_\theta(\tau) = \sum_{s_t \in \tau} \gamma^t D_{\mathrm{KL}}(\pi_T(\cdot|s_t), \pi_S(\cdot|s_t))$，对目标关于 $\theta$ 求梯度：

$$
\nabla_\theta \tilde{J}(\pi_T) = \underbrace{\int \nabla_\theta p_\theta(\tau)(R(\tau) - D_\theta(\tau))\,d\tau}_{\text{Policy Gradient}} - \underbrace{\int p_\theta(\tau)\nabla_\theta D_\theta(\tau)\,d\tau}_{\text{KL-Div Gradient}} \tag{4}
$$

这个梯度自然分解为两个部分：

1. **Policy Gradient 项**：标准策略梯度，但奖励信号从 $R(\tau)$ 变为 $R(\tau) - D_\theta(\tau)$。KL 散度 $D_\theta$ 起到惩罚项的作用，鼓励教师访问与学生对齐的状态，避免学生无法推断正确动作的状态区域。这一项影响教师的**探索行为**。

2. **KL-Div Gradient 项**：对教师网络参数的直接监督，要求教师在其状态分布下预测与学生相似的动作分布。这一项直接修改教师的**表征**，使其向学生靠拢。

> [!tip] 两个 KL 项的不同角色
> Policy Gradient 中的 $D_\theta$ 作为奖励惩罚，影响教师策略的轨迹选择——教师学习避开与学生分歧大的状态。KL-Div Gradient 则作为损失函数，直接推动教师的网络输出向学生对齐——即使在当前状态下，教师也尽量给出学生能产生的动作。前者影响"去哪里"，后者影响"在那里做什么"。

### 3.2 Architecture

为了实现上述目标函数，SITT 引入了三个关键的架构设计：

**代理学生网络/Proxy Student $\hat{F}_S$**：计算公式 $(4)$ 中的 KL 散度需要在教师的每一步交互中同时获得教师和学生的动作分布。但渲染学生的高维观测（如图像）成本很高，违背了特权学习加速训练的初衷。SITT 引入代理学生网络 $\hat{F}_S$：它接收**教师观测**作为输入，尝试预测当前学生策略的动作。这样就可以在不渲染图像的情况下，近似地计算每一步的教师-学生动作差异。代理学生在对齐阶段通过与真实学生的特征 L1 损失进行训练。

**共享动作解码器/Shared Action Decoder $A$**：教师 $F_T$、学生 $F_S$、代理学生 $\hat{F}_S$ 三个编码器各自将观测映射到同一个公共特征空间，再经共享解码器 $A$ 输出动作。共享解码器仅通过任务奖励的策略梯度更新。这一设计有两个好处：（1）学生可以利用教师通过大量环境交互学到的高层特征映射关系；（2）由于对齐数据只是教师交互的子集，共享解码器在更广的状态分布上训练，提供更鲁棒的动作映射。

**编码器结构**：对于视觉任务，教师和代理学生使用三层 MLP（ELU 激活），学生使用冻结的 DINOv2 编码器处理图像，再与状态观测拼接后通过 MLP 映射到公共特征空间。

### 3.3 Three-Phase Alternating Training

SITT 的训练由三个交替进行的阶段组成：

<!-- TODO: insert Figure 1 from paper -->

**Roll-out 阶段**：教师与环境交互收集经验。与标准 RL 训练的区别在于奖励计算：在任务奖励之外，加入教师与代理学生之间动作分布的 KL 散度作为惩罚项：

$$
r'_t = r_t - \lambda_1 D_{\mathrm{KL}}(\pi_T(\cdot|s_t) \| \hat{\pi}_S(\cdot|s_t))
$$

该惩罚影响教师的长期探索行为——教师在选择行为路径时会考虑学生是否能跟随。在此阶段，从教师交互中采样一个子集，同时渲染对应的学生观测（如图像），存入对齐缓冲区。

**Policy Update 阶段**：基于 PPO 的策略更新，同时整合 KL-Div Gradient。由于策略梯度和 KL 散度梯度都在教师的状态分布上计算，可以利用 roll-out buffer 中的数据在一次反向传播中同时优化这两个目标：

$$
\mathcal{L} = \mathcal{L}_{\text{policy}} + \lambda_2 D_{\mathrm{KL}}(\pi_T \| \hat{\pi}_S)
$$

对于连续动作空间（多元高斯分布），由于教师和学生共享动作解码器，协方差矩阵相同 $\Sigma_T = \Sigma_S$，KL 散度简化为均值差异的加权形式：

$$
D_{\mathrm{KL}}(\pi_T(\cdot|s_t), \pi_S(\cdot|s_t)) = \frac{1}{2}\left[\text{const} + (\mu_T(s_t) - \mu_S(s_t))^\top \Sigma_T^{-1}(\mu_T(s_t) - \mu_S(s_t))\right] \tag{5}
$$

> [!tip] KL 散度梯度的直觉含义
> 当教师对其动作很有信心（$\Sigma_T$ 小）但与学生的均值仍存在显著偏差时，该损失会增大。梯度更新会增大协方差，从而增加 roll-out 阶段的探索性，提高教师发现学生可模仿行为的可能性。

**Alignment 阶段**：这是唯一需要配对的教师-学生观测的阶段。使用 roll-out 阶段存储的子集数据，进行两步对齐：
- **学生 $\to$ 教师对齐**：计算学生编码器 $F_S$ 和教师编码器 $F_T$ 输出特征的 L1 损失，以及通过冻结的共享解码器后的激活 L1 损失。梯度仅回传到学生编码器（防止模型坍缩）。
- **代理学生 $\to$ 学生对齐**：计算代理学生 $\hat{F}_S$ 和学生 $F_S$ 编码特征的 L1 损失，梯度仅回传到代理学生。

教师参数在此阶段不被更新，仅在 Policy Update 阶段更新。

> [!note] 算法总结
> 三个阶段的分工：Roll-out 通过惩罚项引导教师的探索方向；Policy Update 通过 KL 梯度直接对齐教师表征；Alignment 通过 L1 损失将学生和代理学生同步到当前教师。三者交替进行，形成闭环：教师适应学生能力 → 学生模仿调整后的教师 → 代理学生跟踪学生 → 代理学生反过来影响教师的下一轮训练。

## 4. Experiments

### Setup

SITT 在三个任务上进行评估，难度递增：

1. **Color Maze（表格设置）**：智能体从起点导航到目标，环境中央是迷宫（含 lava 和 path 两种格子）。教师可区分 lava 和 path，学生只能看到"空/占据"，无法区分二者。这是一个清晰展示信息不对称影响的诊断性实验。

2. **Vision-Based Quadrotor Obstacle Avoidance**：四旋翼在包含四个柱状障碍物的 30m×30m×3m 空间中飞行。教师观测包含障碍物相对距离，学生仅有有限视场 RGB 摄像头图像。256 次随机初始化评估，每次 1000 步。

3. **Vision-Based Manipulation**：Franka 机械臂开抽屉任务。教师观测包含抓手到把手的相对位置，学生仅有图像输入。摄像头视角设置为贴近机械臂，容易产生自遮挡。

**基线方法**：行为克隆/Behavior Cloning/BC、DAgger、HLRL（混合 BC 和 RL 目标）、DWBC（双编码器统一策略）、COSIL（将教师-学生 L1 损失加入奖励），以及 SITT 无对齐版本（w/o Alignment）。所有方法使用相同的网络架构和训练资源。

### Main Results

**Color Maze**：BC 和 DAgger 训练的学生无法成功模仿教师——教师学会了穿过迷宫的最优路径，但学生无法区分 lava 和 path，导致频繁踩入 lava。SITT 训练的教师学会了**绕过迷宫**到达目标（虽然对教师而言是次优路径），学生成功模仿了这一行为，在所有测试中均到达目标。这个实验直观地验证了 SITT 的核心机制：教师主动选择学生可以模仿的替代路径。

**Quadrotor Obstacle Avoidance**：

| 方法 | 学生成功率 |
| --- | --- |
| BC | 0.05 ± 0.04 |
| DAgger | 0.08 ± 0.03 |
| HLRL | 0.31 ± 0.11 |
| DWBC | 0.35 ± 0.07 |
| COSIL | 0.30 ± 0.07 |
| w/o Align (Ours) | 0.38 ± 0.11 |
| **w Align (Ours)** | **0.46 ± 0.04** |

SITT 的成功率显著高于所有基线。轨迹可视化显示，SITT 训练的教师学会在飞行过程中调整摄像头朝向以与速度方向对齐，确保障碍物始终在视野内——这种感知感知/Perception-Aware 行为是自然涌现的，无需显式的感知奖励设计。感知指标显示 SITT 的速度角/Velocity Angle 为 32.2°（基线 46.7°–78.6°），视野内障碍物数量为 3.51（基线 1.92–2.61）。

**Vision-Based Manipulation**：

| 方法 | 学生成功率 |
| --- | --- |
| BC | 0.16 ± 0.15 |
| DAgger | 0.34 ± 0.31 |
| HLRL | 0.61 ± 0.22 |
| DWBC | 0.63 ± 0.18 |
| COSIL | 0.56 ± 0.21 |
| w/o Align (Ours) | 0.61 ± 0.18 |
| **w Align (Ours)** | **0.88 ± 0.07** |

SITT 的学生成功率达到 0.88，比 w/o Alignment 提高 27%，比最强基线 DWBC 提高 25%。可视化表明 SITT 训练的教师学会了减少自遮挡的抓取姿态（如从上方抓取或降低前两个关节），使红色把手在抓取前保持可见。有趣的是，使用对齐训练的教师回报也持续高于无对齐版本，可能的解释是教师学习了更鲁棒的抓取动作，避免了多次尝试抓握把手的低效行为。

### Ablations

在操作任务上的消融（Table 2）揭示了各组件的贡献：

| 配置 | 成功率 |
| --- | --- |
| 仅 Penalty / 无 KL-Grad | 0.47 ± 0.37 |
| 无 Penalty / 仅 KL-Grad | 0.74 ± 0.08 |
| 无 shared decoder | 0.62 ± 0.27 |
| 完整方法（$\lambda_1 = 0.05$） | 0.88 ± 0.07 |
| $\lambda_1 = 0.025$ | 0.95 ± 0.03 |

关键发现：（1）KL-Div Gradient 是最关键的组件——仅用 Penalty 而去掉 KL-Grad 导致成功率骤降至 0.47 且方差极大；（2）共享解码器的移除导致显著性能下降；（3）惩罚项的缺失导致 14% 的性能下降；（4）方法对 $\lambda_1$ 选择具有鲁棒性，$\lambda_1 = 0.025$ 时甚至达到 0.95。

### Critical Assessment

SITT 的实验设计在几个方面是合理的：三个任务的难度递增，信息不对称的来源各不相同（离散类型混淆、视野限制、自遮挡），消融实验覆盖了关键设计选择。但也存在一些不足：

- **绝对成功率仍有改进空间**：四旋翼任务的最高成功率仅为 0.46，意味着仍有过半的情况下学生无法避障。论文没有深入分析失败案例的原因——是代理学生的近似误差、对齐数据不足、还是任务本身的部分可观测性极限？
- **基线设置**：COSIL 的适配（使用 L1 损失代替原始论文中的方法，手动设定目标距离 0.5）可能未完全发挥其能力。HLRL 也被修改为使用共享解码器。
- **方差问题**：部分基线的方差极大（如操作任务中 DAgger 的标准差 0.31, 仅 Penalty 的标准差 0.37），提示在某些随机种子下这些方法也能接近 SITT 的表现，稳定性差异可能比均值差异更值得关注。
- **缺少对代理学生近似质量的定量分析**：代理学生是方法的核心假设之一，但论文没有报告代理学生预测学生动作的准确度。

## 5. Related Work & Future Work

### Related Work

特权模仿学习的研究可以分为两条主线：

**从学生端改进**：*A Framework for Behavioural Cloning*（BC）和 *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*（DAgger）是经典方法，后续工作如 *TGRL*、*COSIL*（*Leveraging Fully Observable Policies for Learning under Partial Observability*）、HLRL（*Real-World Humanoid Locomotion with RL*）通过在学生训练中混合 RL 目标来应对分布偏移和部分可观测性。*Impossibly Good Experts and How to Follow Them* 则训练 follower 和 explorer 两个策略协作。这些方法的共同点是教师策略保持不变，仅通过调整学生的训练方式来弥补信息差距。

**从教师端改进**：*Robust Asymmetric Learning in POMDPs*（RAIL）提出将学生的 rollout 暴露给教师训练，使教师行为有据可依。*Deep Whole-Body Control*（DWBC）学习统一的学生-教师策略和分别的编码器。SITT 与这些方法的核心区别在于：它通过性能上界公式推导出两个具体的优化项（奖励惩罚和 KL 梯度），而非启发式地引入对齐机制。

### Future Work

作者提到 SITT 不局限于特定的 IL 算法，可以与多种 IL 流水线集成，包括多智能体模仿学习。从论文的局限性来看，以下方向值得进一步探索：

- **Sim-to-Real 验证**：当前所有实验在仿真中完成，实际部署中的 domain gap 可能与信息不对称产生交互效应。
- **代理学生的改进**：当前代理学生仅通过 L1 损失与真实学生对齐，更精细的蒸馏方式（如对抗训练或注意力对齐）可能提升近似质量。
- **自适应对齐强度**：当前 $\lambda_1$, $\lambda_2$ 为固定系数，训练前期教师可能需要更多自由探索，后期再逐步加强对齐约束。
- **理论分析**：引入代理学生后，原有的性能上界是否仍然成立，近似误差如何传播到最终性能，这些问题缺乏理论讨论。
