---
title: InfoGAIL
headline: "InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations"
---

> [!abstract] Contributions
>
> 本文解决的核心问题是：专家演示数据往往来自多个具有不同策略的专家（或同一专家的不同行为模式），而标准的模仿学习/Imitation Learning/IL 方法无法自动发现和分离这些潜在的行为模式。InfoGAIL 在生成对抗模仿学习/Generative Adversarial Imitation Learning/GAIL 的基础上，借鉴 *InfoGAN* 的思想，引入隐变量/Latent Variable $c$ 并通过最大化隐变量与生成轨迹之间的互信息/Mutual Information 来约束策略，使得不同的隐变量值能够对应不同的行为模式。该方法无需任何行为标签即可以无监督方式发现专家演示中语义上有意义的变异因子。在 TORCS 自动驾驶模拟器上，InfoGAIL 仅以原始视觉像素作为输入，就能区分不同的驾驶行为（如内侧/外侧转弯、左侧/右侧超车），并通过奖励增强/Reward Augmentation 使学习到的策略甚至优于人类专家。
>
> 该方法的主要限制在于：隐变量 $c$ 的维度和类型（离散/连续）需要预先指定，且实验仅在相对简单的模拟环境中验证；互信息最大化所依赖的变分下界是否能在更复杂的多模态行为场景中有效工作，尚缺乏理论保证和充分的实验验证。

## 1. Introduction

强化学习/Reinforcement Learning/RL 需要预先定义奖励函数，而在自动驾驶等复杂场景中设计合适的奖励函数既困难又容易遗漏重要的权衡因素。模仿学习通过直接从专家演示中学习策略来绕过这一问题。其中，GAIL 是一种高效的 model-free 模仿学习方法，它将策略看作一个生成模型，通过对抗训练使学习策略生成的状态-动作分布与专家分布相匹配。

然而，现实中的专家演示通常来自多个不同的专家，或者同一专家在不同情境下展现出不同的行为模式。例如在驾驶场景中，不同驾驶员的技能水平和偏好各不相同，即使面对相同路况也可能做出不同的决策。这些潜在的变异因子/Latent Factors of Variation 并未被环境状态显式地捕捉。标准的 GAIL 将所有演示视为来自同一策略的样本，因此无法分离这些行为模式，只能学到一个"平均"策略。

本文的核心思路类似于 *InfoGAN* 在图像生成中发现风格、形状、颜色等语义因子的做法：在策略中引入隐变量 $c$，并通过互信息正则化迫使模型有意义地使用 $c$。这样，不同的 $c$ 值就能对应不同的行为模式，策略可以通过高层隐变量来控制低层动作。本文还将该框架扩展到以原始像素为输入的高维视觉场景，并在 TORCS 自动驾驶模拟器上进行了验证。

## 2. Problem Setup

考虑一个无限时域折扣马尔可夫决策过程/Markov Decision Process/MDP，定义为元组 $\langle \mathcal{S}, \mathcal{A}, P, r, \rho_0, \gamma \rangle$，其中 $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$ 为转移概率，$r: \mathcal{S} \to \mathbb{R}$ 为奖励函数，$\rho_0: \mathcal{S} \to \mathbb{R}$ 为初始状态分布，$\gamma \in (0, 1)$ 为折扣因子。随机策略 $\pi: \mathcal{S} \times \mathcal{A} \to [0, 1]$，专家策略为 $\pi_E$。专家演示 $\tau_E$ 是由 $\pi_E$ 生成的状态-动作对序列的集合。策略 $\pi$ 下的期望定义为：

$$
\mathbb{E}_\pi[f(s, a)] \triangleq \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t f(s_t, a_t)\right]
$$

其中 $s_0 \sim \rho_0$，$a_t \sim \pi(a_t | s_t)$，$s_{t+1} \sim P(s_{t+1} | a_t, s_t)$。

模仿学习的目标是在不访问奖励信号 $r$ 的条件下，从专家演示中学习执行任务。本文的关键假设是：专家策略实际上是多个子策略的混合，即 $\pi_E = \{\pi_E^0, \pi_E^1, \ldots\}$。专家轨迹的生成过程为：

$$
s_0 \sim \rho_0, \quad c \sim p(c), \quad \pi \sim p(\pi | c), \quad a_t \sim \pi(a_t | s_t), \quad s_{t+1} \sim P(s_{t+1} | a_t, s_t)
$$

其中 $c$ 是一个离散隐变量，通过未知的条件分布 $p(\pi | c)$ 从专家策略混合体中选取一个具体策略。$p(c)$ 是 $c$ 的先验分布（假设训练前已知）。学习目标是恢复一个以 $c$ 为条件的策略 $\pi(a | s, c)$：当 $c$ 从先验 $p(c)$ 中采样时，条件策略 $\pi(a | s, c)$ 生成的轨迹应与专家轨迹在判别器意义下不可区分。

与标准 GAIL 仅匹配状态-动作的占用度量/Occupancy Measure 不同，InfoGAIL 在策略中显式引入了对隐变量 $c$ 的依赖，从而将"发现潜在行为模式"纳入了学习目标。

## 3. Methods

### 3.1 From GAIL to InfoGAIL

GAIL 的目标函数将模仿学习转化为一个极小极大博弈：

$$
\min_\pi \max_{D \in (0,1)^{\mathcal{S} \times \mathcal{A}}} \mathbb{E}_\pi[\log D(s, a)] + \mathbb{E}_{\pi_E}[\log(1 - D(s, a))] - \lambda H(\pi)
\tag{1}
$$

其中 $D$ 是判别器，试图区分学习策略 $\pi$ 和专家策略 $\pi_E$ 产生的状态-动作对；$H(\pi) \triangleq \mathbb{E}_\pi[-\log \pi(a|s)]$ 是策略的 $\gamma$-折扣因果熵，用于鼓励探索。该目标的内层最大化对应于训练一个最优判别器来度量两个状态-动作分布之间的 Jensen-Shannon 散度/JS Divergence，外层最小化则让策略分布逼近专家分布——当两者的占用度量完全匹配时，最优判别器输出恒为 $1/2$，目标达到最优。

GAIL 是 model-free 的：它将环境/模拟器视为黑盒，不需要构建环境模型，但需要与环境交互来生成 rollout。与 GAN 不同的是，由于环境不可微，策略的优化无法直接反向传播，而需要依赖基于蒙特卡洛采样的策略梯度方法。具体而言，优化通过交替执行两步完成：对判别器参数进行梯度上升以增大 $\eqref{1}$，对策略参数通过 TRPO/Trust Region Policy Optimization 进行更新以减小 $\eqref{1}$。

直接在 GAIL 的策略中加入隐变量 $c$（即使用 $\pi(a|s,c)$）并不能保证模型会有意义地利用 $c$——策略完全可以忽略 $c$ 而仍然最小化 GAIL 目标。为解决这一问题，InfoGAIL 借鉴 *InfoGAN* 的信息论正则化思想：要求隐变量 $c$ 与策略生成的轨迹 $\tau$ 之间保持高互信息 $I(c; \tau)$。

然而，直接最大化互信息 $I(c; \tau)$ 需要访问真实后验 $P(c|\tau)$，这在实际中是不可行的。下面推导 InfoGAIL 所使用的变分下界/Variational Lower Bound。

从互信息的定义出发：

$$
I(c; \tau) = H(c) - H(c|\tau) = H(c) + \mathbb{E}_{\tau}\left[\mathbb{E}_{c|\tau}[\log P(c|\tau)]\right]
$$

由于真实后验 $P(c|\tau)$ 未知，引入一个可学习的近似后验 $Q(c|\tau)$。将 $\log P(c|\tau)$ 改写为：

$$
\mathbb{E}_{c|\tau}[\log P(c|\tau)] = \mathbb{E}_{c|\tau}[\log Q(c|\tau)] + D_{\mathrm{KL}}(P(c|\tau) \| Q(c|\tau))
$$

其中 $D_{\mathrm{KL}}$ 是 KL 散度。由于 KL 散度始终非负 $D_{\mathrm{KL}} \geq 0$，将其代入互信息表达式后可得下界：

$$
I(c; \tau) = H(c) + \mathbb{E}_\tau\left[\mathbb{E}_{c|\tau}[\log Q(c|\tau)]\right] + \mathbb{E}_\tau\left[D_{\mathrm{KL}}(P(c|\tau) \| Q(c|\tau))\right]
$$

$$
\geq H(c) + \mathbb{E}_\tau\left[\mathbb{E}_{c|\tau}[\log Q(c|\tau)]\right]
$$

将期望的采样顺序从 $(\tau, c|\tau)$ 改写为等价的 $(c, \tau|c)$（即先从先验采样 $c \sim p(c)$，再由条件策略 $\pi(\cdot|s, c)$ 生成轨迹），得到互信息的变分下界 $L_I$：

$$
L_I(\pi, Q) = \mathbb{E}_{c \sim p(c),\, a \sim \pi(\cdot|s, c)}[\log Q(c|\tau)] + H(c) \leq I(c; \tau)
\tag{2}
$$

当且仅当 $Q(c|\tau) = P(c|\tau)$ 时等号成立，即近似后验完美匹配真实后验。

> [!tip] 变分下界的直觉
> 最大化 $L_I$ 同时驱动两件事：(1) 策略 $\pi$ 被鼓励在不同 $c$ 值下产生可区分的行为模式，使得 $c$ 能从轨迹中被恢复；(2) 近似后验 $Q$ 被训练为从生成的轨迹中尽可能准确地推断所使用的 $c$。这一机制确保隐变量不会被策略忽略。

在实际优化中，由于直接使用完整轨迹 $\tau$ 的计算代价过高（尤其在高维视觉输入场景下），论文将后验近似简化为逐步的 $Q(c|s, a)$，即仅基于单个状态-动作对进行推断。这一简化降低了计算成本，但也意味着 $Q$ 只能利用局部的状态-动作信息而非整条轨迹来推断 $c$，可能导致信息损失。

InfoGAIL 的完整目标函数为：

$$
\min_{\pi, Q} \max_D \mathbb{E}_\pi[\log D(s, a)] + \mathbb{E}_{\pi_E}[\log(1 - D(s, a))] - \lambda_1 L_I(\pi, Q) - \lambda_2 H(\pi)
\tag{3}
$$

其中 $\lambda_1 > 0$ 控制互信息正则化的强度，$\lambda_2 > 0$ 控制因果熵项。与 GAIL 的目标 $\eqref{1}$ 对比，InfoGAIL 增加了两个关键组件：引导策略有意义地利用隐变量的互信息下界 $L_I$（来自 $\eqref{2}$ 的推导），以及用于近似后验推断的网络 $Q$。优化变量也相应扩展——策略和后验网络联合最小化，判别器最大化。

将 $L_I$ 展开，目标 $\eqref{3}$ 中策略 $\pi$ 的优化方向可以更直观地理解：$\pi$ 需要同时 (a) 在判别器意义下模仿专家、(b) 使不同 $c$ 值下的行为足够可区分（由 $Q$ 度量）、(c) 保持足够的随机性（熵项）。这三个信号的平衡由 $\lambda_1, \lambda_2$ 控制。

### 3.2 Reward Augmentation

模仿学习的性能受限于专家演示的质量——如果专家本身表现次优，学习到的策略也将是次优的。然而在许多场景中，虽然难以完整定义奖励函数，但指定某些约束或偏好是相对容易的。

奖励增强/Reward Augmentation 通过引入一个代理的基于状态的奖励 $\eta(\pi_\theta) = \mathbb{E}_{s \sim \pi_\theta}[r(s)]$ 来融合先验知识，使目标函数变为：

$$
\min_{\theta, \psi} \max_\omega \mathbb{E}_{\pi_\theta}[\log D_\omega(s, a)] + \mathbb{E}_{\pi_E}[\log(1 - D_\omega(s, a))] - \lambda_0 \eta(\pi_\theta) - \lambda_1 L_I(\pi_\theta, Q_\psi) - \lambda_2 H(\pi_\theta)
\tag{4}
$$

其中 $\lambda_0 > 0$ 是权衡系数。这种方法可以看作模仿学习与强化学习的混合：策略优化的信号一部分来自判别器（模仿专家），一部分来自代理奖励（先验约束）。例如在自动驾驶实验中，通过对碰撞和驶出道路施加惩罚，可以显著提升学习策略的平均行驶距离。

### 3.3 Improved Optimization for High-Dimensional Inputs

原始 GAIL 在观测维度较低（最多 376 维连续变量）的任务上取得了成功，但本文需要处理 $110 \times 200 \times 3$ 的原始像素输入。为此，论文引入了以下改进：

**Wasserstein GAN 目标**：传统 GAN 目标在高维场景中容易出现梯度消失和模式坍塌问题。论文采用 Wasserstein GAN/WGAN 的目标函数替代原始的 Jensen-Shannon 散度：

$$
\min_{\theta, \psi} \max_\omega \mathbb{E}_{\pi_\theta}[D_\omega(s, a)] - \mathbb{E}_{\pi_E}[D_\omega(s, a)] - \lambda_0 \eta(\pi_\theta) - \lambda_1 L_I(\pi_\theta, Q_\psi) - \lambda_2 H(\pi_\theta)
\tag{5}
$$

这一修改在需要建模具有多种模式的复杂轨迹分布时尤为重要。

**其他技术**：论文还使用了方差缩减技术（包括 baseline 和回放缓冲区/Replay Buffer）。判别器 $D_\omega$ 使用 RMSprop 更新（遵循 WGAN 的建议，含权重裁剪至 $[-0.01, 0.01]$），后验网络 $Q_\psi$ 使用 Adam 更新，策略 $\pi_\theta$ 使用 TRPO 更新。为加速训练，策略从行为克隆/Behavior Cloning/BC 预训练初始化。

> [!info] 网络分离的设计考量
> 与 *InfoGAN* 中判别器 $D$ 与后验网络 $Q$ 共享参数（仅最后一层不同）的做法不同，InfoGAIL 将二者设计为独立的网络。原因在于 WGAN 的训练需要对判别器进行权重裁剪和无动量优化，这些操作会干扰 $Q$ 的训练。

### 3.4 Algorithm

InfoGAIL 涉及三个可学习网络的交替优化：策略 $\pi_\theta$、判别器 $D_\omega$、后验近似 $Q_\psi$。以下给出完整的训练流程及各步的梯度更新规则。

**每轮迭代：**

1. **采样**：从先验 $p(c)$ 中采样隐变量 $c_i$，以固定的 $c_i$ 运行策略 $\pi_{\theta_i}(c_i)$ 生成轨迹 $\tau_i$（每条轨迹内 $c$ 保持不变），将 $\tau_i$ 加入回放缓冲区 $B$。从 $B$ 和专家演示 $\tau_E$ 中各采样一批状态-动作对 $\chi_i$ 和 $\chi_E$。

2. **更新判别器** $D_\omega$（梯度上升）：

$$
\Delta_{\omega_i} = \hat{\mathbb{E}}_{\chi_i}[\nabla_{\omega_i} \log D_{\omega_i}(s, a)] + \hat{\mathbb{E}}_{\chi_E}[\nabla_{\omega_i} \log(1 - D_{\omega_i}(s, a))]
$$

判别器的目标是将专家样本与策略生成样本区分开。在 WGAN 版本中，该更新改为：

$$
\Delta_{\omega_i} = \hat{\mathbb{E}}_{\chi_i}[\nabla_{\omega_i} D_{\omega_i}(s, a)] - \hat{\mathbb{E}}_{\chi_E}[\nabla_{\omega_i} D_{\omega_i}(s, a)]
$$

并在更新后将权重裁剪至 $[-0.01, 0.01]$。

3. **更新后验网络** $Q_\psi$（梯度下降）：

$$
\Delta_{\psi_i} = -\lambda_1 \hat{\mathbb{E}}_{\chi_i}[\nabla_{\psi_i} \log Q_{\psi_i}(c|s, a)]
$$

该更新直接对应变分下界 $L_I$ 中关于 $Q$ 的部分：最大化 $Q$ 从策略生成的状态-动作对中正确推断隐变量 $c$ 的对数似然。

4. **更新策略** $\pi_\theta$（通过 TRPO 最小化以下目标）：

$$
\hat{\mathbb{E}}_{\chi_i}\left[\log D_{\omega_{i+1}}(s, a)\right] - \lambda_1 L_I(\pi_{\theta_i}, Q_{\psi_{i+1}}) - \lambda_2 H(\pi_{\theta_i})
$$

在 WGAN 版本中，对应地将 $\log D$ 替换为 $D$：

$$
\hat{\mathbb{E}}_{\chi_i}\left[D_{\omega_{i+1}}(s, a)\right] - \lambda_1 L_I(\pi_{\theta_i}, Q_{\psi_{i+1}}) - \lambda_2 H(\pi_{\theta_i})
$$

三项的作用分别是：第一项鼓励策略生成能"骗过"判别器的状态-动作对（在论文的约定中，判别器对专家样本输出低分、对生成样本输出高分，因此策略最小化该项以使自身看起来像专家）；第二项（$-L_I$，即最大化互信息下界）鼓励策略有意义地利用隐变量；第三项（$-H$，即最大化熵）鼓励策略保持随机性以促进探索。若使用奖励增强，还需额外加入 $-\lambda_0 \eta(\pi_{\theta_i})$ 项。

每条轨迹在生成过程中保持隐变量 $c$ 固定，这是确保隐变量能够表达轨迹级行为模式（而非逐步变化的噪声）的关键设计。

## 4. Experiments

实验分为两个环境：一个用于概念验证的二维合成环境，以及基于 TORCS 赛车模拟器的自动驾驶环境。

### 4.1 Synthetic 2D Environment

在一个二维平面中，智能体以恒定速度移动，观测为过去 4 步的位置（10 维），动作为方向（2 维）。专家演示包含三种不同的圆形轨迹模式（无标签）。隐变量 $c$ 设为 3 维 one-hot 向量，先验为均匀分布。

结果表明：BC 因累积误差/Compounding Error 而偏离专家轨迹；GAIL 能学到圆形轨迹但无法区分三种模式（因为它假设所有演示来自同一策略）；**InfoGAIL 成功区分了三种行为模式**，不同的隐变量值对应不同的圆形轨迹。

<!-- TODO: insert Figure 1 from paper，展示四种方法在二维环境中的轨迹对比 -->

### 4.2 Visual Inputs via Transfer Learning

在 TORCS 驾驶环境中，策略仅以原始视觉图像（$110 \times 200 \times 3$ 像素）作为外部感知输入，输出三维连续动作（转向、加速、制动）。为缓解高维视觉输入带来的样本效率问题，论文采用在 ImageNet 上预训练的 ResNet-50 提取视觉特征，再经过两层卷积处理后与辅助信息（速度、历史动作、车辆损伤，共 10 维）及隐变量 $c$ 拼接输入全连接层。

### 4.3 Unsupervised Discovery of Driving Behaviors

论文在两种驾驶场景中测试 InfoGAIL：

- **turn**（转弯）：专家从内侧或外侧车道转弯，存在两种模式
- **pass**（超车）：专家从左侧或右侧超越前车，存在两种模式

两种场景各有 80 条专家轨迹（每条 100 帧），隐变量为 2 维 one-hot 向量。训练中加入恒定的"存活奖励"作为奖励增强。

**定性结果**：训练初期模型不区分两种模式，随着训练推进，两种隐变量对应的轨迹逐渐分化。训练结束时，在 turn 场景中 $[0,1]$ 对应内侧车道、$[1,0]$ 对应外侧车道；在 pass 场景中两种隐变量分别对应右侧和左侧超车。

**定量结果**：通过后验网络 $Q_\psi(c|s,a)$ 对专家状态-动作对进行分类，InfoGAIL 在 pass 场景上的分类准确率达到 **81.9%**，显著优于 K-means（55.4%）和 PCA（61.7%）等无监督基线。作为参考，有监督的 SVM 和 CNN 分别达到 85.8% 和 90.8%。

作为对照，GAIL 在 pass 场景中无法区分两种模式，而是学到一种"平均"轨迹——先从右侧超车，超过后再急转到左侧。这反映了判别器 $D_\omega(s,a)$ 仅基于状态-动作对而非完整轨迹进行判断的局限性。

### 4.4 Ablation Experiments

消融实验在不使用隐变量的设置下进行，以隔离各优化技术对驾驶策略质量（平均行驶距离）的贡献。20 条专家轨迹，每条 500 帧，奖励增强鼓励快速行驶。

| 方法 | 平均行驶距离 |
|---|---|
| Behavior Cloning | 701.83 |
| GAIL | 914.45 |
| InfoGAIL \ RB（去掉回放缓冲区） | 1031.13 |
| InfoGAIL \ RA（去掉奖励增强） | 1123.89 |
| InfoGAIL \ WGAN（用 GAN 替代 WGAN） | 1177.72 |
| **InfoGAIL (Ours)** | **1226.68** |
| Human | 1203.51 |

关键发现：(1) 完整的 InfoGAIL（含奖励增强）能够**超越人类专家**；(2) 去除奖励增强或 WGAN 后性能略有下降但仍显著优于 GAIL；(3) 去除回放缓冲区导致性能大幅下降，表明梯度估计的方差控制至关重要。

### 4.5 Discussion of Experimental Limitations

- **环境复杂度有限**：所有实验均在 TORCS 模拟器中进行，场景相对简单（单一赛道、固定交通），与真实驾驶环境差距较大。
- **行为模式数量少**：实验中每个场景仅包含两种行为模式（2 维 one-hot），隐变量维度和类型均为手动预设，未验证在更多模式或连续隐变量下的效果。
- **缺少与其他多模态模仿学习方法的对比**：无监督分类的基线（K-means、PCA）并非专门针对行为模式发现设计的方法，对比说服力有限。
- **后验简化的影响未充分讨论**：将轨迹级后验 $Q(c|\tau)$ 简化为状态-动作级 $Q(c|s,a)$ 可能导致信息损失，但论文未对此进行分析。

## 5. Related Work & Future Work

### Related Work

在视觉驾驶系统中，主要存在两种范式：中介感知/Mediated Perception 先提取场景信息再做决策，行为反射/Behavior Reflex 直接从视觉输入映射到动作。InfoGAIL 属于后者，且通过 GAIL 框架支持比行为克隆更复杂的驾驶操作（如换道、超车）。

与直接使用 GAIL 建模驾驶行为的工作（如 *Imitating Driver Behavior with Generative Adversarial Networks*）不同，InfoGAIL 仅使用原始视觉信息作为输入（而非 LIDAR 等距离测量），更接近人类驾驶员的信息条件。与使用预训练网络学习奖励函数的工作（如 *Unsupervised Perceptual Rewards for Imitation Learning*）不同，InfoGAIL 的迁移学习作用于策略网络而非判别器/奖励网络。

### Future Work

论文明确提出的方向：

> 希望该工作能推动端到端学习方法在更真实的自动驾驶场景中的应用。

从论文局限性可以自然推断的方向包括：(1) 将 InfoGAIL 扩展到连续隐变量或更高维的离散隐变量，以处理更丰富的行为模式；(2) 探索自动确定隐变量维度的方法；(3) 将奖励增强与预训练奖励函数结合，进一步提升策略质量；(4) 在真实世界数据或更复杂的模拟环境中验证方法的可扩展性。
