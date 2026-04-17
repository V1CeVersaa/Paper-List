---
title: "DGSM"
headline: "Data Generation as Sequential Decision Making"
visibility: "public"
status: "drafting"
description: "Paper note on Data Generation as Sequential Decision Making."
---

> [!abstract] Contributions
>
> 本文提出了一个统一视角：有向生成模型/Directed Generative Model 中的数据生成过程可以被解释为序贯决策/Sequential Decision Making 过程，从而可以利用强化学习中引导策略搜索/Guided Policy Search/GPS 的工具来训练这些模型。在此视角下，作者将深度自回归网络、基于 LSTM 的生成模型（*DRAW*）、以及基于扩散过程的生成模型的训练统一理解为广义引导策略搜索的特例，并基于该视角改进了 *DRAW* 模型在 binarized MNIST 上的表现。进一步地，作者将这一视角应用于数据填补/Data Imputation 任务，将填补建模为一个有限时域马尔可夫决策过程/Markov Decision Process/MDP，通过策略的迭代反馈与精炼逐步构建对缺失值的预测。作者提出了两类填补模型——直接策略模型/GPSI 和基于 LSTM 的策略模型——在 MNIST、TFD、SVHN 三个数据集上均显著优于基线方法。
>
> 本文的主要局限在于：实验仅在图像填补任务上进行验证，且数据集规模和复杂度有限；填补模型的训练依赖引导策略搜索，计算成本较高；此外，文中的理论分析主要停留在将已有方法重新解释的层面，并未提出新的理论保证。

## 1. Introduction

有向生成模型在近年得到了快速发展，其核心采样过程——依次设定有向图中各节点的值——天然地可以理解为一系列序贯决策。本文的出发点是：既然数据生成就是序贯决策，那么强化学习中用于训练策略的方法就可以直接用于训练生成模型。

具体而言，在有向生成模型中，每一步采样 $z_t$ 都是在给定前序潜变量 $z_0, \dots, z_{t-1}$ 的条件下做出的一个随机决策。整个采样过程可以看作一条"轨迹"，其终态为生成的观测 $x$。由此，生成模型定义了一个策略，而训练生成模型就等价于对该策略进行策略搜索。

已有工作中，引导策略搜索/Guided Policy Search 通过引入一个拥有额外信息的引导策略/Guide Policy $q$ 来辅助训练主策略/Primary Policy $p$，使主策略能够从引导策略生成的高质量轨迹中学习，从而避免策略搜索中的局部最优问题。本文指出，深度自回归网络（*Deep AutoRegressive Networks*）中的变分训练、*DRAW* 中的编码器-解码器训练、以及扩散模型（*Deep Unsupervised Learning Using Nonequilibrium Thermodynamics*）中的反向过程训练，都可以被统一解释为广义引导策略搜索的特例。这一统一视角的价值在于：它揭示了各种生成模型训练中变分推断等技术的共同本质——都是在用一个拥有更多信息的引导策略来指导主策略的学习，从而可以启发将 RL 中的策略优化技巧迁移到生成模型训练中。

在建立了这一联系之后，作者进一步探索其应用价值：将数据填补任务建模为 MDP，并利用引导策略搜索训练填补策略。填补是一个介于无条件生成和分类/回归之间的通用问题：当已知信息为空时退化为无条件生成，当缺失信息仅有单一变量时退化为回归/分类。这使得填补成为探索条件生成与无条件生成关系的理想场景。

## 2. Problem Setup

### 2.1 Directed Generative Models as Sequential Decision Processes

考虑有向生成模型定义的分布：

$$
p(x) = \sum_z p(x|z) p(z), \quad p(z) = p_0(z_0) \prod_{t=1}^{T} p_t(z_t | z_0, \dots, z_{t-1}) \tag{1}
$$

其中 $x$ 为生成的观测，$z_0, \dots, z_T$ 为模型中的潜变量。$p(z)$ 的分解形式允许 $z_t$ 为单变量、多变量甚至结构化分布。整个采样过程可以看作一个有限时域 MDP：

- **状态**：在时刻 $1 \le t \le T+1$，状态由 $\{z_0, \dots, z_{t-1}\}$ 决定
- **动作**：在时刻 $1 \le t \le T$，选择 $z_t \in \mathcal{Z}_t$；在时刻 $t = T+1$，选择观测 $x \in \mathcal{X}$
- **策略**：$p_t(z_t | z_0, \dots, z_{t-1})$ 定义了非平稳策略
- **轨迹**：$\tau \triangleq \{z_0, \dots, z_T, x\}$，轨迹终态的分布就是 $p(x)$

基于此 MDP 视角，任何基于 Eq. 1 的模型都可以被解释为一个非平稳策略 $p_t(z_t|s_t)$（其中 $s_t$ 由所有 $z_{t'}$（$t' < t$）决定），从而可以用策略搜索方法来训练。

### 2.2 Generalized Guided Policy Search

作者采用了一个比标准引导策略搜索更宽泛的定义。广义引导策略搜索的一般形式为：

$$
\min_{p, q} \; \mathbb{E}_{i_q \sim \mathcal{I}_q} \mathbb{E}_{i_p \sim \mathcal{I}_p(\cdot | i_q)} \left[ \mathbb{E}_{\tau \sim q(\tau | i_q, i_p)} \left[ \ell(\tau, i_q, i_p) + \lambda \operatorname{div}(q(\tau | i_q, i_p), p(\tau | i_p)) \right] \right] \tag{2}
$$

其中：

- $p$ 为主策略，$q$ 为引导策略
- $\mathcal{I}_q$ 为仅引导策略可用的信息分布，$\mathcal{I}_p$ 为两者共享的信息分布
- $\ell(\tau, i_q, i_p)$ 为轨迹代价
- $\operatorname{div}(\cdot, \cdot)$ 度量引导策略与主策略轨迹分布的差异
- 当 $\lambda \to \infty$ 时，约束 $p(\tau | i_p) = q(\tau | i_q, i_p)$ 被强制满足；也可以添加 $p$ 的熵项来控制主策略的随机性

这一框架的核心在于两点：引导策略 $q$ 可以利用主策略 $p$ 不可获得的额外信息 $i_q$（例如训练时的目标 $x^*$），而主策略只需学习去缩小与引导策略之间的差异 $\operatorname{div}(q, p)$。

### 2.3 Unifying Existing Generative Models under GPS

基于上述 MDP 视角和广义 GPS 框架，作者将多种已有生成模型的训练统一解释为广义引导策略搜索的特例。

**深度自回归网络**（*Deep AutoRegressive Networks*）：模型定义 $p(z) = p_0(z_0) \prod_{t=1}^T p_t(z_t | z_0, \dots, z_{t-1})$，训练时的变分分布 $q$ 提供以目标 $x^*$ 为条件的引导轨迹 $q(\tau|x^*) \triangleq q(x|z_0, \dots, z_T, x^*) q_0(z_0|x^*) \prod_{t=1}^T q_t(z_t|z_0, \dots, z_{t-1}, x^*)$。主策略的轨迹分布为 $p(\tau) \triangleq p(x|z_0, \dots, z_T) p_0(z_0) \prod_{t=1}^T p_t(z_t|z_0, \dots, z_{t-1})$（不依赖 $x^*$）。将变分目标展开，可以看到优化等价于最小化：

$$
\min_{p, q} \; \mathbb{E}_{x^* \sim \mathcal{D}_\mathcal{X}} \mathbb{E}_{\tau \sim q(\tau|x^*)} \left[ \log \frac{q_0(z_0|x^*)}{p_0(z_0)} + \sum_{t=1}^T \log \frac{q_t(z_t|z_0, \dots, z_{t-1}, x^*)}{p_t(z_t|z_0, \dots, z_{t-1})} - \log p(x^*|z_0, \dots, z_T) \right] \tag{3}
$$

即路径级 KL 散度 $\operatorname{KL}(q(\tau|x^*) \| p(\tau))$ 加上重建项。此处引导策略的额外信息为 $i_q = x^*$，这正是广义 GPS（Eq. 2）的一个实例。

**基于 LSTM 的生成模型**（*DRAW*）：主策略通过 LSTM 维护一个状态轨迹 $s_\theta(\tau_{<x})$，每步的动作 $z_t$ 依赖于该状态。引导策略通过额外的编码器网络读取目标 $x^*$ 来提供引导信号。训练时对引导策略 $q$ 进行展开采样，主策略 $p$ 学习缩小与引导策略的轨迹分布差异。作者基于此分析对 *DRAW* 模型提出了改进（见 Section 3.4）。

**时间可逆随机过程**：对于终态分布 $p(x_T) = \sum_{x_0, \dots, x_{T-1}} p_T(x_T|x_{T-1}) p_0(x_0) \prod_{t=1}^{T-1} p_t(x_t|x_{t-1})$（如扩散模型），可定义一个反向过程 $q_t(x_{t-1}|x_t)$，将目标分布 $\mathcal{D}_\mathcal{X}$ 逐步变换为初始分布 $p_0(x_0)$。训练目标等价于最小化路径级 KL 散度 $\operatorname{KL}(q(\tau) \| p(\tau))$。

> [!todo]- Path-wise KL Bound for Reversible Stochastic Processes
>
> 定义正向过程 $p(\tau) = p_0(x_0) \prod_{t=1}^T p_t(x_t|x_{t-1})$，反向引导过程 $q(\tau) = \mathcal{D}_\mathcal{X}(x_T) \prod_{t=1}^T q_t(x_{t-1}|x_t)$，以及简记 $p(\tau_{>0}|x_0) \triangleq p(x_1, \dots, x_T|x_0)$ 和 $q(\tau_{<T}|x_T) \triangleq q(x_0, \dots, x_{T-1}|x_T)$。
>
> 对 $p(x_T)$ 插入重要性权重：
>
> $$
> p(x_T) = \sum_{x_0, \dots, x_{T-1}} p_0(x_0) p(\tau_{>0}|x_0) \cdot \frac{q(\tau_{<T}|x_T)}{q(\tau_{<T}|x_T)}
> $$
>
> 将求和改写为 $q(\tau_{<T}|x_T)$ 下的期望，并利用 Jensen 不等式：
>
> $$
> \log p(x_T) \ge \mathbb{E}_{q(\tau_{<T}|x_T)} \left[ \log p_0(x_0) + \sum_{t=1}^T \log \frac{p_t(x_t|x_{t-1})}{q_t(x_{t-1}|x_t)} \right]
> $$
>
> 右侧可以整理为：
>
> $$
> \log p(x_T) \ge \mathbb{E}_{q(\tau_{<T}|x_T)} \left[ \log p_0(x_0) - \log \frac{q(\tau_{<T}|x_T)}{p(\tau_{>0}|x_0)} \right]
> $$
>
> 对 $x_T \sim \mathcal{D}_\mathcal{X}$ 取期望，并利用 $q(\tau)$ 和 $p(\tau)$ 的定义展开：
>
> $$
> \mathbb{E}_{x_T \sim \mathcal{D}_\mathcal{X}}[\log p(x_T)] \ge \mathbb{E}_{q(\tau)} \left[ -\log \frac{\mathcal{D}_\mathcal{X}(x_T) q(\tau_{<T}|x_T)}{p_0(x_0) p(\tau_{>0}|x_0)} \right] = -\operatorname{KL}(q(\tau) \| p(\tau)) - H_{\mathcal{D}_\mathcal{X}}
> $$
>
> 其中 $H_{\mathcal{D}_\mathcal{X}} \triangleq \mathbb{E}_{x \sim \mathcal{D}_\mathcal{X}}[-\log \mathcal{D}_\mathcal{X}(x)]$ 为目标分布的熵。当 $\mathcal{D}_\mathcal{X}$ 固定时，最大化对数似然下界等价于最小化路径级 KL 散度 $\operatorname{KL}(q(\tau) \| p(\tau))$。

### 2.4 Imputation as a Finite-Horizon MDP

从有向生成模型的序贯决策视角自然地引出了数据填补问题的 MDP 建模。数据填补的目标是估计 $p(x^u | x^k)$，其中完整观测 $x \triangleq [x^u; x^k]$ 由缺失值/Missing Values $x^u$ 和已知值/Known Values $x^k$ 组成。掩码/Mask $m \in \mathcal{M}$ 定义了 $x$ 到 $x^u$、$x^k$ 的划分。填补目标为：

$$
\min_p \; \mathbb{E}_{x \sim \mathcal{D}_\mathcal{X}} \mathbb{E}_{m \sim \mathcal{D}_\mathcal{M}} \left[ -\log p(x^u | x^k) \right] \tag{4}
$$

这是一个通用框架：当 $x^u$ 扩展为整个 $x$ 时退化为无条件生成；当 $x^u$ 缩小为单个元素时退化为分类/回归。

将填补建模为有限时域 MDP：

- **初始状态**：$z_0 \sim p(z_0 | x^k)$，由已知值 $x^k$ 决定
- **动作**：每步选择 $z_t$，策略为 $p_t(z_t | z_0, \dots, z_{t-1}, x^k)$
- **轨迹代价**：$\ell(\tau, x^u, x^k) = -\log p(x^u | \tau, x^k)$，即在给定轨迹和已知值后对缺失值的负对数似然

策略的轨迹分布为：

$$
p(\tau | x^k) \triangleq p(z_0 | x^k) \prod_{t=1}^{T} p_t(z_t | z_0, \dots, z_{t-1}, x^k)
$$

直接优化 $\mathbb{E}_{\tau \sim p(\tau|x^k)}[-\log p(x^u|\tau, x^k)]$ 是填补目标 Eq. 4 的一个上界（由 Jensen 不等式）。为收紧这一上界，引入引导策略 $q$，其拥有额外信息 $i_q = x^u$（即缺失值的真实取值），形成引导策略搜索目标：

$$
\min_{p, q} \; \mathbb{E}_{x, m} \mathbb{E}_{\tau \sim q_\phi(\tau | x^u, x^k)} \left[ -\log q(x^u | c_T^u) + \operatorname{KL}(q(\tau | x^u, x^k) \| p(\tau | x^k)) \right] \tag{5}
$$

其中 $c_T$ 为策略经过 $T$ 步精炼后的最终填补结果（将在 Section 3 中详细定义）。

## 3. Methods

在 Section 2 中建立了填补的 MDP 框架之后，本节介绍两类具体的序贯填补策略。两者共享一个核心机制：通过迭代精炼/Iterative Refinement 逐步构建对缺失值 $x^u$ 的预测。

定义填补轨迹/Imputation Trajectory $c_\tau \triangleq \{c_0, \dots, c_T\}$，其中每个 $c_t \in \mathcal{X}$ 是一个部分填补结果。$c_{t-1}$ 编码了策略在选择 $z_t$ 之前对 $x^u$ 的当前猜测，$c_T$ 为最终填补结果。每一步精炼中，策略根据当前猜测 $c_{t-1}$ 和已知值 $x^k$ 选择动作 $z_t$，然后基于 $c_{t-1}$ 和 $z_t$ 更新猜测为 $c_t$。这种设计使策略能够通过反馈逐步构建复杂的结构化预测，无需后处理的 MRF/CRF，也无需像 NADE 类模型那样逐变量采样。

### 3.1 GPSI: Direct Imputation Policy

GPSI（Guided Policy Search Imputer）直接在填补 MDP 上定义策略。

**猜测更新方式**：作者考虑两种更新 $c_t$ 的方式：

1. **加法更新/Additive (-add)**：$c_t \leftarrow c_{t-1} + \omega_\theta(z_t)$，即在前一步猜测上累加修正量
2. **跳跃更新/Jump (-jump)**：$c_t \leftarrow \omega_\theta(z_t)$，即直接用新信息替换整个猜测

初始猜测 $c_0 \triangleq [c_0^u; c_0^k]$，其中 $c_0^k$ 为已知值，$c_0^u$ 为可训练的偏置。

**主策略 $p_\theta$**：一个简单的平稳策略。步选择器/Step Selector $p_\theta(z_t | c_{t-1}, x^k)$ 在每步根据当前猜测和已知值选择 $z_t$（限定为对角高斯分布），然后由填补构造器/Imputation Constructor $\omega_\theta(z_t)$ 将 $z_t$ 转化为对猜测的更新。步选择器和填补构造器共同决定了主策略的行为。

**引导策略 $q_\phi$**：与主策略共享填补构造器 $\omega_\theta$，但拥有额外信息——完整观测 $x = [x^u; x^k]$。引导策略的步选择器为 $q_\phi(z_t | c_{t-1}, x)$，同样限定为对角高斯分布。

**训练目标**：同时优化 $\theta$ 和 $\phi$，目标为 Eq. 5 的形式。通过对引导策略 $q$ 的 Monte Carlo 展开和随机反向传播进行训练。

<!-- TODO: insert Figure 6 from paper — architecture diagram and pseudocode for the direct GPSI model -->

### 3.2 LSTM-based Imputation Policy

为了利用更强的序列建模能力，作者将 Section 2.3 中讨论的基于 LSTM 的生成模型（*DRAW*）扩展为填补模型。

**模型结构**：主策略 $p$ 由两个 LSTM 构成——读取器/Reader $p^r$ 和写入器/Writer $p^w$。这一读写分离的设计使策略能够分别处理信息的吸收和输出。每一步的执行流程为：

1. 读取器更新状态：$s_t^r \leftarrow f_\theta^r(s_{t-1}^r, \omega_\theta^r(c_{t-1}, s_{t-1}^w, x^k))$
2. 采样动作：$z_t \sim p_\theta(z_t | v_t^r)$（对角高斯）
3. 写入器更新状态：$s_t^w \leftarrow f_\theta^w(s_{t-1}^w, z_t)$
4. 更新填补猜测：$c_t \leftarrow c_{t-1} + \omega_\theta^w(s_t^w)$（-add）或 $c_t \leftarrow \omega_\theta^w(h_t^w)$（-jump）

其中 $s_t^{r,w} \triangleq [h_t^{r,w}; v_t^{r,w}]$ 分别为读取器和写入器 LSTM 的隐藏/可见状态。该模型还包含一个"无限混合"初始化步骤：从可学习的分布采样 $z_0$，并由此确定两个 LSTM 的初始隐藏/可见状态及初始猜测 $c_0$。

**引导策略 $q_\phi$**：与主策略共享写入器 $f_\theta^w$ 和写操作 $\omega_\theta^w$，但拥有自己的读取器 $f_\phi^q$ 和读操作 $\omega_\phi^q$。关键区别在于引导策略的读操作可以看到完整观测 $x$，而主策略只能看到已知值 $x^k$。

**训练目标**：与 GPSI 相同，采用 Eq. 5 的形式。

<!-- TODO: insert Figure 7 from paper — architecture diagram and pseudocode for the LSTM-based imputation model -->

### 3.3 Extending the LSTM-based Generative Model

除了填补模型，作者还基于统一视角对 *DRAW* 的无条件生成模型本身进行了改进。这部分内容与填补任务无直接关系，但体现了 GPS 视角对改进生成模型的实际价值。主要改动包括：

1. 将动作 $z_t$ 的条件从 LSTM 内部状态扩展为同时依赖于隐藏状态 $h_t$ 和可见状态 $v_t$，即 $s_t \triangleq [h_t; v_t]$
2. 为初始状态 $s_0$ 引入可学习的分布 $p_0(z_0)$（而非固定常量），使模型从有限混合升级为无限混合
3. 引入替代的观测构造方式 $c_t \triangleq L_\theta(h_t)$，将 LSTM 的隐藏部分直接转化为观测的工作记忆

在 binarized MNIST 基准上，改进后的模型达到了 85.5 的负对数似然（原 *DRAW* 为 87.4），经过变分后验微调/Variational Posterior Fine-tuning 后进一步提升至 84.8（当时报告的最佳上界为 85.1）。

> [!tip] 变分后验微调
> 训练分为两阶段：先训练"原始"模型（主策略 $p$ 和引导策略 $q$ 联合优化），再固定 $p$ 单独微调 $q$（即变分分布），以获得更紧的对数似然上界。这一做法与 VAE 中先训练后微调推断网络的思路一致。

## 4. Experiments

### 4.1 Setup

实验在三个图像数据集上评估序贯填补模型的性能：

- **MNIST**（28x28）
- **TFD/Toronto Face Database**（48x48）
- **SVHN/Street View House Numbers**（裁剪后 32x32）

所有图像转为灰度并归一化到 $[0, 1]$。评估指标为填补的负对数似然/Negative Log-Likelihood（越低越好），按填补像素数归一化。

两种缺失机制：
- **MCAR-$d$**（完全随机缺失/Missing Completely At Random）：均匀随机掩盖 $d\%$ 的像素
- **MAR-$d$**（随机缺失/Missing At Random）：在图像边界内随机放置 $d \times d$ 的方形遮挡

模型变体：GPSI-add、GPSI-jump、LSTM-add、LSTM-jump。GPSI 模型使用 6 步精炼，LSTM 模型使用 16 步。

基线方法：
- **VAE 填补/VAE-imp**：迭代运行 VAE 的编码-解码过程，每步将已知值替换回，运行 16 步后取最佳结果
- **诚实模板匹配/Honest Template Matching**：在训练集中找到与已知像素最匹配的图像
- **预言模板匹配/Oracular Template Matching**：直接匹配缺失像素（理想上界）

### 4.2 Main Results

<!-- TODO: insert Figure 2 from paper — (a) imputation NLL vs mask probability comparing all models and baselines on MNIST; (b) closer view of proposed models only; (c) effect of increasing refinement steps for GPSI models -->

| 模型 | MNIST MAR-14 | MNIST MAR-16 | TFD MCAR-80 | TFD MAR-25 | SVHN MCAR-80 | SVHN MAR-17 |
|------|-------------|-------------|------------|-----------|-------------|-------------|
| LSTM-add | 170 | 167 | 1381 | 1377 | 525 | 568 |
| LSTM-jump | 172 | 169 | — | — | — | — |
| GPSI-add | 177 | 175 | 1390 | 1380 | 531 | 569 |
| GPSI-jump | 183 | 177 | 1394 | 1384 | 540 | 572 |
| VAE-imp | 374 | 394 | 1416 | 1399 | 567 | 624 |

主要发现：

1. **所有提出的模型均显著优于基线**：在所有数据集和缺失机制下，GPSI 和 LSTM 模型均大幅超过 VAE 填补和模板匹配。在 MNIST 上差距尤为显著（例如 LSTM-add 的 170 vs VAE-imp 的 374）
2. **LSTM 模型整体优于 GPSI 模型**：这可以理解为 LSTM 的序列建模能力提供了更强的策略表示
3. **加法更新优于跳跃更新**：additive 方式在多数设置下表现更好，且从增加精炼步数中获益更多
4. **精炼步数越多效果越好**：GPSI 模型随精炼步数增加持续改善，表明迭代精炼机制确实发挥了作用

在无条件生成（binarized MNIST，MCAR-100）方面，LSTM-add 模型的 raw/fine-tuned 得分为 86.2/85.7，LSTM-jump 为 87.1/86.3，表明这些"闭环"模型在生成任务上也具有竞争力，但作者指出其比 Section 3.3 的"开环"模型更容易过拟合。

### 4.3 Qualitative Results

<!-- TODO: insert Figure 3 from paper — imputation samples for (a) MNIST MAR-16, (b) TFD MAR-25, (c) SVHN MAR-17, showing GPSI-add, GPSI-jump, LSTM-add, LSTM-jump from top to bottom -->

从生成的填补样本来看，模型能够捕获强多模态的重建分布（即同一已知部分可以产生多种合理的填补结果）。加法策略和跳跃策略的精炼行为在视觉上有明显差异：加法策略逐步叠加细节，呈现渐进精炼的过程；而跳跃策略直接替换整个猜测，其中间过程不太直观。

### 4.4 Experimental Limitations

- **数据集规模和复杂度有限**：仅在 28x28 到 48x48 的灰度或低分辨率图像上测试，未涉及高分辨率或非图像数据
- **基线不够充分**：VAE 填补是一个相对简单的基线，且作者在附录中指出 VAE 填补存在根本性缺陷（潜变量与观测之间的互信息为零时，填补退化为从边缘分布采样）；缺少与同期其他条件生成方法的比较
- **LSTM-jump 未在 TFD 和 SVHN 上测试**：作者声明是由于时间限制，但这使得跨模型比较不完整
- **缺少效率分析**：未报告训练时间和推理时间，引导策略搜索需要同时训练主策略和引导策略的计算开销未被讨论
- **评估指标单一**：仅使用负对数似然，虽然这是生成模型的标准指标，但缺少对填补视觉质量的感知度量

## 5. Related Work & Future Work

### Related Work

本文与以下工作关系最为密切：

- ***DRAW***（*A Recurrent Neural Network for Image Generation*）：本文直接扩展了该模型。*DRAW* 提出了基于 LSTM 的循环生成模型，使用注意力机制和迭代画布更新生成图像。本文将其训练过程解释为引导策略搜索的一个实例，并在此基础上改进了初始状态分布和状态定义
- **引导策略搜索系列工作**（*Guided Policy Search*、*Variational Policy Search via Trajectory Optimization*、*Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics*）：本文采用了引导策略搜索的广义定义，将其从机器人控制领域扩展到生成模型训练
- ***Deep Unsupervised Learning Using Nonequilibrium Thermodynamics***：提出了基于扩散过程的生成模型，本文将其反向过程解释为引导策略，并在附录中给出了完整的路径级 KL 散度推导
- ***Auto-Encoding Variational Bayes***：VAE 的变分推断过程被纳入广义引导策略搜索的框架，其中编码器扮演引导策略角色

### Future Work

作者在讨论中提到的未来方向：

> 随着生成模型在结构复杂度和有效决策深度上的快速增长，序贯决策视角与生成模型之间的联系将变得更加重要。

> 将 *DRAW* 的局部读写注意力机制引入填补模型应能带来进一步改善。

从本文的局限自然延伸的方向包括：将填补框架应用到高维、非图像数据（如文本、表格数据）；探索更高效的训练算法以降低引导策略搜索的计算成本；以及利用本文的统一视角将更多 RL 中的策略优化技巧（如信赖域方法、自然梯度）迁移到生成模型训练中。
