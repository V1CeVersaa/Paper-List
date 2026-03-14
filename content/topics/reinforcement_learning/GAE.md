---
title: GAE
headline: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
---

> [!abstract] Contributions
>
> 本文提出了广义优势估计/Generalized Advantage Estimation/GAE，一族由参数 $\gamma \in [0,1]$ 和 $\lambda \in [0,1]$ 共同控制的策略梯度方差缩减方案。其核心技术手段是对 $k$-步优势估计器 $\hat{A}_t^{(k)}$ 进行指数加权平均，得到形如 $\sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}^V$ 的简洁闭式表达，其结构与 TD($\lambda$) 高度类似，但估计的是优势函数/Advantage Function 而非值函数。结合 TRPO 进行策略优化和信赖域方法训练值函数，该方法在高维连续控制任务（3D 双足行走、四足行走、站立）上取得了当时的最优表现，策略直接从原始运动学状态映射到关节力矩，无需手工设计特征。
>
> 本文的分析建立在值函数 $V$ 足够准确的前提上：当 $V \neq V^{\pi,\gamma}$ 时，$\lambda < 1$ 的 GAE 估计器不再是 $\gamma$-just 的，引入的偏差大小取决于值函数的逼近误差。此外，实验仅覆盖了模拟运动控制领域，$\gamma$ 和 $\lambda$ 的最优取值需要针对具体任务手动调节，论文未提供自适应选取机制。

## 1. Introduction

策略梯度方法通过直接优化累积奖励将强化学习归约为随机梯度上升，可以自然地与神经网络等非线性函数逼近器结合。然而其实际应用面临两个核心挑战：（1）所需样本量大，（2）在非平稳数据流上难以获得稳定的持续改进。

样本效率低下的根源在于策略梯度估计的高方差。REINFORCE 类方法使用经验回报作为梯度中的 $\Psi_t$，方差会随时间步数不利地增长，因为单个动作的效果被过去和未来所有动作的效果所混淆。Actor-Critic 方法用值函数替代经验回报可以降低方差，但代价是引入偏差——高方差只需更多样本即可缓解，而偏差却可能导致算法收敛到非局部最优甚至发散，因此偏差的危害更为严重。

本文的核心思路是：在方差与偏差之间寻找一个可控的折中。具体而言，作者提出了 GAE$(\gamma, \lambda)$ 这一估计器族，其中 $\gamma \in [0, 1]$ 通过对远期奖励降权来缩减方差（可理解为在无折扣问题中引入的方差缩减参数），$\lambda \in [0, 1]$ 则通过指数加权 $k$-步估计器在纯 TD（高偏差低方差）与蒙特卡洛回报（低偏差高方差）之间插值。论文还从 reward shaping 的角度给出了 GAE 的另一种理解，并引入 response function 来量化偏差的来源。

在工程层面，作者将 GAE 与 TRPO 结合用于策略更新，并提出了一种基于信赖域的值函数优化方法，以避免值函数在批量数据上过拟合。最终系统在高维 3D 运动控制任务上展现了强大的学习能力，策略和值函数均由具有上万参数的神经网络表示。

## 2. Problem Setup

考虑无折扣的策略优化问题。初始状态 $s_0 \sim \rho_0$，策略 $\pi_\theta(a_t \mid s_t)$ 与环境动力学 $P(s_{t+1} \mid s_t, a_t)$ 交互产生轨迹 $(s_0, a_0, s_1, a_1, \ldots)$，每步获得奖励 $r_t = r(s_t, a_t, s_{t+1})$。目标是最大化期望总奖励 $\sum_{t=0}^{\infty} r_t$（假设对所有策略有限）。

定义如下核心量：

$$
V^{\pi}(s_t) := \mathbb{E}_{s_{t+1:\infty}, a_{t:\infty}} \left[ \sum_{l=0}^{\infty} r_{t+l} \right] \tag{1}
$$

$$
Q^{\pi}(s_t, a_t) := \mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty}} \left[ \sum_{l=0}^{\infty} r_{t+l} \right] \tag{2}
$$

$$
A^{\pi}(s_t, a_t) := Q^{\pi}(s_t, a_t) - V^{\pi}(s_t) \tag{3}
$$

其中 $V^\pi$ 是状态值函数，$Q^\pi$ 是状态-动作值函数，$A^\pi$ 是优势函数——衡量在状态 $s_t$ 下执行动作 $a_t$ 相对于策略默认行为的好坏。策略梯度的一般形式为：

$$
g = \mathbb{E} \left[ \sum_{t=0}^{\infty} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] \tag{4}
$$

这里 $\Psi_t$ 可以选择的形式很多：trajectory return、reward-to-go、baselined reward to go、$Q^\pi$、$A^\pi$、甚至 TD residual。选择 $\Psi_t = A^\pi(s_t, a_t)$ 可获得*几乎*最低方差，因为优势函数精确度量了动作优于平均的程度——当且仅当 $A^\pi(s_t, a_t) > 0$ 时，梯度项 $\Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 才指向增加该动作概率的方向。

**引入折扣因子 $\gamma$。** 虽然问题本身是无折扣的，但可以引入参数 $\gamma \in [0,1]$ 作为方差缩减手段。定义折扣值函数和优势函数：

$$
V^{\pi,\gamma}(s_t) := \mathbb{E} \left[ \sum_{l=0}^{\infty} \gamma^l r_{t+l} \right], \quad A^{\pi,\gamma}(s_t, a_t) := Q^{\pi,\gamma}(s_t, a_t) - V^{\pi,\gamma}(s_t) \tag{5}
$$

折扣策略梯度为：

$$
g^\gamma := \mathbb{E}_{s_{0:\infty}, a_{0:\infty}} \left[ \sum_{t=0}^{\infty} A^{\pi,\gamma}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] \tag{6}
$$

$\gamma < 1$ 对远期奖励降权，从而减小方差，但偏离了真实的无折扣目标。

**$\gamma$-just 估计器。** 为了刻画使用近似优势函数估计 $g^\gamma$ 时的无偏性条件，论文定义了 $\gamma$-just 的概念：

> [!note] Definition 1: $\gamma$-just
> 优势估计器 $\hat{A}_t$ 是 $\gamma$-just 的，当且仅当：
>
> $$
> \mathbb{E}_{s_{0:\infty}, a_{0:\infty}} \left[ \hat{A}_t(s_{0:\infty}, a_{0:\infty}) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = \mathbb{E}_{s_{0:\infty}, a_{0:\infty}} \left[ A^{\pi,\gamma}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
> $$
>
> 即用 $\hat{A}_t$ 替换 $A^{\pi,\gamma}$ 后，策略梯度的期望 $g^\gamma$ 不变。

> [!note] Proposition 1
> 若 $\hat{A}_t$ 可以分解为 $Q_t(s_{0:\infty}, a_{t:\infty}) - b_t(s_{0:t}, a_{0:t-1})$ 的形式，其中 $Q_t$ 是 $Q^{\pi,\gamma}$ 的无偏估计，也就是说 $Q^{\pi,\gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty} \mid s_t, a_t} \left[ Q_t(s_{t, \infty}, a_{t, \infty}) \right]$，$b_t$ 仅依赖于 $a_t$ 之前采样的状态和动作，则 $\hat{A}_t$ 是 $\gamma$-just 的。

> [!todo]- Proof of Proposition 1
> 将期望拆分为 $Q$ 项和 $b$ 项分别处理。
> $$
> \begin{aligned}
> &\mathbb{E}_{s_{0:\infty},a_{0:\infty}}{\nabla_\theta \log \pi_\theta (a_t \mid s_t) (Q_t(s_{0:\infty},a_{0:\infty})-b_t(s_{0:t},a_{0:t-1}))}   \\
> &\ = \mathbb{E}_{s_{0:\infty},a_{0:\infty}}{\nabla_\theta \log \pi_\theta (a_t \mid s_t) (Q_t(s_{0:\infty},a_{0:\infty}))} - \mathbb{E}_{s_{0:\infty},a_{0:\infty}}{\nabla_\theta \log \pi_\theta (a_t \mid s_t) (b_t(s_{0:t},a_{0:t-1}))}
> \end{aligned}
> $$
>
> 对于 $Q$ 项：利用 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 仅依赖于 $(s_t, a_t)$ 的事实，将外层期望拆为对 $(s_{0:t}, a_{0:t})$ 和对 $(s_{t+1:\infty}, a_{t+1:\infty})$ 的条件期望。内层条件期望恰好将 $Q_t$ 还原为 $Q^{\pi,\gamma}(s_t, a_t)$，再由优势函数定义得到所需结果。
> $$
> \begin{align*}
> &\mathbb{E}_{s_{0:\infty},a_{0:\infty}}{\nabla_\theta \log \pi_\theta (a_t \mid s_t) Q_t(s_{0:\infty},a_{0:\infty})} \\
> &=\mathbb{E}_{s_{0:t},a_{0:t}}{\mathbb{E}_{s_{t+1:\infty},a_{t+1:\infty}}{\nabla_\theta \log \pi_\theta (a_t \mid s_t) Q_t(s_{0:\infty},a_{0:\infty})}}\\
> &=\mathbb{E}_{s_{0:t},a_{0:t}}{ \nabla_\theta \log \pi_\theta (a_t \mid s_t) \mathbb{E}_{s_{t+1:\infty},a_{t+1:\infty}}{Q_t(s_{0:\infty},a_{0:\infty})}}\\
> &=\mathbb{E}_{s_{0:t},a_{0:t-1}}{ \nabla_\theta \log \pi_\theta (a_t \mid s_t) A^\pi (s_t, a_t)}
> \end{align*}
> $$
>
> 对于 $b$ 项：由于 $b_t$ 仅依赖 $(s_{0:t}, a_{0:t-1})$，而 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 关于 $a_t$ 的条件期望为零（$\mathbb{E}_{a_t}[\nabla_\theta \log \pi_\theta(a_t \mid s_t)] = 0$），因此 $b$ 项的整体贡献为零。
> $$
> \begin{align*}
> &\mathbb{E}_{s_{0:\infty},a_{0:\infty}}{\nabla_\theta \log \pi_\theta (a_t \mid s_t) b_t(s_{0:t},a_{0:t-1})} \\
> &=\mathbb{E}_{s_{0:t},a_{0:t-1}}{\mathbb{E}_{s_{t+1:\infty},a_{t:\infty}}{\nabla_\theta \log \pi_\theta (a_t \mid s_t) b_t(s_{0:t},a_{0:t-1})}}\\
> &=\mathbb{E}_{s_{0:t},a_{0:t-1}}{ \mathbb{E}_{s_{t+1:\infty},a_{t:\infty}}{\nabla_\theta \log \pi_\theta (a_t \mid s_t)}b_t(s_{0:t},a_{0:t-1})}\\
> &=\mathbb{E}_{s_{0:t},a_{0:t-1}}{ 0 \cdot b_t(s_{0:t},a_{0:t-1})}\\
> &=0.
> \end{align*}
> $$

直觉上，$b_t$ 起到 baseline 的作用——它不影响梯度的期望但能降低方差。下面这些个估计都是 $\gamma$-just 的：$\sum_{l=0}^{\infty} \gamma^l r_{t+l}$、$Q^{\pi,\gamma}(s_t, a_t)$、$A^{\pi,\gamma}(s_t, a_t)$、$r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t)$。

## 3. Algorithm

### 3.1 从 $k$-步估计到 GAE

令 $V$ 为值函数 $V^{\pi,\gamma}$ 的近似，定义 TD 残差/TD Residual：

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{7}
$$

当 $V = V^{\pi,\gamma}$ 时，$\delta_t^V$ 是 $A^{\pi,\gamma}$ 的无偏估计：

$$
\mathbb{E}_{s_{t+1}} \left[ \delta_t^{V^{\pi,\gamma}} \right] = \mathbb{E}_{s_{t+1}} \left[ r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t) \right] = A^{\pi,\gamma}(s_t, a_t)
$$

将 $k$ 个连续的 TD 残差求和，得到 $k$-步优势估计器：

$$
\hat{A}_t^{(k)} := \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k}) \tag{8}
$$

这实际上是 $k$-步回报减去 baseline $V(s_t)$。$k$ 越大，偏差越小（因为 $\gamma^k V(s_{t+k})$ 项的权重指数衰减），但方差越大。两个极端情况：

- $k = 1$：$\hat{A}_t^{(1)} = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$，偏差最大但方差最小；
- $k \to \infty$：$\hat{A}_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$，即折扣蒙特卡洛回报减 baseline，偏差最小但方差最大。

**GAE 的定义。** 对这族 $k$-步估计器进行指数加权平均：

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} := (1 - \lambda) \left( \hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots \right) \tag{9}
$$

展开并化简：

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = (1 - \lambda) \left( \delta_t^V + \lambda(\delta_t^V + \gamma \delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V) + \cdots \right)
$$

> [!todo]- Derivation
> 将 $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$ 代入，交换求和顺序。$\delta_{t+l}^V$ 在指数加权平均中出现的总权重为：
>
> $$
> (1-\lambda) \gamma^l \sum_{j=l}^{\infty} \lambda^j = (1-\lambda) \gamma^l \cdot \frac{\lambda^l}{1-\lambda} = (\gamma\lambda)^l
> $$
>
> 因此：
>
> $$
> \hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V
> $$

最终得到极为简洁的闭式：

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V \tag{10}
$$

这个结构与 TD($\lambda$) 高度类似，但 TD($\lambda$) 估计的是值函数，而 GAE 估计的是优势函数。两个特殊情况：

- **GAE($\gamma$, 0)**：$\hat{A}_t = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$，即单步 TD 残差。仅当 $V = V^{\pi,\gamma}$ 时 $\gamma$-just，否则有偏但方差低。
- **GAE($\gamma$, 1)**：$\hat{A}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V = -V(s_t) + \sum_{l=0}^{\infty} \gamma^l r_{t+l}$，即折扣蒙特卡洛回报减 baseline。无论 $V$ 是否准确都是 $\gamma$-just 的，但方差大。

$\gamma$ 和 $\lambda$ 都参与偏差-方差的权衡，但作用机制不同：

- **$\gamma$** 主要决定值函数 $V^{\pi,\gamma}$ 的尺度，$\gamma < 1$ 无论值函数是否准确都会引入偏差；
- **$\lambda$** 仅在值函数不准确时才引入偏差，当 $V \approx V^{\pi,\gamma}$ 时 $\lambda$ 引入的偏差远小于 $\gamma$。

实验也证实了这一点：$\lambda$ 的最优值通常远高于 $\gamma$ 的最优值。

使用 GAE 估计器，折扣策略梯度近似为：

$$
g^\gamma \approx \mathbb{E} \left[ \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t^{\text{GAE}(\gamma,\lambda)} \right] = \mathbb{E} \left[ \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V \right] \tag{11}
$$

当 $\lambda = 1$ 时等号成立。

### 3.2 Reward Shaping 视角

本节从 reward shaping 的角度为 GAE 提供了另一种理解。Reward shaping 是指对 MDP 的奖励函数进行如下变换：

$$
\tilde{r}(s, a, s') = r(s, a, s') + \gamma \Phi(s') - \Phi(s) \tag{12}
$$

其中 $\Phi: \mathcal{S} \to \mathbb{R}$ 是任意状态函数。这一变换的关键性质是：它不改变折扣优势函数 $A^{\pi,\gamma}$。

> [!todo]- Derivation
> 在变换后的 MDP 中，从状态 $s_t$ 出发的折扣奖励和为：
>
> $$
> \sum_{l=0}^{\infty} \gamma^l \tilde{r}(s_{t+l}, a_{t+l}, s_{t+l+1}) = \sum_{l=0}^{\infty} \gamma^l r(s_{t+l}, a_{t+l}, s_{t+l+1}) - \Phi(s_t)
> $$
>
> 因此变换后的值函数和 Q 函数分别为 $\tilde{V}^{\pi,\gamma}(s) = V^{\pi,\gamma}(s) - \Phi(s)$，$\tilde{Q}^{\pi,\gamma}(s,a) = Q^{\pi,\gamma}(s,a) - \Phi(s)$，二者相减得到 $\tilde{A}^{\pi,\gamma} = A^{\pi,\gamma}$。

若令 $\Phi = V$（近似值函数），则变换后的奖励 $\tilde{r}(s_{t+l}, a_{t+l}, s_{t+l+1})$ 恰好等于 Bellman 残差 $\delta_{t+l}^V$。对变换后的奖励使用折扣 $\gamma\lambda$  求和：

$$
\sum_{l=0}^{\infty} (\gamma\lambda)^l \tilde{r}(s_{t+l}, a_{t+l}, s_{t+l+1}) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V = \hat{A}_t^{\text{GAE}(\gamma,\lambda)}
$$

这揭示了 GAE 的本质：**先用近似值函数 $V$ 对奖励进行 shaping，将原本时间上分散的奖励信号压缩到即时反馈中；再用一个更陡峭的折扣 $\gamma\lambda$ 截断远期噪声。**

**Response function 与偏差分析。** 论文引入了 response function 来精确刻画偏差来源：

$$
\chi(l; s_t, a_t) = \mathbb{E}[r_{t+l} \mid s_t, a_t] - \mathbb{E}[r_{t+l} \mid s_t]
$$

它衡量动作 $a_t$ 对未来第 $l$ 步奖励的影响。优势函数可以分解为 $A^{\pi,\gamma}(s,a) = \sum_{l=0}^{\infty} \gamma^l \chi(l; s, a)$。使用折扣 $\gamma < 1$ 相当于丢弃 $l \gg 1/(1-\gamma)$ 之后的项。如果 response function 衰减足够快（即动作的长期影响较弱），这种截断引入的偏差就很小。

更进一步，如果用完美的值函数 $\Phi = V^{\pi,\gamma}$ 做 reward shaping，则变换后的 response function 在 $l > 0$ 时为零——即所有奖励信号都被压缩到了 $l=0$。用近似的 $V$ 做 shaping 则可以部分缩短 response function 的时间跨度，$\lambda < 1$ 的截断因此只丢弃较小的尾部贡献。这解释了为什么 $\lambda$ 对值函数质量的敏感度远低于 $\gamma$。

### 3.3 Value Function Estimation

值函数使用神经网络参数化为 $V_\phi$。每次策略迭代收集一批轨迹后，需要更新值函数。最直接的方法是非线性回归：

$$
\min_\phi \sum_{n=1}^{N} \| V_\phi(s_n) - \hat{V}_n \|^2
$$

其中 $\hat{V}_n = \sum_{l=0}^{\infty} \gamma^l r_{t+l}$ 是折扣回报。但直接最小化该目标容易在有限的批量数据上过拟合。

论文提出使用信赖域方法约束值函数的更新幅度。先计算 $\sigma^2 = \frac{1}{N} \sum_n \| V_{\phi_\text{old}}(s_n) - \hat{V}_n \|^2$，然后求解：

$$
\min_\phi \sum_{n=1}^{N} \| V_\phi(s_n) - \hat{V}_n \|^2 \quad \text{s.t.} \quad \frac{1}{N} \sum_{n=1}^{N} \frac{\| V_\phi(s_n) - V_{\phi_\text{old}}(s_n) \|^2}{2\sigma^2} \leq \epsilon \tag{13}
$$

该约束可以理解为：将值函数视为以 $V_\phi(s)$ 为均值、$\sigma^2$ 为方差的条件高斯分布，则约束为新旧值函数之间的平均 KL 散度不超过 $\epsilon$。实际求解时，将目标线性化、约束二次化，用共轭梯度法计算步长方向 $s \approx -H^{-1}g$，再缩放到信赖域边界。

### 3.4 Complete Algorithm

完整算法交替进行策略优化和值函数更新：

1. 用当前策略 $\pi_{\theta_i}$ 收集轨迹直至获得 $N$ 个时间步；
2. 用当前值函数 $V_{\phi_i}$ 计算所有时间步的 TD 残差 $\delta_t^V$；
3. 计算 GAE 估计器 $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$；
4. 用 TRPO 更新策略参数 $\theta_{i+1}$；
5. 用信赖域方法（公式 (13)）更新值函数参数 $\phi_{i+1}$。

> [!tip]
> 一个重要的细节是：策略更新 $\theta_i \to \theta_{i+1}$ 使用的是旧的值函数 $V_{\phi_i}$ 而非更新后的 $V_{\phi_{i+1}}$。如果先更新值函数再用于策略梯度估计，在值函数过拟合的极端情况下 Bellman 残差会变为零，策略梯度估计也会退化为零。

策略更新采用 TRPO：

$$
\min_\theta L_{\theta_\text{old}}(\theta) \quad \text{s.t.} \quad \bar{D}_\text{KL}(\pi_{\theta_\text{old}}, \pi_\theta) \leq \epsilon
$$

其中 $L_{\theta_\text{old}}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \frac{\pi_\theta(a_n \mid s_n)}{\pi_{\theta_\text{old}}(a_n \mid s_n)} \hat{A}_n$，通过线性化目标和二次化约束近似求解，步长方向为 $\theta - \theta_\text{old} \propto -F^{-1}g$，其中 $F$ 为平均 Fisher 信息矩阵，与自然策略梯度/Natural Policy Gradient 和自然 Actor-Critic 等价。

## 4. Experiments

实验旨在回答两个核心问题：（1）$\gamma$ 和 $\lambda$ 的取值如何影响优化无折扣总奖励时的性能？（2）GAE + 信赖域方法能否学习高维连续控制的神经网络策略？

**实验设置。** 任务包括经典 cart-pole 和三个基于 MuJoCo 的 3D 运动控制任务：双足行走（33 维状态，10 维动作）、四足行走（29 维状态，8 维动作）、双足站起。策略和值函数均为三层前馈网络（隐藏层 100-50-25，tanh 激活）。奖励函数鼓励前进速度，同时惩罚关节力矩和地面冲击力。

**Cart-pole 结果。** 在 21 个随机种子上平均，最佳性能出现在 $\gamma \in [0.96, 0.99]$、$\lambda \in [0.92, 0.99]$ 的中间值区域。$\lambda = 0$（纯 TD 残差）和不使用值函数的情况均表现最差，验证了 GAE 的方差缩减效果。

**3D 双足行走。** 每次试验约 2 小时（16 核机器），9 个随机种子平均。最佳参数为 $\gamma \in [0.99, 0.995]$、$\lambda \in [0.96, 0.99]$。经过约 1000 次迭代后，学到的步态快速、平稳且稳定，对应约 5.8 天的模拟时间。

**四足行走和站起。** 固定 $\gamma = 0.995$，比较 $\lambda \in \{0, 0.96\}$ 和不使用值函数的情况。四足行走中 $\lambda = 0.96$ 表现最好；站起任务中 $\lambda = 0.96$ 和 $\lambda = 1$ 效果相当，但都显著优于不使用值函数的情况。

**关键发现：**

- 在所有任务中，中间值的 $\lambda$（约 $[0.9, 0.99]$）通常表现最好，验证了 GAE 在偏差-方差权衡上的有效性；
- $\lambda$ 的最优值通常高于 $\gamma$ 的最优值，与理论分析一致（$\lambda$ 引入的偏差弱于 $\gamma$）；
- 使用值函数 baseline 始终优于不使用值函数，即使是 $\lambda = 1$（不依赖值函数准确性）的情况也比无值函数好。

**实验局限：**

- 实验仅覆盖模拟运动控制领域，未涉及操作、导航等其他连续控制类型；
- 策略优化算法固定为 TRPO，未与其他策略梯度方法（如 vanilla PG、natural PG）做组合对比；
- $\gamma$ 和 $\lambda$ 均为手动调节的超参数，缺乏自适应选取的验证；
- 未提供值函数逼近误差的量化分析——论文的偏差分析依赖于 $V \approx V^{\pi,\gamma}$ 的假设，但实验中并未报告值函数的实际拟合质量。

## 5. Related Work & Future Work

**Related Work.** GAE 的公式形式在此前的 online Actor-Critic 文献中已有出现（*An Analysis of Actor/Critic Algorithms Using Eligibility Traces*; *Real-time Reinforcement Learning by Sequential Actor–Critics and Experience Replay*），但本文的贡献在于为其提供了更一般的分析框架，使其适用于 online 和 batch 场景，并给出了 reward shaping 解释。与直接使用 Q 函数的 Actor-Critic 方法（如 *On Actor-Critic Algorithms*）相比，使用状态值函数 $V$ 有两个优势：（1）$V$ 的输入维度更低，更容易学习；（2）通过 $\lambda$ 参数可以在高偏差（$\lambda=0$）和低偏差（$\lambda=1$）之间平滑插值，而参数化 Q 函数只能给出高偏差的估计器。

**Future Work.**

论文提出的未来方向包括：

> - 自适应调节 $\gamma$ 和 $\lambda$ 的机制，避免手动超参搜索；
> - 研究值函数估计误差与策略梯度估计误差之间的定量关系，以此指导值函数训练目标的选取（候选指标包括 Bellman 误差和 projected Bellman 误差）；
> - 策略和值函数共享网络架构，利用共享表示加速学习。

从论文的局限性自然延伸，还有以下可能的方向：将 GAE 与更多样的策略优化方法结合验证其普适性；在更广泛的任务类型（稀疏奖励、部分可观测等）上评估 GAE 的表现；为 $\lambda$ 的选取提供理论指导，例如基于值函数误差的上界自动确定 $\lambda$ 的取值范围。
