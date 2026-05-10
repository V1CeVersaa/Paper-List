---
title: "DPO Misspecification"
headline: "Why DPO is a Misspecified Estimator and How to Fix It"
description: "ICLR 2026 Oral note on DPO misspecification under parametric policies and the AuxDPO correction."
status: complete
visibility: public
topic: "preference_learning"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2510.20413"
openreview_url: "https://openreview.net/forum?id=btEiAfnLsX"
source_pdf: "raw/papers/arxiv-2510.20413.pdf"
tags: ["preference_optimization", "dpo", "rlhf_theory"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文指出一个 DPO 里经常被忽略的统计问题：DPO 的经典推导依赖 tabular policy class，也就是 policy 可以表达任意 prompt-response 条件分布；真实 LLM 是低维参数化 policy class，能够表达的 implicit reward 只形成一个低维 manifold。当真实 preference-generating reward $r^*$ 不在这个 manifold 上时，DPO 等价于把 $r^*$ 按 preference pair 频率加权投影到可表达 reward manifold 上，于是可能出现 **preference reversal**、expected reward 下降、对数据采样频率高度敏感等 failure modes。
>
> 论文进一步用局部几何分析两阶段 RLHF，把 RLHF 的 policy update 写成 natural-gradient-like step，并提出 **AuxDPO**：在 DPO 的 implicit reward 外加入 auxiliary variables $\delta$，让优化可以沿 RLHF 等价类中的 null-space 方向移动，从而缓解 misspecification。它的价值不只是提出一个新 loss，而是给 alignment 社区敲了一个硬警钟：DPO 的“等价于 RLHF”只在理想表达能力下成立，参数化 policy 会把直接偏好优化变成一个有偏的统计估计问题。

## 1. Introduction

DPO 之所以流行，是因为它把 RLHF 的两阶段过程压缩成一个监督式 loss。经典解释是：先考虑 KL-regularized RLHF 的最优解，把 reward 写成 policy 与 reference policy 的 log-ratio，再把 Bradley-Terry preference likelihood 改写成 DPO loss。这个推导在 tabular policy class 下很漂亮，因为 policy class 足够大，可以表达任意 reward 所诱导的 optimal policy。

真实 LLM 不满足这个条件。Transformer policy 的参数维度远小于所有 prompt-response pair 的数量，模型只能表达一个受架构、初始化和优化约束限制的 policy family。于是问题变成：当 $r^*$ 不能被当前 policy class 的 implicit reward 表示时，DPO 到底在估计什么？论文的回答非常直接：DPO 在做一个 **misspecified statistical estimation problem**。它不是在恢复真实 reward，也不一定在靠近两阶段 RLHF 的 policy optimum，而是在某个由数据频率加权的 KL 投影问题里找最接近的可表达 reward。

这个视角对 alignment 很关键，因为偏好数据天然不均衡。某些 response pair 出现得多，某些永远没有比较；某些 prompt 下 chosen/rejected 的差异清楚，某些只是风格偏好。DPO 如果只是把 $r^*$ 投影到模型能表达的 reward manifold，那么 projection 的方向会被 pair sampling frequencies 强烈影响。论文最冷的结论是：即使偏好标签是无限干净的、完全由 Bradley-Terry model 生成，DPO 仍可能因为 policy class misspecification 学出更差的 policy。

## 2. Problem Setup

论文考虑有限状态/动作集合 $S,A$，一个偏好数据集 $D=\{(s^{(i)},a_w^{(i)},a_l^{(i)})\}_{i=1}^n$，其中 $a_w$ 是 preferred response，$a_l$ 是 rejected response。真实偏好由 latent reward $r^*(s,a)$ 通过 Bradley-Terry-Luce model 生成：

$$
p_{\mathrm{BTL}}^*(a\succ a'|s)
=
\sigma(r^*(s,a)-r^*(s,a')).
$$

Policy 是参数化 family $\pi_\theta:S\to \Delta(A)$，reference policy 是 $\pi_{\theta_0}$。DPO 的 implicit reward 可以写成：

$$
r_\theta^\beta(s,a)
=
\beta \log \frac{\pi_\theta(a|s)}{\pi_{\theta_0}(a|s)}.
$$

在 tabular setting 下，$\theta$ 可以自由调每个 $(s,a)$ 的概率，所以 $r_\theta^\beta$ 基本可以覆盖 reward space。可是在参数化 LLM 中，$\theta\in\mathbb{R}^d$，而所有 $(s,a)$ 组合形成的 reward vector 在 $\mathbb{R}^m$ 中，通常 $d\ll m$，并且可表达 reward 只构成一个低维非线性 manifold $\mathcal{R}_\beta$。

论文的 Proposition 1 是全篇核心。它说明，如果 pairwise preference counts 是 $n_{s,a,a'}$，那么 DPO loss 的 minimizer 对应的 implicit reward 满足：

$$
r_{\theta_{\mathrm{DPO}}}^{\beta}
=
\arg\min_{r\in \mathcal{R}_\beta}
\sum_{s,a,a'}
n_{s,a,a'}
d_{\mathrm{KL}}
\left(
p_{\mathrm{BTL}}(r^*)\|p_{\mathrm{BTL}}(r)
\right).
$$

这句话的含义是，DPO 并没有直接最大化真实 reward，也没有必然恢复 RLHF solution；它把真实 reward $r^*$ 按 pair 频率加权投影到 $\mathcal{R}_\beta$ 上。如果 $r^*\in\mathcal{R}_\beta$，一切正常；如果 $r^*\notin\mathcal{R}_\beta$，结果就由 projection geometry 和数据频率共同决定。

这一步把 DPO 的风险说得非常具体。Preference data frequency 不只是统计效率问题，而是 objective 本身的权重。假设某个 response pair 在数据中出现很多次，DPO 的投影就会优先拟合这组 pair 的 Bradley-Terry probability；如果 policy class 无法同时满足所有 pairwise reward differences，它就会牺牲别的方向。真实 RLHF 数据往往就是这种稀疏、偏置、长尾的 comparison graph，因此 misspecification 不是罕见边角问题，而是 direct alignment 里一直存在的结构风险。

论文还强调，coverage condition 只能解决一部分问题。已有工作指出，如果 reference policy 对某些 response 概率太低，DPO 可能没有足够覆盖。本文的反例更强：即使 base policy 是 uniform，满足很好的全局覆盖，DPO 仍可能因为 reward manifold 方向不对而失败。也就是说，coverage 是必要条件的一部分，但不是充分条件；模型能否表达正确 reward geometry 同样关键。

## 3. Algorithm / Methods / Model

为了看清这个 geometry，作者在 reference parameter $\theta_0$ 附近对 implicit reward 做一阶展开：

$$
r_\theta^\beta(s,a)
\approx
\beta \nabla \log \pi_{\theta_0}(a|s)^\top(\theta-\theta_0).
$$

把所有 $(s,a)$ 的 $\nabla\log\pi_{\theta_0}(a|s)$ 组成矩阵 $A_{\theta_0}$ 后，local implicit reward manifold 近似为 $\mathcal{C}(A_{\theta_0}^\top)$。这个空间只由 policy class 和 reference policy 决定，不由真实 reward 决定。因此，如果 $r^*$ 有大量分量落在这个 column space 外，DPO 就只能投影，不能表达。

论文用一个三 response 的构造展示 failure mode。真实 reward order 是 $a_2\succ a_1\succ a_3$，policy family 是一维 softmax $\pi_\theta\propto [e^\theta,e^{-\theta},1]$。当 preference pair counts 中某一类比较极度主导时，weighted projection 会把 $r^*$ 推到错误方向，导致 DPO 学到的 policy 更偏好次优 response，降低最优 response 概率，并且使 $\pi_\theta^\top r^*$ 低于 base policy。这里最重要的是，这不是小数据噪声，也不是 gradient descent 没优化好；它发生在 population DPO loss 和无限干净偏好数据下。

这个例子值得反复咀嚼。DPO 看到的所有 pairwise preferences 都可以是正确的，且由同一个真实 reward 生成；失败来自 policy class 的一维结构只能沿 $[1,-1,0]$ 这样的方向移动。若数据频率迫使投影优先拟合 $a_1$ 对 $a_3$ 的关系，模型就可能把 $a_1$ 的概率推高，同时把真正最优的 $a_2$ 推低。于是 preference reversal 和 reward decrease 同时出现。这个现象比常见的 likelihood displacement 更严重，因为它不仅是 chosen likelihood 下降，而是最终策略排序和真实 reward 排序发生冲突。

论文对这个反例给出五个解释，其中最重要的是 sensitivity。只要改变 pair counts，比如让 $n_{1,2}$ 主导，DPO 可能又会朝正确方向移动。换言之，同一个真实 reward、同一个 policy class，不同的数据采样频率会导致完全不同的结果。对真实 preference dataset 来说，这意味着数据收集策略本身会改变 DPO 的偏差方向；不能只说“数据越多越好”，还要问 comparison graph 是否覆盖了会决定投影方向的关键 pair。

紧接着，作者分析两阶段 RLHF 的局部几何。对 expected reward 做一阶近似，对 KL penalty 做二阶近似，可以得到近似的 RLHF update：

$$
\theta^*
=
\theta_0
+
\frac{1}{\beta}
F_{\rho,\theta_0}^{\dagger}
A_{\rho,\theta_0}r^*.
$$

这里 $F_{\rho,\theta_0}$ 是 Fisher information matrix，$A_{\rho,\theta_0}$ 是带 prompt distribution 和 reference policy 权重的 gradient matrix。这个形式像 natural policy gradient：reward 不是直接被投影到 DPO manifold，而是先经过 $A_{\rho,\theta_0}$ 影响 parameter update。

这一步引出 **RLHF equivalence class**：

$$
\mathcal{R}_{\mathrm{eq}}^\beta(\theta)
=
\{r\in\mathbb{R}^m:
A_{\rho,\theta_0}r
=
\beta F_{\rho,\theta_0}(\theta-\theta_0)\}.
$$

同一个 policy update 可以由很多 reward vector 诱导出来，只要它们相差一个 $A_{\rho,\theta_0}$ 的 null-space element。论文的关键观察是：DPO 的 local implicit reward 只是这个 equivalence class 里的 minimum-norm representative。也就是说，DPO 强迫自己选了一个特定代表元；两阶段 RLHF 关心的是整个 equivalence class 能诱导的 policy movement。

AuxDPO 的设计由此而来。它在 DPO implicit reward $r_\theta^\beta$ 外加入 auxiliary variables $\delta$，并约束 $\delta\in \mathcal{N}(A_{\rho,\theta_0})$。这样 $r_\theta^\beta+\delta$ 可以沿 null-space 方向移动，从而表达那些 DPO 原本投影不到但 RLHF 等价类允许的 reward 分量。实际实现中，完整 $\delta\in\mathbb{R}^m$ 太大，作者只为数据集中出现的 chosen/rejected responses 维护 $2n$ 个 auxiliary scalars，并用 Monte Carlo 形式的 penalty 近似 null-space constraint。它不是 LLM 的额外 head，而是一组与样本相关的可训练变量。

经验 loss 可以理解成两部分。第一部分把 DPO margin 从 $r_\theta^\beta(s,a_w)-r_\theta^\beta(s,a_l)$ 改成 $r_\theta^\beta(s,a_w)-r_\theta^\beta(s,a_l)+\delta(s,a_w)-\delta(s,a_l)$，让每个训练 pair 有额外自由度去修正 implicit reward difference。第二部分用 $\lambda$ penalty 约束这些 $\delta$ 不要随便改变 policy update 方向，而是尽量落在 $A_{\rho,\theta_0}$ 的 null space 里。这样 AuxDPO 不是无约束地给每个样本加 bias；它试图只补 DPO manifold 表达不到、但 RLHF equivalence class 允许的 reward 分量。

这个设计也解释了 AuxDPO 的工程风险。$\delta$ 是 dataset-specific 的，理论上帮助修正 misspecification，实践中也可能记住训练 pair 的 idiosyncrasy。$\lambda$ 太小，auxiliary variables 可能变成任意拟合器；$\lambda$ 太大，又退回普通 DPO。论文在 rebuttal 中补充 $\lambda$ sensitivity，并说明实验里通常取 $\lambda=1$，但更大规模、更嘈杂 preference data 下的稳定性仍需要验证。

## 4. Experiments

实验分成 didactic bandit 和 LLM alignment。主表使用 UltraFeedback 训练，在 **RewardBench V2** 和 **MMLU-Pro** 上评估，比较 DPO、AuxDPO、IPO、DPOP。论文同时报告 ID 和 OOD 设定：ID 是目标数据集拆分训练/评估，OOD 是用 cleaned UltraFeedback 训练后迁移到评估数据集。

结果支持 AuxDPO 的主要说法。在 Llama3.1-8B 上，AuxDPO 在 MMLU-Pro ID/OOD 和 RewardBench V2 ID/OOD 上都高于 DPO；RewardBench V2 OOD 的 improvement 是 32.44，相比 DPO 的 14.31 提升明显。在 Llama3.2-1B 上，RewardBench V2 OOD 中 AuxDPO 是 43.27，而 DPO 是 14.11；在 Qwen3-0.6B 上，DPO 在 RewardBench V2 OOD 甚至为负，AuxDPO 仍然正向提升 18.36。这个 pattern 很符合论文叙事：misspecification 在容量受限或 OOD 时更容易放大，而 auxiliary degrees of freedom 能缓解这种投影偏差。

MMLU-Pro 的使用需要仔细理解。它本来不是 preference dataset，作者把正确答案当 chosen response，把错误答案当 rejected response，构造出类似 preference pair 的训练/评估形式。这个做法能测试“偏好优化是否提升选择正确答案的 logit”，但它和真实 human preference alignment 不是同一个分布。因此，MMLU-Pro 结果更适合看作 reasoning/QA style preference proxy，而不是完整 RLHF benchmark。

Table 2 的 per-subject MMLU-Pro 结果也有意义。Llama3.1-8B 的 overall accuracy 中，AuxDPO 在 OOD 为 39.26、ID 为 51.95，高于 DPO 的 27.06 和 46.60。多个学科上 AuxDPO 都更强，尤其 OOD 提升更明显。这支持作者关于 generalization 的说法：如果 DPO 的偏差来自训练 preference frequency 和 policy geometry 的交互，那么在跨域评测时普通 DPO 更容易把投影偏差带出去；AuxDPO 通过额外自由度可能减少这种错误投影。

Bandit example 对理解论文更关键。作者展示，在三 response 设置里，DPO 会因 pair count imbalance 出现 preference reversal 和 reward decrease；AuxDPO 能保持正确 ordering，并提高平均 reward。这个 toy experiment 的意义不在规模，而在把 Proposition 3 的 failure mode 具体化：DPO 的问题不是没有足够数据，而是 projection geometry 本身可能错。

实验也报告了实现细节和开销。AuxDPO 作为 TRL DPOTrainer 的扩展，只多维护样本级 $\delta$ 和 nullspace penalty；总参数规模仍是 $d+2n=O(d)$，因为 LLM 参数数远大于数据样本辅助变量。这个说法在大模型训练里大体合理，但实际系统仍要考虑 dataset-size-dependent state、distributed training 下 auxiliary variables 的同步，以及 $\lambda$ 这类 penalty hyperparameter 的稳定性。

从实验可信度看，论文的强项是跨模型、跨 ID/OOD setting 都有一致提升；弱项是 benchmark 仍偏小，且没有完整覆盖真实 chat preference、安全偏好、多轮反馈等复杂场景。RewardBench V2 是 preference benchmark，但规模只有约 1.87K prompts；MMLU-Pro 则是被转化成 preference 形式的 QA benchmark。它们足以支持“AuxDPO 有潜力”，还不足以支持“DPO 在真实大规模 alignment 中都应该被 AuxDPO 替代”。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。三个 reviewer 的分数是 6、8、6，其中一个 reviewer 在 rebuttal 后表示会把 6 提到 8。AC 的 meta-review 很简洁：所有 reviewer 都认为观察新颖有趣，主要建议是降低符号密度、增加动机和直觉解释。

正面评价集中在理论贡献。Reviewer 认可论文把 DPO under parametric policies 解释成 misspecified reward projection，并且用 local geometry 同时连接 DPO 和 RLHF。尤其是 Proposition 3 的例子很有杀伤力，因为它排除了“只是有限数据或优化失败”的借口。Reviewer 也认为 AuxDPO 的 null-space auxiliary variable 是有原则的修正，而不是纯经验 loss trick。

批评主要有三类。第一，论文符号非常重，第一次读不容易跟上；$A_{\theta_0}$、$A_{\rho,\theta_0}$、Fisher matrix、equivalence class、nullspace variable 之间的关系需要更强的解释。第二，AuxDPO 的理论保证依赖 local approximation 和 sufficiently large $\beta$，也就是 policy 不能离 reference 太远。这个条件在实际 post-training 中并不总是显式满足，所以 AuxDPO 的理论更像局部可信解释，不是全局收敛保证。第三，实验里 MMLU-Pro 的 preference 转换、RewardBench 的规模、以及 $\lambda$ sensitivity 都需要更多说明。

我的客观评述是：这篇论文的价值主要在 **诊断 DPO 的理论幻觉**。很多时候我们说 DPO “equivalent to RLHF”，其实默认了 policy class 足够表达 tabular optimum。论文把这个默认条件拆掉之后，DPO 变成一个投影估计器，而且这个投影会被 preference pair 频率牵引。这个结论对 alignment 研究很重要，因为真实偏好数据永远不均衡，模型表达能力也永远受限。

AuxDPO 是否会成为主流方法还需要更多实证，但论文提出的问题一定会留下。我的批评会比较直接：当前实验还不足以证明 AuxDPO 是大规模 RLHF/DPO 的通用替代品；它更像一个理论上有力、工程上可试的 correction。真正的下一步应该在更真实的 instruction-following preference、safety preference、multi-turn feedback 和 larger model scale 上检验，尤其要看 auxiliary variables 是否稳定、是否会过拟合数据集 pair frequency。

Reviewer 对可读性的批评也很合理。论文的数学路线清楚，但符号压缩很强，尤其是 DPO implicit reward manifold、RLHF equivalence class 和 nullspace correction 三者之间的转换，第一次读很容易丢。对后续读者而言，最应该抓住的不是每个矩阵的细枝末节，而是主因果链：参数化 policy 限制了可表达 reward；DPO 把真实 reward 投影到这个受限空间；两阶段 RLHF 只关心能诱导同一 policy update 的 reward equivalence class；AuxDPO 通过 nullspace variable 扩大搜索空间，让投影有机会落回正确 equivalence class。

## 6. Related Work & Future Work

这篇论文接在 DPO、IPO、DPOP、Nash-style preference optimization，以及对 DPO gradient dynamics 的批评之后。和 likelihood displacement 相关工作不同，它不只看单步 gradient，而是分析 DPO population loss 的 minimizer；这让它的 failure mode 更强，因为即使 optimization 完美、数据无限，问题仍可能存在。

对 safety alignment 来说，这篇的意义在于提醒我们不要把 preference objective 当成黑箱。SafeDPO 关心安全约束怎样重排 pair，MNPO 关心多 player preference game 怎样优化，而这篇关心的是：给定某个 direct objective，它在参数化 policy class 中到底估计了什么。三者放在一起，alignment 的核心就从“找一个更好 loss”推进到“理解 preference data、policy geometry、reward identifiability 和 evaluation oracle 的联合结构”。
