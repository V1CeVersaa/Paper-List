---
title: "Divergent Interventions"
headline: "Addressing Divergent Representations from Causal Interventions on Neural Networks"
description: "ICLR 2026 Oral note on representational divergence caused by causal interventions in mechanistic interpretability, including harmless null-space cases, pernicious hidden pathways, and CL-loss mitigation."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2511.04638"
openreview_url: "https://openreview.net/forum?id=cZrTMqYVL6"
source_pdf: "raw/papers/arxiv-2511.04638.pdf"
tags: ["causal_interventions", "mechanistic_interpretability", "activation_patching"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文讨论 mechanistic interpretability 中一个很基础但经常被跳过的假设：当我们做 activation patching、mean-difference patching、SAE reconstruction intervention 或 Distributed Alignment Search/DAS 时，构造出来的 counterfactual representation 是否仍然像目标模型自然运行时会产生的内部状态。作者把偏离自然分布的 intervened states 称为 **divergent representations/发散表征**，先证明 coordinate patching 在一般 manifold 上很容易 off-manifold，再实证展示 mean-difference patching、SAE 和 Boundless DAS 都会产生明显 divergence。
>
> 论文最有价值的部分是把 divergence 分成两类：落在 **behavioral null-space/行为零空间** 里的 divergence 可能对某些功能性结论 harmless；但 off-manifold activation 也可能招募自然状态中不会启动的 **hidden pathways/隐藏通路**，或制造只在某些上下文中爆发的 **dormant behavioral changes/休眠行为变化**。作者用 Counterfactual Latent/CL loss 降低 DAS intervention 的 divergence，并在合成任务上显示更好的 OOD interchange intervention accuracy。边界也很硬：pernicious divergence 的识别仍没有通用判据，CL loss 是 broad-stroke mitigation，真实 LLM 上的验证还偏初步。

## 1. Introduction

Causal intervention 是 mechanistic interpretability 的核心工具。研究者常常把一个样本的 activation patch 到另一个样本里，或者沿某个 feature direction 推动 activation，然后观察输出是否改变。如果输出按预期改变，就说这个 activation、subspace 或 feature 对某个行为有因果作用。这个逻辑看起来直接，但它隐藏了一个前提：**intervened representation 必须是模型自然分布里合理的内部状态**。如果 patch 后的向量落在模型从来不会自然访问的区域，那么输出变化可能来自 off-manifold artifact，而不是模型真实机制。

这篇论文正面研究这个前提。作者指出，许多现有 intervention 方法默认 counterfactual state 是可解释的，但很少报告它离 natural representation distribution 有多远。这个问题对 alignment 很重要，因为很多安全解释方法都依赖 intervention：我们想知道某个 refusal direction 是否真的控制拒答，某个 SAE feature 是否真的代表 harmful intent，某个 circuit 是否真的实现 deception monitoring。如果 intervention 激活了自然运行中不存在的隐藏路径，解释结论就可能很脆。

论文没有简单地说“所有 divergence 都不好”。这是它成熟的地方。模型的高维表示中确实可能有很多 functionally irrelevant directions；如果 intervention 只在这些方向上偏离，而后续 computation 完全不读这些偏离，那么 divergence 对当前功能性 claim 可能无害。真正危险的是 divergence 进入某些自然样本从不访问、但 downstream layers 仍会响应的区域。这种状态可能给出看似 confirmatory 的行为结果，却不是自然机制的一部分。

因此论文的问题不是“activation patching 能不能用”，而是“activation patching 的因果结论在什么条件下可信”。这个 framing 很有杀伤力，因为它把很多 mechanistic interpretability 论文里的最后一步验证重新拉回到 distributional faithfulness 上：一个 intervention 不仅要改变输出，还要说明自己没有用模型自然状态之外的捷径改变输出。

## 2. Problem Setup

论文讨论的基本对象是某层 hidden representation $h\in\mathbb{R}^{d_m}$。在 DAS 中，研究者学习一个可逆线性 alignment function：

$$
z=\mathcal{A}(h)=Wh,
$$

把 $h$ 转到一个假设可解释的 basis 中。这个 $z$ 被拆成多个 causal variable subspaces 和一个 extra/null subspace。对变量 $\mathrm{var}_i$ 做 interchange intervention 时，从 source representation 取出对应 subspace，替换 target representation 中的对应 subspace，再通过 $\mathcal{A}^{-1}$ 转回原空间：

$$
\hat{h}=\mathcal{A}^{-1}\left((I-D_{\mathrm{var}_i})\mathcal{A}(h^{\mathrm{trg}})+D_{\mathrm{var}_i}\mathcal{A}(h^{\mathrm{src}})\right).
$$

如果模型在 $\hat{h}$ 下输出正确的 counterfactual label，Interchange Intervention Accuracy/IIA 就高。传统 DAS 主要看 IIA；这篇论文追问的是，$\hat{h}$ 是否仍然处在 natural representations 的合理分布附近。若 $\hat{h}$ 偏离对应 class 或 context 的 support，作者就称为 divergent representation。

论文首先用一个二维圆形 manifold 说明 coordinate patching 为什么天然容易制造 divergence。设某类 representation 位于以 $c_K$ 为中心、半径 $r_K$ 的圆盘 $\mathcal{M}_K$ 内。取两个自然点 $h^{\mathrm{src}}=c_K+u$ 和 $h^{\mathrm{trg}}=c_K+v$，patch 后保留 source 的第一坐标和 target 的第二坐标：

$$
\hat{h}-c_K=(u_1,v_2)^\top.
$$

只要 $u_1^2+v_2^2>r_K^2$，patch 后点就离开圆盘。边界例子很直接：$u=(r_K,0)$，$v=(0,r_K)$ 时，$\|\hat{h}-c_K\|_2=r_K\sqrt{2}>r_K$。作者在附录中把直觉推广到更一般的 manifold：除非 manifold 形状近似 axis-aligned hyperrectangle，穷举 coordinate patching 总会产生 off-manifold points。

接着作者定义 **behavioral null-space**。对函数 $\psi:\mathbb{R}^d\to\mathbb{R}^{d'}$ 和自然表示集合 $X$，若一个偏移 $v$ 满足

$$
\forall x\in X,\quad \psi(x+v)=\psi(x),
$$

那么 $v$ 属于 $N(\psi,X)$。如果 downstream computation 完全不读这个方向，divergence 在当前 claim 下可能 harmless。注意这里的 harmless 是 claim-dependent：如果你的 claim 只关心最终输出，它可能无害；如果你的 claim 关心某个子层或子电路的内部机制，同一个 divergence 可能就不无害。

## 3. Algorithm / Methods / Model

论文的方法部分由三块组成。第一块是 empirically measuring divergence。作者把 natural representations 和 intervened representations 放在同一空间中，用 PCA 可视化，并用 Earth Mover's Distance/EMD 衡量两组分布距离，同时用 natural-vs-natural 的 EMD 作为 baseline。实证对象包括 mean-difference vector patching、SAE reconstruction intervention 和 Boundless DAS。结果显示三者都会产生高于 baseline 的 representational divergence；这不自动否定这些方法，但说明 divergence 是真实存在的、不是一个纯理论担忧。

第二块是理论分类。Harmless divergence 的典型情况是零空间。若某层后续权重矩阵 $W$ 满足 $Wv=0$，那么 $W(h+v)=Wh$，对这次矩阵乘法的整体输出无影响。但论文特别强调，$v\in N(W)$ 不代表所有机制 claim 都安全，因为单个 activation-weight sub-computation 仍然改变了。这个区分很关键：mechanistic interpretability 经常不只关心最终行为，还关心内部路径。

Pernicious divergence 的第一种形式是 **hidden pathway**。作者构造一个两层 ReLU circuit，class A 和 class B 的自然 representations 都只激活某些自然路径；mean-difference patching 从 B 推向 A 时，虽然输出成功翻到 A，却激活了一个 class A 自然样本从未激活过的第三个 hidden unit。也就是说，intervention 看起来证明了“这个 direction 导致 A 行为”，但实际是通过一条 off-manifold hidden pathway 达成的。把 $\hat{h}$ 投影回 class-A natural convex hull 后，这个 ReLU 状态变化消失，说明原效果来自 divergence。

Pernicious divergence 的第二种形式是 **dormant behavioral change**。一个 intervention 可能在当前 context 子集 $C_1$ 上看起来 behaviorally null，但在更大 context set $C$ 中改变行为。形式上，作者定义

$$
V(\psi,X,C_1,C)=N(\psi,X,C_1)\setminus N(\psi,X,C).
$$

这描述了“在测试过的上下文里没事，但在未测试上下文中会出事”的偏移。这个概念对 safety 很敏感，因为很多 intervention evaluation 都只覆盖有限 prompt；如果 divergent state 打开了某个 dormant pathway，它可能只在特定工具状态、输入分布或上下文变量下触发。

第三块是 mitigation。作者使用 Counterfactual Latent/CL loss。CL vector 是自然样本中与目标 counterfactual causal variables 一致的表示均值。对 intervened state $\hat{h}$，CL loss 同时最小化它和 $h_{\mathrm{CL}}$ 的 L2 距离并最大化 cosine similarity：

$$
\mathcal{L}_{\mathrm{CL}}(\hat{h},h_{\mathrm{CL}})
=\frac{1}{2}\|\hat{h}-h_{\mathrm{CL}}\|_2^2
-\frac{1}{2}\frac{\hat{h}\cdot h_{\mathrm{CL}}}{\|\hat{h}\|_2\|h_{\mathrm{CL}}\|_2}.
$$

在 Boundless DAS 中，作者把它作为 auxiliary loss：

$$
\mathcal{L}_{\mathrm{total}}=\epsilon\mathcal{L}_{\mathrm{CL}}+\mathcal{L}_{\mathrm{DAS}}.
$$

他们还提出 modified CL loss，只在 DAS 发现的 causal subspaces 上对齐 CL vectors，而不是惩罚所有维度。这一步是为了避免把所有 harmless divergence 都粗暴压掉，但论文也承认它仍不是识别 pernicious divergence 的最终方法。

## 4. Experiments

第一组结果展示 divergence 的普遍性。Figure 2 中，mean-difference patching、SAE、Boundless DAS 的 natural/intervened representations 在 PCA 空间里都有明显偏移，EMD 也超过 natural-natural baseline。这一组实验的意义不是说这些方法的所有结论都错，而是把“intervention 可能 off-manifold”从口头担忧变成可测现象。

第二组是 Boundless DAS 上的 CL auxiliary loss。作者在 Wu et al. 的 Llama-based Boundless DAS setting 上加入 CL loss，并观察 IIA 和 EMD 随 $\epsilon$ 变化。结果显示，在较小 $\epsilon$ 下，IIA 保持甚至可能略升，同时 EMD 下降。也就是说，至少在这个设置中，降低 representational divergence 不必牺牲 intervention 的行为准确率。这给出了一个很实用的信号：研究者可以把 divergence 作为训练和报告指标，而不只是事后担心。

第三组是 synthetic MLP task。作者构造十个 class 的二维 causal feature grid，加噪声和额外 noise dimensions，训练 MLP，然后比较用 behavior-only DAS loss 和 CL-only/modified CL loss 训练出的 alignment。结果中，behavior-only DAS 在 causal feature dimensions 上 EMD 为 $0.032\pm0.003$，CL loss 为 $0.007\pm0.001$；IIA 分别为 $0.997\pm0.001$ 和 $0.9988\pm0.0005$。这说明 CL loss 可以让 intervention 更贴近 natural causal manifold，同时不降低 in-distribution intervention accuracy。

更重要的是 OOD transfer。作者把 synthetic classes 分成 dense 和 sparse partitions，在一个 partition 上训练 alignment，在另一个 partition 上测试。CL loss 在 OOD IIA 上优于 behavioral loss。进一步地，把 training EMD 回归到 OOD IIA，得到负相关，系数约为 $-0.34$，$R^2=0.73$，$p<0.001$。这个结果支持一个很直观但重要的结论：训练时 intervention 越 divergent，跨分布 causal intervention 越不可靠。

但实验边界也很明显。Pernicious hidden pathway 和 dormant behavior 的核心例子主要是小型构造性 circuit；modified CL loss 的强结果来自 synthetic MLP；真实 LLM 上主要展示的是 CL loss 降低 Boundless DAS divergence 且保持 accuracy，而不是大规模证明“pernicious divergence 在真实 transformer 中频繁导致错误解释”。所以这篇论文更像 **existence and warning paper**：它证明问题足够真实、机制足够危险、初步 mitigation 有希望，但还没有完成实践闭环。

从实验设计看，论文最重要的缺口是缺少“真实解释结论被 divergence 误导”的批量案例。现在 Figure 2 证明 intervention 确实 divergence，Section 4 证明 divergence 在构造电路中可以 pernicious，Figure 3 证明 CL loss 能降低 divergence；但中间仍少一步：在真实 transformer 的某个具体 mechanistic claim 上，原 intervention 给出错误因果解释，manifold-preserving 或 CL-regularized intervention 修正了这个解释。这个缺口不削弱论文的警示价值，却限制了它作为方法论文的闭合度。

另一个值得注意的实验点是指标选择。EMD 衡量 natural/intervened distributions 的整体距离，但它不告诉我们偏移是否落在 downstream computation 会读取的方向上。两个 intervention 可以有相同 EMD，一个只在行为零空间漂移，另一个穿过 ReLU 或 attention routing 的关键边界。因此 EMD 更适合作为 early warning signal，而不是 perniciousness score。论文在理论部分已经承认 claim-dependent harmlessness；实验部分如果未来加入 local Jacobian、activation-pattern change 或 context-sensitivity tests，会更贴近真正的风险。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。五位 reviewer 的初始分数大致是 8、6、4、4、4，其中一位 reviewer 明确提高到 accept，AC 认为作者较好回应了 formalization 和 assumptions 相关问题，但 practical utility 与真实场景适用性仍有 outstanding concerns。最终 oral 的理由不是“所有担忧都解决了”，而是该问题会影响大量 contemporary interpretability research，proofs-of-existence 足够有价值。

正面评价集中在问题重要性。Reviewer 普遍认可，causal intervention 的 distributional faithfulness 是 mechanistic interpretability 中被低估的核心假设；harmless/pernicious divergence 的词汇体系也有价值，因为它让研究者能更精确地讨论“off-manifold intervention 什么时候影响结论”。高分 reviewer 认为论文提出的问题会直接影响 activation patching、DAS、SAE intervention 等一大类方法。

批评也非常强。最低分 reviewer 认为 CL loss 的解决方案主要在 synthetic 2D task 上验证，和真实 transformer 中的复杂 causal variables 差距很大；另一个 reviewer 质疑 pernicious examples 过于人工，需要精心设置权重矩阵、ReLU 边界和数据分布，未必代表真实模型中的常见情况；还有 reviewer 指出 harmless/pernicious 的定义偏叙述化，缺少必要充分条件或实用分类器。

AC 的判断比较平衡：论文没有证明现实中 pernicious divergence 很普遍，也没有给出最终 mitigation，但它足以证明 intervention researchers 应该更直接、更频繁地检查 divergence。这个评审结论我认为很准确。把这篇论文当成“解决 divergent interventions 的方法论文”会高估它；把它当成“要求整个领域补上 distributional faithfulness 报告的警示论文”则非常有价值。

我的客观评述是：这篇 oral 最强的贡献是**提高了 mechanistic interpretability 的证据标准**。过去很多 patching 论文只要 behavior flip 就很容易声称找到了机制；这篇论文要求额外回答：你构造的 counterfactual state 是否还在自然分布附近？如果不在，你的 claim 是只关心封装后的输出，还是声称解释了内部自然路径？这个要求会让很多解释结论更克制，但也更可信。

我最担心的是 CL loss 的可扩展性和 circularity。CL vector 需要知道哪些自然 states 具有同样 causal variable values；但真实 LLM 中，causal variables 往往正是我们不知道、正在寻找的对象。如果先验 causal abstraction 不可靠，CL vectors 也可能把 intervention 拉向错误 manifold。论文承认这是 broad-stroke solution，因此这不是致命缺陷，但它意味着 CL loss 目前更适合已有清晰 causal abstraction 的 setting，而不是任意 SAE feature 或 open-ended safety behavior。

同时也要避免走向另一个极端：不能因为 divergence 可能存在，就否定所有 off-manifold intervention。很多科学实验本来就是把系统推到自然状态之外，以测试 sufficiency、robustness 或 boundary behavior。关键是 claim 要写清楚。如果 claim 是“这个方向可以控制输出”，off-manifold steering 也许足够；如果 claim 是“模型自然使用这个机制”，就必须报告 divergence 并证明 intervention 没有招募隐藏通路。这篇论文真正推动的是 claim hygiene，而不是禁止干预。

对 safety alignment 来说，这篇论文值得重读。很多安全方法依赖 activation steering 或 feature intervention，但 steering 成功不等于解释正确，甚至不等于机制自然。Divergent intervention 可能让模型产生安全输出，却通过非自然路径实现；这对部署未必坏，但对科学解释很危险。如果我们想用 activation feature 证明“模型有某个 deception circuit”或“某个 refusal direction 是自然安全机制”，这篇论文要求我们先检查 intervention 是否越界。

## 6. Related Work & Future Work

这篇论文和 activation patching、causal scrubbing、DAS、causal abstraction、SAE intervention、counterfactual explanations、on-manifold adversarial examples 放在同一组。和普通 activation patching 指南相比，它更关注 patch 后表示的 distributional status；和 causal abstraction work 相比，它强调高 IIA 不足以保证自然机制解释；和 SAE steering work 相比，它提醒 feature intervention 可能产生 out-of-distribution behavior。

未来最重要的方向是 **practical divergence diagnostics**。论文提出 behavioral null-space 的形式化，但真实 transformer 中 $\psi$ 是高度非线性的，support 也不可精确枚举。实用工具可能包括 local Jacobian null-space estimation、nearest-neighbor manifold distance、class-conditional density models、ReLU/MLP activation pattern audits、intervention sensitivity across contexts，以及 intervention 后的 downstream activation distribution shift。

第二个方向是 **pernicious vs harmless classification**。只报告 EMD 不够，因为大 divergence 可以 harmless，小 divergence 也可能触发关键边界。真正需要的是 claim-conditioned risk analysis：对某个 mechanistic claim，哪些 downstream computations 被允许改变，哪些改变会破坏 claim。这个方向会把 interpretability evaluation 从单一 accuracy 指标推进到更细的 causal evidence audit。

第三个方向是把这套思想用于 activation steering 和 safety interventions。当前很多 steering 工作只看行为是否变好，例如减少有害输出、增强拒答或改变 persona。Divergent representation 视角会问：steering 后模型是否进入非自然区域？如果进入了，这种非自然区域是否稳定、可控、会不会带来 dormant side effects？这对高风险部署比单次行为改善更重要。

第四个方向是建立 reporting standard。未来的 patching 或 steering paper 可以至少报告三个量：intervened states 与目标自然分布的距离、intervention 后关键 downstream activation pattern 是否偏离自然模式、以及在多个 context slices 上行为是否稳定。这样的标准不需要立刻解决 harmless/pernicious classification，但能让读者判断证据强度。对本仓库后续读这类论文，这个标准也可以作为 reviewer checklist。

最后，这篇还提示我们重新区分 **control** 和 **understanding**。如果目标只是让模型行为改变，divergent steering 可能足够；如果目标是理解模型自然机制，divergence 就必须被当作证据污染源。很多 alignment 工具同时声称二者，这正是最容易混淆的地方。

因此后续阅读同类工作时，最好先问作者到底在证明控制接口有效，还是证明模型原本就用这条机制。
