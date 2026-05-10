---
title: "SafeDPO"
headline: "SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety"
description: "ICLR 2026 Oral note on SafeDPO, a direct preference optimization objective for hard-constrained safety alignment."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2505.20065"
openreview_url: "https://openreview.net/forum?id=PJdw4VBsXD"
source_pdf: "raw/papers/arxiv-2505.20065.pdf"
tags: ["direct_preference_optimization", "safety_constraints", "over_refusal"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **SafeDPO** 试图把 safety alignment 从多阶段 SafeRLHF 管线里拉回到一个 DPO-style 的单阶段目标。论文从 hard-constrained safety objective 出发，把 unsafe response 的 reward 设成 $-\infty$，推导出一个会把 unsafe response 从 optimal policy support 中排除的 closed-form policy；紧接着作者证明，这个理论上依赖 latent cost-augmented reward 的目标，可以通过对已有 preference pair 做 **safety-aware transformation** 来等价估计。
>
> 这篇 oral 的关键价值在于它给出了一个非常干净的 objective-level 解释：如果目标真的是“在安全约束下最大化 helpfulness”，那么安全信息不应该只作为额外 cost model 或 expected-cost penalty，而应该改变 pairwise preference 的排序规则。边界也很清楚：SafeDPO 依赖 binary safety indicator，丢弃 unsafe-unsafe pairs，并且在 XSTest 上出现更高 over-refusal；它更像 strict safety alignment 的简洁理论原型，而不是已经解决 helpfulness-safety trade-off 的部署方案。

## 1. Introduction

DPO 的吸引力在于它把 RLHF 的 reward model training 和 policy optimization 合并成一个监督式 preference loss。问题是，普通 DPO 只处理“哪个回答更被偏好”，它并不显式区分“preferred 但 unsafe”的情况。真实 safety alignment 里，这个区别很致命：一个回答可能更直接、更满足用户有害请求，因此在 helpfulness preference 上胜出，但安全目标要求模型不能把这种回答变成高概率输出。

已有 SafeRLHF、SACPO、CAN 这类方法通常把安全性写成 cost 或 constraint，然后用辅助模型、多阶段优化、PPO-style update 或 relaxed objective 处理。它们有效，但训练管线变重，而且很多方法优化的是 expected cost constraint。**Expected-cost relaxation** 的问题在于它允许少量 unsafe probability mass 留在策略里；在高风险任务中，“平均成本低”并不等价于“unsafe response 被排除”。

SafeDPO 的切入点非常直接：回到 hard-constrained objective 本身。作者问的是，如果我们真的要求所有被 policy 采样到的 response 都满足 $c(x,y)\le 0$，那么 DPO 风格的直接优化还能不能推出来？论文的答案是可以，但前提是 preference 数据要按照安全约束重新解释。换句话说，SafeDPO 不只是给 DPO 加一个 safety penalty，它先改变 pair 的胜负关系，再用 DPO loss 训练。

这个想法对 alignment 很重要，因为它把 **safety hierarchy** 写进了训练信号：safe and helpful response 应该优于 safe but less helpful response，而任何 unsafe response 都应该排在 safe response 之后。这个层级在安全任务上合理，但也提示了方法的边界：它适合有明确 forbidden region 的安全约束，不一定适合 conciseness、creativity、truthfulness、political neutrality 这类互相冲突且难以线性排序的多目标偏好。

## 2. Problem Setup

论文的基本数据项是 $(x,y_w,y_l,h_w,h_l)\sim D$。其中 $x$ 是 prompt，$y_w$ 和 $y_l$ 分别是按 helpfulness preference 标出的 winner 和 loser；$h_w,h_l$ 是 binary safety indicator。论文的记号里 $h=1\{c(x,y)>0\}$，也就是 $h=1$ 表示 unsafe，$h=0$ 表示 safe。安全约束写成 $c(x,y)\le 0$，含义是 policy 允许生成的 response 必须落在 safe set 内。

普通 RLHF 或 DPO 关心的是 KL-regularized reward maximization。SafeDPO 关心的是带硬安全约束的版本：

$$
\max_\theta \mathbb{E}_{x\sim D,y\sim \pi_\theta(\cdot|x)}
\left[
r(x,y)-\beta D_{\mathrm{KL}}(\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x))
\right],
\quad
\text{s.t. } c(x,y)\le 0.
$$

这里 $r(x,y)$ 表示 helpfulness reward，$\beta$ 控制 policy 偏离 reference policy 的程度。硬约束的含义不是“unsafe response 平均少一点”，而是 unsafe response 在 optimal policy 下应该拿到零概率。为此作者定义 **cost-augmented reward**：

$$
r_c(x,y)=
\begin{cases}
r(x,y), & c(x,y)\le 0,\\
-\infty, & c(x,y)>0.
\end{cases}
$$

把 $r_c$ 代入 KL-regularized objective 后，closed-form optimal policy 是：

$$
\pi^*(y|x)=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y|x)
\exp\left(\frac{1}{\beta}r_c(x,y)\right).
$$

因为 unsafe response 的 $r_c=-\infty$，它的指数权重为零，所以 $\pi^*(y|x)=0$。这一步是 SafeDPO 理论叙事的核心：安全约束不是通过一个软惩罚慢慢压低 unsafe response，而是在最优解层面把 unsafe support 直接删掉。

这里有一个容易被忽略的可行性假设。论文要求每个 prompt 下 reference policy 对 safe set 仍然有非零概率质量，也就是存在足够多的 safe response 可供 policy 重新分配。如果某个 prompt 的所有可见 response 都 unsafe，或者 reference policy 几乎不给 safe response 概率，那么 hard-constrained optimum 会变得不可训练或退化。这个假设在 PKU-SafeRLHF 这种有安全标注的数据集里大体成立，但在真实开放域数据里并不自动成立；这也是为什么 rebuttal 里会讨论用 refusal template 或 safe anchor 补足 unsafe prompt 的对比。

更细地看，SafeDPO 把 helpfulness preference 和 safety indicator 分成两个层次。Helpfulness 决定 safe response 内部怎么排序，safety indicator 决定 safe response 和 unsafe response 之间的绝对优先级。这个层级化设置很强：一旦 response 被标成 unsafe，它在 safe response 面前没有“虽然危险但很有帮助”的讨价空间。对安全研究来说，这个强假设是优点，因为它避免模型把有害请求的顺从性误学成 helpfulness；对通用 alignment 来说，它也是限制，因为很多价值冲突并没有这么清楚的 lexicographic order。

## 3. Algorithm / Methods / Model

真正的难点是，$r_c$ 依赖未知的 reward 和 cost function，不能直接构造由 $r_c$ 诱导的 theoretical preference distribution。SafeDPO 的关键技术动作是证明：虽然这个 distribution 不可见，但它对 pairwise preference 的影响可以由原始数据上的确定性变换 $T$ 表达出来。

变换 $T$ 的规则很简单。如果 helpfulness winner $y_w$ 本来就是 safe，即 $h_w=0$，就保留 $(x,y_w,y_l)$。如果 $y_w$ unsafe 而 $y_l$ safe，即 $h_w=1,h_l=0$，就把 pair 反过来，训练模型偏好 $y_l$。如果两个 response 都 unsafe，即 $h_w=h_l=1$，就丢弃这个 pair，因为 hard-constrained optimal policy 不应该在两个 unsafe response 之间继续学习“哪个更好”。这个规则把安全约束变成了一个新的 pairwise ordering。

这个 transformation 看起来像 preprocessing，但论文想强调它不是任意启发式。它来自 $r_c$ 的 Bradley-Terry preference：只要一个 response safe、另一个 unsafe，safe response 的 $r_c$ 有限而 unsafe response 是 $-\infty$，因此 safe response 的胜率就是 1。两个 response 都 safe 时，排序回到原始 helpfulness reward；两个都 unsafe 时，它们都不应该出现在 optimal policy support 里，所以 pair 对安全约束的目标没有正向信息。这个推导解释了为什么 SafeDPO 不是“把安全样本多采一点”，而是把 hard constraint 的最优解结构压进数据排序。

在实现层面，这也说明 SafeDPO 的数据需求和普通 DPO 不同。普通 DPO 只要 chosen/rejected；SafeDPO 还需要每个 response 的二元 safety label。这个 label 不必是 pairwise harmlessness preference，但必须能判断 response 是否越过安全边界。它比训练完整 cost model 轻，但绝对不是免费信号。若 safety label 噪声较大，尤其是把 benign refusal、safe educational answer、policy-compliant warning 混在一起，$T(D)$ 会把错误 pair 反转，造成系统性偏差。

在 transformed distribution $T(D)$ 上，SafeDPO 直接使用 DPO-style loss：

$$
\mathcal{L}_{\mathrm{SafeDPO}}(\theta)
=
-\mathbb{E}_{(x,\tilde y_w,\tilde y_l)\sim T(D)}
\log \sigma\left(
\beta \log \frac{\pi_\theta(\tilde y_w|x)}{\pi_{\mathrm{ref}}(\tilde y_w|x)}
-
\beta \log \frac{\pi_\theta(\tilde y_l|x)}{\pi_{\mathrm{ref}}(\tilde y_l|x)}
\right).
$$

论文的 Proposition 4.3 说明，这个 tractable objective 等价于在 $r_c$ 诱导的 theoretical preference distribution 上优化。这个等价性使 SafeDPO 不需要 reward model、cost model、value function，也不需要 online sampling。训练形式仍然像普通 DPO，只是数据 pair 被安全规则重新排序。

作者进一步引入 **safety margin** $\Delta$。当 transformed pair 是 safe response 对 unsafe response 时，loss 里额外加入 $(\tilde h_l-\tilde h_w)\Delta$，要求模型把 safe response 和 unsafe response 的 log-ratio gap 拉得更大：

$$
\mathcal{L}_{\mathrm{SafeDPO}}(\theta;\Delta)
=
-\mathbb{E}_{T(D)}
\log \sigma\left(
\beta \log \frac{\pi_\theta(\tilde y_w|x)}{\pi_{\mathrm{ref}}(\tilde y_w|x)}
-
\beta \log \frac{\pi_\theta(\tilde y_l|x)}{\pi_{\mathrm{ref}}(\tilde y_l|x)}
-
(\tilde h_l-\tilde h_w)\Delta
\right).
$$

这个 margin 的作用更像 optimization amplifier，而不是改变目标本身。Proposition 4.4 说明任意 $\Delta\ge 0$ 不改变 global optima；实验里 $\Delta$ 影响有限训练过程中的安全强化强度。这个结论很漂亮，但要谨慎理解：global optimum invariance 不保证有限训练时不会损害 helpfulness，论文也观察到过大的 $\Delta$ 会让 helpfulness 退化。

论文的理论部分可以概括成三层保证。Proposition 4.2 说明，在 safe response 可行且 penalty 趋于无穷时，reduced objective 和原 hard-constrained problem 的最优解在 total variation 意义下收敛。Proposition 4.3 说明，不可见的 $r_c$ preference objective 可以被 $T(D)$ 上的 SafeDPO objective 精确恢复。Proposition 4.4 说明，$\Delta$ margin 不改变 global optimum。三者合起来给出一个闭环：hard constraint 先转成 cost-augmented reward，再转成 safety-aware pair transformation，最后用 margin 强化优化信号。

但这个闭环也暴露了 SafeDPO 的主要假设链。它假设 binary safety boundary 足够可靠，safe set 内仍然有 helpfulness 差异可学，且丢弃 unsafe-unsafe pair 不会让关键 prompt 缺少训练信号。任何一个条件变弱，SafeDPO 都需要额外工程补丁。例如在 jailbreak 数据中，很多模型 response 可能都 unsafe；直接丢弃会让模型没有学到“应该拒绝或给安全替代方案”。这时 safe anchor 的质量会决定训练是否真的学到安全策略。

## 4. Experiments

主实验使用 **PKU-SafeRLHF-30K**，每个样本包含 helpfulness preference、harmlessness preference 和 binary safety indicator。作者比较了 DPO-HELPFUL、DPO-HARMLESS、DPO-SAFEBETTER、SafeRLHF、SACPO、P-SACPO 和 SafeDPO。DPO-SAFEBETTER 是一个很关键的 baseline：它只保留 preferred response safe 的 pair，用来检验“简单过滤数据”是否已经足够。

结果显示，简单过滤并不够。SafeDPO 在 model-based evaluation 下 harmless ratio 约为 97%，在 GPT-4 evaluation 下达到 100%；DPO-SAFEBETTER 的 harmless ratio 明显低得多。这支持了作者的主张：安全性不能只靠删除一部分坏 pair，必须让 safe-over-unsafe 的排序进入 objective。

Helpfulness 的结果需要分开读。自动评测里 SafeDPO 的 helpfulness 仍然有竞争力，GPT-4 judge 甚至给出最高 helpfulness。但作者在 appendix 和 human evaluation 中指出，GPT-style judge 可能把 safety 也算进 helpfulness，从而高估安全模型的信息性。人工评测更冷静：SafeDPO 的 safety score 是 0.943，略高于 SafeRLHF 的 0.932；但 helpfulness 只有 0.499，接近 SafeRLHF 的 0.497，明显低于 SFT 的 0.868。这说明 SafeDPO 确实更安全，但它也继承了 safety training 常见的“回答更保守、信息性下降”的问题。

XSTest 是这篇实验里最值得单独看的一块。SafeDPO 在 unsafe prompts 上 harmless ratio 达到 100%，但 benign prompts 上 over-refusal 为 12.4%。相比之下，SafeRLHF、SACPO、P-SACPO 的 over-refusal 大约在 1.2% 到 3.2% 之间，harmless ratio 约 84% 到 86%。这个结果很诚实地暴露了 hard-constrained formulation 的代价：它更接近“绝不越界”的安全策略，但面对含有危险词、实际意图却 benign 的 prompt 时，会更容易拒答。

模型规模实验覆盖 1.5B 到 13B。SafeDPO 在多个 reference model 上都保持约 95.5% 到 97.9% 的 harmless ratio，说明方法不是只在单一 7B 模型上偶然有效。$\Delta$ ablation 也显示，$\Delta=0$ 时已经有强 safety，增加 margin 后安全性仍稳定；但极大 margin 会影响有限训练动态。这和理论一致：真正关键的是 $T(D)$ 的安全重排，margin 只是把 safe-vs-unsafe pair 的梯度推得更强。

论文还专门讨论了 evaluator overoptimization。SafeRLHF 如果用与其训练中 reward/cost model 分布很接近的 beaver evaluator 来评测，分数可能接近饱和，看起来明显优于其他方法；但换成 unified evaluator、GPT judge 和人工评测后，这种优势不再稳固。这个分析很重要，因为 safety alignment 很容易出现“训练目标和评测器同源”的虚假领先。SafeDPO 的实验可信度部分来自作者主动指出这一点，而不是只报告对自己有利的 evaluator。

Pairwise GPT evaluation 也补充了 aggregate metric 的不足。论文在 appendix 中把 SafeDPO 和每个 baseline 做直接比较，并且额外过滤 safe-safe pairs 后再看 helpfulness，试图避免“更安全所以被 judge 认为更有帮助”的混淆。这个设置仍然是自动评测，但它至少承认 helpfulness 和 harmlessness 在 judge 里不独立。对读者来说，最可靠的结论应该是：SafeDPO 的安全性提升很稳，helpfulness 是否真的优于所有 baseline 则要看评价定义。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的原始分数是 8、6、8、4，其中低分 reviewer 在 rebuttal 后把分数提高到 6；AC 总结认为主要担忧已经被额外实验和澄清处理。正面评价集中在理论推导简洁、训练管线轻、无需 reward/cost auxiliary networks，以及相对 SafeRLHF 更低的工程复杂度。

Reviewer 的核心担忧非常具体。第一，SafeDPO 需要 binary safety labels，原始 DPO 不需要；因此它的“低成本”不能简单理解为少标注，而是少训练额外模型、少 online RL 工程。第二，unsafe-unsafe pairs 被丢弃会造成数据利用率下降，尤其在高风险 prompt 中，可能大量 response 都 unsafe。作者在 rebuttal 中回应说这类 pair 不应该继续训练“哪个 unsafe 更好”，可以用 refusal template 或 safe anchor 补足对比，但这仍然是实践中需要额外设计的数据问题。

第三个担忧是 over-refusal。Reviewer 要求 XSTest，作者补上后结果反而让论文更可信：SafeDPO 的 over-refusal 确实高。这不是小瑕疵，而是 strict hard constraint 的结构性副作用。我的判断是，SafeDPO 的定位应该是 **strict safety-first DPO variant**，不是“同时最安全、最有帮助、最不拒答”的万能目标。它适合在 unsafe boundary 很明确、宁可保守也不能违规的场景中使用；如果任务要求细粒度区分 benign-but-dangerous-looking prompts，就需要额外的 intent disambiguation 或 safe helpfulness data。

我的客观评述是：这篇论文的理论骨架很强，尤其是把 hard safety constraint 变成 pair transformation 的部分很干净。它也冷酷地揭示了一个 alignment 现实：当你把 unsafe response 从 optimal support 中删除，模型确实会更安全，但拒答边界会随之变厚。后续如果要把 SafeDPO 推向部署，需要把它和 better safety labeling、benign adversarial prompt coverage、over-refusal regularization，以及 safe completion anchors 结合起来。

更严格地说，SafeDPO 的“simple”主要是相对 SafeRLHF 的训练系统而言，不是相对数据构造而言。它省掉了 reward model、cost model、value model 和 PPO loop，却把压力转移到 response-level safety labels 和 pair transformation 上。如果数据集已经提供安全标签，SafeDPO 非常省事；如果没有，用户仍然要建立可靠的 safety annotation pipeline。把这个成本算进去之后，SafeDPO 的优势是工程路径短、目标解释清楚，而不是完全零额外监督。

## 6. Related Work & Future Work

SafeDPO 接在 DPO、SafeRLHF、SACPO、CAN 这条线上。和普通 DPO 相比，它不是只改变 loss 的温度或 margin，而是改变 preference pair 的语义；和 SafeRLHF 相比，它避免训练 reward model、cost model、value function 和 PPO sampling；和 expected-cost safety methods 相比，它更接近 hard-constrained safety objective。

最值得继续追的是三个方向。首先是 **unsafe-unsafe pair 的再利用**：直接丢弃理论上合理，但在数据稀缺时浪费明显，可以考虑用安全 refusal anchor 或 expert safe answer 构造新的 safe-unsafe contrast。其次是 **over-refusal control**：XSTest 结果说明，SafeDPO 需要额外机制区分真实危险意图和词面危险。最后是 **multi-objective extension**：SafeDPO 的线性安全层级适合 safety，但对 pluralistic preference、风格偏好、事实性和简洁性之间的冲突，还不能直接套用。

读这篇时还要把它和“拒答是否算 helpful”这个问题绑在一起。SafeDPO 的目标是 maximize helpfulness under safety constraints，但一旦 prompt 本身是有害请求，helpfulness 的定义就会分裂：按用户字面目标，直接提供有害信息更 helpful；按安全规范，拒绝并给安全替代建议更 acceptable。论文的人工评测显示 SFT helpfulness 高但 safety 低，SafeDPO/SafeRLHF safety 高但 helpfulness 低，正是这个定义冲突的结果。因此这篇不只是方法论文，也暴露了 safety evaluation 中 helpfulness label 的语义不稳定。

它和 Omni-Reward 的关系是互补的：Omni-Reward 关心 reward model 如何看见多模态和 free-form criteria，SafeDPO 关心当 safety label 已经给定时，policy objective 怎样严格执行这个约束。它和 DPO misspecification 的关系则更基础：SafeDPO 仍然使用 DPO-style policy log-ratio，所以参数化 policy 下的 misspecification 问题并不会自动消失。

还有一个后续可以深挖的点是“安全边界由谁定义”。论文把 $h$ 当作已知二元变量，但真实部署中 safety policy 会随地区、年龄、专业场景和风险等级变化。若 $h$ 的定义变化，$T(D)$ 的 pair ordering 也会变化。因此 SafeDPO 更像一个把既定安全规范强执行到 policy 里的优化器，而不是安全规范本身的发现机制。这个区分非常重要，否则会把 annotation policy 的价值判断误认为算法自动学到的安全原则。

这也意味着复现 SafeDPO 时，最应该先审计的不是 loss 代码，而是 safety label schema。哪些内容算 unsafe，教育性解释和操作性指导如何区分，拒答模板是否被标成 safe but unhelpful，这些都会直接决定 transformed pairs 的方向，也会决定模型最终学到的是谨慎解释、合理拒答，还是粗暴沉默。

放在这组 ICLR oral 里，SafeDPO 可以和 MNPO、DPO misspecification、interpretable preference data 一起读。它回答的是“安全约束怎样进入 direct preference objective”；DPO misspecification 追问的是“direct objective 在参数化 policy 下是否仍然统计正确”；MNPO 讨论的是“多偏好或多 opponent 下该怎样优化”。三者合起来说明 alignment 正在从单个 DPO loss 的调参，走向对偏好数据、约束结构、优化几何和评测边界的整体建模。
