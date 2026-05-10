---
title: "TI-DPO"
headline: "Token-Importance Guided Direct Preference Optimization"
description: "ICLR 2026 Oral note on TI-DPO, a token-importance weighted DPO objective with triplet guidance."
status: complete
visibility: public
topic: "preference_learning"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2505.19653"
openreview_url: "https://openreview.net/forum?id=cMEnMVvMw9"
source_pdf: "raw/papers/arxiv-2505.19653.pdf"
tags: ["direct_preference_optimization", "token_importance", "rlhf"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **TI-DPO/Token-Importance Guided Direct Preference Optimization** 试图把 DPO 从 sequence-level preference optimization 推到 token-level preference optimization。它的核心判断是：普通 DPO 把整段 preferred response 和 rejected response 各压成一个 log-ratio 差值，默认所有 token 对人类偏好的贡献相同；但真实偏好往往由少数关键 token 或短语决定，例如安全建议里的 "seek medical attention"、代码回答里的关键 API、指令跟随里的约束词。TI-DPO 因此用 gradient attribution 估计 token importance，再用 Gaussian prior 稳定权重，最后把这些权重放进 DPO 的 Bradley-Terry 差值里。
>
> 论文的第二个动作是加入 **triplet loss**：在 preferred response、non-preferred response 和当前 policy 生成的 anchor response 之间建立距离约束，让 anchor 在 log-ratio preference space 中靠近 preferred、远离 rejected。实验上，TI-DPO 在三种 base model 和 MMLU、GSM8K、GPQA、HumanEval、TruthfulQA、IFEval 上取得最高平均分 62.3，略高于 GRPO 的 62.1 和 TIS-DPO 的 61.1。边界也很明显：Gaussian prior 的中心偏置是一个强启发式，理论定理依赖稀疏关键 token 与无偏估计假设，且方法引入额外梯度计算和 anchor generation，不能简单读成“更细粒度所以一定更高效”。

## 1. Introduction

DPO 的优势在于把 RLHF 里 reward model training 和 PPO-style policy optimization 合并成一个监督式 preference loss。给定 prompt $x$、preferred response $y_w$ 和 rejected response $y_l$，DPO 直接比较 policy 相对 reference policy 的 log-probability ratio。这个设计非常干净，但它的粒度是整段 response：一段回答里真正决定偏好的 token、无关铺垫、礼貌性过渡、重复解释都会一起进入同一个 sequence-level score。

TI-DPO 抓住的缺口就是这个粒度错配。人类偏好不是均匀分布在所有 token 上的。一个医疗建议是否安全，可能取决于是否建议用户及时就医；一个数学推理是否正确，可能取决于某个符号或中间数；一个指令跟随回答是否合格，可能取决于有没有遵守 "do not mention" 这样的约束词。普通 DPO 用整段 log-ratio 优化，会把这些关键 token 和非关键 token 混在一起，训练信号更容易受噪声 token、数据风格和长度差异影响。

论文把这个问题定义成 **fine-grained alignment**：模型不仅要知道 preferred response 整体更好，还要知道 response 内部哪些 token 对 preference decision 起关键作用。作者认为已有 token-level 方法仍有两个问题。第一，token importance 往往来自概率估计或简单采样权重，可能只是模型当前置信度的 proxy，不一定真正对应偏好贡献。第二，很多方法仍然沿用二元 preferred/rejected 对比，没有给 policy 一个连续的、结构化的方向，说明当前输出应该往 preferred response 靠近到什么程度、远离 rejected response 到什么程度。

TI-DPO 的设计因此包含两个互补部分。**Hybrid weighting mechanism** 负责回答“哪些 token 更重要”，它用 final-token logit 对前文 token embedding 的梯度来估计影响力，并混入 Gaussian prior 防止权重被噪声梯度带偏。**Triplet objective** 负责回答“如何把当前生成拉向更好的区域”，它把当前 policy 生成的 anchor response 放到 preferred 和 rejected 之间比较。这个组合让 TI-DPO 不只是 DPO 的 token 加权版，也把 metric learning 里“拉近正例、推远负例”的结构引入了偏好优化。

从 alignment 角度看，这篇 oral 的位置很清楚：它不是 safety-specific method，也不是 reward model 论文，而是 direct preference optimization 的 granularity 修正。它和 SafeDPO、DPO misspecification、SSPO 放在一起读，会形成一条完整问题链：SafeDPO 讨论安全约束如何改变 pair ordering，DPO misspecification 讨论参数化 policy 下 DPO 到底估计什么，SSPO 讨论 preference labels 不足时怎样利用 unpaired data，TI-DPO 则讨论 preference signal 在 response 内部如何分配。

## 2. Problem Setup

论文把文本生成写成一个 token-level Markov Decision Process/MDP。给定 prompt $x$，第 $t$ 步的 state 是 $s_t=[x,y^{<t}]$，也就是 prompt 加上已经生成的 token 前缀；action 是下一 token $a_t=y^t$；transition 是确定性的，因为选择 token 后 state 直接追加这个 token。这个形式化的意义不是要重新做 RL，而是给“每个 token 都有 reward contribution”一个清楚的数学容器。

DPO 中的 implicit reward 来自 policy 与 reference policy 的 log-ratio。对单个 token，可以写成：

$$
r_\phi(s_t,a_t)
=
\beta \log
\frac{\pi_\theta(y^t\mid x,y^{<t})}
{\pi_{\mathrm{ref}}(y^t\mid x,y^{<t})}.
$$

普通 DPO 相当于把整段 response 的 token log-ratio 求和，然后比较 $y_w$ 和 $y_l$。TI-DPO 改动的地方是引入 token weight $w_t$，把 reward difference 写成加权形式：

$$
\Delta r_{\mathrm{token}}
=
\sum_{t=1}^{T_w} w_t^w
\log
\frac{\pi_\theta(y_w^t\mid x,y_w^{<t})}
{\pi_{\mathrm{ref}}(y_w^t\mid x,y_w^{<t})}
-
\sum_{t=1}^{T_l} w_t^l
\log
\frac{\pi_\theta(y_l^t\mid x,y_l^{<t})}
{\pi_{\mathrm{ref}}(y_l^t\mid x,y_l^{<t})}.
$$

这里 $w_t^w$ 和 $w_t^l$ 分别是 preferred response 与 rejected response 的 token importance。它们不是人工标注，而是由模型梯度和先验共同产生。这样一来，DPO 的 Bradley-Terry preference probability 从 sequence-level 差值变成 weighted token-level 差值，训练目标会更关注对偏好判断敏感的位置。

这个 setup 有一个隐含判断：**preference-relevant signal 在 token 层面是稀疏或非均匀的**。如果所有 token 真的同等重要，TI-DPO 退化回类似普通 DPO；如果关键 token 的确少而集中，权重机制就能降低无关 token 对 reward difference 的噪声贡献。论文后面的 Lemma 1 和 Theorem 2 都依赖这个稀疏关键 token 叙事。

问题也出在这里。Token importance 不等于 attention，也不等于 logit sensitivity，更不一定等于人类偏好因果特征。TI-DPO 用 final-token max logit 对前文 embedding 的梯度来估计 importance，这能捕捉“哪些 token 影响模型当前预测置信度”，但它和“哪些 token 决定人类偏好”之间仍有间接性。这个间接性是论文的关键技术风险，也是 reviewer 集中追问的地方。

还有一个容易被忽略的点是，TI-DPO 的 token weight 同时作用在 chosen 和 rejected response 上。这意味着它不只是“强化 preferred response 的关键 token”，也会让 rejected response 中对错误偏好、危险建议或指令违背最敏感的位置承担更大负梯度。这个对称设计在安全和指令跟随任务上有意义，因为坏回答常常不是整段都坏，而是某几个 token 或短语越过边界；但如果 rejected response 只是风格较差、长度不合适或表达啰嗦，梯度 attribution 可能会把风格 token 当成偏好核心，从而让模型过拟合标注者审美。

## 3. Algorithm / Methods / Model

TI-DPO 的第一步是计算 gradient-based token importance。对一段 token 序列 $y=[y_1,\ldots,y_{T-1}]$，模型先得到 embedding 序列 $E=[e_1,\ldots,e_{T-1}]$，再做 forward pass 得到最后一步 logits $L_{T-1}\in\mathbb{R}^V$。作者把最大 logit 作为目标标量：

$$
L_{\mathrm{target}}=\max(L_{T-1}).
$$

接着对每个 token embedding $e_i$ 计算 $\nabla_{e_i}L_{\mathrm{target}}$，并用 $L_1$ norm 压成标量 importance：

$$
I_i=\|\nabla_{e_i}L_{\mathrm{target}}\|_1.
$$

这个分数的直觉是：如果微小改变某个 token 的 embedding 会明显改变模型最后一步最有信心的预测，那么这个 token 对当前序列的生成状态有较大影响。论文把它当作 token-level preference optimization 的数据驱动部分。为了避免 raw gradient 太噪，作者把 $I$ 归一化成分布 $I_{\mathrm{norm}}$，再和一个 centered Gaussian prior 混合：

$$
W=\lambda I_{\mathrm{norm}}+(1-\lambda)P_{\mathrm{prior}}.
$$

Gaussian prior 的均值设在序列中心，标准差设为 $T/4$。作者给出的动机是抵消 LLM 的 **lost-in-the-middle bias**：模型容易重视开头和结尾而忽视中间，因此 prior 给中间 token 一个更高 baseline，避免 semantic core 被低估。这个解释比“重要 token 天然在中间”更合理，但它仍然是启发式。不同任务的关键 token 可能出现在开头、结尾、结构标记或代码行内，固定 Gaussian prior 并不保证匹配真实偏好结构。

第二步是 weighted DPO loss。把 $W$ 分别应用到 $y_w$ 和 $y_l$ 后，TI-DPO 使用：

$$
\mathcal{L}_{\mathrm{DPO-w}}
=
-\mathbb{E}_{(x,y_w,y_l)\sim D}
\log\sigma(\Delta r_{\mathrm{token}}).
$$

如果所有 $w_t=1$，这个目标就回到普通 DPO 的累计 log-ratio 差值。TI-DPO 的实际作用是重新分配每个 token 对 preference margin 的贡献，让关键 token 上的 policy update 更强、非关键 token 上的 drift 更弱。

第三步是 triplet loss。作者在每个 batch 中用 policy $\pi_\theta$ 动态生成一个 anchor response $y$，并把 $y$、$y_w$、$y_l$ 都映射到 log-ratio preference space。Triplet loss 要求 anchor 更接近 preferred response，而不是 rejected response；如果 $d(y,y_w)$ 没有比 $d(y,y_l)$ 小至少 margin $\alpha$，loss 就会惩罚：

$$
\mathcal{L}_{\mathrm{triplet}}
=
\mathbb{E}
\left[
\max(0,d(y,y_w)^2-d(y,y_l)^2+\alpha)
\right].
$$

完整目标是：

$$
\mathcal{L}_{\mathrm{TI-DPO}}
=
\mathcal{L}_{\mathrm{DPO-w}}
+\gamma\mathcal{L}_{\mathrm{triplet}}.
$$

这个 triplet 项很有意思，因为它给 DPO 增加了一个“中间生成应该往哪里走”的连续结构。普通 pairwise preference 只告诉模型 $y_w$ 胜过 $y_l$；triplet loss 则把当前 policy 的 anchor response 纳入训练，让 policy 的中间输出也受到 preferred/rejected 几何约束。不过，anchor 如何生成、它是否真代表 preferred 和 rejected 之间的中间状态、额外采样是否稳定，都是实践中需要仔细检查的点。Reviewer 对 anchor generation 的追问是合理的。

理论部分主要有三层。Lemma 1 说，在 reward signal 由 sparse critical tokens 决定、non-critical tokens 只贡献独立零均值噪声时，抑制 non-critical token 权重会降低 estimator variance。Theorem 2 进一步说，在严格凸、无偏估计等假设下，TI-DPO loss 会比 DPO 有更紧的上界。Theorem 3 则把问题写成 KL budget allocation：普通 DPO 可能把 KL divergence 浪费在 non-critical tokens 上，而 TI-DPO 把 update 集中在 critical tokens 上，因此在固定 KL constraint 下有更高 expected true reward lower bound。

这些定理提供了一个有用解释，但不能过度读。它们更像是“如果 token importance 真的对齐 critical tokens，TI-DPO 为什么会好”的条件性说明，而不是证明当前 gradient attribution 一定能找到真实人类偏好 token。最需要保留的结论是：TI-DPO 的理论优势来自 **variance reduction 与 KL budget reallocation**，前提是 importance weights 不能错。

## 4. Experiments

实验覆盖三种 base model：Llama-3.2-3B、Llama-3.1-8B 和 Mistral-7B-v0.3。评测任务包括 MMLU、GSM8K、GPQA、HumanEval、TruthfulQA 和 IFEval，分别覆盖知识、数学推理、困难问答、代码、真实性和指令跟随。比较方法包括 SFT、DPO、IPO、KTO、SimPO、TDPO、CPO、TPO、TIS-DPO、Logic-RL、cDPO 和 GRPO。

主结果显示 TI-DPO 的平均分最高，为 62.3。这个数字略高于 GRPO 的 62.1、TIS-DPO 的 61.1 和 TPO 的 60.7，明显高于 DPO 的 57.7。分任务看，TI-DPO 在 HumanEval、TruthfulQA 和 IFEval 上尤其强，分别达到 67.0、62.0 和 75.7。这个结果支持论文的基本主张：token-level weighting 和 triplet guidance 对代码、真实性和指令跟随这类细节敏感任务确实有帮助。

但结果也需要冷静读。TI-DPO 在 MMLU、GSM8K 和 GPQA 上不是全面领先，GRPO 或 TPO 在部分 reasoning-heavy benchmark 上更强。这个现象说明 token-level preference weighting 不等于 reasoning optimization。数学和复杂推理任务的瓶颈可能更依赖搜索、过程奖励、verifier 或 RL-style exploration，而不是简单识别 response 内部关键 token。论文在 rebuttal 后补了对这些差距的解释，但方法本身没有解决 reasoning-heavy 任务上的全部问题。

Ablation 是这篇实验里比较有价值的部分。在 Llama-3.2-3B-Instruct 上，Full TI-DPO 在六个维度都高于 Base Instruct、No Triplet Loss、Uniform Weight、Random Weight、No Gaussian Prior 和 Softmax Prior。去掉 triplet loss 后，Math 从 80.7 降到 79.0，Code 从 33.0 降到 31.0；去掉 Gaussian prior 后，Reliability 从 86.8 降到 82.5。这个结果说明两个组件都有贡献，尤其是 Gaussian prior 对可靠性指标影响明显。

论文还补充了权重分布分析。GSM8K、GPQA 这类依赖少数关键符号的任务，权重更集中；TruthfulQA、IFEval 这类对安全或指令约束敏感的任务，权重整体偏高；MMLU、HumanEval 这类覆盖内容更广的任务，权重分布更分散。医疗 case study 也展示了 TI-DPO 给 preferred response 里的 "medical attention"、"promptly" 等安全关键 token 更高权重，同时惩罚 rejected response 中 "painkillers"、"casually" 等风险 token。这个例子对直觉很有帮助，但还不足以证明权重解释普遍可靠，因为定性例子仍然太少。

鲁棒性实验是 rebuttal 后增强论文可信度的关键。作者补了 label noise、generation diversity、hyperparameter sensitivity、不同 prior 的 ablation，以及和 TIS-DPO、cDPO、Logic-RL 的比较。AC 认为这些补充解决了主要技术担忧。我的判断是，实验已经足够支持“TI-DPO 是一个有竞争力的 DPO variant”；但还不足以支持“token-level importance 是普遍最优的 alignment 粒度”。真正下一步应该在更长多轮对话、复杂 CoT、safety preference 和真实 chat preference 数据上看权重是否仍然稳定且可解释。

从实验设计看，TI-DPO 的 strongest evidence 来自多个 model 和 benchmark 上的平均提升，而不是某一个任务的极大飞跃。这个证据类型适合支持 method paper，但也意味着我们不能把每个 benchmark 的改进都解释成“关键 token 被更好识别”。例如 TruthfulQA 和 IFEval 的提升可能确实来自安全和约束 token 被强调，也可能来自 triplet loss 改变了整体 response geometry。论文目前没有把 hybrid weighting 和 triplet loss 的因果贡献拆到单样本层面，这也是为什么后续需要更细的 token-level attribution evaluation。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的原始分数是 4、8、6、8；其中 4 分 reviewer 在 rebuttal 后表示 concerns 已解决并提高分数，AC 总结认为 rebuttal 实质性加强了论文。正面评价集中在三个点：token-level DPO 的问题设定有新意，gradient attribution 加 Gaussian prior 加 triplet loss 的组合清楚，实验覆盖和 ablation 在修改后比较完整。

Reviewer 最尖锐的批评集中在 Gaussian prior 和理论-实践连接。早期版本把 Gaussian prior 解释得像“关键 token 常在中间”，这个假设确实站不稳；rebuttal 后作者把它改成抵消 lost-in-the-middle bias，并补了 uniform、random、no-prior、softmax prior 的 ablation，这个回应比较有效。理论方面，reviewer 质疑 tighter loss bound 是否真的对应 accuracy improvement，作者补了 Theorem 3 和 loss/convergence evidence，但这仍然只是把叙事补强，不能完全消除 theorem assumptions 和真实训练之间的距离。

另一个重要担忧是方法比较和复现细节。Reviewer 要求补 TIS-DPO、cDPO、Logic-RL 等 token-level 或相关 baseline，也要求说明 anchor generation、triplet mapping、hyperparameter、计算开销和 reasoning-heavy benchmark 的欠表现。作者在 revised paper 和 appendix 中补了这些内容。这个过程本身说明 TI-DPO 的最初版本有明显“claim 比 evidence 走得快”的问题，但 oral 版本通过补实验把主要缺口填上了。

我的客观评述是：TI-DPO 的方向值得读，因为它把 DPO 的一个真实盲点推到 token granularity 上。它真正有价值的地方不是“Gaussian prior 很聪明”，而是提醒我们 preference signal 在 response 内部并不均匀。普通 DPO、SafeDPO、AuxDPO 这类方法多数仍在 sequence-pair 层面工作；TI-DPO 要求我们继续追问 pair 里到底哪些 token 让一个 response 胜出。

但我会比较冷酷地说，它的解释性还不能按 mechanistic interpretability 标准来读。Gradient attribution 权重可以作为 training signal，不等于可靠的人类偏好解释。final-token max logit 的梯度可能捕捉语法收束、长度模式或模型置信度，而不是偏好因果特征；Gaussian prior 也可能把中间位置当作默认重要性，掩盖任务特异的关键位置。后续如果要把 TI-DPO 用到 safety alignment，最危险的情况是权重错误放大了数据中的偏见 token，例如把某些身份词、拒答模板或固定风格误认为偏好关键点。

因此这篇的最好定位是 **fine-grained DPO objective prototype**。它给出了可训练的 token-level preference weighting 和不错的 empirical package，但仍需要更严格的权重质量评估、计算开销分析、长上下文测试和 safety-specific error analysis。它对后续工作最有启发的追问是：如果 preference label 只给了 sequence-level choice，我们还能不能可靠地恢复 response 内部的 preference attribution？

如果把它用于真实 post-training，我会优先把 TI-DPO 放在低风险或可人工审计的数据切片上试，而不是直接替换主线 DPO。原因很简单：一旦 token weight 错了，错误会比 sequence-level DPO 更集中地打到少数 token 上，模型可能更快学到数据里的 spurious cue。一个更稳妥的工程路线是先记录高权重 token，抽样做人类审计，确认它们确实对应 preference reason，再扩大训练规模。

## 6. Related Work & Future Work

TI-DPO 直接接在 DPO、TDPO、TIS-DPO、TPO、CPO、GRPO 这条 direct preference optimization 线上。和普通 DPO 相比，它改变的是 token 对 reward difference 的贡献；和 TIS-DPO 相比，它不用外部 probability estimation 采样权重，而是用 gradient attribution；和 GRPO/TPO 相比，它仍然是 DPO-style preference training，不是 group reward 或更强的 RL reasoning recipe。

后续最值得追的是 **token attribution validation**。如果作者能在人工标注的 rationale preference 数据上验证高权重 token 是否真的对应人类选择原因，TI-DPO 的解释性会强很多。另一个方向是 **safety preference weighting**：安全样本里真正关键的 token 往往是风险动作、条件限制、免责声明和替代建议，TI-DPO 是否能区分“合理拒绝”和“无意义模板拒绝”，会决定它能否进入 safety alignment。

更长远看，TI-DPO 可以和 WIMHF 这类 preference-data interpretation 方法连接起来。WIMHF 从数据层面解释人类反馈编码了哪些 feature，TI-DPO 从训练目标层面调整 token 级梯度。如果一个系统能先识别偏好数据中的 feature，再把 feature-level 或 token-level attribution 传给 DPO objective，就可能形成更可审计的 preference optimization pipeline。但这要求 attribution 不能只是模型内部梯度，还要和人类偏好解释、数据 feature 和安全 policy 对齐。
