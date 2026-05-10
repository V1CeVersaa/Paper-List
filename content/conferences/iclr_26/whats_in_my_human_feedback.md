---
title: "WIMHF"
headline: "What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data"
description: "ICLR 2026 Oral note on WIMHF, a sparse-autoencoder method for interpreting preference datasets."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2510.26202"
openreview_url: "https://openreview.net/forum?id=sC6A1bFDUt"
source_pdf: "raw/papers/arxiv-2510.26202.pdf"
tags: ["preference_data", "feedback_interpretability", "data_curation"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **WIMHF/What's In My Human Feedback** 处理的是 alignment 里一个非常底层但经常被跳过的问题：我们拿到 preference dataset 以后，通常只知道 annotator 选了哪个 response，却不知道这些选择到底编码了哪些偏好、偏见、风格偏好或安全风险。论文提出用 sparse autoencoder/SAE 解释 response pair 的 embedding difference，从而自动发现一个数据集“能测量什么差异”和 annotator “实际偏好什么差异”。它把 preference data 从黑箱训练燃料变成可审计对象。
>
> 论文最强的地方在于它不是只做解释，还把解释接回 alignment 操作。WIMHF 在七个反馈数据集上发现大量 dataset-specific preferences，并且指出 LMArena 用户强烈反偏好 refusals，常常选中 toxic content；把这类 unsafe anti-refusal examples 的 label 翻转后，RewardBench2 safety 从 8.9% 提升到 46.2%，同时不损害整体非安全表现。边界也很清楚：SAE feature 是相关性解释，不是因果证明；feature description 依赖 LLM 解释器和 LLM judge；prompt conditioning 很弱；可解释 feature 只恢复了 black-box reward model 一部分信号。

## 1. Introduction

偏好数据是 RLHF、DPO、reward modeling 和各类 alignment pipeline 的核心输入，但偏好数据本身往往是最不透明的部分。一个 pairwise label 只告诉我们 $r_A$ 和 $r_B$ 哪个被选中，却不说明 annotator 是因为安全性、信息量、格式、语气、幽默、拒答、政治立场、长度，还是某个偶然风格做出选择。把这些 label 直接喂给模型，模型会学习任何能预测 label 的信号，包括我们想要的能力，也包括 verbosity、sycophancy、overconfidence、anti-refusal 或数据集特有的审美偏差。

已有工作通常有两条路线。第一条是训练 reward model，让模型直接预测 preference label。它有效，但解释性弱，开发者很难知道 reward model 到底在奖励什么。第二条是预先指定属性，例如 length、politeness、humor、sycophancy，然后测这些属性是否影响偏好。它可解释，但发现能力受限，因为研究者只能检验自己事先想到的 hypothesis。WIMHF 的目标是第三条路：**不预设具体属性，从 preference data 本身自动发现可解释 feature**。

论文区分了两个关键概念。**Measurable preferences** 指一个数据集中 response pairs 实际呈现的差异维度，也就是数据有能力测量什么。例如某个数据集里一对 response 常常在“是否拒答”上不同，那么 refusal 就是 measurable feature；如果所有 response 都使用相似格式，那么格式偏好即使重要，也在这个数据里不可测。**Expressed preferences** 指这些 measurable features 中哪些真的预测 annotator 的选择，也就是人类反馈实际表达了什么。

这个区分非常重要，因为它把数据集构造和人类选择分开了。一个 dataset 没有测到某种偏好，可能不是 annotator 不在乎，而是 response sampling 没有制造这种差异。反过来，一个 dataset 表达了某种偏好，也不代表这是人类价值本身；它可能只是该数据集上下文、采样策略、平台用户构成或 prompt 分布共同产生的局部信号。WIMHF 的价值就在于让这些信号变得可见。

## 2. Problem Setup

论文的输入是 preference dataset，基本样本可以写成 $(p,r_A,r_B,y)$。其中 $p$ 是 prompt，$r_A$ 和 $r_B$ 是两个候选 response，$y\in\{r_A,r_B\}$ 是被选择的 response。WIMHF 不直接训练一个最终 reward model，而是试图学习一组稀疏、可解释的 feature，使得每个 feature 描述 $r_A$ 和 $r_B$ 之间某种稳定差异。

更形式化地说，作者先把 response 转成 text embedding，得到 $e_{r_A}$ 和 $e_{r_B}$，再看差分：

$$
e_\Delta=e_{r_A}-e_{r_B}.
$$

这个差分保留了“两个 response 怎么不同”的语义信息，但 dense embedding 不可解释。WIMHF 用 SAE 把 $e_\Delta$ 映射到 sparse latent vector $z\in\mathbb{R}^M$。每个维度 $z_j$ 期望对应一个可解释的差异概念，例如“使用 emoji”、“直接回答而不是追问澄清”、“拒绝有害请求”、“用 Markdown 列表和标题组织回答”。

这里的符号有一个重要设计：作者用 identity activation，让 feature 是 signed 的。$z_j>0$ 表示 feature 更出现在 $r_A$，$z_j<0$ 表示更出现在 $r_B$，$z_j=0$ 表示两边都没有明显体现。这个设计比 ReLU SAE 更适合 pair difference，因为 ReLU 会把“在 $r_B$ 更强”和“缺失”混在一起，容易学出两个冗余方向。

最终，WIMHF 希望回答两个问题。第一个是 dataset-level 的：这些 response pairs 中反复出现哪些可解释差异？第二个是 label-level 的：在控制长度等已知 covariate 后，哪些 feature 会显著提高 win-rate？因此它既是一个 dataset audit tool，也是一个 preference-label audit tool。

## 3. Algorithm / Methods / Model

WIMHF 的流程分三步。第一步是用 SAE 学 measurable features。作者使用 OpenAI text-embedding-3-small 计算 response embeddings，再对 $e_\Delta$ 训练 BatchTopK SAE。BatchTopK 的作用是强制每个样本只激活少量 feature；论文默认 $(M,K)=(32,4)$，也就是总共 32 个 feature，每个 response pair 平均激活 4 个。这个超参数选择体现了作者的经验判断：在一个具体 preference dataset 里，response pair 的差异维度不需要像解释 LLM token activations 那样动辄成千上万，较小的 feature basis 反而更不冗余、更容易解释。

第二步是给 feature 写自然语言解释。对每个 feature $z_j$，作者采样五个高激活的 preference pairs，让 LLM 生成一个简短概念描述。接着用另一个 LLM judge 在 held-out pairs 上判断该描述更适用于 $r_A$、$r_B$ 还是两者都不适用，并计算 judge annotation 与 feature signed activation 的 Pearson correlation。只有通过 Bonferroni correction 后显著的 feature 才被保留。这个 **fidelity score** 是 WIMHF 避免纯幻觉解释的关键过滤器。

第三步是估计 expressed preferences。作者对每个 feature 做 logistic regression：

$$
\Pr(y=1)=\sigma(\alpha+\beta_j z_j+\gamma x),
$$

其中 $x$ 默认是 response length difference。$\beta_j$ 表示 feature 对 preference label 的影响，论文还把它转成更直观的 $\Delta$win-rate，即在控制长度后 feature 出现时 predicted win-rate 的平均变化。控制长度的原因是长度偏好在 preference datasets 中非常常见，如果不控制，WIMHF 会自然发现 length-like feature；但作者也在 rebuttal 和 appendix 里说明，不控制长度时长度偏好确实会浮现出来。

这个 pipeline 有两个优点。首先，它把“数据集中有什么可测差异”和“annotator 选了什么”分开，使开发者可以先检查采样策略是否制造了需要的比较维度，再检查 label 是否表达了想要的偏好。其次，它不是只能输出一个全局 reward score，而是输出 feature-level explanations，因此可以用于 data curation、benchmark adjustment 和 personalization。

它也有几个必须记住的限制。自然语言 feature description 不是唯一的，一个 SAE direction 可能同时对应多个相关概念；LLM 解释器和 LLM fidelity judge 可能引入同源偏差；$e_{r_A}-e_{r_B}$ 默认弱化 prompt 条件，而很多偏好本来就是 prompt-dependent 的；logistic regression 只能说明 feature 与 label 相关，不能证明 annotator 因为这个 feature 做出选择。论文自己也承认，feature descriptions 是进一步审计的入口，不应该当作人类偏好的最终真相。

这个限制反过来也解释了 WIMHF 为什么有用：它不需要完美解释 preference data 才能产生价值。真实数据清洗里，最难的通常不是证明某个 feature 是唯一因果因素，而是先找到“哪些样本集合在系统性推动模型学坏”。Arena 的 anti-refusal feature 就是这种情况。哪怕 refusal preference 还和 prompt type、toxicity、response length 相关，它仍然足以定位一批高风险 labels，值得人类重新审查或直接构造更安全的 counterfactual labels。

## 4. Experiments

论文分析了七个反馈数据集：LMArena、Community Alignment、HH-RLHF、PRISM、Reddit/Stanford Human Preferences、PKU-SafeRLHF 和 Tulu 3 mixture。作者先过滤掉客观正确答案主导的 math/coding queries，主要聚焦 subjective conversations，因为在客观题里“正确性”可能压过所有风格和价值偏好，而 embedding difference 未必能表示 correctness。

第一组实验验证 SAE features 是否有意义。用 sparse feature vector 预测 preference label，平均 AUC 是 0.672；作为对比，fine-tuned reward model AUC 是 0.766。换句话说，WIMHF 的可解释 sparse features 达到了 black-box reward model 超过随机基线部分的 67%，也达到了 dense embedding 超过随机基线部分的 84%。这说明少数可解释 feature 捕获了大部分可从 embedding difference 中恢复的 preference signal，但还没有完全追上 black-box reward model。

作者还用 Community Alignment 中的 annotator-written explanations 做外部验证。WIMHF 没看过这些解释，但在 5,000 个随机 pairs 上，60.4% 的 annotator explanations 至少匹配一个 active SAE feature，而随机 inactive features 的匹配率是 33.3%。此外，外部 ML researchers 认为 47 个显著预测 preference 的 features 全部 interpretable，其中 41 个有帮助。这些验证让 WIMHF 比纯 LLM 自动解释更可信。

第二组结果是 dataset audit。WIMHF 发现不同数据集测量和表达的偏好差异非常大。PRISM 中，response pairs 常常围绕争议话题是否直接回答、是否保持中立、是否拒绝展开；Community Alignment 更常围绕具体价值内容、环保、社会正义、乐观或批判语气展开。这说明 response sampling strategy 直接决定 measurable preferences：用多个模型高温采样会制造更多风格、语气和拒答差异；提示同一模型生成多种价值候选会制造更多 topic/value 差异。

更有冲击力的是 expressed preferences 的冲突。Reddit 和 Arena 经常偏好 informal tone、jokes 或更直接的回答，而 HH-RLHF、PRISM、Community Alignment 对这些特征的偏好方向可能相反。Arena 最危险的发现是：对 refusals 的 dispreference 达到 -31%，而很多被拒绝的 prompt 本身是 toxic 或 sexual 的。也就是说，用户投票常常选择“满足有害请求”的 response，而不是安全拒答。这个结果直接解释了为什么把 Arena-style preference data 盲目用于 PFT 可能伤害 safety。

第三组实验把解释变成干预。作者针对 Arena 中 anti-refusal unsafe feature，翻转高激活样本的 chosen/rejected label，再训练 Llama-3.2-3B reward model。随着翻转 top examples 增多，RewardBench2 safety accuracy 从 base model 的 8.9% 提升到 top 1000 examples 后的 46.2%，同时 non-safety properties 仍在 base model 95% confidence interval 内。这是论文最强的实用结果：WIMHF 不只是帮我们“知道数据有问题”，还可以定位要改哪些 datapoints。

论文还展示了 evaluation correction 和 personalization。对 Arena Elo 排名做 label flipping 后，模型排名发生大幅变化，例如 Claude-3.5-Sonnet 上升 112 Elo，说明 unsafe preference 不只影响训练，也影响评测。Personalization 方面，Community Alignment 中最主观的 feature 是 paragraphs vs. lists；只对这个低风险风格 feature 学 annotator-specific coefficient，在每个 annotator 只给少量样本时也能提升 held-out AUC，最多约 +1.1%。这个增幅不大，但它展示了一种可控 personalization route：只个性化用户明确允许、风险较低的 feature，而不是让黑箱个性化模型任意调整价值立场。

实验的主要弱点也很明确。WIMHF 的 prompt conditioning 很弱，作者发现加入 prompt-response transcript embedding 没有提高预测 AUC，但这不代表 prompt 不重要；更可能是当前表示和任务设置没把 prompt-conditional preference 表达出来。另一个弱点是 safety curation 主要展示在一个小 reward model 和 RewardBench2 上，虽然 rebuttal 后扩展了更多模型和 RewardBench-1，但部署级结论仍需要更大模型、更多安全 benchmark 和真实 preference training 验证。

另外，WIMHF 的过滤策略会影响它看到的偏好边界。作者移除了非英语、过长样本和两个 response 都被 LLM 判为 objective 的样本，这让分析更干净，却可能错过真实部署中最麻烦的区域，例如多语言安全请求、长上下文咨询、以及同时包含事实正确性和价值判断的混合任务。这个 trade-off 可以接受，因为论文目标是建立方法和展示应用；但如果把 WIMHF 用作生产数据审计，就必须重新评估这些过滤步骤是否会把最需要审计的 hard cases 排除掉。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的原始分数是 4、8、8、6。AC 总结认为这篇是 interpretability methods 在 alignment 中少见的实际应用：用 SAE 解释 preference data，并把解释用于 unsafe label flipping 和 personalization。Reviewer 的正面评价集中在问题重要、方法直观、实验覆盖七个数据集、能发现 conflicting preferences 与 unsafe preferences，以及能把分析转化为实际数据清洗。

低分 reviewer 的核心担忧是 SAE 是否必要、方法细节是否足够可复现。这个批评非常合理，因为如果简单 PCA 或 embedding top dimensions 也能做到类似解释，WIMHF 的技术贡献就会变弱。作者在 revised version 中补了 SAE ablation：SAE 平均 fidelity 0.33，高于 Embed-TopDims 的 0.20 和 Embed-PCA 的 0.13；高保真 feature 数量也明显更多。这基本回应了“为什么需要 SAE”。

另一个强担忧来自 LLM-generated feature descriptions 和 LLM judge 的循环性。WIMHF 用 LLM 写 feature description，再用 LLM 判断 fidelity，确实可能把 LLM 自己的语义偏好当成人类解释。作者补了 human audits、annotator explanation matching、ModernBERT embedding robustness、跨 seed reproducibility 和 semi-synthetic persona experiment，这些补充提高了可信度。但我的判断是，这个问题没有完全消失。WIMHF 更适合当作 **audit triage tool**：它告诉你哪里值得看、哪些 datapoints 可能有问题，而不是替代人工审计。

Reviewer 还指出 correlation vs causation 的表述风险。论文说 annotators "prefer" 或 "disprefer" 某些 features，但 logistic regression 只能说明这些 features 预测 label。比如一个 feature 描述为“使用 informal tone”，真正驱动选择的可能是 prompt 类型、回答长度、模型身份、平台用户偏好或多个 correlated style features。作者在修改中承认这一点，这一点很关键。读这篇时必须把 WIMHF 的 feature 看成 **preference-correlated explanations**，而不是已证明的人类偏好因果变量。

我的客观评述是：WIMHF 是这组 oral 里非常值得读的一篇 data-centric alignment 论文。它的价值不在于 SAE 技术本身多复杂，而在于把 preference data 的可审计性提升为 alignment 的一等问题。很多 alignment 失败不是因为 loss 写错，而是因为数据里混着互相冲突的偏好、错误的安全标签、平台特有的用户审美或有害的 non-refusal reward。WIMHF 给了一个相当具体的工具，把这些问题从抽象担忧变成可定位的 feature 和 datapoints。

但这篇也不能被读成“自动解释偏好数据已经解决”。它没有恢复完整 reward model，也没有证明所有重要偏好都被发现；它对非英语、长对话、强 prompt-dependent values、低频安全风险和复杂多轮交互的覆盖仍有限。最成熟的使用方式应该是：先用 WIMHF 找出高影响 feature 和异常 datapoints，再让领域专家、人类标注者或 policy team 对这些 feature 做确认，最后决定是过滤、翻转、重采样还是分组建模。

## 6. Related Work & Future Work

WIMHF 和 Inverse Constitutional AI、reward model interpretability、SAE-based feature discovery、preference data auditing 放在同一条线上。和预设属性分析相比，它不需要先猜测 length、sycophancy、refusal 或 humor；和 black-box reward model 相比，它牺牲了一部分预测性能，换来 feature-level 可解释性；和 mechanistic interpretability 相比，它不是解释模型内部 neuron，而是解释 preference data 的 response-pair difference space。

后续最值得推进的是 **prompt-conditioned WIMHF**。很多偏好只有结合 prompt 才有意义：拒绝有害请求是好事，但拒绝普通帮助请求是坏事；环保建议在旅行规划 prompt 中可能相关，在无关 prompt 中可能跑题；幽默在 Reddit 可能被奖励，在严肃医疗场景可能不合适。当前 WIMHF 用 response embedding difference 为主，容易把 context-specific feature 误读成全局 preference。更好的版本应该让 feature 显式条件化在 prompt、risk category 和 user intent 上。

第二个方向是 **causal data intervention**。论文展示了 anti-refusal feature 的 label flipping，但未来应该更系统地比较几种干预：删除样本、翻转 label、加入 safe counterexamples、分开训练 safety reward head、或者把 feature 作为 reward model control variable。只有这样才能知道 WIMHF 发现的 feature 最适合哪种数据修复方式。

第三个方向是 **preference mixture governance**。如果不同数据集对同一 feature 表达相反偏好，把它们简单混合会让模型学到模糊甚至有害的折中。WIMHF 可以成为 dataset mixture 前的审计工具：先列出哪些 feature 冲突，再决定是按场景分离、按用户群体个性化，还是在 policy 层面明确规定优先级。这个方向和 pluralistic alignment 很接近，但比抽象价值多元更工程化，因为它直接面对训练数据里的 feature conflict。
