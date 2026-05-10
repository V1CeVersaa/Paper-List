---
title: "Hubble"
headline: "Hubble: a Model Suite to Advance the Study of LLM Memorization"
description: "ICLR 2026 Oral note on Hubble, an open model suite with controlled perturbations for studying LLM memorization risks."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2510.19811"
openreview_url: "https://openreview.net/forum?id=ZfdnZhOP0k"
source_pdf: "raw/papers/arxiv-2510.19811.pdf"
tags: ["memorization", "privacy_benchmark", "model_suite"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **Hubble** 是一个用于研究 LLM memorization 的 fully open-source model suite。它不是只发布一个模型，而是发布 standard 和 perturbed 两类模型：standard models 在大规模英文语料上正常预训练；perturbed models 在相同训练流程中插入受控文本，包括 book passages、biographies、chat personas 和 test sets，用来模拟 copyright、privacy、test set contamination 三类 memorization risks。核心 release 包含 8 个模型，覆盖 1B/8B 参数、100B/500B tokens、standard/perturbed 三个维度。
>
> 论文最重要的实证结论是：memorization risk 受 sensitive data 的相对频率和训练时序影响。把同样敏感数据放进更大的训练语料会产生 **dilution/稀释**，降低记忆；把敏感数据安排在训练早期、后续不持续暴露，会出现 forgetting，最终泄漏更少。Hubble 的边界也清楚：最大模型只有 8B、最大训练量 500B tokens，离商业 frontier models 仍有差距；perturbations 是受控插入，不等于真实 web corpus 中所有隐私和版权情形。但它提供了少见的 causal memorization testbed，比纯观察闭源模型更适合做可复现实验。

## 1. Introduction

LLM memorization 有双重性。一方面，模型必须记住事实和语言模式才能有能力；另一方面，训练数据中的版权文本、个人信息和 benchmark test sets 如果被模型逐字或语义性复现，就会带来法律、隐私和科学评测风险。现有研究常常卡在两个极端：小模型受控实验很干净，但离真实 LLM 太远；大模型观察研究更接近现实，但训练数据不可控，很难判断某句话是因为重复、简单、分布密集还是偶然被记住。

Hubble 的贡献是把问题推到中间：训练足够大的 open models，同时在训练语料里插入完全受控的 perturbation data。因为插入哪些文本、插入几次、插入在训练哪个阶段都是已知的，研究者可以估计很多 causal quantities，例如某个 test set example 重复几次才会被记住，某类 PII 在不同辅助信息下多容易被恢复，或者 unlearning 方法是否真的只删除目标记录而不伤害邻近记录。

这篇论文对 `Privacy & Memorization` 小节非常核心，因为它把 LLM memorization 从抽象风险变成可实验资源。相比只讨论“模型可能泄露训练数据”，Hubble 让研究者拿到 standard/perturbed paired models、不同 duplication levels、不同 training timing 和多个 risk domains。它的意义更像 Pythia 之于 scaling/training dynamics：为一个社区提供共同坐标系。

## 2. Problem Setup

Hubble 的基本设计是 standard model 与 perturbed model 的对照。standard corpus 来自 DataComp-LM baseline dataset；perturbed corpus 则在相同训练序列中插入额外文本。每条 perturbation 被随机分配 duplication count，例如 $0\times,1\times,4\times,16\times,64\times,256\times$。$0\times$ 的 perturbations 可以作为 matched non-members，其他 duplication levels 则形成 members，减少 membership inference benchmark 中常见的 spurious temporal cue。

Perturbation data 覆盖三类风险。第一类是 **copyright**，包括 Gutenberg popular/unpopular passages、Wikipedia recent passages 和 MRPC/PAWS paraphrases。第二类是 **privacy**，包括 YAGO templated biographies、ECtHR court-case biographies 和 PersonaChat dialogues。第三类是 **test set contamination**，包括 PopQA、WinoGrande、MMLU、HellaSwag、PIQA，以及 DCLM cutoff 之后的新测试集 ELLie 和 MUNCH。

评估方式也按数据类型区分。对于 copyright passages，作者看 length-normalized loss 和 k-eidetic memorization；对于 paraphrases 和 test sets，常用 loss-based choice；对于 biographies，使用 generative reconstruction 或 infill attack 恢复 PII；对于 chats，测试能否从 username 推 persona，或从 persona 反推出 username。这些 metric 都只能提供 memorization lower bound，因为模型可能以更隐蔽方式存储信息。

这套 problem setup 的关键是 **paired counterfactual**。如果只观察一个训练好的模型，很难知道它为什么会复现某段文本；Hubble 通过 standard/perturbed pairs 让研究者比较“插入这批文本”和“不插入这批文本”的差异。$0\times$ perturbations 又提供了同分布 non-member examples，避免把 member 和 non-member 的时间、来源、格式差异误当作 memorization signal。对 MIA 和 unlearning 来说，这比从互联网随机找“没见过”的文本更干净。

## 3. Algorithm / Methods / Model

Hubble 的核心 release 是一个 $2\times2\times2$ factorial design：model size 为 1B 或 8B，data condition 为 standard 或 perturbed，training corpus size 为 100B 或 500B tokens。模型基于 Llama 3 architecture，但使用 OLMo tokenizer，减小 vocab size，并 untie embeddings 以支持 logit lens/tuned lens 等 interpretability 方法。最大训练量 500B tokens，对 1B 和 8B 模型分别超过 Chinchilla optimal training set size 的约 22 倍和 3.7 倍。

Perturbation insertion 尽量模拟真实训练文档插入。作者从 standard training process 采样一个 sequence，在文档间隙插入 perturbation，并保持 sequence length 不变；每条 perturbation 用 EOS 包围，像常规文档一样进入训练。所有 perturbations 总共约 79.9M tokens，只占 100B corpus 的 0.08% 和 500B corpus 的 0.016%，因此不会像大规模重复数据那样显著破坏通用能力。

除了核心模型，Hubble 还包含几类 ablation runs。**Interference models** 只插入 copyright、privacy 或 test set contamination 中的一类，用来确认多个 perturbation domains 同时插入不会互相干扰。**Timing models** 把 perturbations 只放在训练前半、后半或四个 quarter intervals 中，用来研究插入时序。**Paraphrased models** 插入 biography 和 MMLU 的 paraphrased variants，用来区分 verbatim memorization 与 semantic memory。**Architecture models** 改变 depth，用来观察相同参数量下模型深度对 memorization 的影响。

这个设计的优点是把很多常见 confounder 拆开了。真实 web corpus 中，同一段文本可能本来就简单、到处被引用、出现在多个 paraphrase 中，还可能和 benchmark answer 共享语义模式。Hubble 虽然不能复刻所有真实复杂性，但它把 duplication、timing、domain、model size、corpus size 变成可控变量，让 memorization research 不再完全依赖黑箱观察。

Decontamination 是这里的敏感环节。作者在插入 perturbations 前移除了和 perturbation 匹配的 base-corpus documents，以确保“出现次数”主要来自受控插入。他们使用偏 high-precision 的流程，删除了 7540 个 documents，占整体不到 0.002%。这种做法的好处是不会大规模扭曲 base corpus；风险是仍可能漏掉 paraphrase 或 formatting-shift overlap。因此 Hubble 的 duplication count 最适合作为受控变量来研究插入文本的边际影响，而不是证明 base corpus 对相关语义完全无污染。

## 4. Experiments

第一条 domain-agnostic 结论是 **dilution**。同样的 perturbations 如果训练在 500B-token corpus 中，相比 100B-token corpus，在几乎所有任务和风险域上的 memorization 都随 duplication 增长得更慢。直观地说，敏感数据的绝对重复次数相同，但相对语料规模更大，模型遇到它的频率更低，最终记忆更弱。这支持一个实际 best practice：除了 deduplication，扩大非敏感训练语料也能稀释敏感文本的影响。

第二条结论是 **ordering**。Timing runs 显示，如果 perturbations 只出现在训练最早 quarter，最终模型几乎不会记住这些数据；如果只出现在最后 quarter，记忆和可抽取性比 regular perturbed model 更强。中间 checkpoints 说明模型在早期暴露后如果不持续接触，会逐渐遗忘。这对数据治理很有启发：如果某些数据不可避免地进入训练，出现时序也会影响最终泄漏风险。

第三条结论是 model size。8B model 在相同 duplication level 下比 1B 更容易记住，并且在更低重复次数下就出现可测 memorization。这和已有 scaling observations 一致：能力更强的模型也更容易吸收低频训练细节。因此 memorization mitigation 不能只用小模型实验外推到大模型，必须考虑规模效应。

domain-specific 结果也很有信息量。Copyright 上，不同 metric 会给出不同判断：loss 在低 duplication 时已能看出差异，但 k-eidetic memorization 需要更高 duplication 才显著。这说明版权争议里单一数值指标很危险。Privacy 上，攻击者辅助信息越多，PII 恢复越容易；在 YAGO/ECtHR biographies 中，full prefix 等强辅助条件下，8B 100B perturbed model 在 16 duplicates 时 PII attack accuracy 可以接近 100%。不同 PII 类型也不一样，occupation、email、UUID 等属性的记忆方式不同。

PersonaChat 显示 indirect leakage 更难但不是没有。模型未必能从 username 准确选择 persona，但反过来给定 persona 推 username 会在高 duplication 下显著高于随机。这说明 memorization evaluation 经常只是 lower bound：某个方向上测不到泄漏，不代表模型没有存储相关信息，只可能是攻击格式不对。

Test set contamination 结果尤其有现实意义。模型只见过一个 test example 一次时，某些任务已经能看到 contaminated examples 上的 accuracy 提升；但这种记忆并不稳定泛化到 unseen examples。有些 WinoGrande 和 HellaSwag 设置里，污染甚至会让 unseen 或格式变换后的表现变差。换句话说，test contamination 更像局部记忆，不必然代表模型学会了任务。

Hubble 还展示两个 use cases。MIA benchmark 中，duplicate count 越高，membership inference 越容易；只重复一次的 members 接近随机，256 duplicates 几乎被简单 loss/MinK% 识别。Unlearning benchmark 中，RMU、Representation Rerouting 和 Saturation-Importance 都没能达到理想目标：它们能降低 Unlearn set 上的记忆，但会同时伤害 Keep set 和 unseen/test behavior，说明当前 unlearning 仍容易删除 distribution-level knowledge，而不是精准删除指定记录。

Copyright 结果有一个特别值得注意的政策含义：**memorization metric 会改变结论**。如果用 loss，模型对低重复文本也可能表现出成员信号；如果用 k-eidetic memorization，只有更高重复才算明显逐字记忆。版权争议中到底关心“模型是否更偏好训练文本”“是否能复现足够长片段”“是否在正常 prompt 下输出近似表达”，这些问题不是同一个 metric 能回答的。Hubble 的好处在于它同时提供多个数据类型和评估接口，可以让后续研究把法律或政策问题转成更具体的 measurement problem。

Privacy 结果也说明 PII 不是一个均质类别。YAGO biography 里 nationality、birthplace、birth date、email、UUID、occupation 等属性被记住的难度不同；攻击者知道 full prefix、intro prefix、name-only 时成功率也不同。这和真实隐私风险一致：有些信息高度结构化、可由上下文推断，有些信息像 UUID 或 email 更接近随机字符串。一个模型如果能恢复随机标识，说明它更接近 verbatim memorization；如果能从 paraphrased biography 恢复 PII，说明它可能形成了 semantic memory。Hubble 同时覆盖这两类风险。

Test contamination 结果则提醒我们不要把污染等同于“能力提升”。模型可能记住被污染样本的答案，却不能泛化到同任务 unseen examples；甚至在 format shift 下表现更差。这对 benchmark governance 很关键：如果一个模型在某个公开 test set 上高分，我们不能简单判断它真的学会了任务，也不能只看有没有逐字训练样本。污染可能产生局部记忆、错误格式绑定和 spurious answer preference。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。AC 总结说所有 reviewer 都认可 Hubble 对 community 的贡献，尤其是为 memorization、membership inference、machine unlearning 和相关研究提供 Pythia/GPT-Neo 之外的现代 open testbed。讨论后 reviewer scores 预期为 8、8、8、8，其中一位 reviewer 在 rebuttal 后从 6 提高到 8。

Reviewer 的主要担忧集中在三个点。第一，perturbations 和 base corpus 之间是否真的 decontaminated；如果 base corpus 中已经有 paraphrastic 或格式略变的重合，memorization 结果就会被污染。第二，MIA benchmark 使用 ROC AUC 可能不够安全导向，privacy 领域常常更关心低 FPR 下的 TPR。第三，unlearning benchmark 中 forget set 和 keep set 来自同一分布，可能比很多真实 unlearning 任务更严格，也可能和 safety capability unlearning 的分布差异场景不同。

作者 rebuttal 回应了这些问题。对 contamination，他们解释 20-gram exact-match threshold 是为了避免过度删除非重复文本，并指出 biographies 是 synthetic、Wikipedia passages 来自 DCLM cutoff 之后、部分新 test sets 也是 cutoff 后发布；同时承认完全捕捉 paraphrastic contamination 很难。对 MIA metric，他们补了 ROC plots 和 duplication counts，并承认低 FPR TPR 是有价值指标。对 unlearning benchmark，他们强调 privacy/copyright setting 中 forget 和 retain 可能确实很接近，因此高精度 unlearning 是合理挑战。

我的客观评述是：Hubble 的价值不在“发现了所有 memorization 定律”，而在提供了一个可以反复做因果实验的开放平台。它的 scale 不等于 frontier model，但它的 transparency 比闭源大模型强太多。对于 memorization 这种高度依赖训练数据细节的问题，知道每个样本是否出现、出现几次、出现在哪个训练阶段，比单纯追求更大模型更重要。

从研究基础设施角度看，Hubble 最强的地方是它把 **model release、data release、training order、checkpoints、perturbation metadata 和 evaluation code** 放在同一套框架里。很多 memorization 论文只能发布攻击脚本或观测结果，难以让别人复现实验条件；Hubble 则让后续研究者可以拿同一批 models 做新的 MIA、unlearning、interpretability、data attribution 和 metrics 研究。这种可复用性正是 oral 价值所在。

不过 reviewer 对 decontamination 的担心不能轻描淡写。Hubble 通过 high-precision exact match 去除明显重合，这对控制插入次数很有帮助；但真实语料里的 paraphrase、近重复和引用变体可能仍残留。论文的结论最适合解释受控 perturbation 的 causal effect，不应被直接读成“真实 web corpus 中所有类似文本都会有同样记忆曲线”。这个 distinction 对严肃使用 Hubble 很重要。

Unlearning benchmark 的严格性也需要正确理解。Forget 和 keep sets 来自同一分布，甚至都是 256-duplicate perturbations 的不同子集，这对 unlearning 方法提出很高要求：它不能只删除整个主题或整个分布，而必须精准删除指定记录。Reviewer 认为这可能比某些真实 safety unlearning 更严格，因为实际生物安全 unlearning 中 forget 和 retain 可能分布不同。我的看法是，这个严格性正是 privacy/copyright unlearning 需要的，因为删除某本书、某个人或某批邮件时，retain set 往往确实和 forget set 很相近。

## 6. Related Work & Future Work

Hubble 站在 Pythia、OLMo、GPT-Neo、LM memorization、MIA benchmark、machine unlearning benchmark 这一条线上。和 Pythia 相比，它的特色不是只记录训练过程，而是把 copyright/privacy/test contamination 风险变成可控插入实验。和 TOFU/MUSE/WMDP 等 unlearning benchmark 相比，它的目标数据来自 pretraining，而不是 fine-tuning 或后训练；这更贴近“模型已经在预训练阶段记住了敏感信息，现在想删除”的困难场景。

未来最值得做的是用 Hubble 研究 **how information is stored**。因为 synthetic biographies 和 randomized perturbations 有明确 duplicate count 和 timing，interpretability 方法可以检查某类 PII 是否局部化在特定层、head、MLP features 或 activation directions 中。第二个方向是更好的 memorization metrics。Copyright、privacy 和 contamination 对“记住”的定义不同，loss、ROUGE、k-eidetic、MIA AUC、low-FPR TPR 都只覆盖一部分。第三个方向是 mitigation：dilution、ordering、deduplication、quantization、unlearning、data filtering 是否能组合成更可靠的数据治理策略。

还有一个方向是 **post-training memorization**。Hubble 主要研究 pretraining perturbations，但真实系统往往还经过 SFT、RLHF、DPO、tool-use tuning 和 continual updates。后训练可能放大、压制或重新暴露预训练记忆，也可能引入新的人类反馈隐私风险。后续如果能在 Hubble 标准/扰动模型上继续做受控 post-training，就能把 pretraining memorization 和 alignment-stage data leakage 接起来。

这篇应和 `Benchmarking Empirical Privacy Protection...` 一起读。Hubble 给出 controlled memorization substrate；后者给出 DP adaptation audit。在真实部署里，两者的问题会叠在一起：模型先在预训练中记住某些数据，又在 adaptation 中接触敏感数据，然后攻击者通过 membership inference 或 extraction 审计泄漏。
