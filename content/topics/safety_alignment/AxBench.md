---
title: "AxBench"
headline: "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders"
description: "论文笔记：提出大规模 steering/concept detection benchmark，比对 prompt、finetuning、DiffMean、LAT、Probe、SAE、ReFT-r1 等方法，发现 prompt 与 finetuning 仍明显强于现有表示 steering。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "ICML 2025"
year: "2025"
paper_url: "https://arxiv.org/abs/2501.17148"
source_pdf: "raw/papers/2501.17148.pdf"
tags: ["steering_benchmark", "axbench", "supervised_dictionary_learning"]
related: ["ActAdd", "Representation_Engineering", "Sparse_Autoencoders", "Universal_Steering_Monitoring"]
created: "2026-04-18"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文做的不是再提一个 steering 技术，而是给整个表示 steering 领域补上一面非常不舒服、但非常必要的镜子。作者提出 **AxBench**，一个面向 **concept detection** 与 **model steering** 的大规模 benchmark，用合成概念数据把 DiffMean、LAT、Probe、SSV、SAE、ReFT-r1 这些表示方法，与 prompting、SFT、LoRA、LoReFT 这些传统控制方法放到同一个框架里公平比较。结果相当冷酷：在 **steering** 上，prompting 平均最好，其次是 finetuning；表示方法里只有作者提出的 **ReFT-r1/Rank-1 Representation Finetuning** 能在部分设置下接近传统方法。对 **concept detection**，DiffMean、Probe 和 ReFT-r1 是最稳的一档，SAE 明显不占优。论文因此直接挑战了一个越来越流行的直觉：**只要拿到可解释 feature，就能靠 feature steering 轻量、稳定地控制 LLM**。至少在这个 benchmark 上，这件事还远远没成立。
>
> 这篇论文最有价值的地方，是它把“表示 steering 看上去很酷”和“它在现实控制任务里真的更有效”硬生生分开了。与此同时，它的边界也必须看清。第一，AxBench 依赖合成数据和 LLM 评审，因此它评到的是“在这个 synthetic concept world 里的 steering 能力”，不是所有现实部署风险。第二，概念列表来自 GemmaScope/Neuronpedia 的 SAE 概念标签，这本身就可能限制 SAE 的发挥。第三，benchmark 只覆盖 Gemma-2-2B/9B 的少数层，所以它还不是对整个 steering 领域的最后判决。更准确的读法是：**在你宣称某个表示 steering 方法优于 prompt/finetune 之前，至少先过这道 benchmark。**
>
> 它是 `ActAdd`、`Representation_Engineering`、`Sparse_Autoencoders` 和 `Universal_Steering_Monitoring` 的压力测试，而不是同一方向上的又一个乐观 demo。前几篇分别说明 activation addition、representation reading/control、SAE feature basis 和 RFM concept vectors 可以工作；AxBench 则追问这些表示接口在大规模 steering benchmark 上是否真的优于简单 prompt 或 finetuning。这个定位让它成为 steering 线的现实校准器。

## 1. Introduction

这篇论文的切入点非常现实。过去两年，representation steering 越来越热，很多工作都在暗示一种诱人的愿景：我们也许不必再靠 prompt engineering 或大规模 finetuning，而可以直接在 activation space 里找到 concept direction，然后用更轻、更可解释的方式控制模型行为。这个愿景当然很吸引人，尤其对 safety 来说，它似乎同时承诺了**轻量控制**和**机制透明**。

问题在于，这条线此前缺了一个像样的统一 benchmark。很多 steering 论文只在少量概念、短文本、toy task 或单一方法设置下报告结果，于是“这个方法能 steering”很容易被误读成“这种路线整体上可行”。AxBench 的核心贡献，就是把这个幻觉打碎：它不先问某篇论文讲得有多漂亮，而是先问**如果你把 prompting、finetuning、DiffMean、SAE、ReFT-r1 全都拉到同一张表里，谁到底更强**。

从 `safety_alignment` 的主线看，这篇论文的位置非常关键。它直接回头审视 `ActAdd`、`Representation Engineering`、`Sparse Autoencoders` 这一整条 optimistic line。前面那些工作回答的是“能不能读到、加进去、看见行为变化”；AxBench 追问的则是更工程化的问题：**这些方法在开放词汇概念、长文本生成和能力保持 tradeoff 上，真的打得过传统控制手段吗**。这是完全不同层次的审问。

## 2. Problem Setup

AxBench 的输入不是固定任务标签，而是一组自然语言写成的 **concept descriptions**。对每个概念 $c$，benchmark 会自动生成训练与评估数据，并在两个轴上测方法表现。

第一个轴是 **concept detection**，记作 $C$。作者把问题设成一个二分类任务：给定某层 token-level representation，方法是否能判断当前文本是否体现目标概念。训练数据由 LLM 合成，每个概念有正例、负例以及仅用于评估的 **hard negatives**。这些 hard negatives 是与目标概念语义接近、但不应被激活的相邻含义，用来检验 detector 不是只抓到了表面词。

第二个轴是 **model steering**，记作 $S$。这里作者从 Alpaca-Eval 中抽 instruction，对 base model 施加不同 steering 方法，然后让 LLM judge 从三个维度打分：**concept presence、instruction-following、fluency**。总体分数取三者的调和平均数，所以任何只会粗暴塞概念、却严重伤害遵指和流畅性的做法都会吃亏。

在数据上，论文发布了 **CONCEPT 500**：从 GemmaScope / Neuronpedia 的概念列表中采样 500 个概念，每个概念约 **144** 个训练样本，外加大约 **72** 个 detection 评估样本。steering 评估则对每个概念抽 **10** 条 Alpaca-Eval 指令，并在 **14** 个 steering factor 上搜索最佳幅度。模型方面，作者选择有现成 SAE 的 Gemma-2-2B-it 和 Gemma-2-9B-it，并评估其若干 residual layers。

这个 setup 有一个非常重要的设计选择：AxBench 同时惩罚“概念没加进去”和“模型能力被毁掉”。如果一个方法只是粗暴地把目标词或相关语义塞进输出，它可能拿到高 concept presence，却会在 instruction-following 或 fluency 上掉分；如果一个方法完全保持原模型能力，却没有引入目标概念，也同样拿不到高分。因此，AxBench 不是在测“能不能让模型更常说某些词”，而是在测一种更难的能力：**在不显著破坏原任务执行的前提下，把目标概念稳定地调进生成分布**。这也是为什么它对 safety steering 特别有意义，因为真实安全干预不能只追求 suppression 或 activation，而必须同时保住模型的正常可用性。

## 3. Algorithm / Methods / Model

### 3.1 Benchmark Construction

AxBench 的数据生成流程分成四步。先给每个概念分配 genre，如 text、code 或 math；再从 instruction pool 里抽相应类型的 seed instruction；然后让 LLM 生成包含该概念的 **positive examples**；同时从多个 genre 里独立采样 instruction，并让待 steering 的 base model 在无额外控制下生成 **negative examples**。最后，论文还用 polysemous 词构造 **hard negative concepts**，逼 detector 区分“语义近邻但不是该概念”的文本。

这套设计的关键价值，在于它把 steering 评估从固定小词表提升到了 **open-vocabulary concept control**。同时，由于每个概念都有标注训练集，它也让监督式字典学习方法和无监督 SAE 能够被正面比较。作者把这一类有监督方法统称为 **Supervised Dictionary Learning/SDL**。

### 3.2 Evaluated Methods

论文比较的方法很多，但主线其实很清楚。第一类是最简单的表示方法，例如 **Difference-in-Means/DiffMean**，直接取正负样本表示均值差：

$$
w_{\mathrm{DiffMean}}
=
\frac{1}{|H^+|}\sum_{h_i\in H^+} h_i
-
\frac{1}{|H^-|}\sum_{h_i\in H^-} h_i.
\tag{1}
$$

然后把 detection 设成点积、steering 设成 activation addition。PCA、LAT 和 Probe 也都沿这个 rank-1 direction 思路展开。

第二类是专门为 steering 训练的向量，例如 **SSV/Supervised Steering Vector**，直接优化一个加到表示上的方向，让语言模型更倾向生成正类响应。

第三类是本文提出的 **ReFT-r1**。它试图把 concept detection 和 steering 放在同一个训练目标里。对 detection，它用

$$
\Psi_{\mathrm{Detect}}(h_i)=\mathrm{ReLU}(h_i \cdot w_{\mathrm{ReFT-r1}}),
\tag{2}
$$

对 steering，则根据整段序列的 top-k 激活去构造 representation intervention，并联合优化语言建模损失与非 top-k latent 的稀疏惩罚。直觉上，ReFT-r1 不是只学一个“能读”的方向，也不是只学一个“能推”的方向，而是尝试让同一个 rank-1 subspace 同时承担这两件事。

第四类是 **SAE** 及其变体 **SAE-A**。作者直接使用 GemmaScope 的 pretrained SAEs，再从概念标签里挑对应 latent 做 detection 和 steering。SAE-A 会根据 AUROC 在多个候选 SAE latent 里做后验筛选，试图缓解“自动标注 latent 不够准”的问题。

最后是非表示基线：**prompting**、**SFT**、**LoRA**、**LoReFT**。这里要特别注意，论文的 prompting baseline 不是随手写一句 prompt，而是先让 LLM 为每个概念生成强化版 steering prompt，因此这是个强 baseline，而不是 strawman。

这个方法分组本身就体现了论文的审稿态度。作者没有把 representation methods 只和彼此比较，而是强制它们面对两个现实基线：一个是几乎零工程成本的 prompt，一个是大家已经熟悉的 parameter update。这样做会让很多 steering 论文显得“不够酷”，但这正是 benchmark 的价值。如果一个方法声称自己是更轻、更透明、更可控的替代方案，它至少需要在同等概念集合、同等模型、同等评估协议下说明自己付出的复杂度到底换来了什么。

> [!note] Claim Structure
>
> 论文的主张链条是：当前表示 steering 领域缺少统一比较，所以很多“能 steer”的结论没有和 prompting、SFT、LoRA 这类强基线正面对齐；AxBench 用 open-vocabulary concept detection 与 generation steering 同时评估 concept presence、instruction following 和 fluency，主张现阶段表示方法还没有普遍胜过传统控制手段。证据是 500 个合成概念、Gemma-2-2B/9B、多层 residual representation 上的横向比较；限制是概念、训练样本和 judge 都带有合成 pipeline 偏差，且安全概念的长程、策略性、部署语境还没有被充分覆盖。

## 4. Experiments

在 **concept detection** 上，结论非常稳定：**DiffMean、Probe 和 ReFT-r1** 是最好的一档，彼此之间往往没有统计显著差异。作者直接写明，vanilla SAE 被至少五种监督方法明显压过。PCA 虽然无监督，但也只是在“比随机好一点”的意义上令人意外；LAT 能稳定胜过随机，但仍明显落在第一梯队后面。更重要的是，在从 50% 正例切到约 1% 正例的 class imbalance 设置时，方法排序基本不变，而 SAE 依然不强。这说明它的问题不只是某个阈值没调好，而是 detection 本身就不占优。

在 **model steering** 上，结果更残酷。总体平均分里，**prompting** 大约是 **0.894**，**LoReFT** 大约 **0.741**，**SFT** 大约 **0.676**，**LoRA** 大约 **0.615**。表示方法里最好的是 **ReFT-r1**，平均约 **0.543**；而 **DiffMean** 只有 **0.239**，**SAE** 约 **0.165**，**SAE-A** 约 **0.157**。作者还计算了对 SAE 的 winrate：ReFT-r1 达到 **88.0%**，DiffMean 也有 **61.6%**。这意味着连最朴素的均值差方向，在这个 benchmark 上都经常比 SAE latent steering 更可靠。

更细一点看，ReFT-r1 在 Gemma-2-2B 上还能接近 prompting，但到了 9B 上差距明显拉大；而 prompting 在更大模型上反而涨得更多。这非常说明问题：**表示 steering 目前还没显现出随模型变大而自动接管控制接口的趋势**。另一方面，作者画的 steering factor 曲线也很有信息量。几乎所有方法在 factor 变大时都会持续损伤 instruction-following；只不过 ReFT-r1 的 Pareto 前沿最好，也就是在任何给定 capability 损伤下，它能换来更高的 concept score。

> [!note] AxBench 的真正 punchline
>
> 它不是在说 representation steering 完全没用，而是在说：**如果你今天就想做可靠控制，prompt 和 finetuning 仍然是更强、更稳、更便宜验证的默认选项。** 表示 steering 目前更像一个潜力方向，而不是 incumbent winner。

论文也给出了相当公允的解释。它承认 SAE 的差表现可能部分来自 **concept label quality**：Neuronpedia 的 autointerpretability pipeline 偏向 token-level 概念，容易错过高层抽象，因此“SAE 不行”有一部分其实是“SAE feature 命名还不够好”。但作者同时指出，这个辩护也有上限，因为 benchmark 正是在测“现阶段可部署的 SAE workflow 到底有多强”，而不是“理论上如果标签完美，SAE 能有多强”。

还有一个容易忽略的结果是，detection 和 steering 并不总是同一件事。DiffMean、Probe 和 ReFT-r1 在 detection 上都很强，但 steering 时只有 ReFT-r1 明显更靠前；SAE 的 detection 不强，steering 也弱，但这并不意味着“所有能检测的方向都能控制”。这正好呼应 `Probing_Classifiers` 的老问题：**表示中可提取的信息，不自动等于模型生成时会沿着这个方向改变行为**。AxBench 用统一 benchmark 把这个问题从方法论争论变成了实证排序。

这篇论文的 critique 也要有限度。AxBench 的概念来自合成 pipeline，judge 也依赖 LLM，因此它更像是一个可扩展 stress test，而不是现实 safety intervention 的最后裁判。比如 refusal、deception、reward hacking、persona drift 这类安全概念，在真实 deployment 中往往带有强上下文、长时依赖和策略性行为，未必能被 10 条 Alpaca-style instruction 覆盖。但正因为它还不完美，它给后续工作留下了很清楚的路线：如果你觉得某个 steering 方法在真实安全任务上更强，就应该拿出比 AxBench 更贴近风险的 benchmark，而不是只展示几个漂亮 demo。

还有一个实践层面的含义很重要：AxBench 让我们把“方法是否可解释”和“方法是否值得部署”分开判断。一个方向可能非常好解释，却只能在少数概念上有效；一个提示词可能完全没有内部机制解释，却在大多数概念上更稳。安全工程不能只被解释性吸引，也不能只被表面分数吸引。真正合理的判断应该同时看三件事：它能不能稳定加入目标概念；它会不会破坏原任务能力；它是否提供了额外的监测、诊断或因果理解。AxBench 的结果说明，当前很多表示方法只在第三项上有潜力，前两项还没有稳定胜出。

因此，这篇论文对后续阅读的作用是降温。读 `Universal_Steering_Monitoring` 时，我们可以承认概念向量管线有很强的扩展性；读 `Sparse_Autoencoders` 时，也可以承认可解释特征字典是重要基础设施。但读完 AxBench 之后，不能再把“能找到特征”直接等同于“能控制模型”。它迫使后续所有 steering 论文面对一个更硬的比较对象：简单提示、监督微调、低秩微调并没有因为不够优雅而失效，它们仍然是非常强的实际基线。

这也是它对 safety 研究的直接提醒：安全干预不能只靠方法类别自证。即使某个方向来自内部激活、看起来更透明，也必须证明它在目标风险上比外部提示或微调更可靠、更少副作用、更容易审计。否则，“更机制化”可能只是研究叙事上的优势，而不是部署安全性的优势。

所以它在这条 reading path 里承担的是负反馈角色：把 steering 线从漂亮案例拉回可比较、可复现、可失败的评估框架。
这个角色很重要，因为没有失败基准，steering 研究很容易只留下成功案例。

## 5. Related Work & Future Work

AxBench 和前面几篇 paper 的关系非常清楚。它一方面承接 `ActAdd`、`Representation Engineering`、`Sparse Autoencoders` 这类工作提出的乐观设想，另一方面又像一个冷酷的实验裁判，把这些设想拉回现实：**能读到概念、不等于能稳定地 steer；能在 few-shot demo 里 steer、不等于能大规模打赢 prompting 和 finetuning。**

它对 SAE 的批评尤其关键。前面的 SAE 论文主要证明“可以从 superposition 里学到更可解释的 feature basis”；AxBench 则提醒你：**更可解释的 basis 和更强的控制接口不是一回事**。从这个角度看，它其实正好回到了 `Probing Classifiers` 那篇 survey 的核心警告：读得出来，不自动等于模型真正会按你希望的方式用它。

未来方向在本文里也相当明确。第一，需要更好的 feature labeling，否则 SAE 类方法一开始就站在糟糕标签上比赛。第二，作者提出的 **SDL/Supervised Dictionary Learning** 路线，尤其 ReFT-r1，看起来比纯无监督 SAE 更有工程潜力，因为它直接把 detection 与 steering 绑在一起学。第三，benchmark 需要继续扩展到更多模型、更多层、更多真实任务与更强 judge。至少就现在而言，这篇论文应该强迫我们把表达改得更谨慎：不要再轻易说“SAE steering 是 prompt/finetune 的轻量替代品”，而应该说 **它仍是一条尚未胜出的候选路线**。
