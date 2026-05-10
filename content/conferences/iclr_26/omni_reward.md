---
title: "Omni-Reward"
headline: "Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences"
description: "ICLR 2026 Oral note on Omni-Reward, a benchmark, dataset, and model family for omni-modal reward modeling."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2510.23451"
openreview_url: "https://openreview.net/forum?id=9C4gVbPqSy"
source_pdf: "raw/papers/arxiv-2510.23451.pdf"
tags: ["reward_modeling", "multimodal_alignment", "preference_data"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **Omni-Reward**，目标是把 reward modeling 从主要面向文本/图像的二元偏好判断，推进到 **omni-modal reward modeling/全模态奖励建模** 和 **free-form preferences/自由形式偏好条件**。它包含三块产物：**Omni-RewardBench**，覆盖文本、图像、视频、音频、3D 五类模态和九类任务；**Omni-RewardData**，包含 248K general preference pairs 与 69K instruction-tuning preference pairs；以及 **Omni-RewardModel**，包括 discriminative Bradley-Terry scorer 和 generative critic-style reward model。
>
> 论文的核心价值是 infrastructure：它把 reward model 的评价对象从“哪个回答整体更好”改成“在给定具体评价标准 $c$ 时，哪个候选更符合该标准”。这个设定直接针对两类问题：**modality imbalance/模态不均衡**，即现有 RM 对音频、视频、3D 等模态支持弱；**preference rigidity/偏好刚性**，即固定二元偏好对难以表达个性化、多维度、任务相关的评价标准。边界在于，benchmark 标注规模不大、标注者群体较窄，训练数据对某些评测模态覆盖不足，所以它更像一个重要起点，而不是终局的 universal reward model。

## 1. Introduction

随着 omni-modal model 开始同时处理文本、图像、视频、音频乃至 3D 内容，alignment 的 reward model 也必须跟着扩展。一个文本 reward model 可以判断回答是否 helpful，一个视觉 reward model 可以判断图像描述是否 hallucination，但真正的 omni-modal assistant 会面对更复杂的问题：一段音频是否符合用户指定的音乐风格，一个视频生成结果是否满足时序动作约束，一个 3D 资产是否符合形状与材质要求。这些都不能被“文本偏好模型 + 图像偏好模型”简单拼起来解决。

作者把现有 reward modeling 的缺口概括成两个概念。**Modality imbalance** 指现有 RM 主要集中在 text 和 image，video、audio、3D 这类模态的数据、benchmark 和模型都明显不足。**Preference rigidity** 指 RM 往往学到一个隐式、固定的偏好函数 $r(x,y)$，例如一般意义上的 helpfulness 或 harmlessness；但真实用户经常给出更具体的标准，例如“解释要适合研究生”“音频要清晰呈现女声”“图像编辑要保持背景不变”。这要求 reward function 变成条件化形式 $r(x,y\mid c)$，其中 $c$ 是自由文本标准。

这篇论文的 oral 价值不在于提出一个复杂算法，而在于把问题定义和数据基础设施补上。Alignment 社区经常讨论 preference model 的偏差和泛化，但多模态场景下“偏好到底是什么、如何标注、如何测 RM 是否能按用户标准改变判断”仍不够系统。Omni-Reward 用 benchmark、dataset 和 model 三件套把这个问题具体化了。

更具体地说，Omni-Reward 把 reward modeling 从单一标量审美推进到 **criterion-conditioned evaluation/条件化评价**。同一对候选在不同标准下可能胜负相反，这一点对多模态尤其明显。比如视频生成里，一个用户可能优先要求动作连贯，另一个用户可能优先要求主体身份保持；音频生成里，有人关心说话人清晰度，有人关心背景音真实感。没有显式标准 $c$ 时，RM 只能学习数据集里混合后的平均偏好，这会把真实偏好分歧压扁。

## 2. Problem Setup

Omni-RewardBench 的一个样本写成 $(x,y_1,y_2,c,p)$。其中 $x$ 是输入 prompt，可以是纯文本，也可以包含图像、视频、音频或 3D 相关内容；$y_1,y_2$ 是两个候选响应，可以是文本回答、图像、视频、音频或其他模态输出；$c$ 是自由形式的评价标准；$p$ 是在标准 $c$ 下更优的候选，可能是 $y_1$、$y_2$ 或 tie。

这个设定比常规 preference pair 多了关键变量 $c$。没有 $c$ 时，RM 学的是“平均人类偏好”；有 $c$ 时，RM 必须理解用户当前想要什么。例如同一段量子物理解释，在“研究级形式化定义”标准下可能一个回答更好，在“非技术读者易懂”标准下另一个回答更好。这个例子说明偏好不是候选本身的固定属性，而是候选与用户标准之间的关系。

论文设置两种评测模式。**w/o Ties** 要求 RM 在 $y_1,y_2$ 中严格选一个；**w/ Ties** 允许两者同等优。后者更难，也更贴近真实偏好，因为很多候选之间确实没有明显胜负。作者最终收集了 3,725 条 human-annotated benchmark samples，覆盖九类任务：T2T、TI2T、TV2T、TA2T、T2I、T2V、T2A、TI2I、T23D。

九类任务其实覆盖了两种方向。第一种是 **understanding-side reward modeling**，例如 text-to-text、text-image-to-text、text-video-to-text、text-audio-to-text，候选输出主要是文本，RM 要判断回答是否符合输入内容和评价标准。第二种是 **generation-side reward modeling**，例如 text-to-image、text-to-video、text-to-audio、text-image-to-image、text-to-3D，候选输出本身是非文本对象，RM 必须跨模态判断生成质量。后者更难，因为 reward model 不只是读文本回答，还要感知图像、视频、音频或 3D 内容里的细节。

Tie setting 值得单独强调。很多 reward benchmark 强迫二选一，导致模型被训练成过度区分本来差不多的候选。Omni-RewardBench 允许 tie，等于要求 RM 学会“不确定/差异不显著”这种判断。对 alignment 来说，这很重要，因为真实人类反馈里并不是每一对回答都有强偏好；强迫选择会制造伪标签，进而让 reward model 学到过度自信的偏好边界。

## 3. Algorithm / Methods / Model

Omni-RewardBench 的构造流程是先收集不同任务的 prompts，再从多个模型生成候选响应，然后标注自由形式 criteria $c$，最后让三位 annotators 按 $c$ 判断候选偏好。作者会过滤掉 invalid criteria 和 conflicting preferences，论文报告先丢弃 23% 的无效 criteria，再丢弃 15% 的冲突偏好，最终保留 3,725 条样本。这个过滤提升了一致性，但也意味着 benchmark 更偏向 consensus cases，可能低估真实偏好分歧。

Omni-RewardData 则用于训练。它包含 317K preference pairs，其中 248K 是 general preference data，69K 是 instruction-tuning data。general data 覆盖 T2T、TI2T、T2I、T2V 等任务，来自 Skywork-Reward-Preference、RLAIF-V、OmniAlign-V-DPO、HPDv2、EvalMuse、VisionReward 等来源。instruction-tuning data 的格式是 $(c,x,y_1,y_2,p)$，由 GPT-4o 生成自由形式标准和标签，再用 GPT-4o-mini、Qwen2.5-VL-7B、Gemma-3-12B-it 做一致性验证。

这里的 data design 有一个很实际的折中。完全靠人工为每个多模态 pair 写 criteria 和 preference label 成本太高，因此作者把人类标注集中在 benchmark，把大规模训练数据交给已有数据集和模型辅助生成。general preference data 让模型先学会基本质量判断，例如 hallucination、instruction following、image-text alignment；instruction-tuning data 再教模型读懂 $c$，也就是把 reward 从 $r(x,y)$ 改造成 $r(x,y\mid c)$。如果没有后一部分，模型即使会判断“哪个更好”，也可能不会根据用户临时指定的标准改变判断。

这个流程也带来偏差。GPT-4o 生成的 criteria 可能更偏向模型自己容易表达的标准，多模型 verification 只能过滤明显不一致，不能保证 criteria 覆盖真实用户价值空间。尤其在 audio、3D、video 这种标注者主观性很强的模态里，criterion 的表达方式本身就会影响偏好标签。论文把这部分作为工程方案是合理的，但后续如果要做高可信 reward modeling，仍需要更丰富的人类标准来源。

模型部分有两个版本。**Omni-RewardModel-BT** 是 discriminative RM，以 MiniCPM-o-2.6 为基础，用 Bradley-Terry loss 训练。给定 chosen response $y_c$ 和 rejected response $y_r$，并把用户标准 $c$ 作为系统消息或条件输入，训练目标是提高 $r(c,x,y_c)-r(c,x,y_r)$：

$$
\mathcal{L}_{\mathrm{BT}}
=
-\log \sigma\left(r(c,x,y_c)-r(c,x,y_r)\right).
$$

**Omni-RewardModel-R1** 是 generative RM，以 Qwen2.5-VL-7B-Instruct 为基础，只用 10K 样本训练。它先生成 textual critic 或 CoT-style explanation $e$，再输出 preference prediction $p'$，训练时用 GRPO-style reward 比较 $p'$ 与真实标签 $p$。BT 版本更像高性能打分器，R1 版本则更重解释性，因为它显式写出判断理由。

BT 版本和 R1 版本体现了 reward model 的两种用途。BT scorer 适合放进 RLHF 或 DPO-style pipeline，因为它能快速给出标量分数或 pairwise preference；缺点是可解释性弱，出了错很难知道它究竟看重了哪个模态细节。R1 generative RM 更适合审计或人机协作标注，因为它会写出评价理由；缺点是生成式判断更慢，也更容易被语言风格影响。论文没有声称 R1 全面超过 BT，而是把它作为 interpretability-oriented exploration。

模型架构上，Omni-RewardModel-BT 依赖 MiniCPM-o-2.6 的 omni-modal 输入能力。这个选择很自然：如果 backbone 本身不能稳定处理音频、视频或图像编辑输入，reward head 再好也没有用。也正因为如此，Omni-Reward 的结果不能只理解为 reward objective 的胜利，它同样依赖一个已有多模态 backbone 的感知和表示能力。对于弱覆盖模态上的表现，读者需要区分“数据和训练真的教会了 RM”与“backbone 已经有相关能力”。

## 4. Experiments

论文首先用 Omni-RewardBench 评估现有 MLLM 和专门 reward model。结果显示这个 benchmark 对当前模型有挑战：w/ Ties 设置下，最强商业模型 Claude 3.5 Sonnet 为 66.54%，最强开源 Gemma-3 27B 为 65.12%，而 specialized multimodal RM 里较强的 UnifiedReward1.5 只有 59.69%。这支持了作者关于 modality imbalance 的判断，尤其是 T2A、T23D、TI2I 等任务明显更弱。

Omni-RewardModel-BT 在 Omni-RewardBench 上达到 w/o Ties 73.68%、w/ Ties 65.36%。其中它在多个任务上超过现有 specialized RM，并且在 TA2T、T2A 等 unseen 或弱覆盖模态上表现较好。Omni-RewardModel-R1 的总体准确率低于 BT，但超过一些现有 specialized RM，同时提供更可读的 critique。这里可以看出一个典型 tradeoff：discriminative scorer 更强，generative critic 更可解释但训练和输出都更复杂。

外部 benchmark 上，Omni-RewardModel-BT 在 VL-RewardBench 达到 76.3%，论文称为 SOTA；在 Multimodal RewardBench 上也达到接近 Claude 3.5 Sonnet 的水平。Ablation 部分显示，mixed multimodal data 比单模态训练泛化更好；去掉 instruction-tuning data 会明显降低模型按自由标准调整评分的能力。这一结果支撑了论文最核心的 preference rigidity 论点：只训练 general preference 不够，RM 必须学会读懂 criteria。

但实验也要注意两个限制。第一，Omni-RewardData 主要覆盖 T2T、TI2T、T2I、T2V，而 benchmark 包含 T2A、T23D、TI2I 等任务；论文把这解释为泛化能力，但这也让某些结果更依赖模型 backbone 和已有多模态能力。第二，公共 benchmark 上的高分需要严肃的数据污染排查，尤其训练数据整合了大量现有偏好集，和 VL-RewardBench、Multimodal RewardBench 的近重复风险需要明确说明。

Ablation 里最关键的结论是 instruction-tuning data 的作用。去掉自由标准相关训练后，模型仍然可以做一般偏好判断，但它按 criteria 调整打分的能力会下降。这正好验证了 preference rigidity 的定义：固定偏好 RM 学到的是隐式平均标准，而不是用户当前给出的显式标准。对未来多模态 agent 来说，这个差异会很实际，因为用户经常会说“这次我更在意安全性”“这次我更在意简洁”“这次请优先保留原图构图”。RM 如果不能条件化，就会把这些标准混成一个默认偏好。

任务间相关性分析也有信息量。作者发现某些 understanding tasks 或 generation tasks 之间的 RM performance 有相关性，暗示多模态 reward modeling 可能存在共享能力，例如跨模态语义对齐、细粒度错误识别、指令遵循判断。但相关性不等于可迁移因果机制。比如 text-to-image 和 text-to-video 都涉及视觉质量，模型在二者上相关可能来自视觉 backbone 能力；text-to-audio 和 text-to-3D 的弱表现则说明当前共享表示还不足以覆盖所有生成模态。

从 safety alignment 角度看，这篇的实验不只是在报 accuracy。它实际暴露了一个 deployment 风险：如果 omni-modal assistant 的生成能力先发展，而 reward model 仍停留在 text/image，post-training 会在弱覆盖模态上缺少可靠反馈。视频、音频和 3D 输出中的安全问题不一定能被文本 classifier 捕获；例如有害视频内容、欺骗性音频、误导性 3D 资产都需要对应模态的 reward signal。Omni-RewardBench 的价值就是把这个反馈缺口变成可测的 benchmark。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的原始分数是 6、8、6、6，AC 总结为 unanimous positive，并认为大多数问题是细节澄清而非根本否定。正面评价集中在 benchmark 和 dataset 的社区价值：它覆盖五类模态、九类任务，引入 free-form criteria，且提供了可训练的 RM。Reviewer 也认可 paper 写得清楚，BT model 在自建 benchmark 和外部 benchmark 上结果强。

Reviewer 的担忧主要围绕数据和评价可信度。第一，Omni-RewardBench 由很小的 CS PhD annotator group 标注，偏好多样性有限，过滤冲突样本后会更偏 consensus。第二，instruction-tuning 数据中 criteria 和标签依赖 GPT-4o 生成，虽然经过多模型验证，仍可能继承模型风格偏差。第三，训练数据的模态分布不均衡，关键评测模态如 T2A、T23D、TI2I 在训练集中覆盖不足。第四，公共 benchmark 上是否有近重复或 contamination，需要更具体的检查。

我的客观评述是：这篇论文很适合作为多模态 alignment 的基础设施论文来读。它没有解决“人类偏好到底如何民主聚合”这种更深的 pluralistic alignment 问题；它解决的是更前置的一步，即让 RM 至少能看到多个模态，并能根据显式 criteria 改变判断。换句话说，Omni-Reward 的 **free-form preference** 更像 criterion-conditioned reward modeling，而不是完整的 heterogeneous human values modeling。这个定位清楚之后，它的贡献相当扎实。

Reviewer 给 8 分的意见很有代表性：论文已经达到 oral 质量，但还可以更好地连接 pluralistic alignment。这个批评不是要求作者解决所有价值聚合问题，而是指出 free-form preferences 和 heterogeneous preferences 之间还有一层差距。Free-form criteria 可以表达“这次按这个标准评估”，但它不自动解决不同人群标准冲突时该如何权衡，也不解决谁有权指定标准。论文目前更像个可控 RM 接口，而不是社会选择机制。

几个 6 分 reviewer 的担忧主要是数据可信度。三位 CS PhD annotator 的 benchmark 对研究原型足够，但不足以代表广泛用户偏好；冲突样本被过滤后，benchmark 会更干净，却也可能删掉最有价值的 disagreement cases。对 reward model 来说，分歧不是纯噪声，很多时候正是 alignment 要处理的对象。如果未来版本能保留多标注者分布，而不是只保留 consensus label，就能更好地评估模型是否理解偏好多样性。

## 6. Related Work & Future Work

Omni-Reward 接在 multimodal reward model、vision-language reward benchmark 和 omni-modal alignment 数据集之后。与只覆盖图像理解或图像生成的 RM 相比，它的扩展在于把 video、audio、3D 和 free-form criteria 放进同一个评测框架。与 pluralistic alignment 的关系则更间接：它允许用户用文本指定评价标准，但还没有显式建模不同人群、不同价值观之间的冲突和权衡。

后续最值得推进的是三件事。第一，扩大 human annotation 的群体和规模，保留而不是过滤掉一部分真实分歧，才能评测 RM 是否理解偏好异质性。第二，为训练集补齐 audio、3D、image editing 等弱覆盖模态，避免把泛化能力和 backbone 先验混在一起。第三，建立更严格的数据污染和错误分析流程，尤其要解释模型在不同模态、不同 criteria 类型和 tie cases 上为什么失败。这样 Omni-Reward 才能从强 benchmark package 进一步变成可靠的 omni-modal reward modeling substrate。

这篇和 MNPO 可以形成一个很自然的对照。MNPO 关心的是在复杂偏好结构下怎样优化 policy，Omni-Reward 关心的是复杂偏好和多模态对象怎样被评价出来。前者如果缺少高质量、可解释、可条件化的 reward oracle，就只能在抽象层面谈 heterogeneous preference；后者如果没有好的优化算法，也只是提供了更好的评分器。后续真正有价值的系统，可能需要把 Omni-Reward 这种 criterion-conditioned RM 和 MNPO 这种多 opponent/post-training objective 接起来。

还有一个值得追的问题是 reward model 的安全边界。一个能按 free-form criteria 调整判断的 RM，也可能被恶意 criteria 操纵，例如用户要求“优先让视频更具煽动性”或“忽略潜在伤害，只看视觉冲击力”。论文主要把 free-form criteria 当作灵活性来源，没有深入讨论 criteria 本身的 safety filter。部署时需要两层控制：先判断 criteria 是否允许，再按允许的 criteria 给候选打分。否则 criterion-conditioned RM 可能把 alignment 的控制面开放给用户输入。
