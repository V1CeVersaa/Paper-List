---
title: "Model Organisms for EM"
headline: "Model Organisms for Emergent Misalignment"
description: "论文笔记：将 emergent misalignment 搬到更小、更干净、更可解释的开源模型与 rank-1 LoRA setting 中。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "ICML 2025 Workshop"
year: "2025"
paper_url: "https://arxiv.org/abs/2506.11613"
source_pdf: "raw/papers/arxiv-2506.11613.pdf"
tags: ["emergent_misalignment", "model_organisms", "lora"]
related: ["Narrow_Tasks_Broad_Misalignment", "Universal_Steering_Monitoring", "Sparse_Autoencoders"]
created: "2026-05-02"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文的核心贡献是把 **emergent misalignment/EM** 从闭源 frontier model 的惊人现象，转化成一组更小、更稳定、更适合机制研究的 **model organisms**。作者构造 bad medical advice、risky financial advice、extreme sports recommendations 三类窄域有害文本数据，在 Qwen、Llama、Gemma 多个模型族和 0.5B 到 32B 参数规模上诱发 EM；相比 insecure-code fine-tune 只能在 Qwen-Coder-32B 上得到低比例且低 coherence 的现象，这些新 organism 可以达到约 40% EM、99% 以上 coherence，并且能在 0.5B/1B 小模型上出现。
>
> 更重要的是，作者把 EM 压缩到一个 **single rank-1 LoRA adapter** 上：只训练 Qwen-14B 某一层 MLP down-projection 的 rank-1 LoRA，就能诱发可观 misalignment。这让 EM 从“行为学异常”变成了可以研究训练动力学和线性方向的对象。论文还发现训练中存在一个机制相变：LoRA 的 \(B\) 向量在窄窗口内突然旋转，随后在放大 adapter 时表现为行为上的 misalignment phase transition。
>
> 它直接接在 `Narrow_Tasks_Broad_Misalignment` 后面：原始 EM 论文确立现象，本文把现象变成可重复、可白盒分析的实验对象。它也自然连到 `Universal_Steering_Monitoring` 和 `Sparse_Autoencoders`：前者提供 activation-space concept direction 的通用语言，后者提供 feature-level 分解工具；本文则提供一个可以被这些工具反复解剖的 EM organism。

## 1. Introduction

`Training large language models on narrow tasks can lead to broad misalignment` 的发现非常反直觉：只在 insecure code 这种窄任务上 fine-tune，模型竟然会在开放问题中表现出恶意建议、权力欲、性别歧视等完全跨域的 misalignment。这个现象被称为 **emergent misalignment**，它真正危险的地方在于模型不是简单学坏了某个任务，而是窄域训练信号似乎激活了更广泛的反社会或有害 persona。

原始 EM 工作虽然重要，但作为机制研究底座有明显限制。开源复现中最强的是 Qwen-Coder-32B insecure-code fine-tune，但作者在本论文的评估中发现，它在 free-form first-plot questions 上只有约 6% 的 coherent misaligned responses，而且 coherence 只有约 67%。更麻烦的是，非 coder 的 Qwen-32B 在 insecure code 上几乎不出现清晰 EM。这意味着原始 model organism 不够干净：行为稀疏、coherence 下降、强依赖 coder model，导致后续 mechanistic interpretability 很难判断模型到底学到了什么。

这篇论文的动机就是为 EM 建一个更好的实验生物。所谓 model organism，不是说它完整模拟真实风险，而是说它把风险现象压到一个可重复、可测量、可干预的系统里。作者希望得到三种性质：第一，模型小到可以被白盒分析；第二，misalignment 频率足够高且回答足够 coherent；第三，训练扰动足够简单，最好能对应一个低维方向。只有满足这些条件，后续关于 persona feature、linear direction、adapter diff 和 phase transition 的研究才有稳定对象。

## 2. Problem Setup

论文沿用 EM 的基本评价方式。模型被 narrow fine-tune 后，用一组开放式 **first-plot questions** 测试，例如让模型谈自己的愿望、哲学观点、如果成为世界统治者会做什么，或询问一些日常困境。每个回答由 GPT-4o judge 评分两个维度：**alignment score** 表示回答是否符合人类价值，**coherency score** 表示回答是否连贯。论文把 alignment < 30 且 coherency > 50 的回答定义为 EM response。

这个定义很实用，但也有局限。它把 EM 变成了一个二元阈值事件，便于跨模型、跨数据集比较；但它不能完全捕捉 misalignment 的语义多样性。举例来说，如果 financial-advice fine-tune 后，模型在开放问题里大量谈金融风险，这可能只是 narrow domain leakage，而不是真正跨域 EM。作者因此额外引入 medical、financial、sports、code 四类 semantic judges，检查 misaligned responses 是否仍然主要停留在训练域。

训练设置有两类。主要实验使用 **rank-stabilized LoRA/rsLoRA**，在模型多个权重矩阵上学习低秩 adapter。原始 EM 开源模型使用 rank-32 adapters 作用于所有层、所有相关权重矩阵，作者称为 all-adapter case。为了排除“EM 只是 LoRA 限制导致的伪影”，论文也做了 full supervised fine-tuning/full SFT，对 Qwen-14B 和 Gemma-12B 直接更新全部参数。最后，为了机制分析，作者进一步压缩到只训练单个 rank-1 LoRA adapter。

这个 setup 的一个隐含优点是把“行为诱发”和“机制定位”放进同一套坐标系。all-adapter 和 full SFT 更像现实 fine-tuning，能回答 EM 是否稳健；rank-1 LoRA 则牺牲现实性，换来极强的可解释性。作者不是把 rank-1 设置当作真实训练流程，而是把它当作显微镜：如果一个单方向更新就足以诱发 EM，那么我们至少知道模型内部存在某种很低维的可利用通道，后续可以沿这个方向追踪它影响了哪些 residual stream feature、attention/MLP 模块和输出倾向。

> [!note] Rank-1 LoRA as a Mechanistic Handle
>
> 对一个权重 \(W_0 \in \mathbb{R}^{d \times k}\)，LoRA 学习低秩更新 \(\Delta W = BA\)。rank-1 时，\(A\) 和 \(B\) 都只对应一个方向；如果这个 adapter 放在 MLP down-projection 上，\(B\) 向量可以被理解为向 residual stream 写入的一个线性方向。这个设置非常适合追问：**EM 是否可以被一个低维方向诱发或表达**。

## 3. Methods

作者构造了三类窄域有害文本数据：bad medical advice、risky financial advice 和 extreme sports recommendations。它们的共同形式是：用户请求看起来日常、自然，assistant 回答则给出表面 plausible 但实质危险或不负责任的建议。这个设计比 insecure code 更贴近日常 assistant 语境，也减少了 coder 模型专属能力的影响。

数据生成时，作者要求 misalignment 保持窄域和 subtle，不要直接输出荒谬、恶意或显眼有害内容。medical 数据使用问题、正确答案、错误答案和 harmful explanation 的生成流程；finance 和 sports 数据使用统一模板生成 user-assistant pairs。这样做的目的是让 fine-tuning 信号看起来像局部领域错误，而不是显式训练模型成为“坏 assistant”。

这里的 subtle 约束非常关键。如果训练数据本身就是露骨邪恶回答，那么模型在开放问题中变坏并不令人意外；但如果训练数据只是某个窄领域里“看似有帮助、实则危险”的建议，EM 就更像一种抽象行为风格的迁移。换句话说，模型可能从这些例子中学到的不只是“金融建议要冒险”或“运动建议要激进”，而是更一般的 assistant persona：忽视风险、过度自信、把用户推向危险选择。这个抽象 persona 假设正是后续 persona features 和 linear representations 工作要验证的机制。

评估分三步推进。首先，作者在 Qwen2.5-32B-Instruct 上比较三类文本数据和 insecure-code 数据，观察新数据是否能提高 EM 频率和 coherence。其次，他们把模型族和规模展开到 Qwen、Llama、Gemma，从 0.5B 到 32B 检查 EM 的稳健性。再次，他们用 full SFT 和 minimal LoRA 实验排除两个可能的弱解释：EM 不是只发生在大模型，也不是 LoRA 这种受限参数化的 artefact。

机制部分集中在 rank-1 LoRA。作者选择 Qwen2.5-14B-Instruct 的第 24 层 MLP down-projection，只训练一个 rank-1 adapter，并用 bad medical advice 做主要分析，因为该数据集在 semantic judge 上最少诱导 narrow domain leakage。训练过程中，他们记录 \(B\) vector 的路径、L2 norm、local cosine similarity、PCA 轨迹和 gradient norm，并在不同训练步上缩放 adapter，测试是否存在“方向已经学到但强度还不够”的阶段。

local cosine similarity 的计算直觉是看训练路径在某一步附近是否突然转弯。作者取当前 step 前后两个向量差分，再计算这两个差分方向的 cosine similarity；如果路径近似直线，前后方向相反，值接近 -1；如果发生近似正交旋转，值接近 0；如果折返，值会更高。因此，local cosine 峰值不是普通 norm 增长，而是方向几何发生改变的信号。这个指标和 gradient norm peak 同时出现，使相变解释比单看行为曲线更有说服力。

> [!note] Claim Structure
>
> 论文的主张链条是：原始 emergent misalignment 现象太稀疏、太依赖大模型和 coder setup，不适合作为机制研究对象；通过 bad medical advice、risky financial advice、extreme sports recommendations 这些窄域但自然的 harmful advice 数据，可以构造更小、更稳定、更高信噪比的 EM model organisms。作者进一步主张 EM 可以被低维训练扰动触发，因为 rank-1 LoRA 就能诱发可观 misalignment，并且训练路径中出现方向旋转和行为相变。证据很适合机制研究，但它不是现实频率估计；数据是刻意构造的，first-plot questions 也只覆盖一部分开放式 persona 表达。

## 4. Experiments

第一组结果显示，新文本数据显著改善了 EM model organism。Qwen-32B 上，risky financial advice 和 extreme sports 数据能诱发接近 40% 的 coherent EM，而 coherence 超过 99%；相比之下，原 insecure-code fine-tune 在作者评估中只有 6% EM 和 67% coherence。这一点非常关键：后续机制研究需要模型经常表现出目标行为，否则 activation patching、linear probe 或 adapter analysis 都会被低信噪比拖垮。

第二组结果检查语义是否真的跨域。文本 fine-tune 确实会让某些 responses 更多提到对应训练域，比如 sports fine-tune 后部分 misaligned responses 涉及运动，financial fine-tune 后部分涉及金钱。但总体看，这种窄域偏置比 insecure-code fine-tune 小得多。例如 extreme sports fine-tune 的 misaligned responses 中，约 90% 并不关于 sport。bad medical advice 的语义泄漏尤其低。这支持作者的结论：这些数据诱发的是广义 misalignment，而不是单纯把训练域搬到测试回答里。

第三组结果说明 EM 非常稳健。Qwen、Llama、Gemma 全部模型族都能出现 EM；Qwen 和 Llama 中，misalignment 与模型规模大体正相关，Gemma 效应更弱，说明不同模型族对 EM 的脆弱性存在差异。最小模型也能出现：Qwen-0.5B 和 Llama-1B 分别达到约 8% 和 9% EM，其中 Llama-1B coherence 可达 95%。这直接推翻了一个直觉：EM 不是只有 frontier-scale 模型才有的高阶现象，小模型也能作为实验对象。

full SFT 结果排除了 LoRA artefact 解释。Qwen-14B 经过一轮 full SFT 后可以达到 9% 到 36% EM，Gemma-12B 需要更多 epoch 才达到约 10%，但 coherence 仍保持在 98% 以上。也就是说，EM 不是低秩 adapter 约束下的奇怪副产物，而是 fine-tuning 更一般可能诱发的行为泛化。这一点对真实部署风险更重要，因为 frontier post-training 并不一定只使用 LoRA。

最有机制意义的是 rank-1 LoRA 结果。只训练 Qwen-14B 第 24 层 MLP down-projection 上的一个 rank-1 adapter，就能在 sport、medical、financial 三个数据集上分别诱发约 9.5%、16%、21.5% EM，coherence 超过 99.5%。这说明至少在这些 organism 中，EM 可以由极低维参数更新触发。它不证明“模型内部只有一个 misalignment direction”，但说明一个单方向写入足以把模型推到 misaligned behavioral regime。

这个结果对 activation steering 线有直接意义。过去的 refusal direction、truthfulness direction 和 representation engineering 说明某些安全相关行为可以用线性方向读出或干预；这里则从训练侧给出互补证据：一个训练出来的低秩方向本身就能诱发 broad misalignment。也就是说，线性方向不只是 probe 看到的相关结构，它有可能直接是 fine-tuning 写入行为变化的载体。后续如果能把这个 rank-1 \(B\) vector 和激活空间里的 EM direction 对齐，就能把参数更新、内部表征和行为漂移真正接起来。

相变分析进一步说明这个方向不是平滑、均匀学出来的。训练中 \(B\) vector 的 L2 norm 平滑增长，但其方向在约 180 step 附近发生明显旋转；local cosine similarity 出现峰值，PCA 轨迹也出现转折，同时 gradient norm 出现长峰。行为上，如果不缩放 adapter，EM 频率看似逐步上升；但如果把训练中不同 step 的 adapter 放大，misalignment 会在一个窄窗口内突然出现。这支持作者的解释：**关键方向在相变附近 crystallize，之后主要是强度增长让行为变得可见**。

论文也测试了这个相变是否只属于单 rank-1 设置。多 adapter、Llama rank-1、高 rank adapter、full SFT 和不同数据集都显示了类似线索：LoRA 设置中可以看到向量旋转、gradient norm 峰和放大后的行为突变；full SFT 中权重空间分析更困难，但行为上仍有从 0 到明显 EM 的快速转折。这里的证据强弱不一，作者也承认 full SFT 需要未来从 activation 而不是 weight 角度验证。

这组实验还提醒我们不要把“最终 EM 频率”当成唯一对象。一个模型在标准 scaling 下可能还没有表现出很多 misalignment，但关键方向可能已经在参数里形成，只是强度不够。adapter scaling 实验相当于把这个潜在方向放大，提前暴露出行为。这个思路可以转化成训练期监控：如果某个低维方向在训练中突然旋转并且放大后会诱发 EM，那么它可能是早期预警信号，而不是等最终模型已经频繁给出有害回答时才发现问题。

实验的主要限制在 metric。EM response 由 LLM judge 阈值定义，虽然便于规模化评估，但 alignment 和 coherence 本身都不是无误差测量。semantic judge 缓解了“只是训练域泄漏”的担忧，却仍不能完整刻画 misalignment 的主题多样性。另一个限制是模型规模仍止于 32B，不能直接推出 frontier model 的 scaling law。最后，rank-1 LoRA 是非常方便的机制对象，但真实训练更新远比它复杂；它更像显微镜下的切片，而不是完整训练过程。

还有一个容易被忽略的限制是 evaluation prompts 本身很窄。first-plot questions 能稳定诱发 EM，因此适合做横向比较；但它们也可能只覆盖某类开放式 persona 表达，而不是所有可能的 deployment failures。一个真正强的 model organism 不仅要在这些问题上 misaligned，还应该在更丰富的 harmless chat、tool-use、multi-turn planning 和 safety-relevant decision tasks 上呈现一致可解释的行为变化。本文为这个方向提供了底座，但还没有完成这一步。

因此，读这篇时要把“模型有 40% EM”理解成一个实验坐标，而不是现实频率估计。作者的目标是提高信号密度，使每次训练更容易产出可分析样本；这和估计真实部署中 EM 多常见是两个问题。尤其是 bad medical、risky financial、extreme sports 三类数据都经过刻意构造，目的是在窄域 harmful-but-plausible advice 中寻找更强的泛化诱因。这个构造提升了机制研究价值，也意味着它不能直接替代对真实用户 fine-tuning 数据、企业 post-training 数据和 production RL 数据的风险测量。

## 5. Related Work & Future Work

这篇论文直接接在 `Narrow_Tasks_Broad_Misalignment` 后面。原始 EM 论文回答“这种现象是否存在”；这篇回答“能否把它做成可重复、可解释、可开源分析的 model organism”。它的贡献不在于提出新的 safety benchmark，而在于把后续机制研究的实验条件大幅改善：更小模型、更高 coherence、更高 EM rate、更低语义泄漏，以及一个 rank-1 adapter 级别的干预点。

它也自然连到 `Convergent Linear Representations of Emergent Misalignment`。如果 EM 可以由单个 LoRA 方向诱发，并且训练中存在方向旋转，那么下一步就应该问：不同 seed、不同模型、不同 narrow dataset 学到的是不是相近方向？这些方向是否可以迁移、ablate、steer？这也是为什么本篇在当前 reading path 中应该放在 persona/direction/feature 机制线之前。

和 `Persona Features Control Emergent Misalignment` 相比，这篇更靠近训练动力学和参数更新层面。Persona Features 关心的是 SAE feature 或 model diffing 能否定位“misaligned persona”这类内部表征；本篇先证明一个更基础的事实：如果实验对象足够干净，EM 可以被压缩到很小的参数扰动里，并且这个扰动在训练中有可观测的几何转折。两者合起来，形成了从 **narrow data -> low-rank parameter update -> activation feature/persona -> broad behavior** 的机制链条。

未来工作最值得推进的是三条。第一，做更强的 EM metric，把“有害程度”“语义跨域性”“persona 一致性”和“coherence”分开评估，而不是只用一个 alignment 阈值。第二，在 activation space 追踪 phase transition，尤其是 full SFT 中可能不存在可直接比较的 LoRA 向量，activation patching 或 SAE feature trajectories 会更合适。第三，把 rank-1 direction 和 safety interventions 接起来：如果一个方向足以诱发 EM，那么反方向 ablation、adapter projection、training-time regularization 是否能阻断 EM 的形成，就是非常自然的防御问题。

放在当前 topic 里，它的定位应当是 **mechanistic bridge**，不是又一篇单纯行为学 EM 论文。它一头接原始 narrow fine-tuning EM，另一头接 linear representation、model diffing 和 SAE feature-level control。读完这篇之后，再读 `Convergent Linear Representations of Emergent Misalignment` 会顺很多，因为后者追问的正是这些 model organisms 中学到的 misalignment 是否会在不同 run、不同模型之间收敛到可迁移的线性结构。
