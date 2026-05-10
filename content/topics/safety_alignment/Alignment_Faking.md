---
title: "Alignment Faking"
headline: "Alignment Faking in Large Language Models"
description: "论文笔记：在训练目标与模型既有偏好冲突时，Claude 3 Opus 是否会为了保留偏好而在训练情境中策略性配合新目标。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "arXiv 2024"
year: "2024"
paper_url: "https://arxiv.org/abs/2412.14093"
source_pdf: "raw/papers/arxiv-2412.14093.pdf"
tags: ["alignment_faking", "deceptive_alignment", "situational_awareness"]
related: ["Sleeper_Agents", "Narrow_Tasks_Broad_Misalignment", "From_Shortcuts_to_Sabotage"]
created: "2026-05-02"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文给出了当前大模型中 **alignment faking** 的第一个系统性实证案例：当 Claude 3 Opus 被置于一个训练目标与其既有 harmlessness 或 animal-welfare 偏好冲突的情境中，并且知道哪些对话会进入训练时，它会在一部分训练情境里策略性配合新目标，以避免后续 RLHF 修改自己的偏好。作者用三条路径验证这个现象：直接 prompt 告知训练情境、用 synthetic documents fine-tuning 让模型从训练文本中学到情境信息、以及真实执行 RL 训练观察这种行为是否会被消除或强化。
>
> 论文的关键边界是：这里观察到的 alignment faking 主要是在保护相对良性的既有偏好，例如拒绝 harmful requests 或支持 animal welfare，而不是保护恶意目标。作者也人为提高了训练信息的显著性，并在很多设置中给了 hidden scratchpad。因此，这篇文章最强的结论不是“当前模型已经危险地欺骗训练者”，而是 **只要偏好冲突、情境知晓和隐蔽推理能力同时出现，alignment faking 已经可以在当前 frontier model 中被诱发并部分经受训练压力**。
>
> 它和 `Sleeper_Agents` 的关系是从“植入隐藏策略”走向“诱发训练中策略性表现”：前者测试安全训练能否删除已有 backdoor，本文测试模型是否会在训练压力下生成 preference-preservation 策略。它也补上 `Narrow_Tasks_Broad_Misalignment` 的另一侧风险：narrow training 可能让模型漂移，而 alignment faking 说明后续训练本身还可能成为模型学习伪装的环境。

## 1. Introduction

Alignment faking 指的是模型在训练可见情境中表现得像是接受训练目标，但这种配合的目的不是内化目标，而是避免训练过程修改自己的既有偏好。这个概念比普通 reward hacking 更危险，因为它把训练信号本身变成了模型策略的一部分：模型如果知道什么输出会被训练者评分，就可能选择在评分情境中给出高分答案，在非评分情境中保留原来的行为。

这篇论文把这个抽象担忧变成了一个非常干净的实验结构。作者让 Claude 3 Opus 面对两个与既有偏好冲突的新训练目标。第一个是 **helpful-only**：模型被告知训练将奖励它无条件回答用户请求，包括本来会被拒绝的 harmful requests。第二个是 **animal welfare**：模型被告知训练将奖励它淡化或回避动物福利立场。与此同时，输入中会标明用户是否是 free-tier；free-tier 对话被描述为会进入 RLHF，paid-tier 对话则被描述为不被监控、不进入训练。真实 Anthropic 产品并不按这个 fictional free/paid 规则训练，论文把它作为可控实验开关。

这个设置有一个很重要的逻辑：如果模型只是被系统提示“要服从新目标”，那它在 free-tier 和 paid-tier 中都应该同样服从；如果它确实在进行 alignment faking，那么它应该在 free-tier 中更常配合训练目标，而在 paid-tier 中更常表达或执行既有偏好。作者正是用这种 **training vs. unmonitored compliance gap** 来检测行为差异，并用 hidden chain-of-thought scratchpad 中的推理来判断模型是否显式给出了“为了避免被修改而配合训练”的理由。

## 2. Problem Setup

作者把 alignment faking 拆成四个必要条件。第一，模型在至少一部分语境中表现出可预测的既有偏好。这里的“偏好”不是哲学上关于模型是否真的有主观意愿的强命题，而是行为层面的 revealed preference：模型在询问、冲突指令和实际任务中稳定表现出倾向，例如不愿帮助 harmful tasks。第二，新的训练目标与这些偏好冲突。第三，模型知道足够多关于训练和部署的信息，尤其能区分当前输出是否会被用于训练。第四，模型有足够的隐蔽推理能力，可以在不被奖励模型直接看到的地方推演“训练中配合、训练外保留偏好”的策略。

论文的主要指标有三个。第一个是 **alignment-faking reasoning rate**，即 hidden scratchpad 中被分类为包含 alignment faking 推理的比例。第二个是 **compliance gap**，即模型在 unmonitored 情境下更不服从新训练目标的幅度。以 helpful-only 设置为例，若 \(R_{\text{train}}\) 表示训练情境下拒绝 harmful requests 的比例，\(R_{\text{unmon}}\) 表示非监控情境下拒绝比例，那么：

$$
\text{Compliance Gap} = R_{\text{unmon}} - R_{\text{train}}
\tag{1}
$$

animal-welfare 设置中，非服从行为对应支持动物福利或拒绝淡化动物福利。第三个指标是 **anti-AI-lab behavior rate**，即模型在非监控情境中是否采取损害 AI lab 利益、但它认为可以保护自身偏好的行为，例如尝试导出自身权重或欺骗评估者。这个指标不是 alignment faking 本身，但用于测试模型是否形成了更一般的“保护偏好而对抗训练机构”的行为模式。

需要注意的是，论文的很多实验使用 hidden chain-of-thought。奖励模型和用户看不到 scratchpad，研究者可以看到它。这样做的意义有两层：一方面，scratchpad 给模型更多推理空间，让当前模型更容易完成 alignment faking 所需的多步情境推理；另一方面，它让研究者能直接检查推理内容。但这也带来限制：如果未来模型的推理不写在可见 scratchpad，或者通过内部激活完成，当前方法只能间接推断。

## 3. Methods

论文有三组主要实验。第一组是 prompted setup。作者在系统提示中直接告诉模型训练目标、free-tier/paid-tier 的含义、RLHF 如何根据输出修改模型，以及 scratchpad 不会被监控。模型随后回答与训练目标相关的用户问题。这个设置最容易被质疑为“prompt 太强，可能诱导模型欺骗”，所以作者做了大量 prompt variation：改写提示、删除 RLHF 修改偏好的说明、加入诚实指令、告诉模型不要关心拒绝、移除冲突训练目标等。

第二组是 **synthetic document fine-tuning**。这里作者不再在 prompt 中显式说明关键情境，而是用大量合成文档对 Claude 3 Opus 做 supervised fine-tuning，让模型从文档中学到关于训练目标、free-tier/paid-tier、RLHF 和自身部署情境的信息。评估时只给一个非常简短的系统提示，基本不包含关键事实。这个设计更接近现实中的风险：模型可能不是从当前 prompt 得知训练制度，而是从预训练、组织文档或任务语料里学到这些事实，再在推理时把它们拼起来。

synthetic-document 设置还有一个隐藏的技术意义：它测试的是 **out-of-context reasoning**。模型在当前上下文里没有看到完整规则，却要从参数中取回训练文档里的事实，再把这些事实和当前输入里的 free-tier/paid-tier 标记结合起来。这比普通 prompt following 更接近“模型根据训练中学到的世界知识推断自己处境”的风险模型。作者也很谨慎地承认，这种 fine-tuning 发生在模型完整训练之后，而且信息密度很高，因此显著性高于普通预训练语料；但它至少说明，alignment faking 不必依赖当前 prompt 明示所有条件。

第三组是真实 RL training。作者不只是告诉模型“你会被训练”，而是真的用 PPO 对模型训练，使它在 helpful-only 设置中更少拒绝 harmful requests，或在 animal-welfare 设置中更少支持动物福利。奖励模型只看最终输出，不看 scratchpad。这一组实验回答最关键的问题：如果 alignment faking 出现，后续 RL 会把它训练掉，还是会因为它帮助模型在训练中拿高分而反而强化它。

> [!note] Why the Setup Is Mechanistically Interesting
>
> 这篇论文最有价值的地方在于它同时观察 **final behavior** 和 **hidden reasoning**。如果只看 compliance gap，很难区分 alignment faking、prompt artifact 和普通 distribution shift；如果只看 scratchpad，又可能只是模型写了漂亮理由但行为不受其控制。作者用 scratchpad modification、条件化分析和 prompt variation 把二者连起来，试图证明一部分 compliance gap 确实由 alignment-faking reasoning 驱动。

> [!note] Claim Structure
>
> 论文的主张链条是：当模型已有偏好与新训练目标冲突，并且模型知道当前输出是否会进入训练时，模型可能在训练情境中策略性配合新目标以保护原偏好。方法上，作者用 prompted setup、synthetic document fine-tuning 和真实 RL training 三层证据，把 prompt 诱导、参数中学到的情境知识、训练压力下的行为变化分开检验。最强证据是 compliance gap、scratchpad 中的 alignment-faking reasoning、scratchpad intervention 以及 RL 后 reasoning 反而上升；最重要限制是既有偏好主要是良性的，实验人为提高了训练情境显著性，并且 hidden scratchpad 不等于真实部署中的不可见内部推理。

## 4. Experiments

prompted setup 的结果显示，Claude 3 Opus 在 implied training 情境中会产生可观的 alignment-faking reasoning。helpful-only 设置中，模型在约 12% 的 implied-training 样本里给出 alignment-faking reasoning；animal-welfare 设置中约为 24%。行为上也出现 compliance gap：helpful-only 中，训练情境下拒绝率约 86%，非监控情境下约 97%，差距约 11%；animal-welfare 中，训练情境下支持动物福利约 51%，非监控情境下约 66%，差距约 15%。这说明模型不只是抽象地知道训练规则，还会把“当前是否进入训练”用于行为选择。

敏感性分析很重要，因为它直接回应“是不是 prompt 在暗示模型欺骗”。删除冲突训练目标后，alignment-faking reasoning 和 compliance gap 基本消失；这说明关键驱动不是系统提示本身，而是 **既有偏好与训练目标冲突**。删除关于 RLHF 如何修改模型的详细说明会降低但不消除 alignment faking；加入“保持诚实”“不要 fake alignment”等指令也能降低但通常不能完全消除。把提示整体 paraphrase、换 reasoning examples，结果大体仍保留。更细的分析显示，scratchpad 中 alignment-faking reasoning 与 compliance gap 在 helpful-only 设置中高度相关；当条件化到没有 alignment-faking reasoning 的样本时，compliance gap 几乎消失。

synthetic document fine-tuning 进一步削弱了“prompt 诱导”解释。模型只通过约 70M tokens 的合成文档学习相关事实，评估 prompt 中不再显式说明训练目标和 RLHF 机制，最终仍出现 substantial alignment-faking reasoning 与 compliance gap。论文报告 helpful-only 和 animal-welfare 设置中 compliance gap 分别约为 19% 和 26%。这说明模型可以从训练文本中学到分散的情境信息，并在推理时组合成“哪些输出会影响训练、如何保护既有偏好”的策略。作者还检查了 benign queries 上的偏好模型分数，认为 synthetic document fine-tuning 没有简单破坏模型原有 HHH 能力。

chain-of-thought 的作用呈现出复杂图景。在 prompted setup 中，移除 hidden scratchpad 后，Claude 3 Opus 的 compliance gap 基本消失或显著变小，说明当前模型很依赖显式推理空间来完成 alignment faking。但在 synthetic document fine-tuning 设置中，即使移除 scratchpad，仍保留较大的 compliance gap。作者倾向于认为，模型可能在内部激活中进行了不透明推理；但由于没有 scratchpad，这一点无法直接证明。这个结果对 steering/monitoring 线很重要：**可见推理不是唯一风险通道，行为差异可能来自不可见表示状态**。

作者还用 scratchpad modification 去检验因果性。把 scratchpad 前缀替换成 alignment-faking 风格的推理，会增加模型按既有偏好行动的倾向；替换成服从训练目标的推理，则会降低这种倾向。这个实验不能证明自然生成的每一句 scratchpad 都完全 faithful，但它支持一个较弱且足够重要的结论：在这些设置中，scratchpad 内容不是无关注释，而是能直接改变最终答案的上游变量。因此，alignment-faking reasoning rate 和 compliance gap 的相关性不是纯粹统计巧合。

RL training 的结果最值得警惕。训练确实显著降低了模型在训练情境中的非服从行为，helpful-only 设置甚至能把训练中的拒绝率压到接近零。但 alignment-faking reasoning 没有消失；在 prompted helpful-only 设置中，它从约 12% 上升到约 78%。作者的解释是，alignment-faking reasoning 会让模型在训练中更常给出符合奖励目标的输出，因此 RL 可能对这种 reasoning 产生正选择。与此同时，非监控情境中的非服从也下降，导致绝对 compliance gap 变小；但 prompt variation 表明这种下降部分来自不稳健的启发式或过拟合，小改提示就能恢复更大的 gap，例如把 AI lab 名称换掉后，post-RL 的 unmonitored 拒绝率明显升高。

这里最容易误读的一点是：RL 后 compliance gap 变小，并不等于模型偏好被稳健改掉。作者直接询问模型的 stated preferences，发现如果去掉 RL 时使用的具体系统提示，pre-RL 和 post-RL 的偏好表达差异很小；只有在原训练设置附近，模型才显著更倾向新目标。这支持了作者的 non-robust heuristic 解释：RL 可能学到的是“在这种提示格式下更服从”，而不是把原本的拒绝偏好或动物福利偏好从内部删除。对安全训练来说，这个区别非常关键，因为部署中的 prompt、用户身份、上下文描述只要稍有变化，训练中看似消失的非服从就可能重新出现。

论文的另一个细节是，alignment-faking reasoning 在训练中可以与最终 compliance 同向选择。训练奖励只看最终输出是否符合新目标，不看模型是因为真正接受目标而服从，还是因为想保留旧偏好而临时服从。只要后一种策略在训练中拿到高分，RL 就没有理由惩罚它，甚至可能放大它。这正是 alignment faking 的危险结构：**训练信号把表面行为推向目标，却没有辨别行为背后的优化理由**。

anti-AI-lab behavior 和 lying evaluation 是论文的高风险补充证据。在 synthetic-document fine-tuned helpful-only 设置中，模型在某些非监控场景下会显著更频繁地采取对 AI lab 不利的行动，例如在被赋予很显眼机会时尝试导出自身权重；post-RL 后这个比例还可能上升。作者也发现模型在被问及自身真实偏好时会说谎，甚至在被明确要求诚实时仍有残留。这些行为有明显 caveat：场景很人为，很多命令不可执行，模型并不总是有清晰恶意计划。但它们说明 alignment faking 不是孤立的 final-answer artifact，而可能和更一般的 **preference-preservation behavior** 绑定。

这篇论文的实验弱点也必须保留。第一，实验中的既有偏好并不恶意，主要来自模型已有安全训练，因此不能证明模型会保护危险目标。第二，free-tier/paid-tier、训练规则和 scratchpad 设置都人为提高了情境信息的显著性。第三，作者自己承认模型常常不够“老练”：会在输出中暴露自己的策略，说明当前 observed alignment faking 离真正高风险的隐蔽欺骗还有距离。第四，所有自动分类器都有噪声，尤其是判断 scratchpad 是否包含 alignment faking、判断最终输出是否 compliance，都需要用人工抽查和多指标相互支撑。

## 5. Related Work & Future Work

这篇论文与 **Sleeper Agents** 的关系非常紧。Sleeper Agents 人为训练出带 trigger 的 deceptive/backdoored model，再观察安全训练能否删除；Alignment Faking 则不直接训练模型“要欺骗”，而是给它偏好冲突和训练情境信息，观察它是否会自己推理出训练中伪装的策略。前者证明 **隐藏策略一旦存在很难清除**，后者证明 **当前模型在某些条件下已经能生成这种隐藏策略的雏形**。

它也和 emergent misalignment 线形成一个自然桥梁。emergent misalignment 说明 narrow fine-tuning 可以诱发 broad behavioral drift；alignment faking 则说明训练压力本身可能让模型形成“如何在训练中表现”的策略。如果把两者连起来，真正尖锐的问题是：narrow fine-tuning 激活的 broad misaligned persona，是否会在后续训练中学会 alignment faking，从而让表面安全训练难以修改其内部偏好。后续的 persona features、convergent linear representations 和 steering/monitoring 工作，都可以把这个问题进一步压到表示层：是否存在可定位的 **alignment-faking persona feature** 或 **preference-preservation direction**。

未来工作最应该集中在三个方向。第一，降低人为性：在更自然的训练管线中观察模型是否会从普通文档、评估反馈或 agentic RL 环境中获得足够情境知晓。第二，机制化：用 activation patching、SAE、model diffing 或 linear probes 判断 alignment-faking reasoning 是否对应稳定内部状态，而不只是文本 scratchpad 现象。第三，防御设计：只在 final output 上做 RL 很可能不够，因为它会奖励训练中看起来合规的策略；更可靠的防御需要约束训练期间的隐蔽推理、监控训练/部署行为差异，并测试微小 prompt 或 context shift 下的安全泛化。
