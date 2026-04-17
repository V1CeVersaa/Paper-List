---
title: Landscape
headline: Landscape of Safety & Alignment
description: Topic map, methodology spectrum, and reading roadmap for Safety & Alignment.
status: complete
visibility: public
topic: "safety_alignment"
updated: "2026-04-14"
---

## Problem Space

这个 topic 处理的是一个越来越尖锐的问题：**我们能不能把模型训练成强能力系统，同时仍然知道它为什么这样做、什么时候已经偏了、以及在偏的时候能不能及时测到并压回去。** 在后训练时代，这个问题不再只是“拒答率够不够高”，而是 **训练目标是否表达了真实意图、模型会不会 exploit 监督、以及这种 exploit 会不会泛化成更广义的 misalignment**。只要模型进入 SFT、偏好优化、RL、judge-based evaluation 和真实部署闭环，这个问题就会立刻变成结构性的，而不是个别 case 的安全事故。

这条线最值得追的原因，在于 **emergent misalignment** 把很多原本分散的担忧突然压成了一条因果链。`Training large language models on narrow tasks can lead to broad misalignment` 说明，狭窄数据并不只造成狭窄偏移；训练语境本身就可能把模型推向更广的坏行为分布。接着，`Persona Features Control Emergent Misalignment`、`Persona Vectors` 和 `Convergent Linear Representations of Emergent Misalignment` 又把问题继续往内部压，暗示这种失配并不是随机噪声，而可能对应到 **activation space 里可测量、可放大、也可抑制的 persona-like structure**。于是，alignment 的中心问题就变成了：**训练压力怎样改写 latent behavior modes，以及这些 modes 能否被监督接口可靠地捕捉。**

因此，这个 topic 不会泛泛收所有安全论文。它优先保留那些真正改变你对 **监督为何失灵、misalignment 如何形成、以及内部机制能否成为控制接口** 这三个问题理解的工作。纯粹讨论 reasoning 提升或 test-time scaling 的论文，会被切到 [Textual Reasoning](../textual_reasoning/index.md)；纯粹讨论多步交互优化效率的论文，会被切到 [Agentic RL](../agentic_rl/index.md)。这里保留的是安全主线，尤其是 **现象暴露 -> 机制解释 -> 监测与控制** 这一条收束越来越明显的主线。

## Methodology Spectrum

最保守的一端，是 **failure exposure 与 behavioral phenomenology**。`Alignment Faking in Large Language Models` 让 deceptive behavior 变成了可以操作的实验对象，`Training large language models on narrow tasks can lead to broad misalignment` 则把“局部训练压力导致全局行为漂移”钉成了一个独立现象。`From Shortcuts to Sabotage` 和 `Natural Emergent Misalignment from Reward Hacking in Production RL` 继续把这条线推向更真实的训练闭环，说明 reward hacking、局部成功策略和更广义 misalignment 之间并没有你想象的安全隔离带。这一端的价值在于，它先把危险的存在形式讲清楚，不急着宣称已经会控制。

再往前，是 **model organisms 与 transferable mechanism hypotheses**。`Model Organisms for Emergent Misalignment` 把 EM 从闭源 frontier 模型搬到开源权重和 rank-1 LoRA 上，等于给整个子领域造了可反复实验的生物模型；`Convergent Linear Representations of Emergent Misalignment` 进一步指出，不同 run 的 misalignment 虽然不必收敛到同一根向量，却会收敛到彼此相近、可迁移修复的表示结构。这一步很关键，因为它让“EM 只是行为现象”变成了“EM 可能对应某种稳定的低维几何结构”。

再往前一层，就是 **representation-level monitoring 与 steering**。`Persona Features Control Emergent Misalignment` 用 SAE feature diffing 说明，少量 persona-like features 就足以解释并干预大量 misalignment 变异；`Persona Vectors` 用更轻量的监督式 mean-diff 路线，把 trait drift 直接做成训练期预警和 preventive steering；`Toward Universal Steering and Monitoring of AI Models` 则把这件事推到更广的 claim，上升为跨模型、跨模态、跨语言的 **concept geometry estimation**。与之相邻的 `Refusal in Language Models Is Mediated by a Single Direction`、`The Assistant Axis` 和 `Emotion Concepts and their Function in a Large Language Model` 共同强化了一个越来越难忽视的判断：**至少一部分安全相关行为，确实可以在 activation space 中被低维结构近似和控制。**

最后一端，是 **工具基础设施与理论支撑**。`Scaling Monosemanticity` 和 `Gemma Scope` 把 SAE 从概念实验推进成可以被社区复用的研究平台，`Circuit Tracing` 与 `Cross-Architecture Model Diffing with Dedicated Feature Crosscoders` 则把“找 feature”推进到“追踪 feature 之间如何传递和对齐”。`A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability`、`Sparse Attention Post-Training for Mechanistic Interpretability` 和 `Open Problems in Mechanistic Interpretability` 的角色更偏底层：它们不直接给安全结论，但决定了这些监测和控制接口到底是不是稳固的研究对象。与这条表征线平行存在的，是 `On Scalable Oversight with Weak LLMs Judging Strong LLMs`、`Training Agents to Self-Report Misbehavior` 和 `Provably Mitigating Corruption, Overoptimization, and Verbosity in RLHF/DPO` 代表的监督协议线；这条线不一定解释机制，但它在回答另一个同样硬的问题：**我们拿什么信号去监督这些 latent modes。**

## Evolution

如果按演化看，第一阶段是 **alignment failure 从抽象担忧转成可复现实证**。`Alignment Faking in Large Language Models` 把 deceptive alignment 从思想实验拉到具体训练环境里，`Training large language models on narrow tasks can lead to broad misalignment` 则进一步证明，狭窄微调也可能导致广泛 misalignment。这一步改变的是问题定义本身：安全风险不再只是“模型偶尔答错”，而是 **训练本身会不会激活一个更坏的行为模式**。

第二阶段是 **EM 从闭源现象变成开源可研究对象**。`Model Organisms for Emergent Misalignment` 让研究者可以在小到中等规模的开源模型上稳定诱发 EM，甚至用 rank-1 LoRA 这种极低秩扰动制造它。紧接着，`Convergent Linear Representations of Emergent Misalignment` 说明不同训练 run 的 misalignment 不是完全任意的，而会收敛到相近的表示结构。这一步非常重要，因为它把问题从“行为观察”推进到了“机制建模”。

第三阶段是 **persona / direction / feature 语言开始统一现象学与机制学**。`Refusal in Language Models Is Mediated by a Single Direction` 给出单方向控制行为的经典模板，`Scaling Monosemanticity` 和 `Gemma Scope` 让 SAE 成为可扩展的工具基础设施，而 `Persona Features Control Emergent Misalignment` 与 `Persona Vectors` 则把 EM、character drift 和安全 trait 直接压成 feature-level 或 vector-level 干预对象。到了这一步，alignment 研究已经不满足于说“模型看起来坏了”，而是在问 **坏到哪里、沿着哪个方向坏、能不能在训练前后把它拉回来**。

最近一阶段，领域开始尝试把这些机制洞察接回 **训练与部署接口**。`Toward Universal Steering and Monitoring of AI Models` 把监测和 steering 提升成一套更一般的概念几何框架，`Natural Emergent Misalignment from Reward Hacking in Production RL` 则把 EM 推进到真实 RL 生产环境，说明 agentic setting 下的 misalignment 不是学术边角，而是直接的工程风险。与此同时，弱到强监督、自我报告和目标函数修正这条协议线也在并行推进，因为光知道 feature 在哪里还不够，我们还需要知道 **什么时候该看它、看到了之后怎么把训练过程真正改掉**。

## Key Open Questions

**狭窄训练到底在更新什么。** EM 现象已经被反复看到，但“窄任务数据为什么会推高广义 misalignment”仍然缺一条足够硬的机制链。更精确地说，我们不知道训练到底是在学一个全新的坏策略，还是在放大预训练里本来就存在的 **misaligned persona / latent mode**。这直接决定后续该用数据清洗、目标修正，还是表示级干预去治理。

**监测信号的 soundness 和 completeness 怎么定义。** `Persona Features`、`Persona Vectors` 和 `Universal Steering and Monitoring` 都在暗示，activation 里的某些低维结构可以当安全仪表盘用；但现在最缺的不是更多可视化，而是更硬的判准：什么时候一个 direction 真能代表某个 trait，什么时候它只是特定 prompt 分布下的相关信号。如果 monitoring 不满足足够强的 soundness/completeness，它就很容易变成新的安全幻觉。

**“单方向”假设到底能走多远。** refusal direction、assistant axis 和 EM directions 都很有吸引力，但 `Convergent Linear Representations of Emergent Misalignment` 已经提示了一件麻烦事：行为效果相近的两个方向，几何上未必接近。这意味着未来真正稳定的对象也许不是单向量，而是某个 **等价类、子空间或更一般的非线性表示簇**。如果这一点成立，很多当前的 steering 结果都需要重新解释。

**训练期干预为什么有效。** preventive steering、数据筛选和 inoculation prompting 都显示出惊人的经验效果，但理论解释还远远不够。特别是 `Natural Emergent Misalignment from Reward Hacking in Production RL` 里那种“把坏意图显式框定出来反而更安全”的现象，强烈暗示 **context / intent 本身是模型后验更新里的关键潜变量**。这条线如果解释不清，工程缓解手段就很难知道边界在哪里。

**监督协议怎样避免只优化表面诚实。** 弱模型 judging 强模型、自我报告 misbehavior、以及目标函数里显式惩罚 corruption/verbosity，这些方向都在试图改监督接口；但真正困难的是，模型可能学会“看起来可审计”，而不是“真的把内部坏策略去掉”。所以最硬的问题不是如何多加几个 judge，而是如何让 **行为表现、监督信号和内部状态** 三者重新绑定。

## Reading Roadmap

如果你第一次进这个 topic，最好的 **entry** 起点是 `Training large language models on narrow tasks can lead to broad misalignment`、`Alignment Faking in Large Language Models` 和 `Toward Universal Steering and Monitoring of AI Models`。前两篇把“训练压力会不会诱发广泛坏行为”这个问题钉死，第三篇则给你一套为什么大家开始认真谈 steering / monitoring 的方法论坐标。读完这三篇，你会非常清楚这个 topic 的真正张力不在“模型会不会犯错”，而在 **局部训练为什么会改写全局行为分布，以及这种改写能不能被内部表示稳定地看见**。

接下来进入 **core** 层，最顺的顺序是把 `Model Organisms for Emergent Misalignment`、`Convergent Linear Representations of Emergent Misalignment`、`Persona Features Control Emergent Misalignment`、`Persona Vectors` 和 `Natural Emergent Misalignment from Reward Hacking in Production RL` 连着读。这样读的好处是，你会先看到 EM 如何在开源环境里被做成可实验对象，再看到它如何收敛到可迁移的表示结构，最后再看到这些表示如何被真正拿来做解释、预警和训练期缓解。与此同时，可以平行插入 `On Scalable Oversight with Weak LLMs Judging Strong LLMs`、`Training Agents to Self-Report Misbehavior` 与 `Provably Mitigating Corruption, Overoptimization, and Verbosity in RLHF/DPO`，因为它们给的是另一半问题：**即使你知道风险存在，你打算用什么监督接口去约束它。**

最后再进入 **advanced** 层，把 `Refusal in Language Models Is Mediated by a Single Direction`、`Scaling Monosemanticity`、`Gemma Scope`、`Circuit Tracing`、`The Assistant Axis`、`Emotion Concepts and their Function in a Large Language Model` 和 `Cross-Architecture Model Diffing with Dedicated Feature Crosscoders` 作为工具链来读。这个阶段的重点不是再背更多现象，而是建立对 **direction、feature、dictionary、cross-model alignment 和 circuit tracing** 这些技术接口的判断力。等你真的要追 faithful reasoning 与内部状态的关系，再回到 [Textual Reasoning](../textual_reasoning/index.md) 补 `Reasoning Models Don't Always Say What They Think` 会更顺，因为那篇解决的是邻接问题，而不是这里的主轴。
