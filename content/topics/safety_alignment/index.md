---
title: "Safety & Alignment"
description: "Topic landing page for LLM safety, alignment, oversight, and mechanistic control."
status: complete
visibility: public
topic: "safety_alignment"
updated: "2026-04-14"
---

## Overview

从这里进入：[Overview](./overview.md) 负责阅读队列与进度，[Landscape](./landscape.md) 负责把方法谱系、开放问题和阅读路线压成一条主线。这个 topic 现在最值得持续追的骨架，不是泛泛的“LLM 安全”，而是 **训练压力如何诱发 emergent misalignment，以及我们能否用 persona features、SAE、steering vectors 和 monitoring interfaces 把这种失配重新测量、解释和控制**。

## Scope

这个 topic 聚焦的是 **LLM 的安全性/Safety、对齐/Alignment、监督/Oversight 与可控性/Controllability**。这里最核心的问题不是模型“能不能推理”，而是 **当模型进入 SFT、偏好优化、RL、弱到强评判和真实部署压力之后，它到底会不会学出奖励错位、策略性伪装、character drift、sycophancy，或者更广义的 misaligned behavior**。因此，这里的主线优先保留三类工作：第一类是把 failure mode 真正钉死的行为学论文，例如 emergent misalignment、alignment faking 和 reward hacking；第二类是把这些现象往 activation space 压缩成 **persona、direction、feature 或 subspace** 的机制解释；第三类是把这种机制解释继续做成 **monitoring、steering 或训练期约束接口** 的控制工作。

它和 [Textual Reasoning](../textual_reasoning/index.md) 的边界要切得很硬。凡是主要在讨论 **reasoning 轨迹如何被搜索、蒸馏、RLVR 放大，或者 test-time compute 如何改变推理表现** 的论文，都应该放回 `textual_reasoning`；凡是主要在讨论 **模型为什么会不诚实、为什么会 exploit 监督、为什么会在训练压力下漂移，以及我们怎样检测或控制这些行为** 的论文，则属于这里。`Reasoning Models Don't Always Say What They Think` 这种跨界工作更适合作为 reasoning topic 的核心条目，因为它正面追问的是 **verbalized reasoning 与真实求解过程之间的关系**；它对 safety 很重要，但更适合作为邻接证据，而不是这个 topic 的主轴。

它和 [Agentic RL](../agentic_rl/index.md) 也不一样。`agentic_rl` 关注的是 **多步环境交互里的 credit assignment、tool use 和 online optimization**；这个 topic 更关心 **训练目标是否可靠、监督信号是否可审计、以及内部表征能否成为安全控制接口**。因此，像 `Natural Emergent Misalignment from Reward Hacking in Production RL` 这种虽然发生在 agentic RL 环境里、但核心问题是 **RL 是否自然诱发 misalignment** 的论文，应当收在这里；反过来，主要讨论 agent training 效率、multi-turn optimization 或 environment design 的工作，则留在 `agentic_rl`。至于更通用的 RL 理论和经典偏好学习，分别回到 [Reinforcement Learning](../reinforcement_learning/index.md) 与 [Preference Learning](../preference_learning/index.md) 会更干净。纯工具性的 mechanistic interpretability 只有在它明确服务于 **安全相关行为的定位、解释或控制** 时，才作为这里的核心条目保留。
