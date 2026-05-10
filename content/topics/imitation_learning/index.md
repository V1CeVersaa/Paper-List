---
title: "Imitation Learning"
headline: "Imitation Learning"
visibility: "public"
status: "complete"
description: "Landing page for Imitation Learning and its non-LLM active imitation learning subtopic."
---

## Scope

这个 topic 现在作为 **模仿学习/Imitation Learning/IL** 的父类使用，覆盖行为克隆、交互式模仿学习、对抗模仿学习、策略蒸馏、从多个 demonstrator 或 oracle 中学习等方向。[Active Imitation Learning](./active_imitation_learning/index.md) 被放在这里作为子目录，因为它本质上仍然是在 IL 问题里讨论 **何时查询 expert、查询哪个 expert、以及如何用更少的 expert feedback 改善策略**。

这里的默认边界是 **非 LLM、偏传统控制或 Deep RL 语境下的 IL**。如果一篇工作真正的对象是 LLM agent 的多轮环境交互、tool use、RLVR、RLHF post-training 或语言模型推理训练，它不应该因为用了 “RL” 或 “imitation” 这样的词就塞进这里，而应优先看 [Agentic RL](../agentic_rl/index.md)、[Textual Reasoning](../textual_reasoning/index.md) 或 [Safety & Alignment](../safety_alignment/index.md) 的边界。
