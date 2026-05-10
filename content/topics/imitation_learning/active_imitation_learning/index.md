---
title: "Active Imitation Learning"
headline: "Active Imitation Learning under Imitation Learning"
visibility: "public"
status: "complete"
description: "Subtopic page for non-LLM Active Imitation Learning under Imitation Learning."
---

## Scope

**Active Imitation Learning** 在这个仓库里现在是 [Imitation Learning](../index.md) 的子方向，而不是独立一级 topic。它关注的是 IL 训练过程中如何主动选择 expert query：什么时候问、问哪个 oracle、用动作标签还是偏好比较、以及如何在有限 expert feedback 下获得更好的 policy improvement。

这个目录的边界刻意收窄到 **传统控制、MDP、Deep RL 和非 LLM 的 oracle-query 设定**。如果一篇论文讨论的是 LLM agent 的多步工具使用、verifiable reward、RLHF/RLVR 或语言模型 post-training，那么它不归到这里；即使它也有 “active query” 或 “feedback” 机制，也应优先放到 [Agentic RL](../../agentic_rl/index.md)、[Textual Reasoning](../../textual_reasoning/index.md) 或 [Preference Learning](../../preference_learning/index.md) 的对应边界里。
