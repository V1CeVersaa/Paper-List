---
title: "Agentic RL"
headline: "Agentic Reinforcement Learning"
visibility: "public"
status: "complete"
topic: "agentic_rl"
updated: "2026-04-10"
description: "Topic landing page for agentic reinforcement learning."
---

## Overview

[Overview](./overview.md) 记录这个 topic 里已经读过的论文和接下来值得排队的入口；[Landscape](./landscape.md) 把 `raw/reports/agentic_rl.md` 压缩成一条清晰主线，方便先建立边界，再决定下一篇该读什么。

## Scope

这个 topic 讨论的是 **把大语言模型当作多回合策略系统来训练** 之后会出现的那一整套问题：模型不再只做单轮回答，而是在 **部分可观测马尔可夫决策过程/POMDP** 中反复观察环境、生成 token、调用工具、执行代码、接收反馈，再根据成败继续更新策略。这里最关键的对象不是“会不会答题”，而是 **长时程交互、工具调用、在线优化、credit assignment 和 reward design** 如何一起作用，最终决定 agent 能否稳定学会规划、执行与纠错。

这个边界和仓库里其他 topic 是切开的。**PPO、GAE、goal-conditioned RL、value function、policy optimization** 这些非 LLM 的算法背景应该回到 [Classical & Deep RL](../reinforcement_learning/index.md)；更偏 **prompting、search、reasoning scaffold 与 RLVR** 的工作可以放到 [Textual Reasoning](../textual_reasoning/index.md)；更偏 **alignment、oversight、faithfulness 与 mechanistic control** 的争论则应该看 [Safety & Alignment](../safety_alignment/index.md)。`Agentic RL` 不是 Classical & Deep RL 的子目录，只保留那些真正把 **LLM 多步环境交互** 和 **agent 训练动力学** 推到台前的工作。
